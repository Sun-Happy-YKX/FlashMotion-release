"""Build a separate complete-clip HuBERT/fastText cache; never mutate BEAT2.

Requires explicit local model/vocabulary assets. Missing assets fail closed,
not synthetic audio, empty text, or covert model downloads.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import pickle
from pathlib import Path
import numpy as np

from data import EXPECTED_COUNTS, PERSON_IDS, atomic_json, build_manifest, sha256, train_statistics, roundtrip_audit


class _Vocabulary:
    pass


class VocabularyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # The released BEAT2 asset was serialized by executing build_vocab
        # as a script, so its inert state holder is named __main__.Vocab.
        # Resolve only this exact legacy name; never import pickle globals.
        if name == "Vocab" and (module == "__main__" or module.endswith("build_vocab")):
            return _Vocabulary
        if module in ("numpy", "numpy.core.multiarray", "numpy._core.multiarray") and name in ("ndarray", "dtype", "_reconstruct", "scalar"):
            return super().find_class(module, name)
        if module in ("numpy.core.numeric", "numpy._core.numeric") and name == "_frombuffer":
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Unsupported vocabulary type: {module}.{name}")


def read_vocabulary(path):
    with Path(path).open("rb") as handle:
        vocab = VocabularyUnpickler(handle).load()
    weights = np.asarray(vocab.word_embedding_weights, dtype=np.float32)
    assert weights.ndim == 2 and weights.shape[1] == 300 and np.isfinite(weights).all()
    assert isinstance(vocab.word2index, dict)
    assert all(0 <= int(i) < len(weights) for i in vocab.word2index.values())
    return vocab.word2index, weights


def frame_text(path, frames, mapping, embeddings):
    import textgrid
    grid = textgrid.TextGrid.fromFile(str(path))
    assert len(grid) > 0
    tokens = np.zeros(frames, dtype=np.int64)
    for interval in grid[0]:
        word = interval.mark
        for punctuation in (",", ".", "?", "!"):
            word = word.replace(punctuation, " ")
        start = max(0, int(np.ceil(interval.minTime * 30)))
        end = min(frames, int(np.ceil(interval.maxTime * 30)))
        tokens[start:end] = mapping.get(word, 3)
    return embeddings[tokens], tokens


def hubert_features(model, processor, speech, frames, device):
    import torch
    import torch.nn.functional as F
    # SemTalk's long-waveform extraction uses 320 stride and 400 kernel.
    values = processor(speech, sampling_rate=16000, return_tensors="pt").input_values.to(device)
    stride, kernel, clip = 320, 400, 320000
    expected = (values.shape[-1] - (kernel - stride)) // stride
    assert expected > 0
    pieces = []
    with torch.inference_mode():
        for start in range(0, values.shape[-1], clip):
            segment = values[:, start:min(start+clip-stride+kernel, values.shape[-1])]
            if segment.shape[-1] >= kernel:
                pieces.append(model(segment).last_hidden_state[0].float())
        result = torch.cat(pieces, dim=0)
        assert abs(len(result)-expected) <= 1, (len(result), expected)
        result = F.pad(result, (0, 0, 0, max(0, expected-len(result))))[:expected]
        result = F.interpolate(result.T[None], size=frames, mode="linear", align_corners=True)[0].T
    assert result.shape == (frames, 1024) and torch.isfinite(result).all()
    return result.cpu().numpy().astype(np.float16)


def atomic_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def audit_evaluation_budgets(manifest, cache_root):
    """Reject unsupported val/test BC budgets before training, not epoch50.

    This is a coverage check, NOT a completed metric or actual data result.
    Training clips can be much shorter: masked padding retains every clip.
    """
    from evaluation import ALIGN_MASK, rollout_frames
    reports = {}
    for split in ("val", "test"):
        reports[split] = []
        for ident in manifest["splits"][split]:
            with np.load(Path(cache_root) / (ident + ".npz"), allow_pickle=False) as z:
                frames = len(z["poses"])
                if z["audio"].shape != (frames, 1024) or z["text"].shape != (frames, 300):
                    raise ValueError(f"Feature coverage differs from complete motion: {ident}")
            evaluation_frames = rollout_frames(frames)
            if evaluation_frames <= 2*ALIGN_MASK:
                raise ValueError(f"{split} {ident}: {frames} source frames gives {evaluation_frames} rollout frames; official BC's +/-60-frame crop is unsupported. No silent omission; review evaluation protocol before training.")
            reports[split].append({"id": ident, "source_frames": frames,
                                   "evaluation_frames": evaluation_frames,
                                   "fgd_frames": evaluation_frames//32*32})
    return {"passed": True, "kind": "complete_clip_frame_budget_inventory_not_metrics", "splits": reports,
            "split_counts": {key: len(records) for key, records in reports.items()},
            "minimum_frames_by_split": {key: min((r["evaluation_frames"] for r in records), default=0)
                                         for key, records in reports.items()}}


def bc_audio_onset_evidence(audio_path, frames):
    """CPU-only audio path, exactly matching both branches in evaluator.evaluate.

    The legacy original-WAV end argument is intentional. This is an onset
    availability audit, not BC scoring, generated motion, or metric completion.
    """
    import librosa
    import soundfile as sf
    from evaluation import ALIGN_MASK, AUDIO_SR, FPS

    audio_path = Path(audio_path).resolve()
    if type(frames) is not int or frames <= 2 * ALIGN_MASK or frames % 4:
        raise ValueError(f"Unsupported BC audio frame budget: {frames}")
    audio_info = sf.info(audio_path)
    if audio_info.frames <= 0 or audio_info.samplerate <= 0:
        raise ValueError(f"Missing/nonempty audio contract failed: {audio_path}")
    wave_legacy, legacy_rate = librosa.load(audio_path)
    wave_legacy = librosa.resample(wave_legacy, orig_sr=legacy_rate, target_sr=AUDIO_SR)
    offset = int(ALIGN_MASK * AUDIO_SR / FPS)
    legacy_segment = wave_legacy[:int(frames * AUDIO_SR / FPS)][offset:len(wave_legacy) - offset]
    wave_corrected = librosa.load(audio_path, sr=AUDIO_SR)[0]
    corrected_segment = wave_corrected[offset:int((frames - ALIGN_MASK) * AUDIO_SR / FPS)]
    segments = {"legacy": legacy_segment, "corrected": corrected_segment}
    evidence = {}
    for protocol, segment in segments.items():
        if segment.ndim != 1 or not len(segment) or not np.isfinite(segment).all():
            raise ValueError(f"{audio_path.name}: {protocol} BC audio segment is empty/nonfinite; no silent omission")
        onsets = librosa.onset.onset_detect(y=segment, sr=AUDIO_SR, hop_length=512, units="time")
        if not len(onsets) or not np.isfinite(onsets).all():
            raise ValueError(f"{audio_path.name}: {protocol} BC has no finite detected audio onsets; no silent zero or omission")
        evidence[protocol] = {"sample_count": len(segment), "onset_count": len(onsets),
                              "first_onset_seconds": float(onsets[0]), "last_onset_seconds": float(onsets[-1])}
    return {"audio_path": str(audio_path), "audio_sha256": sha256(audio_path),
            "original_audio_samples": int(audio_info.frames), "original_audio_sample_rate": int(audio_info.samplerate),
            "evaluation_frames": frames, **evidence}


def audit_bc_audio_onsets(manifest, raw_root, budgets, *, manifest_path, evaluation_budget_path):
    """Require both actual audio-onset branches for every val106/test265 clip.

    No CUDA imports/context, motion generation, metric values, fallback, or
    sample removal. All IDs and frame budgets are bound to their source JSONs.
    """
    import librosa
    import soundfile as sf
    import evaluation

    manifest_path, evaluation_budget_path = Path(manifest_path).resolve(), Path(evaluation_budget_path).resolve()
    manifest_hash = sha256(manifest_path)
    if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("BC audio audit manifest differs from its on-disk evidence")
    if json.loads(evaluation_budget_path.read_text(encoding="utf-8")) != budgets:
        raise ValueError("BC audio audit budgets differ from their on-disk evidence")
    if budgets.get("passed") is not True or budgets.get("manifest_sha256") != manifest_hash:
        raise ValueError("BC audio audit needs the current passing manifest-bound frame inventory")
    expected_counts = {split: EXPECTED_COUNTS[split] for split in ("val", "test")}
    if budgets.get("split_counts") != expected_counts:
        raise ValueError("BC audio audit must cover the full val106/test265 inventory")
    raw_root = Path(raw_root).resolve()
    reports, seen_ids = {}, set()
    for split in ("val", "test"):
        records = evaluation.manifest_records(manifest, split, EXPECTED_COUNTS[split])
        budget_records = budgets.get("splits", {}).get(split, [])
        if [record.get("id") for record in budget_records] != [record["id"] for record in records]:
            raise ValueError(f"BC audio budget IDs omit/reorder {split} clips")
        reports[split] = []
        for record, budget in zip(records, budget_records):
            ident, frames = record["id"], budget.get("evaluation_frames")
            if ident in seen_ids:
                raise ValueError(f"Cross-split duplicate in BC audio inventory: {ident}")
            seen_ids.add(ident)
            if type(budget.get("source_frames")) is not int or frames != evaluation.rollout_frames(budget["source_frames"]):
                raise ValueError(f"BC audio budget does not match the complete cached clip: {ident}")
            value = bc_audio_onset_evidence(raw_root / "wave16k" / (ident + ".wav"), frames)
            with np.load(raw_root / "smplxflame_30" / (ident + ".npz"), allow_pickle=False) as motion:
                available = min(len(motion["poses"]) // evaluation.FPS,
                                math.floor(value["original_audio_samples"] / value["original_audio_sample_rate"])) * evaluation.FPS
            if evaluation.rollout_frames(available) != frames:
                raise ValueError(f"Raw evaluator and complete-cache BC frame budgets differ: {ident}")
            reports[split].append({"id": ident, "source_frames": budget["source_frames"], **value})
    return {"schema": 1, "passed": True, "kind": "complete_val_test_bc_audio_onset_inventory_not_metrics",
            "splits": reports, "split_counts": expected_counts, "total_clips": len(seen_ids),
            "manifest_sha256": manifest_hash, "evaluation_budget_sha256": sha256(evaluation_budget_path),
            "evaluation_source_sha256": sha256(evaluation.__file__), "prepare_source_sha256": sha256(__file__),
            "raw_data_root": str(raw_root), "cuda_used": False, "motion_predictions_used": False,
            "metric_values_computed": False, "audio_sample_rate": evaluation.AUDIO_SR,
            "hop_length": 512, "onset_units": "time", "align_mask": evaluation.ALIGN_MASK,
            "protocol": {"legacy": "librosa default load 22050->16000; generated-length trim then original-WAV end argument",
                         "corrected": "librosa direct 16000; synchronized +/-60-frame crop"},
            "dependency_versions": {"librosa": librosa.__version__, "soundfile": sf.__version__, "numpy": np.__version__},
            "minimum_onsets_by_protocol": {protocol: min(record[protocol]["onset_count"]
                                          for records in reports.values() for record in records)
                                          for protocol in ("legacy", "corrected")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--hubert", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--smplx", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--statistics-only", action="store_true")
    args = parser.parse_args()
    assert 0 <= args.shard < args.shards
    assert args.cache_root.resolve() != args.raw_root.resolve()
    assert args.hubert.is_dir() and args.vocab.is_file() and args.smplx.is_dir()
    args.cache_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.cache_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        assert Path(manifest["raw_data_root"]).resolve() == args.raw_root.resolve()
        assert manifest["split_csv_sha256"] == sha256(args.raw_root / "train_test_split.csv")
    elif args.shard == 0:
        manifest = build_manifest(args.raw_root, manifest_path)
    else:
        raise RuntimeError("Create and audit manifest once before parallel shards")
    if not args.statistics_only:
        import librosa
        import smplx
        import torch
        from smplx.lbs import blend_shapes, vertices2joints
        from transformers import HubertModel, Wav2Vec2Processor
        processor = Wav2Vec2Processor.from_pretrained(args.hubert, local_files_only=True)
        audio_model = HubertModel.from_pretrained(args.hubert, local_files_only=True).eval().requires_grad_(False).to(args.device)
        smpl = smplx.create(str(args.smplx), model_type="smplx", gender="NEUTRAL_2020",
                            num_betas=300, num_expression_coeffs=100, use_pca=False,
                            use_face_contour=False, ext="npz").eval().requires_grad_(False).to(args.device)
        mapping, embeddings = read_vocabulary(args.vocab)
        ids = sum((manifest["splits"][split] for split in ("train", "val", "test")), [])
        metadata = {"hubert_model": str(args.hubert), "vocab_sha256": sha256(args.vocab),
                    "fps": 30, "features": "HuBERT-Large1024 + released fastText300 frame words",
                    "audio_resampling": "official long HuBERT chunks then linear align_corners=True",
                    "no_emotion_or_sentence_clip_condition": True, "no_motion_seed_condition": True,
                    "geometry_schema": "unposed-shaped-rest-v2; official axis-angle pose_mean applied only inside FK"}
        atomic_json(args.cache_root / f"feature_source_shard{args.shard}.json", metadata)
        for ident in ids[args.shard::args.shards]:
            path = args.cache_root / (ident + ".npz")
            if path.exists():
                with np.load(path, allow_pickle=False) as existing:
                    assert str(existing["source_id"].item()) == ident
                    assert str(existing["vocab_sha256"].item()) == metadata["vocab_sha256"]
                    assert str(existing["hubert_model"].item()) == str(args.hubert)
                    assert str(existing["geometry_schema"].item()) == metadata["geometry_schema"]
                    assert existing["pose_mean"].shape == (165,)
                continue
            with np.load(args.raw_root / "smplxflame_30" / (ident + ".npz"), allow_pickle=False) as archive:
                pose = np.asarray(archive["poses"], dtype=np.float32)[:, :165]
                trans = np.asarray(archive["trans"], dtype=np.float32)
                betas = np.asarray(archive["betas"], dtype=np.float32).reshape(-1)
                if "mocap_frame_rate" in archive:
                    assert float(archive["mocap_frame_rate"].item()) == 30
            speech, _ = librosa.load(args.raw_root / "wave16k" / (ident + ".wav"), sr=16000, mono=True)
            # Whole common seconds as SemTalk loader; no max1200 truncation.
            seconds = min(len(pose)//30, len(trans)//30, len(speech)//16000)
            frames = seconds * 30
            assert frames >= 30, (ident, frames)  # Keep short train clips; masked padding handles them.
            pose, trans = pose[:frames], trans[:frames]
            # Raw AA rotations with nonzero SMPL-X pose_mean must be on a
            # unique branch: adding the mean after an AA +/-2pi wrap is not
            # equivalent. Reject that data, never silently canonicalize it.
            pose_mean = smpl.pose_mean.detach().cpu().numpy().reshape(165)
            mean_active = np.linalg.norm(pose_mean.reshape(55, 3), axis=1) > 1e-8
            if mean_active.any():
                max_mean_active_angle = float(np.linalg.norm(pose.reshape(-1, 55, 3)[:, mean_active], axis=2).max())
                if max_mean_active_angle >= np.pi - 1e-5:
                    raise ValueError(f"{ident}: raw mean-active hand AA crosses pi ({max_mean_active_angle}); 6D plus mean is not invertible here. Representation must be reviewed, no silent wrap.")
            else:
                max_mean_active_angle = 0.
            speech = speech[:seconds*16000]
            audio = hubert_features(audio_model, processor, speech, frames, args.device)
            text, word_index = frame_text(args.raw_root / "textgrid" / (ident + ".TextGrid"), frames, mapping, embeddings)
            betas = np.pad(betas, (0, max(0, 300-len(betas))))[:300]
            with torch.inference_mode():
                shape = torch.as_tensor(betas[None], device=args.device)
                shaped = smpl.v_template[None] + blend_shapes(shape, smpl.shapedirs[:, :, :300])
                rest = vertices2joints(smpl.J_regressor, shaped)[0, :55].cpu().numpy()
                pose_mean = smpl.pose_mean.detach().cpu().numpy().reshape(165)
            atomic_npz(path, poses=pose, trans=trans, audio=audio, text=text.astype(np.float16),
                       word_index=word_index, betas=betas, rest_joints=rest.astype(np.float32), pose_mean=pose_mean,
                       geometry_schema=np.asarray(metadata["geometry_schema"]),
                       max_mean_active_raw_angle=np.asarray(max_mean_active_angle),
                       source_id=np.asarray(ident), person_id=np.asarray(int(ident.split("_", 1)[0])),
                       vocab_sha256=np.asarray(metadata["vocab_sha256"]), hubert_model=np.asarray(str(args.hubert)),
                       height_axis=np.asarray("y"), rotation_space=np.asarray("local"))
            print(json.dumps({"prepared": ident, "frames": frames, "shard": args.shard}), flush=True)
        atomic_json(args.cache_root / f"features_complete_shard{args.shard}.json", {"shard": args.shard, "shards": args.shards, "assigned": len(ids[args.shard::args.shards])})
    else:
        ids = sum((manifest["splits"][s] for s in ("train", "val", "test")), [])
        assert all((args.cache_root / (i + ".npz")).exists() for i in ids), "Incomplete feature cache"
        budgets = audit_evaluation_budgets(manifest, args.cache_root)
        budgets["manifest_sha256"] = sha256(manifest_path)
        budgets_path = args.cache_root / "evaluation_budget_audit.json"
        atomic_json(budgets_path, budgets)
        audio_audit = audit_bc_audio_onsets(manifest, args.raw_root, budgets,
                                          manifest_path=manifest_path, evaluation_budget_path=budgets_path)
        audio_audit_path = args.cache_root / "bc_audio_onset_audit.json"
        atomic_json(audio_audit_path, audio_audit)
        reports = {}
        for mode in ("full55", "body22"):
            standardizer = train_statistics(manifest, args.cache_root, mode, args.cache_root / f"stats_{mode}.npz")
            reports[mode] = []
            for ident in manifest["splits"]["train"][:4]:
                with np.load(args.cache_root / (ident + ".npz"), allow_pickle=False) as z:
                    reports[mode].append({"id": ident, **roundtrip_audit(z["poses"], z["trans"], standardizer, mode)})
        atomic_json(args.cache_root / "normalization_audit.json", reports)
        atomic_json(args.cache_root / "ready.json", {"passed": True, "full_clips": 1464,
                                                     "manifest_sha256": sha256(manifest_path),
                                                     "evaluation_budget_audit_path": str(budgets_path.resolve()),
                                                     "evaluation_budget_audit_sha256": sha256(budgets_path),
                                                     "evaluation_budget_audit": {
                                                         "passed": True, "kind": budgets["kind"],
                                                         "split_counts": budgets["split_counts"],
                                                         "minimum_frames_by_split": budgets["minimum_frames_by_split"],
                                                         "manifest_sha256": budgets["manifest_sha256"],
                                                         "evidence": str(budgets_path.resolve()), "sha256": sha256(budgets_path)},
                                                     "bc_audio_onset_audit": {
                                                         "passed": True, "kind": audio_audit["kind"],
                                                         "split_counts": audio_audit["split_counts"],
                                                         "total_clips": audio_audit["total_clips"],
                                                         "manifest_sha256": audio_audit["manifest_sha256"],
                                                         "evaluation_budget_sha256": audio_audit["evaluation_budget_sha256"],
                                                         "evaluation_source_sha256": audio_audit["evaluation_source_sha256"],
                                                         "minimum_onsets_by_protocol": audio_audit["minimum_onsets_by_protocol"],
                                                         "evidence": str(audio_audit_path.resolve()), "sha256": sha256(audio_audit_path)},
                                                     "coverage": {s: len(manifest["splits"][s]) for s in ("train", "val", "test")},
                                                     "statistics": "train-only", "axis": "Y-up; no rotations", "no_missing_short_clips": True})


if __name__ == "__main__":
    main()
