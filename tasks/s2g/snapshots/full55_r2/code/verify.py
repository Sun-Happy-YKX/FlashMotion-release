"""Real-data/real-GPU gates; never accepts synthetic tests as training clearance."""
from __future__ import annotations
import argparse
import csv
import io
import json
import socket
import subprocess
from pathlib import Path
import numpy as np
from data import Standardizer, atomic_json, decode_motion, encode_motion, roundtrip_audit, sha256
from evaluation import SemTalkEvaluator, pose_mask
from render import fixed_validation_ids, render_bundle
from train import SOURCE_FILES, load_config


def assert_idle_eight(config, output):
    assert socket.gethostname() == config["expected_hostname"], "Wrong host: no GPU operation permitted"
    fields = "index,uuid,utilization.gpu,memory.used,memory.total"
    value = subprocess.check_output(["nvidia-smi", "--query-gpu=" + fields,
                                    "--format=csv,noheader,nounits"], text=True)
    rows = list(csv.reader(io.StringIO(value)))
    assert len(rows) == 8, rows
    compute = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,process_name",
                                      "--format=csv,noheader"], text=True).strip()
    assert not compute, "Existing compute workload; do not stop it or co-run"
    assert all(int(row[2]) <= 5 and int(row[3]) < 1024 for row in rows), rows
    result = {"passed": True, "kind": "real_gpu", "task_name": config["task_name"],
              "hostname": socket.gethostname(), "gpus": rows, "compute_processes": compute}
    atomic_json(output, result)
    return result


def cache_integrity_inventory(config, manifest, output):
    """Bind clearance to every actual cache file and both train-only statistics."""
    cache_root = Path(config["cache_root"]).resolve()
    evidence = (output / "cache_integrity.json").resolve()
    if evidence.exists():
        raise FileExistsError("Cache proof already exists; inspect partial precheck before retrying")
    manifest_hash = sha256(config["manifest"])
    stats = {}
    for mode in ("full55", "body22"):
        path = cache_root / ("stats_" + mode + ".npz")
        with np.load(path, allow_pickle=False) as z:
            assert z["split"].item() == "train" and z["mode"].item() == mode
            assert z["split_csv_sha256"].item() == manifest["split_csv_sha256"]
        stats[mode] = {"path": str(path), "sha256": sha256(path)}
    ids = [ident for split in ("train", "val", "test") for ident in manifest["splits"][split]]
    assert len(ids) == len(set(ids)) == 1464
    clips = []
    for index, ident in enumerate(ids):
        path = cache_root / (ident + ".npz")
        assert path.is_file() and path.resolve().parent == cache_root
        clips.append({"id": ident, "path": str(path), "sha256": sha256(path)})
        if (index + 1) % 100 == 0:
            print(f"cache_integrity: {index + 1}/{len(ids)} files hashed", flush=True)
    inventory = {"passed": True, "kind": "complete_feature_cache_integrity_inventory",
                 "manifest_sha256": manifest_hash, "clip_count": len(clips),
                 "stats": stats, "clips": clips}
    atomic_json(evidence, inventory)
    return {"passed": True, "kind": inventory["kind"], "evidence": str(evidence),
            "sha256": sha256(evidence), "manifest_sha256": manifest_hash,
            "clip_count": len(clips), "stats_sha256": {mode: value["sha256"] for mode, value in stats.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--native-jvp-evidence")
    p.add_argument("--data-only", action="store_true",
                   help="Run real GT/axis/FK/render diagnostics in a separate directory; NEVER writes training clearance")
    p.add_argument("--data-version", type=int, default=1,
                   help="Versioned GT-only diagnostic output; previous outputs remain immutable")
    a = p.parse_args()
    config = load_config(a.config)
    if not a.data_only and not a.native_jvp_evidence:
        p.error("Formal precheck requires --native-jvp-evidence")
    if a.data_version < 1:
        p.error("--data-version must be positive")
    output = Path(config["output_root"]) / (f"precheck_data_v{a.data_version}" if a.data_only else "precheck")
    output.mkdir(parents=True, exist_ok=True)
    gate_path = Path(config["precheck_path"])
    if gate_path.exists():
        raise FileExistsError("Clearance already exists; no silent overwrite/reuse after source changes")
    if a.data_only and (output / "DATA_DIAGNOSTICS_COMPLETE.json").exists():
        raise FileExistsError("Data diagnostic already exists; do not silently overwrite")
    assert_idle_eight(config, output / "gpu8_idle.json")
    source = Path(__file__).resolve().parent
    native_path = None
    if not a.data_only:
        native_path = Path(a.native_jvp_evidence).resolve()
        native = json.loads(native_path.read_text())
        assert native["passed"] is True and native["kind"] == "real_gpu"
        assert native["hostname"] == config["expected_hostname"]
        assert native["task_name"] == config["task_name"]
        for name in ("model.py", "loss.py", "native_jvp_check.py", "native_higherorder_jvp.py"):
            assert native["source_hashes"][name] == sha256(source / name), name
        from native_jvp_check import kernel_binary_bindings
        import mamba_ssm.ops.selective_scan_interface as scan_module
        import causal_conv1d.causal_conv1d_interface as conv_module
        assert native.get("kernel_binary_sources") == kernel_binary_bindings(scan_module, conv_module), "Native kernel binary changed"
        from native_higherorder_jvp import install
        assert native.get("native_derivative_repair") == install(), "Native derivative repair/runtime binding changed"
        from native_jvp_check import require_extended_native_checks
        require_extended_native_checks(native)
    manifest = json.loads(Path(config["manifest"]).read_text())
    assert {s: len(v) for s, v in manifest["splits"].items()} == {"train": 1093, "val": 106, "test": 265}
    assert manifest["height_axis"] == "y" and manifest["statistics_split"] == "train"
    cache_root, raw_root = Path(config["cache_root"]), Path(config["raw_root"])
    ready = json.loads((cache_root / "ready.json").read_text())
    assert ready["passed"] is True and ready["full_clips"] == 1464
    assert ready["manifest_sha256"] == sha256(config["manifest"])
    budget_path = Path(ready["evaluation_budget_audit"]["evidence"])
    assert sha256(budget_path) == ready["evaluation_budget_audit"]["sha256"]
    budgets = json.loads(budget_path.read_text())
    assert budgets["passed"] is True and budgets["manifest_sha256"] == sha256(config["manifest"])
    assert budgets["split_counts"] == {"val": 106, "test": 265}
    for split in ("val", "test"):
        assert [v["id"] for v in budgets["splits"][split]] == manifest["splits"][split]
        assert all(v["evaluation_frames"] > 120 for v in budgets["splits"][split])
    onset = ready["bc_audio_onset_audit"]
    assert onset["passed"] is True and onset["kind"] == "complete_val_test_bc_audio_onset_inventory_not_metrics"
    assert onset["split_counts"] == {"val": 106, "test": 265} and onset["total_clips"] == 371
    assert onset["manifest_sha256"] == sha256(config["manifest"])
    assert onset["evaluation_budget_sha256"] == sha256(budget_path)
    assert onset["evaluation_source_sha256"] == sha256(source / "evaluation.py")
    assert min(onset["minimum_onsets_by_protocol"].values()) > 0
    assert sha256(onset["evidence"]) == onset["sha256"]
    onset_inventory = json.loads(Path(onset["evidence"]).read_text())
    assert onset_inventory["passed"] is True
    for split in ("val", "test"):
        assert [v["id"] for v in onset_inventory["splits"][split]] == manifest["splits"][split]
    cache_integrity = cache_integrity_inventory(config, manifest, output)
    evaluator = SemTalkEvaluator(config["metric_assets"], config["evaluator_assets"], device="cuda:0")
    ids = fixed_validation_ids(manifest)
    records = []
    normalization, axes, metric_results, renders, geometry = {}, [], {}, {}, {}
    for mode in ("full55", "body22"):
        stats = cache_root / ("stats_" + mode + ".npz")
        with np.load(stats, allow_pickle=False) as z:
            assert z["split"].item() == "train" and z["mode"].item() == mode
            assert z["split_csv_sha256"].item() == manifest["split_csv_sha256"]
        standardizer = Standardizer.load(stats)
        predicted = output / (mode + "_gt_roundtrip")
        predicted.mkdir(exist_ok=True)
        audits = []
        geometry[mode] = []
        for ident in ids:
            with np.load(cache_root / (ident + ".npz"), allow_pickle=False) as z:
                poses, trans, betas = z["poses"], z["trans"], z["betas"]
                rest_joints, pose_mean = z["rest_joints"], z["pose_mean"]
                assert z["source_id"].item() == ident
            data_audit = roundtrip_audit(poses, trans, standardizer, mode)
            physical = encode_motion(poses, trans, mode)
            restored = standardizer.denormalize(standardizer.normalize(physical))
            poses_back, trans_back = decode_motion(restored, mode, trans[0, (0, 2)])
            audits.append({"id": ident, **data_audit,
                           "real_aesk_smplx": evaluator.audit_roundtrip(poses, poses_back, betas, pose_policy=mode)})
            import torch
            from loss import motion_to_joints
            sample = poses[[0, len(poses)//2, len(poses)-1]]
            masked_sample = pose_mask(sample, mode)
            actual = evaluator.positions(masked_sample, betas)
            actual = actual - actual[:, :1]
            physical_sample = encode_motion(sample, np.zeros((len(sample), 3), np.float32), mode)
            with torch.inference_mode():
                fk = motion_to_joints(torch.as_tensor(physical_sample[None], device="cuda:0"),
                                      torch.as_tensor(rest_joints[None], device="cuda:0"), mode,
                                      pose_mean=torch.as_tensor(pose_mean, device="cuda:0"))[0].cpu().numpy()
            fk_error = float(np.max(np.abs(actual-fk)))
            assert fk_error <= 2e-4, (ident, mode, fk_error)
            geometry[mode].append({"id": ident, "fk_vs_actual_smplx_max_error_m": fk_error,
                                   "tolerance_m": 2e-4, "unposed_shaped_rest": True,
                                   "source_cache_sha256": sha256(cache_root / (ident + ".npz"))})
            np.savez(predicted / ("res_" + ident + ".npz"), poses=poses_back, trans=trans_back,
                     betas=betas, expressions=np.zeros((len(poses_back), 100), np.float32),
                     mocap_frame_rate=30, generation_kind="GT normalization roundtrip; NOT generated motion")
            if mode == "full55":
                sample = poses[[0, len(poses)//2, len(poses)-1]]
                positions = evaluator.positions(pose_mask(sample, "full55"), betas)
                # Head minus mean ankle height in actual raw-data poses. No
                # score-driven rotation is tested, selected, or applied.
                upright = np.mean(positions[:, 15] - positions[:, [10, 11]].mean(1), axis=0)
                assert upright[1] > .5 and upright[1] > 1.5 * max(abs(upright[0]), abs(upright[2])), (ident, upright)
                axes.append({"id": ident, "head_minus_ankles_xyz_m": upright.tolist(), "passed_y_up": True})
        normalization[mode] = audits
        metrics = evaluator.evaluate(predicted, raw_root / "smplxflame_30", raw_root / "wave16k",
                                     ids, pose_policy=mode, expected_count=3, model_pose_policy=mode)
        tol = metrics["fgd_numerical_audit"]["numerical_tolerance"]
        assert abs(metrics["fgd"]) <= min(tol, 1e-5), (mode, metrics["fgd"])
        assert abs(metrics["div"] - metrics["div_gt"]) <= 1e-5 * max(1., abs(metrics["div_gt"]))
        metric_results[mode] = metrics
        atomic_json(output / (mode + "_gt_self_metrics.json"), metrics)
        renders[mode] = render_bundle(config["manifest"], predicted, raw_root, config["smplx"],
                                      output / (mode + "_gt_roundtrip_renders"), ids, mode,
                                      "cuda:0", 10., 4, prediction_label="GT roundtrip")
    atomic_json(output / "normalization.json", {"passed": True, "kind": "real_data", "clips": normalization})
    atomic_json(output / "axis.json", {"passed": True, "kind": "real_data", "height_axis": "y",
                                       "coordinate_transform": "none", "clips": axes})
    atomic_json(output / "gt_self_fgd.json", {"passed": True, "kind": "real_data", "metrics": metric_results})
    atomic_json(output / "smplx_fk.json", {"passed": True, "kind": "real_data", "geometry": geometry,
                                         "official_default_hand_mean": True})
    atomic_json(output / "actual_three_clip_render.json", {"passed": True, "kind": "smplx_mesh", "renders": renders,
                "content": "real GT normalization roundtrip, NOT learned model outputs", "visual_quality_review": "pending"})
    checks = {name: {"passed": True, "evidence": str((output / (name + ".json")).resolve()), "kind": kind}
              for name, kind in (("gpu8_idle", "real_gpu"), ("gt_self_fgd", "real_data"),
                                 ("normalization", "real_data"), ("axis", "real_data"),
                                 ("actual_three_clip_render", "smplx_mesh"), ("smplx_fk", "real_data"))}
    if a.data_only:
        for check in checks.values():
            check["sha256"] = sha256(check["evidence"])
        atomic_json(output / "DATA_DIAGNOSTICS_COMPLETE.json", {
            "passed": False, "data_checks_passed": True, "training_clearance": False,
            "kind": "real_GT_data_geometry_metrics_render_diagnostics_NOT_training_clearance",
            "checks": checks, "task_name": config["task_name"], "hostname": socket.gethostname(),
            "cache_integrity": cache_integrity, "manifest_sha256": sha256(config["manifest"]),
            "code_hashes": {name: sha256(source / name) for name in SOURCE_FILES},
            "actual_render_pixels_reviewed": False, "native_cuda_jvp_passed": False,
            "note": "GT normalization roundtrip, not learned predictions. Formal native/visual/BS gates still required."})
        print("Real GT-only diagnostics completed; no training clearance written.", flush=True)
        return
    checks["native_cuda_jvp"] = {"passed": True, "kind": "real_gpu", "evidence": str(native_path)}
    for check in checks.values():
        check["sha256"] = sha256(check["evidence"])
    hashes = {name: sha256(source / name) for name in SOURCE_FILES}
    atomic_json(gate_path, {"passed": True, "task_name": config["task_name"], "hostname": socket.gethostname(),
                           "checks": checks, "code_hashes": hashes, "manifest_sha256": sha256(config["manifest"]),
                           "evaluation_budget_audit": ready["evaluation_budget_audit"],
                           "bc_audio_onset_audit": onset,
                           "cache_integrity": cache_integrity,
                           "native_jvp_evidence_sha256": sha256(native_path), "not_training_results": True,
                           "visual_review": {"passed": False, "kind": "actual_render_frame_review", "evidence": None,
                               "note": "Technical checks only. Root must view first/middle/last actual PNG frames from all six paired videos, record hash-bound review evidence, then attach it before any BS probe or training."}})


if __name__ == "__main__":
    main()
