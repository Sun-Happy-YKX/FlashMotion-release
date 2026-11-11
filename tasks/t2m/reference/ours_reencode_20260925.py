import os,sys,json,time,pathlib,fcntl,subprocess,socket,gc
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='1'
B=pathlib.Path('/efs/yangkaixing/FlowerDance_T2M_Long');C=B/'runs/t2m_restart_8000ep_20260922/code';R=B/'eval/comparison_mixed_20260925/ours_reencoded'
R.mkdir(exist_ok=True)
lock=open(R/'worker.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if '--worker' not in sys.argv:
 assert not (R/'launch.json').exists(),'Existing launch requires explicit inspection'
 free=int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>=30000
 with open(R/'worker.log','ab',buffering=0) as log:
  p=subprocess.Popen([sys.executable,__file__,'--worker'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 (R/'launch.json').write_text(json.dumps(dict(pid=p.pid,host=socket.gethostname(),gpu=0,cap_gib=6,free_mib_before=free,started=time.time(),purpose='Regenerate four historically selected checkpoints and uniformly reencode; no reranking of full history')));print('LAUNCHED',p.pid);sys.exit()
import numpy as np,torch
sys.path.insert(0,str(C));os.chdir(C)
from evaluate_t2m_ablation import load_objective,sample_episode,stable_seed,MotionTextDataset,effective_segments,recover_joints_from_ric
from t2m_long.evaluator import FloodT2MEvaluator
from t2m_long.babel_long_benchmark import MultiPredictionEmbeddingCollector,centered_transition,TRANSITION_FRAMES,score_embedding_subset,transition_metrics
torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(6*1024**3/torch.cuda.get_device_properties(0).total_memory)
flood=B/'third_party/FloodDiffusion';data=flood/'raw_data';ds=MotionTextDataset(data,B/'data/umt5_xxl_sentence_features','babel','val',max_length=None)
canonical=np.load(R.parent/'scores/canonical_gt.npz');allowed=set(canonical['episode_id'])
def status(**kw):
 (R/'status.json').write_text(json.dumps(dict(pid=os.getpid(),updated=time.time(),**kw)));print(json.dumps(kw),flush=True)
for mode,steps,epoch in [('mixed',4,5850),('multi',4,5900),('mixed',1,7150),('multi',1,7100)]:
 tag=f'{mode}_{steps}step_e{epoch}';dest=R/tag;dest.mkdir(exist_ok=True)
 if (dest/'metrics.json').exists():continue
 ck=B/f'runs/t2m_adain_selection_8000ep_20260923/adain_selection/checkpoints/step-{epoch*14:07d}.pt'
 torch.set_float32_matmul_precision('high');model,record,variant=load_objective(ck,data,torch.device('cuda'));del record
 ev=FloodT2MEvaluator(flood,torch.device('cuda'),seed=42);names=['single','multi'] if mode=='mixed' else ['multi'];col=MultiPredictionEmbeddingCollector(ev,names,32)
 trans={'transition__ground_truth':[],'transition__multi':[],'transition_key':[],'transition_episode_id':[]};count=0
 status(status='generating',tag=tag,completed=0)
 for idx,name in enumerate(ds.ids):
  if name not in allowed:continue
  item=ds[idx];segments=effective_segments(item['text_ends'],item['length']);gt=item['motion']*torch.from_numpy(ds.std)+torch.from_numpy(ds.mean)
  raw=dest/(name+'.npz')
  if raw.exists():
   with np.load(raw) as f:arrays={k:f[k].copy() for k in f.files}
  else:
   torch.set_float32_matmul_precision('high')
   with torch.inference_mode():
    arrays={'multi':sample_episode(model,item['condition'],steps,stable_seed(42,variant,name,steps))[:item['length']].cpu().numpy()}
    if mode=='mixed':
     for i,s in enumerate(segments):arrays[f'single_{i}']=sample_episode(model,item['condition'][s.start:s.end],steps,stable_seed(42,variant,name,i,'single',steps)).cpu().numpy()
   assert all(np.isfinite(a).all() for a in arrays.values());np.savez_compressed(raw,**arrays)
  multi=torch.from_numpy(arrays['multi']);torch.set_float32_matmul_precision('highest')
  for i,s in enumerate(segments):
   preds={'multi':multi[s.start:s.end]}
   if mode=='mixed':preds['single']=torch.from_numpy(arrays[f'single_{i}'])
   col.add(gt[s.start:s.end],item['text_tokens'][s.annotation_index],preds,dict(segment_key=f'{name}::{i}',episode_id=name,caption=item['captions'][s.annotation_index],segment_length=s.end-s.start))
  gj=recover_joints_from_ric(gt[None])[0];pj=recover_joints_from_ric(multi[None])[0]
  for i,s in enumerate(segments[1:]):
   g=centered_transition(gj,s.start,TRANSITION_FRAMES);p=centered_transition(pj,s.start,TRANSITION_FRAMES)
   if g is None or p is None:continue
   trans['transition__ground_truth'].append(g);trans['transition__multi'].append(p);trans['transition_key'].append(f'{name}::{i}');trans['transition_episode_id'].append(name)
  count+=1;status(status='generating',tag=tag,completed=count,total=311)
 torch.set_float32_matmul_precision('highest');a=col.finish();a.update({k:np.asarray(v) for k,v in trans.items()})
 assert count==311 and len(a['segment_key'])==621 and len(a['transition_key'])==297
 for k in canonical.files:assert np.array_equal(a[k],canonical[k]),'Canonical mismatch '+tag+' '+k
 np.savez_compressed(dest/'embeddings.npz',**a)
 if mode=='mixed':
  inp={k:np.concatenate([a[k],a[k]]) for k in ['ground_truth','text','caption','episode_id','segment_length']};inp['prediction']=np.concatenate([a['prediction__single'],a['prediction__multi']]);key='prediction'
 else:inp=a;key='prediction__multi'
 met=score_embedding_subset(inp,key,inp['segment_length']>=4,torch.device('cpu'),seed=42)
 (dest/'metrics.json').write_text(json.dumps(dict(tag=tag,epoch=epoch,step=epoch*14,mode=mode,sampling_steps=steps,metrics=met,transition=transition_metrics(a,'multi'),canonical_gt_exact=True,selection='Historical best checkpoint, uniformly reencoded; not a new full-history best search'),indent=2))
 del model,ev,col,a,arrays;gc.collect();torch.cuda.empty_cache();status(status='checkpoint_complete',tag=tag,completed=count)
status(status='complete',checkpoints=4)
