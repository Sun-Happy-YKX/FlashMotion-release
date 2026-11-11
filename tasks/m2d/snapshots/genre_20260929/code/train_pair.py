import os,sys,json,pickle,time,random,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from scipy.stats import rankdata
from EDGE import EDGE
from model.model import GenreGate
from model.metric import (kinetic_features,geometric_features,calculate_pfc,
    calculate_beat_similarity,calculate_avg_distance,calc_fid,align_axis,rotate_pred_like_gt)
from dataset.quaternion import ax_from_6v
from dataset.motion_representation import root_velocity_to_position

R=Path(__file__).resolve().parent.parent
S=Path('/efs/yangkaixing/FlowerDance_x0_ablation')
version,phase=sys.argv[1:3]
assert version in ('genre','no_genre') and phase in ('verify','train')
use_genre=version=='genre';out=R/version/phase
out.mkdir(exist_ok=True,parents=True)
def dump(path,data):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False));os.replace(tmp,path)
class Samples(Dataset):
 def __init__(self,d):self.d=d
 def __len__(self):return len(self.d['pose'])
 def __getitem__(self,i):
  if use_genre:return self.d['pose'][i],self.d['features'][i],int(self.d['genres'][i])
  return self.d['pose'][i],self.d['features'][i]
edge=EDGE(feature_type='baseline',checkpoint_path='',x0_variant='tr_only',cross_modal_fusion='adain',condition_selection=True,direction_fusion='sum',use_genre=use_genre,loss_weights=dict(fm=1,rec=.636,vel=.323,acc=.2,fk=.646,floor=0),learning_rate=2e-4,weight_decay=.02)
a=edge.accelerator;raw=a.unwrap_model(edge.model)
assert sum(isinstance(m,GenreGate) for m in raw.modules())==(4 if use_genre else 0)
with open(S/'data/dataset_backups_stable_v2/train_tensor_dataset.pkl','rb') as f:train=pickle.load(f)
if not use_genre:train.data.pop('genres',None)
edge.normalizer=train.normalizer;edge.flow_matching.set_normalizer(train.normalizer)
loader=a.prepare(DataLoader(Samples(train.data),batch_size=64,shuffle=True,num_workers=0,drop_last=True))
config=dict(version=version,epochs=4000,seed=42,batch_per_gpu=64,world_size=a.num_processes,global_batch=256,steps_per_epoch=len(loader),update_budget=4000*len(loader),pretrained=None,resume=None,selection_split='test',save_every_epochs=50,sampling_steps=[1,4],eval_batch=2,sampling_seed=42,memory_fraction=.4,train_frames=240,test_frames=1020,genre_input=use_genre,ranking='Within each version and sampling-step pool, mean of four equal-weight percentile dimensions; ties earlier epoch',dimensions=['mean(FID_k,FID_g percentiles)','mean(abs(Div_k-GT_k),abs(Div_g-GT_g) percentiles)','negative BeatAlignment percentile','PFC percentile'],weights=[.25]*4)
if a.is_main_process:dump(out/'config.json',config)

def prepare_test():
 with open(S/'data/dataset_backups_stable_v2/test_tensor_dataset.pkl','rb') as f:d=pickle.load(f)
 if not use_genre:d.data.pop('genres',None)
 names=[Path(n).stem for n in d.data['filenames']]
 assert len(names)==84 and len(set(names))==84
 assert set(names)=={p.stem for p in (S/'data/test/baseline_feats').glob('*.npy')}
 qs=[];ps=[];beats=[]
 for n in names:
  with open(S/'data/test/motions_sliced'/f'{n}.pkl','rb') as f:m=pickle.load(f)
  qs.append(m['q']);ps.append(m['pos']);beats.append(np.load(S/'data/test/baseline_feats'/f'{n}.npy')[:,34])
 with torch.no_grad():
  q=torch.as_tensor(np.array(qs),device=a.device).reshape(84,-1,24,3)
  p=torch.as_tensor(np.array(ps),device=a.device)
  joints=align_axis(edge.flow_matching.smpl.forward(q,p).cpu())
 assert joints.shape[1]==d.data['pose'].shape[1]==1020
 gt=dict(k=kinetic_features(joints),g=geometric_features(joints),beats=np.array(beats),pfc=float(calculate_pfc(joints)),ba=float(calculate_beat_similarity(np.array(beats),joints)))
 dump(out/'test_protocol.json',dict(names=names,count=84,original_ids=sorted({n.split('_')[0] for n in names}),selection_split='test',postprocessing='Existing normalized clamp [-1,1], root integration, FK and rotate_pred_like_gt then align_axis',pretrained=None,genre_labels_read=use_genre))
 return d,names,gt

def update_best():
 for n in (1,4):
  rows=[json.loads(p.read_text()) for p in sorted((out/'metrics').glob(f'*_{n}step.json'))]
  if not rows:continue
  vals=np.array([[r['FID_k'],r['FID_g'],abs(r['Diversity_k']-r['GT_Diversity_k']),abs(r['Diversity_g']-r['GT_Diversity_g']),-r['BeatAlignment'],r['PFC']] for r in rows])
  ranks=np.stack([(rankdata(vals[:,i],method='average')-.5)/len(rows) for i in range(6)],axis=1)
  score=(ranks[:,:2].mean(1)+ranks[:,2:4].mean(1)+ranks[:,4]+ranks[:,5])/4
  idx=min(range(len(rows)),key=lambda i:(score[i],rows[i]['epoch']))
  dump(out/f'best_{n}step.json',dict(best=rows[idx],score=float(score[idx]),pool_size=len(rows),pool_epochs=[r['epoch'] for r in rows],weights=[.25]*4,subjective_weights=True,selection_split='test',pool_scope=f'{version}/{n}step',warning='Scores change with pool; do not compare scores across pools'))

@torch.no_grad()
def evaluate(epoch,step,checkpoint):
 edge.eval()
 for n in (1,4):
  # Fork RNG prevents evaluation from changing the subsequent training randomness.
  with torch.random.fork_rng(devices=[a.device.index]):
   torch.manual_seed(42);torch.cuda.manual_seed(42)
   pred=[];outside=[]
   for i in range(0,84,2):
    c=test.data['features'][i:i+2].to(a.device)
    g=torch.as_tensor(test.data['genres'][i:i+2],device=a.device,dtype=torch.long) if use_genre else None
    y=edge.flow_matching.euler_sample((len(c),1020,151),c,g,n_steps=n+1).cpu()
    assert torch.isfinite(y).all();outside.append(float((y.abs()>1).float().mean()))
    y=edge.normalizer.unnormalize(y)[:,:,4:]
    p=root_velocity_to_position(y[:,:,:3].to(a.device));q=ax_from_6v(y[:,:,3:].reshape(len(c),1020,24,6)).to(a.device)
    q,p=rotate_pred_like_gt(q,p)
    pred.append(align_axis(edge.flow_matching.smpl.forward(q,p).cpu()))
   joints=torch.cat(pred)
   feats=dict(k=kinetic_features(joints),g=geometric_features(joints))
   row=dict(epoch=epoch,update_step=step,checkpoint=str(checkpoint.resolve()),sampling_steps=n,sampling_seed=42,coverage=84,test_ids=len({s.split('_')[0] for s in names}),selection_split='test',pretrained=None,version=version,normalized_clip_fraction=float(np.mean(outside)),BeatAlignment=float(calculate_beat_similarity(gt['beats'],joints)),PFC=float(calculate_pfc(joints)),GT_BeatAlignment=gt['ba'],GT_PFC=gt['pfc'])
   for k in ('k','g'):
    mean=gt[k].mean(axis=0);std=gt[k].std(axis=0)
    gn=(gt[k]-mean)/(std+1e-4);pn=(feats[k]-mean)/(std+1e-4)
    fid=calc_fid(pn,gn)
    imag=float(abs(np.imag(fid)))
    assert imag<=1e-3,('FID imaginary residual exceeds tolerance',k,fid)
    row['FID_imaginary_residual_'+k]=imag
    row['FID_'+k]=float(np.real(fid))
    row['Diversity_'+k]=float(calculate_avg_distance(pn));row['GT_Diversity_'+k]=float(calculate_avg_distance(gn))
   assert all(np.isfinite(v) for v in row.values() if isinstance(v,float)),row
   (out/'metrics').mkdir(exist_ok=True)
   np.savez_compressed(out/'metrics'/f'e{epoch}_{n}step_features.npz',kinetic=np.asarray(feats['k']),geometric=np.asarray(feats['g']),names=np.array(names))
   dump(out/'metrics'/f'e{epoch}_{n}step.json',row);print('EVAL',row,flush=True)
   update_best()

test=names=gt=None
if a.is_main_process:test,names,gt=prepare_test()
a.wait_for_everyone();step=0;started=time.time()
for epoch in range(1,(1 if phase=='verify' else 4000)+1):
 if hasattr(loader,'set_epoch'):loader.set_epoch(epoch-1)
 edge.train()
 for batch in loader:
  x,c=batch[:2];g=batch[2] if use_genre else None
  loss,parts=edge.flow_matching(x,c,g,t_override=None)
  finite=torch.tensor(int(torch.isfinite(loss).all()),device=a.device)
  assert int(a.reduce(finite,reduction='sum'))==a.num_processes,'Non-finite training loss'
  edge.optim.zero_grad();a.backward(loss);a.clip_grad_norm_(edge.model.parameters(),1.0);edge.optim.step();step+=1
  if a.is_main_process:
   edge.flow_matching.ema.update_model_average(edge.flow_matching.master_model,edge.flow_matching.model)
   if step<=3 or step%20==0:
    row=dict(epoch=epoch,step=step,loss=float(loss.detach()),parts=[float(v.detach()) for v in parts],seconds=time.time()-started,peak_memory_gib=torch.cuda.max_memory_allocated()/2**30)
    with (out/'train.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
    print('TRAIN',row,flush=True)
  if phase=='verify' and step==2:break
 if epoch%50==0 or phase=='verify':
  a.wait_for_everyone()
  if a.is_main_process:
   (out/'weights').mkdir(exist_ok=True);ckpt=out/'weights'/f'train-{epoch}.pt'
   torch.save(dict(model_state_dict=raw.state_dict(),optimizer_state_dict=edge.optim.state_dict(),normalizer=edge.normalizer,epoch=epoch,update_step=step,config=config,prediction_type='x0',root_representation='vx_vy_z',height_axis='z',x0_variant='tr_only',cross_modal_fusion='adain',condition_selection=True,direction_fusion='sum',use_genre=use_genre,pretrained=None),ckpt)
   evaluate(epoch,step,ckpt)
  a.wait_for_everyone()
if a.is_main_process:dump(out/'completed.json',dict(epoch=epoch,step=step,phase=phase,exit_code=0))
a.wait_for_everyone()
torch.distributed.destroy_process_group()
