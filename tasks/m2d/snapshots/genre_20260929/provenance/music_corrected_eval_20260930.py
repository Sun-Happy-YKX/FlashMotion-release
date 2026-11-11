"""Independent bounded checkpoint reevaluation; never changes training code/results."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
import sys,json,pickle,time,hashlib,fcntl,functools,concurrent.futures,multiprocessing
from pathlib import Path
import numpy as np
import torch
C=Path('/efs/yangkaixing/FlowerDance_x0_ablation/runs/music_genre_ablation_20260929/code')
R=C.parent;S=Path('/efs/yangkaixing/FlowerDance_x0_ablation')
OUT=R/'corrected_eval_20260930'
sys.path.insert(0,str(C));os.chdir(C);torch.set_num_threads(1)

def dump(p,x):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));os.replace(t,p)

def kinetic(j):
    # Valid finite differences only; centered +/-2 window, shortened at boundaries.
    x=np.asarray(j,dtype=np.float64);T=x.shape[1]
    def smooth(d):
        out=np.zeros_like(d)
        for i in range(T):
            lo=max(1,i-2);hi=min(T,i+3)
            out[:,i]=d[:,lo:hi].mean(axis=1)
        return out
    v=np.zeros_like(x);v[:,1:]=np.diff(x,axis=1)*30;v=smooth(v)
    acc=np.zeros_like(v);acc[:,1:]=np.diff(v,axis=1)*30;acc=smooth(acc)
    h=(v[:,1:,:,[0,2]]**2).sum(-1).mean(1)
    vert=(v[:,1:,:,1]**2).mean(1)
    a=np.linalg.norm(acc[:,1:],axis=-1).mean(1)
    return np.stack([h,vert,a],axis=-1).reshape(len(x),-1)[:,:66].astype(np.float32)

def geo_chunk(j):
    from model.features import utils as fu
    from model.features.geometric import geometric_features
    for name in ['velocity_direction_above_threshold','velocity_direction_above_threshold_normal','velocity_above_threshold']:
        fn=getattr(fu,name)
        if not isinstance(fn,functools.partial):setattr(fu,name,functools.partial(fn,time_per_frame=1/30))
    return geometric_features(torch.from_numpy(j)).numpy()

def extract(j,pool):
    chunks=np.array_split(j,4)
    return dict(k=kinetic(j),g=np.concatenate(list(pool.map(geo_chunk,chunks))))

def fid(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    aa=(a-a.mean(0))/np.sqrt(len(a)-1);bb=(b-b.mean(0))/np.sqrt(len(b)-1)
    value=np.sum((a.mean(0)-b.mean(0))**2)+np.sum(aa**2)+np.sum(bb**2)-2*np.linalg.svd(aa@bb.T,compute_uv=False).sum()
    assert value>=-1e-4,value
    return float(max(0,value))

def diversity(a):
    d=np.linalg.norm(a[:,None]-a[None],axis=-1)
    return float(d.sum()/(len(a)*(len(a)-1)))

def pfc(j):
    # Input is native Y-up; rotate +90 X to Z-up for the published PFC routine.
    z=j.copy();z[...,1]=-j[...,2];z[...,2]=j[...,1]
    scores=[]
    for x in z:
        v=np.diff(x[:,0],axis=0)*30;a=np.diff(v,axis=0)*30
        a[:,2]=np.maximum(a[:,2],0);a=np.linalg.norm(a,axis=-1)
        maximum=a.max();a=a/maximum if maximum>1e-8 else np.zeros_like(a)
        feet=x[:,[7,10,8,11],:2];fv=np.linalg.norm(feet[2:]-feet[1:-1],axis=-1)
        scores.append(np.mean(np.minimum(fv[:,0],fv[:,1])*np.minimum(fv[:,2],fv[:,3])*a)*10000)
    return float(np.mean(scores))

def rank_all():
    from scipy.stats import rankdata
    allbest={}
    for version in ['genre','no_genre']:
        for n in [1,4]:
            rows=[json.loads(p.read_text()) for p in (OUT/'metrics').glob(f'{version}_e*_{n}step.json')]
            if not rows:continue
            values=np.array([[r['FID_k'],r['FID_g'],abs(r['Diversity_k']-r['GT_Diversity_k']),abs(r['Diversity_g']-r['GT_Diversity_g']),-r['BeatAlignment'],r['PFC']] for r in rows])
            ranks=np.stack([(rankdata(values[:,i])-.5)/len(rows) for i in range(6)],1)
            scores=(ranks[:,:2].mean(1)+ranks[:,2:4].mean(1)+ranks[:,4]+ranks[:,5])/4
            idx=min(range(len(rows)),key=lambda i:(scores[i],rows[i]['epoch']))
            allbest[f'{version}_{n}step']=dict(best=rows[idx],score=float(scores[idx]),pool_epochs=sorted(r['epoch'] for r in rows),subjective_weights=[.25]*4,selection_split='test',warning='Scores not comparable across pools')
    dump(OUT/'best_so_far.json',allbest)

def main():
    OUT.mkdir(exist_ok=True);lock=(OUT/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for sub in ['metrics','joints','features']:(OUT/sub).mkdir(exist_ok=True)
    from vis import SMPLSkeleton
    from model.model import DanceDecoder
    from model.metric import rotate_pred_like_gt,calculate_beat_similarity
    from dataset.motion_representation import pose_to_root_and_rotations
    from model.flowmatching import FlowMatching
    torch.cuda.set_per_process_memory_fraction(6*1024**3/torch.cuda.get_device_properties(0).total_memory)
    free,_=torch.cuda.mem_get_info();assert free>=30*1024**3,free
    with open(S/'data/dataset_backups_stable_v2/test_tensor_dataset.pkl','rb') as f:test=pickle.load(f)
    names=[Path(n).stem for n in test.data['filenames']];assert len(names)==84
    planfile=OUT/'manifest.json'
    if planfile.exists():plan=json.loads(planfile.read_text())['plan']
    else:
        plan=[]
        for version in ['genre','no_genre']:
            weights=list((R/version/'train/weights').glob('train-*.pt'))
            weights=sorted(weights,key=lambda p:int(p.stem.split('-')[-1]))
            for p in weights:
                epoch=int(p.stem.split('-')[-1])
                for n in [1,4]:plan.append(dict(version=version,epoch=epoch,n=n,checkpoint=str(p)))
        priorities={('genre',600,1):0,('genre',850,4):1,('no_genre',600,1):2,('no_genre',450,4):3}
        plan.sort(key=lambda r:(priorities.get((r['version'],r['epoch'],r['n']),4),-r['epoch'],r['version'],r['n']))
        dump(planfile,dict(plan=plan,created=time.time(),fps=30,axes='native Y-up k/g/BAS; +90X Z-up PFC',k='valid finite differences, centered +/-2 smoothing, frames1:T mean; 22joints',g='existing 32 predicates with all velocity thresholds using 1/30',normalization='GT sample std+1e-4',seed=42,batch=2,clip='unchanged [-1,1]',memory_cap_gib=6,source_code=str(C),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),not_training_change=True))
    x=np.zeros((2,1020,24,3),dtype=np.float64);x[:,:,:,0]=np.arange(1020)[None,:,None]/30
    unit=kinetic(x);assert np.max(abs(unit[:,0::3]-1))<1e-4;assert np.max(abs(unit[:,2::3]))<1e-4
    assert np.max(abs(kinetic(x+100)-unit))<1e-4
    assert pfc(x)<1e-6
    qs=[];ps=[];beats=[]
    for name in names:
        with open(S/'data/test/motions_sliced'/f'{name}.pkl','rb') as f:m=pickle.load(f)
        qs.append(m['q']);ps.append(m['pos']);beats.append(np.load(S/'data/test/baseline_feats'/f'{name}.npy')[:,34])
    smpl=SMPLSkeleton();j=smpl.forward(torch.tensor(np.array(qs),dtype=torch.float32).reshape(84,-1,24,3),torch.tensor(np.array(ps),dtype=torch.float32)).numpy()
    beats=np.array(beats)
    pool=concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'))
    gtpath=OUT/'gt_features.npz'
    if gtpath.exists():
        with np.load(gtpath) as z:gt={k:z[k] for k in ['k','g']}
    else:
        gt=extract(j,pool);np.savez_compressed(gtpath,**gt,names=names)
    stats={k:(v.mean(0),v.std(0,ddof=1)+1e-4) for k,v in gt.items()}
    gtmetrics={'GT_BeatAlignment':float(calculate_beat_similarity(beats,torch.from_numpy(j))),'GT_PFC':pfc(j)}
    for k in gt:
        gn=(gt[k]-stats[k][0])/stats[k][1];gtmetrics['GT_Diversity_'+k]=diversity(gn)
        assert fid(gn,gn)<1e-4
    dump(OUT/'gt_metrics.json',gtmetrics);print('GT',gtmetrics,flush=True)
    model=None;loaded=None
    for entry in plan:
        version=entry['version'];epoch=entry['epoch'];n=entry['n'];key=f'{version}_e{epoch}_{n}step'
        target=OUT/'metrics'/f'{key}.json'
        if target.exists():continue
        dump(OUT/'status.json',dict(state='running',current=entry,completed=len(list((OUT/'metrics').glob('*.json'))),total=len(plan),time=time.time()))
        cache=OUT/'joints'/f'{key}.npz'
        if cache.exists():
            with np.load(cache) as z:joints=z['joints'];clip=float(z['clip'])
        else:
            if loaded!=entry['checkpoint']:
                model=None;torch.cuda.empty_cache()
                ckpt=torch.load(entry['checkpoint'],map_location='cpu',weights_only=False)
                assert ckpt['epoch']==epoch and ckpt['use_genre']==(version=='genre')
                model=DanceDecoder(nfeats=151,seq_len=1200,latent_dim=512,ff_size=1024,num_layers=8,num_heads=8,dropout=.1,cond_feature_dim=35,activation=torch.nn.functional.gelu,num_genres=16,cross_modal_fusion='adain',condition_selection=True,direction_fusion='sum',use_genre=version=='genre')
                model.load_state_dict(ckpt['model_state_dict'],strict=True);normalizer=ckpt['normalizer'];del ckpt
                model=model.cuda().eval();loaded=entry['checkpoint']
            pred=[];clips=[];torch.manual_seed(42);torch.cuda.manual_seed(42)
            with torch.no_grad():
                for i in range(0,84,2):
                    cond=test.data['features'][i:i+2].cuda()
                    genre=torch.as_tensor(test.data['genres'][i:i+2],device='cuda',dtype=torch.long) if version=='genre' else None
                    y=torch.randn((len(cond),1020,151),device='cuda');times=torch.linspace(1,0,n+1,device='cuda')
                    for step in range(n):
                        t=torch.full((len(cond),),times[step],device='cuda');r=torch.full_like(t,times[step+1]);h=t-r
                        y0=model.infer_pred(y,cond,genre,t,h);y,_=FlowMatching._x_r_and_velocity(y,y0,t[:,None,None],r[:,None,None])
                    y=y.cpu();assert torch.isfinite(y).all();clips.append(float((y.abs()>1).float().mean()))
                    p,q=pose_to_root_and_rotations(normalizer.unnormalize(y));q,p=rotate_pred_like_gt(q,p);pred.append(smpl.forward(q,p).numpy())
            joints=np.concatenate(pred);clip=float(np.mean(clips));assert np.isfinite(joints).all()
            np.savez_compressed(cache,joints=joints,clip=clip,names=names)
        fcache=OUT/'features'/f'{key}.npz'
        if fcache.exists():
            with np.load(fcache) as z:feats={k:z[k] for k in ['k','g']}
        else:feats=extract(joints,pool)
        row=dict(**entry,coverage=84,frames=1020,selection_split='test',protocol='corrected_eval_20260930',sampling_seed=42,normalized_clip_fraction=clip,BeatAlignment=float(calculate_beat_similarity(beats,torch.from_numpy(joints))),PFC=pfc(joints),**gtmetrics)
        for k in ['k','g']:
            m,s=stats[k];pn=(feats[k]-m)/s;gn=(gt[k]-m)/s
            row['FID_'+k]=fid(pn,gn);row['Diversity_'+k]=diversity(pn)
        assert all(np.isfinite(v) for v in row.values() if isinstance(v,float))
        np.savez_compressed(OUT/'features'/f'{key}.npz',**feats,names=names)
        dump(target,row);rank_all();print('RESULT',json.dumps(row),flush=True)
    pool.shutdown();rank_all();dump(OUT/'status.json',dict(state='complete',completed=len(plan),total=len(plan),time=time.time()));print('COMPLETE',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        if OUT.exists():dump(OUT/'failure.json',dict(error=repr(exc),time=time.time()))
        raise
