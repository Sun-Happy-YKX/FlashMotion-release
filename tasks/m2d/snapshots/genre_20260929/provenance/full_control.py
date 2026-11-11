import os,sys,json,socket,subprocess,fcntl,time,hashlib
from pathlib import Path
R=Path('/efs/yangkaixing/FlowerDance_x0_ablation/runs/music_genre_ablation_20260929');C=R/'code';P=C/'.venv/bin/python'
assert socket.gethostname()=='bubuyang-dmd-0924-102244-1x8-worker-0'
mode,version,phase=sys.argv[1:4];assert version in ('genre','no_genre') and phase in ('verify','train')
out=R/version/phase;out.mkdir(parents=True,exist_ok=True)
e=os.environ.copy();e.update(CUDA_VISIBLE_DEVICES='0,1,2,3' if version=='genre' else '4,5,6,7',OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1',NCCL_DEBUG='WARN',NCCL_IB_DISABLE='1',WANDB_MODE='offline')
cmd=[str(P),'-m','accelerate.commands.launch','--multi_gpu','--num_processes','4','--num_machines','1','--main_process_port','29974' if version=='genre' else '29975','--mixed_precision','no','--dynamo_backend','no','train_pair.py',version,phase]
if mode=='launch':
 with (R/version/'full_launch.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert not (out/'launch.json').exists(),'Refusing duplicate launch'
  if phase=='train':
   assert json.loads((R/version/'verify/exit.json').read_text())['exit_code']==0
   assert len(list((R/version/'verify/metrics').glob('*.json')))==2
  free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
  cards=[0,1,2,3] if version=='genre' else [4,5,6,7]
  assert min([int(free.split()[i]) for i in cards])>60000,free
  with (out/'stdout.log').open('ab') as log:
   p=subprocess.Popen([str(P),str(R/'full_control.py'),'run',version,phase],cwd=C,env=e,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
  (out/'launch.json').write_text(json.dumps(dict(pid=p.pid,time=time.time(),command=cmd,pretrained=None,resume=None,script_sha256=hashlib.sha256((C/'train_pair.py').read_bytes()).hexdigest()),indent=2));print(version,phase,p.pid)
elif mode=='run':
 with (R/version/'training_worker.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  rc=subprocess.call(cmd,cwd=C,env=e)
  (out/'exit.json').write_text(json.dumps(dict(exit_code=rc,time=time.time())))
  sys.exit(rc)
