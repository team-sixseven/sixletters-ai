"""Run one search trial per unfinished ECG combination per round; gate test at 60 epochs."""
import json,os,sys,traceback,subprocess
from pathlib import Path
import optuna,yaml
from optuna.trial import TrialState
base=Path('/workspace/dl_six');repo=base/'repo'; sys.path[:0]=[str(repo),str(repo/'Members/sang/runpod/dl_optuna_search')]
from Members.sang.runpod.dl_optuna_search.multi_study import COMBINATIONS,write_configs
from Members.sang.model.DL.raw_pipeline import train_raw_ecg
from Members.sang.model.evaluation.evaluation import evaluate_saved_model
os.umask(0o077)
out=base/'six_search';out.mkdir(mode=0o700,exist_ok=True)
search=yaml.safe_load((repo/'Members/sang/runpod/dl_optuna_search/multi_search.yaml').read_text())['search']
samplers={name:optuna.samplers.TPESampler(seed=20261105+i) for i,(name,_,_) in enumerate(COMBINATIONS)}
status_path=out/'status.json'; state={'criteria':'epochs >= 60; early stopping patience 10; validation-loss checkpoint; fixed threshold 0.5; historical test reuse','combinations':{}}
def save():
 tmp=status_path.with_suffix('.tmp');tmp.write_text(json.dumps(state,ensure_ascii=False,indent=2));tmp.replace(status_path)
def render():
 subprocess.run(['/tmp/dl_six_venv/bin/python','/workspace/dl_six/render_dl_six.py'],check=False)
def main():
 for round_index in range(1,2):
  for idx,(name,cnn,sequence) in enumerate(COMBINATIONS):
   previous=state['combinations'].get(name,{})
   if previous.get('test_status')=='complete':continue
   folder=out/name;folder.mkdir(exist_ok=True)
   study=optuna.create_study(direction='maximize',sampler=samplers[name],storage='sqlite:///'+str(folder/'study.sqlite3'),study_name=name,load_if_exists=True)
   completed=[t for t in study.trials if t.state==TrialState.COMPLETE]
   eligible=[t for t in completed if t.user_attrs.get('completed_epochs',0)>=60]
   if not eligible:
    def objective(trial):
     state['combinations'][name]={'stage':'training','trial':trial.number,'round':round_index,'completed_trials':len(completed)};save()
     choice=write_configs(trial,folder,base/'snapshot',cnn,sequence,search)
     cfg=yaml.safe_load((choice.parent/'training.yaml').read_text())
     if cfg['max_segments_per_patient'] != 128: raise RuntimeError('snapshot supports 128 selected segments only')
     run=train_raw_ecg(choice,run_id=f'trial_{trial.number:03d}',device_override='cuda',mlflow_logging=False)
     metadata=json.loads((run/'metadata.json').read_text());metrics=json.loads((run/'metrics.json').read_text())['validation']
     trial.set_user_attr('run_dir',str(run));trial.set_user_attr('completed_epochs',metadata['completed_epochs']);trial.set_user_attr('best_epoch',metadata['best_epoch'])
     print('FINISHED',name,trial.number,'epochs',metadata['completed_epochs'],'val_ap',metrics['average_precision'],flush=True)
     return float(metrics['average_precision'])
    study.optimize(objective,n_trials=1,n_jobs=1)
    completed=[t for t in study.trials if t.state==TrialState.COMPLETE]
    eligible=[t for t in completed if t.user_attrs.get('completed_epochs',0)>=60]
   if not eligible:
    state['combinations'][name]={'stage':'needs_60_epochs','test_status':'held','completed_trials':len(completed),'max_epochs':max(t.user_attrs['completed_epochs'] for t in completed),'trial':completed[-1].number,'run_dir':completed[-1].user_attrs['run_dir'],'best_epoch':completed[-1].user_attrs['best_epoch'],'validation_ap':completed[-1].value};save();render();continue
   best=max(eligible,key=lambda t:t.value);run=Path(best.user_attrs['run_dir']);trial_name=f'trial_{best.number:03d}'
   cfg={'basic_config':str(folder/'configs'/trial_name/'basic.yaml'),'evaluation':{'target':'dl','artifact_dir':str(run),'split':'test','threshold_override':None,'results_directory':str(folder/'evaluation'),'mlflow_logging':False,'metrics':['auroc','average_precision','accuracy','precision','recall','sensitivity','specificity','f1','confusion_matrix']}}
   p=folder/'selected_test.yaml';p.write_text(yaml.safe_dump(cfg,sort_keys=False));state['combinations'][name]={'stage':'test_evaluation','trial':best.number,'completed_epochs':best.user_attrs['completed_epochs'],'completed_trials':len(completed)};save()
   result=evaluate_saved_model(p,run_id=f'{trial_name}_test')
   state['combinations'][name]={'stage':'complete','test_status':'complete','trial':best.number,'run_dir':str(run),'test_dir':str(result),'completed_epochs':best.user_attrs['completed_epochs'],'best_epoch':best.user_attrs['best_epoch'],'validation_ap':best.value,'completed_trials':len(completed)};save();render()
  if len(state['combinations'])==len(COMBINATIONS) and all(x.get('test_status')=='complete' for x in state['combinations'].values()):
   print('ALL_COMBINATIONS_COMPLETE',flush=True);return
 print('SIX_TRIALS_FINISHED_TEST_GATING_REMAINS',flush=True)
try:main()
except Exception as e:
 state['error']={'type':type(e).__name__,'message':str(e)};save();traceback.print_exc();raise
