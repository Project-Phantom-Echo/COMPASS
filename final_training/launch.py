#!/usr/bin/env python3
"""Freeze rotation winners and launch three-seed final COMPASS retraining."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
CAMPAIGN = REPO/'outputs/final-rotations-seeds123'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def prepare():
    grid=REPO/'outputs/proposed-seed1-four-splits/rotations'
    manifest=json.loads((grid/'manifest.json').read_text())
    evidence=[]
    for index,cell in enumerate(manifest['tasks']):
        paths=list((grid/'runs').glob(f'grid-{index:03d}_*/result.json'))
        if len(paths)!=1:raise ValueError(f'Task {index}: expected one result')
        result=json.loads(paths[0].read_text());h=result['validation_history']
        if result['status']!='completed' or result['test_data_opened'] or result['protocol']!=cell['protocol']:
            raise ValueError(f'Task {index}: invalid result')
        if [e['epoch'] for e in h]!=list(range(1,101)) or any(not math.isfinite(e[k]) for e in h for k in ('train_loss','train_accuracy','validation_loss','validation_accuracy')):
            raise ValueError(f'Task {index}: incomplete/nonfinite history')
        evidence.append(dict(task=index,cell=cell,result_path=str(paths[0]),result_sha256=digest(paths[0]),
                             validation_accuracy=h[-1]['validation_accuracy']))
    selected=[max((r for r in evidence if r['cell']['protocol']==p),key=lambda r:r['validation_accuracy'])
              for p in dict.fromkeys(t['protocol'] for t in manifest['tasks'])]
    if len(evidence)!=75 or len(selected)!=3:raise ValueError('Expected 75 runs and three winners')
    inventory=WORKSPACE/'wireless-sensing/experiments-on-xrf55/jobs/proposed-seed1-four-splits/dataset.json'
    inv=json.loads(inventory.read_text());weights={};tasks=[]
    for row in selected:
        protocol=row['cell']['protocol']
        splits=inv['protocols'][protocol]['splits']
        for modality,release in [('wifi','WiFi'),('rfid','RFID'),('mmwave','mmWave')]:
            members={role:splits[role]['modalities'][release]['members'] for role in ('train','validation','test')}
            ids={role:{tuple(inv['files'][p]['identity']) for p in values} for role,values in members.items()}
            if ids['train']&ids['validation'] or (ids['train']|ids['validation'])&ids['test']:
                raise ValueError('Overlapping split identities')
            folder=grid/'backbones'/protocol/modality
            report=json.loads((folder/'result.json').read_text())
            if report['status']!='completed' or report['test_data_opened'] or report['protocol']!=protocol:
                raise ValueError('Invalid pretrained backbone')
            for name in ('result.json','model.state_dict.pt'):weights[str(folder/name)]=digest(folder/name)
        for seed in (1,2,3):
            tasks.append(dict(**row['cell'],seed=seed,backbone_dir=str(grid/'backbones'/protocol),
                              train_samples=len(members['train'])+len(members['validation']),test_samples=len(members['test']),
                              selection_task=row['task']))
    CAMPAIGN.mkdir(parents=True,exist_ok=False);CAMPAIGN.chmod(0o777)
    for folder in ('logs','runs'):
        (CAMPAIGN/folder).mkdir();(CAMPAIGN/folder).chmod(0o777)
    for name,expected in manifest['source_sha256'].items():
        original=WORKSPACE/name
        if digest(original)!=expected:raise ValueError(f'Search source changed: {name}')
        target=CAMPAIGN/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original,target)
    target=CAMPAIGN/'source/compass/XRF55_HAR/train.py'
    target.write_text(target.read_text())  # Normalize the released mixed line endings before patching.
    patch=REPO/'final_training/train-validation.patch'
    subprocess.run(['patch','--batch','--forward','-p0','-i',str(patch)],cwd=CAMPAIGN/'source',check=True)
    shutil.copyfile(inventory,CAMPAIGN/'dataset.json')
    for source,destination in [('launch.py','launcher.py'),('submit.sh','submit.sh'),('train-validation.patch','train-validation.patch')]:
        shutil.copyfile(REPO/'final_training'/source,CAMPAIGN/destination)
    hashes={str(p.relative_to(CAMPAIGN)):digest(p) for p in (CAMPAIGN/'source').rglob('*') if p.is_file()}
    hashes.update({name:digest(CAMPAIGN/name) for name in ('launcher.py','submit.sh','train-validation.patch','dataset.json')})
    write(CAMPAIGN/'manifest.json',dict(tasks=tasks,selected=selected,evidence=evidence,
          source_sha256=hashes,backbone_sha256=weights,raw_root=inv['raw_root_hint'],
          epochs=100,batch_size=32,training='exact train+validation union',
          initialization='same train-only seed-1 backbones as tuning; fusion/fine-tuning seeds 1/2/3',
          selection='epoch-100 validation accuracy; ties by original task order',
          test_policy='fixed final checkpoint, seven modality subsets after training',
          python=str(REPO/'.venv/bin/python')))
    print(f'Prepared {len(tasks)} jobs at {CAMPAIGN}')


def verify(campaign):
    m=json.loads((campaign/'manifest.json').read_text())
    for name,h in m['source_sha256'].items():
        if digest(campaign/name)!=h:raise ValueError(f'Frozen source changed: {name}')
    for name,h in m['backbone_sha256'].items():
        if digest(name)!=h:raise ValueError(f'Backbone changed: {name}')
    return m


def command(campaign,m,index):
    task=m['tasks'][index]
    config=dict(protocol=task['protocol'],seed=task['seed'],max_epoch=100,batch_size=32,
                learning_rate=task['learning_rate'],weight_decay=task['weight_decay'],
                raw_data_dir=m['raw_root'],backbone_dir=task['backbone_dir'],
                save_dir=str(campaign/'runs'),wandb_exp_name=f'final-{index:02d}',
                num_workers=12,val_ratio=0.,final_epoch_only=True,evaluate_test=True,
                train_with_validation=True,freeze_backbone=False,lambda_proxy_cls=.5,drop_prob=.7)
    return [m['python'],'-u',str(campaign/'source/compass/XRF55_HAR/train.py'),'with','task_finetune_xrf55',
            *[f'{k}={v}' for k,v in config.items()]]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,default=CAMPAIGN)
    a=p.add_mutually_exclusive_group(required=True)
    a.add_argument('--prepare',action='store_true');a.add_argument('--submit',action='store_true');a.add_argument('--task',type=int)
    p.add_argument('--inspect',action='store_true');args=p.parse_args();campaign=args.campaign.resolve()
    if args.prepare:prepare();return
    m=verify(campaign)
    if args.submit:
        claim=campaign/'submission.lock';claim.mkdir(exist_ok=False)
        cmd=['sbatch','--parsable','--export=ALL',f'--array=0-{len(m["tasks"])-1}',
             f'--output={campaign}/logs/%x_%A_%a.out',f'--error={campaign}/logs/%x_%A_%a.err',
             str(campaign/'submit.sh'),str(campaign),m['python']]
        result=subprocess.run(cmd,text=True,capture_output=True)
        if result.returncode:claim.rmdir();raise RuntimeError(result.stderr)
        write(campaign/'submission.json',dict(job_id=result.stdout.strip(),user=os.environ.get('USER'),command=cmd))
        print(result.stdout.strip());return
    if args.task<0 or args.task>=len(m['tasks']):raise ValueError('Invalid task')
    cmd=command(campaign,m,args.task)
    if args.inspect:print(json.dumps(cmd));return
    claim=campaign/f'task-{args.task:02d}.lock';claim.mkdir(exist_ok=False)
    raise SystemExit(subprocess.call(cmd,cwd=campaign/'source/compass/XRF55_HAR'))


if __name__=='__main__':main()
