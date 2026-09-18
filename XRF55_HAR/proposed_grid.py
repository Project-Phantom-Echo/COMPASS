#!/usr/bin/env python3
"""Submit split-matched X-Fi pretraining followed by COMPASS validation grids."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
XFI = WORKSPACE / 'x-fi/XRF55_HAR'
CAMPAIGN = ROOT.parent / 'outputs/proposed-seed1-four-splits'
PROTOCOLS = ['unseen_subjects_001'] + [f'unseen_room_subject_rotation_{i}' for i in (1,2,3)]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def source_files():
    files = list(ROOT.glob('*.py'))
    for folder in ('models','backbone_models','dataset'):
        files += list((ROOT/folder).rglob('*.py'))
    files += list((ROOT.parent/'shared').rglob('*.py'))
    files += [ROOT/'submit_proposed_grid.sh']
    shared = WORKSPACE/'wireless-sensing/experiments-on-xrf55'
    files += [shared/name for name in ('baseline_protocols.py','split.py','subject_splits.json')]
    xmanifest = json.loads((XFI/'backbone_grid/seed1-subjects/campaign.json').read_text())
    files += [XFI/name for name in xmanifest['source_sha256']]
    return sorted(set(files))


def prepare():
    original = json.loads((XFI/'backbone_tests/final-four-splits-seeds123/campaign.json').read_text())
    cells = [t['selection']['cell'] for t in original['tasks'] if t['seed']==1]
    assert len(cells)==12 and all(c['seed']==1 for c in cells)
    CAMPAIGN.mkdir(parents=True, exist_ok=False)
    for group, protocols in [('subjects',PROTOCOLS[:1]),('rotations',PROTOCOLS[1:])]:
        folder=CAMPAIGN/group
        folder.mkdir()
        folder.chmod(0o777)  # Shared campaign output, writable by either submitting account.
        backbones=[c for c in cells if c['protocol'] in protocols]
        tasks=[dict(protocol=p,learning_rate=lr,weight_decay=wd)
               for p,lr,wd in itertools.product(protocols,(1e-5,3e-5,1e-4,3e-4,1e-3),(0.,.01,.05,.1,.3))]
        save(folder/'manifest.json',dict(backbones=backbones,tasks=tasks,seed=1,epochs=100,
             source_sha256={str(p.relative_to(WORKSPACE)):digest(p) for p in source_files()},
             backbone_policy='selected seed-1 configurations retrained on grid training samples only',
             selection='epoch-100 validation accuracy',test_data_opened=False))
        (folder/'logs').mkdir()
        (folder/'logs').chmod(0o777)
    print(CAMPAIGN)


def checked(group):
    folder=CAMPAIGN/group
    manifest=json.loads((folder/'manifest.json').read_text())
    for name,expected in manifest['source_sha256'].items():
        if digest(WORKSPACE/name)!=expected:
            raise RuntimeError(f'Source changed: {name}')
    return folder,manifest


def backbone_command(folder,cell):
    output=folder/'backbones'/cell['protocol']/cell['modality']
    args=[sys.executable,'-u',str(XFI/'search_protocol_backbone.py'),
          '--raw-root','/mnt/weka/rmkrtchyan/ws/data/XRF55',
          '--protocol-code-dir',str(XFI/'protocol_snapshot_v2'),
          '--output-dir',str(output),'--epochs','100','--workers','12','--save-model']
    for k,v in cell.items(): args += ['--'+k.replace('_','-'),str(v)]
    return args,output


def compass_command(folder,cell,index):
    values=dict(protocol=cell['protocol'],seed=1,max_epoch=100,batch_size=32,
                learning_rate=cell['learning_rate'],weight_decay=cell['weight_decay'],
                raw_data_dir='/mnt/weka/rmkrtchyan/ws/data/XRF55',
                backbone_dir=str(folder/'backbones'/cell['protocol']),
                save_dir=str(folder/'runs'),wandb_exp_name=f'grid-{index:03d}',
                num_workers=12,val_ratio=0.,final_epoch_only=True,evaluate_test=False,
                freeze_backbone=False,lambda_proxy_cls=.5,drop_prob=.7)
    return [sys.executable,'-u',str(ROOT/'train.py'),'with','task_finetune_xrf55',
            *[f'{k}={v}' for k,v in values.items()]]


def submit(group):
    folder,manifest=checked(group)
    # Each group belongs to the submitting account; fail on accidental duplicates.
    claim=folder/'submission.lock'
    claim.mkdir(exist_ok=False)
    environment=dict(os.environ,COMPASS_GRID_GROUP=group,OMP_NUM_THREADS='1')
    jobs=[]
    for stage, count in [('backbones',len(manifest['backbones'])),('grid',len(manifest['tasks']))]:
        command=['sbatch','--parsable','--export=ALL',f'--job-name=compass-{group}-{stage}',
                 f'--array=0-{count-1}',f'--chdir={ROOT}',
                 f'--output={folder}/logs/%x_%A_%a.out',f'--error={folder}/logs/%x_%A_%a.err']
        if jobs: command += [f'--dependency=afterok:{jobs[0]["job_id"]}']
        command += [str(ROOT/'submit_proposed_grid.sh'),stage]
        result=subprocess.run(command,env=environment,text=True,capture_output=True)
        if result.returncode:
            if not jobs: claim.rmdir()
            raise RuntimeError(result.stderr+'\nAlready submitted: '+str(jobs))
        job=result.stdout.strip().split(';')[0]
        jobs.append(dict(stage=stage,job_id=job,count=count,command=command))
        save(folder/'submission.json',dict(user=os.environ.get('USER'),jobs=jobs))
        print(f'{stage}: {job}, {count} tasks',flush=True)


def run(group,stage,index):
    folder,manifest=checked(group)
    if stage=='backbones':
        command,output=backbone_command(folder,manifest['backbones'][index])
        subprocess.run(command,cwd=XFI,check=True)
        report=json.loads((output/'result.json').read_text())
        if report['status']!='completed' or not (output/'model.state_dict.pt').exists():
            raise RuntimeError('Backbone failed; dependent COMPASS search must not start')
    else:
        subprocess.run(compass_command(folder,manifest['tasks'][index],index),cwd=ROOT,check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--submit',choices=('subjects','rotations'))
    parser.add_argument('--group',choices=('subjects','rotations'))
    parser.add_argument('--stage',choices=('backbones','grid'))
    parser.add_argument('--task',type=int)
    args=parser.parse_args()
    if args.prepare: prepare()
    elif args.submit: submit(args.submit)
    else: run(args.group,args.stage,args.task)


if __name__=='__main__':main()
