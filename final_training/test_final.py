"""Exercise the frozen final-training code without opening experiment test data."""
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import pytest
from types import SimpleNamespace

import numpy as np
from torch import nn

REPO=Path(__file__).resolve().parents[1]
CAMPAIGN=REPO/'outputs/final-rotations-seeds123'
sys.path.insert(0,str(REPO))
sys.path.insert(0,str(REPO/'XRF55_HAR'))
sys.path.append(str(REPO.parent/'wireless-sensing/experiments-on-xrf55'))
from config import BASE_CONFIG


@pytest.fixture
def train(tmp_path):
    root=tmp_path/'source'
    source=root/'compass/XRF55_HAR/train.py'
    source.parent.mkdir(parents=True)
    source.write_text((REPO/'XRF55_HAR/train.py').read_text())
    subprocess.run(['patch','--batch','-p0','-i',str(REPO/'final_training/train-validation.patch')],
                   cwd=root,check=True,capture_output=True)
    spec=importlib.util.spec_from_file_location('compass_final_train_test',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def samples(ids):
    return {mod:[SimpleNamespace(identity=('Scene1',i,1,1),label=0) for i in ids]
            for mod in ('WiFi','RFID','mmWave')}


def test_final_union_and_test_after_training(monkeypatch,tmp_path,train):
    seen=[];steps=[];calls=[]
    class Tiny(nn.Module):
        def __init__(self,**kwargs):super().__init__();self.fc=nn.Linear(1,55)
        def forward(self,inputs,missing_mask=None):
            if self.training:steps.append(1)
            return self.fc(inputs['wifi']),{},{}
    class Data:
        def __init__(self,values):self.values=values['WiFi']
        def __len__(self):return len(self.values)
        def __getitem__(self,index):
            identity=self.values[index].identity[1]
            if identity in (5,7):assert len(steps)==4
            seen.append(identity)
            return (np.ones(1,dtype=np.float32),)*3+(0,)
    def collect(root,protocol,roles=('train','validation')):
        calls.append(roles)
        if roles==('test',):
            assert len(steps)==4
            return {'test':samples([5,7])},{}
        return {'train':samples([1,30]),'validation':samples([3,6])},{
            'protocol':protocol,'train_membership':{},'counts':{'train':2,'validation':2}}
    monkeypatch.setattr(train,'collect_shared',collect)
    monkeypatch.setattr(train,'FusionDataset',Data)
    monkeypatch.setattr(train,'load_custom_encoders',lambda *args,**kwargs:nn.ModuleDict())
    monkeypatch.setattr(train,'XRF55_CMPT_Net',Tiny)
    config=dict(BASE_CONFIG,protocol='unseen_room_subject_rotation_1',raw_data_dir=str(tmp_path),
                save_dir=str(tmp_path),device='cpu',gpu_ids=[],num_workers=0,max_epoch=2,batch_size=2,
                warmup=0,simulate_missing=False,lambda_vicreg=0.,lambda_proxy_cls=0.,
                train_with_validation=True,evaluate_test=True)
    train.main(config)
    assert calls==[('train','validation'),('test',)]
    assert all(seen.count(i)==2 for i in (1,30,3,6))
    assert all(seen.count(i)==1 for i in (5,7))
    path=next(tmp_path.glob('*/result.json'));j=json.loads(path.read_text())
    assert j['status']=='completed' and j['validation_history']==[]
    assert len(j['training_history'])==2
    assert j['train_with_validation'] and j['test_data_opened']
    results=json.loads(path.with_name('robustness_results.json').read_text())
    assert len(results['test_sets']['test'])==7
    assert all(v['samples']==2 for v in results['test_sets']['test'].values())


def test_commands_and_source_guards():
    spec=importlib.util.spec_from_file_location('final_launch',Path(__file__).with_name('launch.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    manifest=json.loads((REPO/'final_training/campaign/manifest.json').read_text())
    assert module.digest(REPO/'final_training/train-validation.patch') == manifest['source_sha256']['train-validation.patch']
    assert len(manifest['tasks'])==9
    assert len({(t['protocol'],t['seed']) for t in manifest['tasks']})==9
    for i,t in enumerate(manifest['tasks']):
        command=module.command(CAMPAIGN,manifest,i)
        assert 'train_with_validation=True' in command
        assert 'evaluate_test=True' in command
        assert 'max_epoch=100' in command
        assert 'freeze_backbone=False' in command
        assert t['seed'] in (1,2,3)
        assert t['train_samples'] in (33000,34100)
        assert t['test_samples']==3300
