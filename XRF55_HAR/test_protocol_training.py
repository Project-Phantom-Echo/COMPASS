import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
import train
from config import BASE_CONFIG


class TinyFusion(nn.Module):
    def __init__(self,**kwargs):super().__init__();self.fc=nn.Linear(1,55)
    def forward(self,inputs,missing_mask=None):return self.fc(inputs['wifi']),{},{}


def test_validation_logged_every_epoch_on_explicit_split():
    with TemporaryDirectory() as tmp:
        ds=[(np.ones(1,dtype=np.float32),)*3+(0,) for _ in range(4)]
        config=dict(BASE_CONFIG,protocol='unseen_subjects_001',raw_data_dir=tmp,
                    save_dir=tmp,device='cpu',gpu_ids=[],num_workers=0,max_epoch=2,batch_size=2,
                    warmup=0,simulate_missing=False,lambda_vicreg=0.,lambda_proxy_cls=0.)
        meta={'protocol':config['protocol'],'train_membership':{}}
        with patch.object(train,'collect_shared',return_value=({'train':ds,'validation':ds},meta)) as collect, \
             patch.object(train,'FusionDataset',side_effect=lambda x:x), \
             patch.object(train,'load_custom_encoders',return_value=nn.ModuleDict()), \
             patch.object(train,'XRF55_CMPT_Net',TinyFusion):
            train.main(config)
        assert collect.call_count==1
        paths=list(Path(tmp).rglob('validation_history.json'))
        assert len(paths)==1
        history=json.loads(paths[0].read_text())
        assert [r['epoch'] for r in history]==[1,2]
        assert all(np.isfinite(r['validation_loss']) for r in history)
        assert not json.loads(paths[0].with_name('result.json').read_text())['test_data_opened']


def test_real_compass_accepts_protocol_backbone_models():
    from backbone_models.mmWave.ResNet import resnet18 as mmwave
    from backbone_models.WIFI.ResNet import resnet18 as wifi
    from backbone_models.RFID.ResNet import resnet18 as rfid
    models={'mmwave':mmwave(),'wifi':wifi(),'rfid':rfid()}
    with patch.object(train,'load_backbone_runs',return_value=(models,{})):
        encoders=train.load_custom_encoders(torch.device('cpu'),backbone_dir='fixture',
                    protocol_metadata={'protocol':'unseen_subjects_001','train_membership':{}})
    model=train.XRF55_CMPT_Net(task_encoders=encoders,task_decoder=([512,55],'classification'),
                              proj_dim=32,embed_dim=512).eval()
    inputs={'mmwave':torch.zeros(1,17,256,128),'wifi':torch.zeros(1,270,1000),'rfid':torch.zeros(1,23,148)}
    with torch.inference_mode():logits,_,_=model(inputs,missing_mask=None)
    assert logits.shape==(1,55) and torch.isfinite(logits).all()
