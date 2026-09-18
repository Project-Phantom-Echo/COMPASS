from pathlib import Path
from proposed_grid import backbone_command, compass_command


def test_backbone_is_train_only_and_saves_weights():
    cell=dict(protocol='unseen_subjects_001',modality='wifi',seed=1,batch_size=16,
              learning_rate=.001,weight_decay=.3,dropout=0.)
    command,output=backbone_command(Path('/tmp/example'),cell)
    assert '--save-model' in command
    assert command[command.index('--seed')+1]=='1'
    assert command[command.index('--epochs')+1]=='100'
    assert command[2].endswith('/search_protocol_backbone.py')
    assert output==Path('/tmp/example/backbones/unseen_subjects_001/wifi')


def test_grid_is_validation_only_final_epoch():
    cell=dict(protocol='unseen_room_subject_rotation_2',learning_rate=.0001,weight_decay=.05)
    command=compass_command(Path('/tmp/example'),cell,7)
    for value in ['seed=1','max_epoch=100','batch_size=32','final_epoch_only=True',
                  'evaluate_test=False','freeze_backbone=False','val_ratio=0.0']:
        assert value in command
    assert 'backbone_dir=/tmp/example/backbones/unseen_room_subject_rotation_2' in command
