import torch
from torch.utils.data import Dataset
import numpy as np
import os
import glob


class XRF55_Dataset(Dataset):
    def __init__(self, root_dir, split='train', scene='all'):
        """
        Args:
            root_dir (string): Dataset root directory.
            split (string): 'train' or 'test'.
            scene: 'all' (Scene1-4) or list like ['Scene1', 'Scene2'].
        """
        super(XRF55_Dataset, self).__init__()

        if scene == "all":
            scene = ["Scene1", "Scene2", "Scene3", "Scene4"]

        if split == 'train':
            self.path = os.path.join(root_dir, 'train_data')
        else:
            self.path = os.path.join(root_dir, 'test_data')

        self.RFID_name_list = []
        for s in scene:
            # X-Fi compatible: {split}/RFID/Scene{N}/Scene{N}/*.npy
            sub_list = glob.glob(os.path.join(self.path, 'RFID', s, s, '*.npy'))
            sub_list.sort()
            self.RFID_name_list += sub_list

        print(f"[XRF55_Dataset] split={split}, scenes={scene}, samples={len(self.RFID_name_list)}")

    def __len__(self):
        return len(self.RFID_name_list)

    def __getitem__(self, idx):
        RFID_file_name = self.RFID_name_list[idx]
        WIFI_file_name = RFID_file_name.replace('RFID', 'WiFi')
        mmWave_file_name = RFID_file_name.replace('RFID', 'mmWave')

        wifi_data = np.load(WIFI_file_name)
        rfid_data = np.load(RFID_file_name)
        mmwave_data = np.load(mmWave_file_name).reshape(17, 256, 128)
        label = int(os.path.basename(RFID_file_name).split('_')[1]) - 1

        return wifi_data, rfid_data, mmwave_data, label
