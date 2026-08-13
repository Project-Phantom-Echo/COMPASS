import torch
from torch.utils.data import Dataset
import numpy as np
import os
import glob
from pathlib import Path


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


def _scene_roots(raw_root, scene, part1_only=False):
    """Return the extracted XRF55 roots that contain one scene's modalities."""
    raw_root = Path(raw_root)
    roots = {
        'Scene1': [
            raw_root / 'part1' / 'Scene1' / 'Scene1',
            raw_root / 'part2' / 'Scene1_part2',
        ],
        'Scene2': [raw_root / 'part1' / 'Scene2' / 'Scene2'],
        'Scene3': [raw_root / 'part1' / 'Scene3' / 'Scene3'],
        'Scene4': [raw_root / 'part1' / 'Scene4' / 'Scene4'],
    }[scene]
    if scene == 'Scene1' and part1_only:
        roots = roots[:1]
    missing = [str(root) for root in roots if not root.is_dir()]
    if missing:
        raise FileNotFoundError(f'Missing XRF55 scene roots: {missing}')
    return roots


class XRF55_Protocol_Dataset(Dataset):
    """Load aligned modalities directly from extracted Part 1/Part 2 data."""

    def __init__(
        self,
        raw_root,
        scenes,
        subjects=None,
        repetitions=None,
        part1_only=False,
        split_name='unspecified',
    ):
        super().__init__()
        subjects = set(subjects) if subjects is not None else None
        repetitions = set(repetitions) if repetitions is not None else None
        samples = []
        identities = set()

        for scene in scenes:
            for scene_root in _scene_roots(raw_root, scene, part1_only):
                rfid_dir = scene_root / 'RFID'
                wifi_dir = scene_root / 'WiFi'
                mmwave_dir = scene_root / 'mmWave'
                for directory in (rfid_dir, wifi_dir, mmwave_dir):
                    if not directory.is_dir():
                        raise FileNotFoundError(directory)

                for rfid_path in sorted(rfid_dir.glob('*.npy')):
                    try:
                        person, action, repetition = map(int, rfid_path.stem.split('_'))
                    except ValueError as error:
                        raise ValueError(f'Unexpected XRF55 filename: {rfid_path}') from error
                    if subjects is not None and person not in subjects:
                        continue
                    if repetitions is not None and repetition not in repetitions:
                        continue

                    identity = (scene, person, action, repetition)
                    if identity in identities:
                        raise ValueError(f'Duplicate XRF55 sample: {identity}')
                    identities.add(identity)

                    wifi_path = wifi_dir / rfid_path.name
                    mmwave_path = mmwave_dir / rfid_path.name
                    missing = [str(path) for path in (wifi_path, mmwave_path) if not path.is_file()]
                    if missing:
                        raise FileNotFoundError(f'Unaligned XRF55 sample {identity}: {missing}')
                    samples.append((wifi_path, rfid_path, mmwave_path, action - 1))

        if not samples:
            raise ValueError(
                f'No XRF55 samples for scenes={scenes}, subjects={subjects}, '
                f'repetitions={repetitions}'
            )
        self.samples = samples
        self.identities = identities
        print(
            f'[XRF55_Protocol_Dataset] split={split_name}, scenes={scenes}, '
            f'samples={len(samples)}'
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wifi_path, rfid_path, mmwave_path, label = self.samples[idx]
        wifi_data = np.load(wifi_path)
        rfid_data = np.load(rfid_path)
        mmwave_data = np.load(mmwave_path).reshape(17, 256, 128)
        return wifi_data, rfid_data, mmwave_data, label


class XRF55_Protocol_Modality_Dataset(Dataset):
    """Single-modality view used to pretrain split-clean COMPASS backbones."""

    def __init__(self, modality, **dataset_kwargs):
        if modality not in {'wifi', 'rfid', 'mmwave'}:
            raise ValueError(f'Unsupported modality: {modality}')
        base = XRF55_Protocol_Dataset(**dataset_kwargs)
        self.modality = modality
        self.samples = base.samples
        self.identities = base.identities

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wifi_path, rfid_path, mmwave_path, label = self.samples[idx]
        path = {
            'wifi': wifi_path,
            'rfid': rfid_path,
            'mmwave': mmwave_path,
        }[self.modality]
        data = np.load(path)
        if self.modality == 'mmwave':
            data = data.reshape(17, 256, 128)
        return torch.from_numpy(np.ascontiguousarray(data)).float(), label


def make_protocol_datasets(raw_root, protocol):
    """Build the two XRF55 generalization splits used outside COMPASS Table 1."""
    if protocol == 'subject_split_21_9':
        trainset = XRF55_Protocol_Dataset(
            raw_root,
            scenes=['Scene1'],
            subjects=range(1, 22),
            split_name='subjects_01_21',
        )
        testsets = {
            'subjects_22_30': XRF55_Protocol_Dataset(
                raw_root,
                scenes=['Scene1'],
                subjects=range(22, 31),
                split_name='subjects_22_30',
            )
        }
        expected = (23100, {'subjects_22_30': 9900})
    elif protocol == 'scene_split':
        trainset = XRF55_Protocol_Dataset(
            raw_root,
            scenes=['Scene1'],
            repetitions=range(1, 15),
            split_name='Scene1_trials_01_14',
        )
        testsets = {
            scene: XRF55_Protocol_Dataset(
                raw_root,
                scenes=[scene],
                split_name=scene,
            )
            for scene in ('Scene2', 'Scene3', 'Scene4')
        }
        expected = (23100, {'Scene2': 3300, 'Scene3': 3300, 'Scene4': 3300})
    else:
        raise ValueError(f'Unsupported raw-data protocol: {protocol}')

    expected_train, expected_tests = expected
    observed = (len(trainset), {name: len(data) for name, data in testsets.items()})
    if observed != expected:
        raise ValueError(f'Unexpected {protocol} counts: expected={expected}, observed={observed}')
    return trainset, testsets
