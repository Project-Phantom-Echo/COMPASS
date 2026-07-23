from pathlib import Path

import torch
from torch import nn

from backbones.RGB_benchmark.RGB_ResNet import RGB_ResNet18
from backbones.depth_benchmark.depth_ResNet18 import Depth_ResNet18
from backbones.mmwave_benchmark.mmwave_point_transformer_TD import mmwave_PointTransformerReg
from backbones.lidar_benchmark.lidar_point_transformer import lidar_PointTransformer_cls
from backbones.lidar_benchmark.pointnet_util import farthest_point_sample, index_points


PROJECT_DIR = Path(__file__).resolve().parent
BACKBONE_DIR = PROJECT_DIR / "backbones"


# --------- 原版 extractor 子模块（不改） ---------
class rgb_feature_extractor(nn.Module):
    def __init__(self, rgb_model):
        super(rgb_feature_extractor, self).__init__()
        self.part = nn.Sequential(*list(rgb_model.children())[:-2])

    def forward(self, x):
        x = self.part(x).view(x.size(0), 512, -1)
        x = x.permute(0, 2, 1)
        return x


class depth_feature_extractor(nn.Module):
    def __init__(self, depth_model):
        super(depth_feature_extractor, self).__init__()
        self.part = nn.Sequential(*list(depth_model.children())[:-2])

    def forward(self, x):
        x = self.part(x).view(x.size(0), 512, -1)
        x = x.permute(0, 2, 1)
        return x


class mmwave_feature_extractor(nn.Module):
    def __init__(self, mmwave_model):
        super(mmwave_feature_extractor, self).__init__()
        self.part = nn.Sequential(*list(mmwave_model.children())[:-1])

    def forward(self, x):
        x, _ = self.part(x)
        return x


class lidar_feature_extractor(nn.Module):
    def __init__(self, lidar_model):
        super(lidar_feature_extractor, self).__init__()
        npoints, nblocks, nneighbor, n_c, d_points = 1024, 5, 16, 51, 3
        self.fc1 = lidar_model.backbone.fc1
        self.transformer1 = lidar_model.backbone.transformer1
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(nblocks - 4):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(lidar_model.backbone.transition_downs[i])
            self.transformers.append(lidar_model.backbone.transformers[i])
        self.nblocks = nblocks

    def forward(self, x):
        xyz = x[..., :3]
        points = self.transformer1(xyz, self.fc1(x))[0]

        xyz_and_feats = [(xyz, points)]
        for i in range(self.nblocks - 4):
            xyz, points = self.transition_downs[i](xyz, points)
            points = self.transformers[i](xyz, points)[0]
            xyz_and_feats.append((xyz, points))
        points = points.view(points.size(0), -1, 512)
        return points


def selective_pos_enc(xyz, npoint):
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint]
    torch.cuda.empty_cache()
    new_xyz = index_points(xyz, fps_idx)
    torch.cuda.empty_cache()
    return new_xyz


class linear_projector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(linear_projector, self).__init__()
        self.rgb_linear_projection = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(49, 32),
            nn.ReLU()
        )
        self.depth_linear_projection = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(49, 32),
            nn.ReLU()
        )
        self.mmwave_linear_projection = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        self.lidar_linear_projection = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        self.pos_enc_layer = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )

    def forward(self, feature_list, lidar_points, modality_list):
        feature_flag = 0
        for i in range(len(modality_list)):
            if modality_list[i] == True:
                if i == 0:
                    rgb_feature = feature_list[feature_flag]
                elif i == 1:
                    depth_feature = feature_list[feature_flag]
                elif i == 2:
                    mmwave_feature = feature_list[feature_flag]
                elif i == 3:
                    lidar_feature = feature_list[feature_flag]
                feature_flag += 1
            else:
                continue

        if sum(modality_list) == 0:
            raise ValueError("At least one modality should be selected")
        else:
            projected_feature_list = []
            if modality_list[0] == True:
                projected_feature_list.append(self.rgb_linear_projection(rgb_feature.permute(0, 2, 1)))
            if modality_list[1] == True:
                projected_feature_list.append(self.depth_linear_projection(depth_feature.permute(0, 2, 1)))
            if modality_list[2] == True:
                projected_feature_list.append(self.mmwave_linear_projection(mmwave_feature.permute(0, 2, 1)))
            if modality_list[3] == True:
                projected_feature_list.append(self.lidar_linear_projection(lidar_feature.permute(0, 2, 1)))

            projected_feature = torch.cat(projected_feature_list, dim=2).permute(0, 2, 1)
            # projected_feature: [B, 32*n, 512]

            if modality_list[3] == True:
                feature_shape = projected_feature.shape
                new_xyz = selective_pos_enc(lidar_points, feature_shape[1])
                pos_enc = self.pos_enc_layer(new_xyz.permute(0, 2, 1)).permute(0, 2, 1)
                projected_feature += pos_enc

        return projected_feature


class EncoderToDict(nn.Module):
    """
    输入：inputs(dict) 例如：
      {
        "rgb":   Tensor or None,
        "depth": Tensor or None,
        "mmwave":Tensor or None,
        "lidar": Tensor or None,   # 注意 lidar points 本身也用于 pos_enc
      }

    输出：dict，每个模态一个 [B, 32, 512]
    """
    def __init__(self):
        super().__init__()

        rgb_model = RGB_ResNet18()
        rgb_model.load_state_dict(
            torch.load(BACKBONE_DIR / "RGB_benchmark" / "RGB_Resnet18.pt", map_location="cpu")
        )
        self.rgb_extractor = rgb_feature_extractor(rgb_model)

        depth_model = Depth_ResNet18()
        depth_model.load_state_dict(
            torch.load(BACKBONE_DIR / "depth_benchmark" / "depth_Resnet18.pt", map_location="cpu")
        )
        self.depth_extractor = depth_feature_extractor(depth_model)

        mmwave_model = mmwave_PointTransformerReg()
        mmwave_model.load_state_dict(
            torch.load(BACKBONE_DIR / "mmwave_benchmark" / "mmwave_all_random_TD.pt", map_location="cpu")
        )
        self.mmwave_extractor = mmwave_feature_extractor(mmwave_model)

        lidar_model = lidar_PointTransformer_cls(root=str(PROJECT_DIR))
        lidar_model.load_state_dict(
            torch.load(BACKBONE_DIR / "lidar_benchmark" / "lidar_all_random.pt", map_location="cpu")
        )
        self.lidar_extractor = lidar_feature_extractor(lidar_model)

        self.projector = linear_projector(512, 512)

        # 固定顺序：与原版 modality_list 对齐
        self.mod_order = ["rgb", "depth", "mmwave", "lidar"]

    def forward(self, inputs: dict) -> dict:
        # 1) 生成 modality_list（按原版顺序）
        modality_list = []
        for m in self.mod_order:
            modality_list.append(m in inputs and inputs[m] is not None)

        if sum(modality_list) == 0:
            raise ValueError("At least one modality should be selected")

        # 2) extractor -> feature_list（顺序与原版一致：按 modality_list True 的顺序 append）
        feature_list = []
        if modality_list[0]:
            feature_list.append(self.rgb_extractor(inputs["rgb"]))       # [B,49,512]
        if modality_list[1]:
            feature_list.append(self.depth_extractor(inputs["depth"]))   # [B,49,512]
        if modality_list[2]:
            feature_list.append(self.mmwave_extractor(inputs["mmwave"])) # [B,32,512]
        if modality_list[3]:
            feature_list.append(self.lidar_extractor(inputs["lidar"]))   # [B,32,512]

        # 3) 原版 projector：concat -> [B, 32*n, 512]，并在 lidar 存在时整体加 pos_enc
        lidar_points = inputs["lidar"] if modality_list[3] else None
        projected = self.projector(feature_list, lidar_points, modality_list)  # [B,32*n,512]

        # 4) 把 concat 的 token 按每模态 32 个 token 切开，返回 dict
        out = {}
        cursor = 0
        for i, m in enumerate(self.mod_order):
            if modality_list[i]:
                out[m] = projected[:, cursor:cursor + 32, :]  # [B,32,512]
                cursor += 32

        return out
