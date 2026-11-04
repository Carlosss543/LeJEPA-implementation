import timm
import torch.nn as nn
from torchvision.ops import MLP


class ViTEncoder(nn.Module):

    def __init__(self, proj_dim=16):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch8_224",
            pretrained=False,
            num_classes=512,
            drop_path_rate=0.1,
            img_size=128,
        )
        self.proj = MLP(512, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x):
        B, V = x.shape[:2]
        emb = self.backbone(x.flatten(0, 1))
        proj = self.proj(emb).reshape(B, V, -1).transpose(0, 1)
        return emb, proj
