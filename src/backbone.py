"""
Acknowledgements:
1. https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
2. https://github.com/IBM/CrossViT
3. https://github.com/rishikksh20/CrossViT-pytorch
"""

from torch import nn
import timm
from timm.models.vision_transformer import VisionTransformer
import torch


class EncoderViT(nn.Module):
    def __init__(self, num_classes=256, feature_dim=768, encoder_backbone='vit_base_patch16_224'):
        super().__init__()
        self.encoder: VisionTransformer = timm.create_model(encoder_backbone, pretrained=True)
        self.num_blocks = 196
        self.mlp_block = nn.Sequential(
            nn.LayerNorm(self.num_blocks),
            nn.Linear(self.num_blocks, 1),
            nn.GELU()
        )
        self.mlp_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Dropout(),
            nn.Linear(feature_dim, num_classes),
            nn.GELU(),
            nn.LayerNorm(num_classes),
            nn.Linear(num_classes, num_classes)
        )
        # self.alpha = nn.Parameter(torch.tensor([1.]))

    def embedding(self, image):
        x = self.encoder.patch_embed(image)
        cls_token = self.encoder.cls_token.expand(x.shape[0], -1, -1)
        if self.encoder.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.encoder.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.encoder.pos_drop(x + self.encoder.pos_embed)
        x = self.encoder.blocks(x)
        x = self.encoder.norm(x)

        return x

    def block_feature(self, feat):
        # return feat[:, 0] + self.alpha * (self.mlp_block(feat[:, 1:].permute(0, 2, 1)).squeeze(-1))
        return feat[:, 0] + self.mlp_block(feat[:, 1:].permute(0, 2, 1)).squeeze(-1)
        # return torch.max(feat, dim=1)[0]
        # return torch.mean(feat, dim=1)
        # return feat[:, 0]

    def forward_feature(self, image):
        vit_feat = self.embedding(image)
        mlp_feat = self.mlp_head(self.block_feature(vit_feat))

        return mlp_feat, vit_feat

    def forward(self, image):
        return self.mlp_head(self.block_feature(self.embedding(image)))


class CrossViT(nn.Module):
    def __init__(self, num_classes=256, feature_dim=768, cross_heads=[12, 12, 12]):
        super().__init__()
        self.cross_blocks = nn.ModuleList([
            CrossFeature(feature_dim=feature_dim, num_heads=cross_heads[i], qkv_bias=True)
            for i in range(len(cross_heads))])

        # self.num_blocks = 196
        # self.mlp_block = nn.Sequential(
        #     nn.LayerNorm(self.num_blocks),
        #     nn.Linear(self.num_blocks, 1),
        #     nn.GELU()
        # )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(feature_dim * 2),
            nn.Dropout(),
            nn.Linear(feature_dim * 2, num_classes),
            nn.GELU(),
            nn.LayerNorm(num_classes),
            nn.Linear(num_classes, num_classes)
        )

    # def block_feature(self, feat):
    #     return feat[:, 0] + self.mlp_block(feat[:, 1:].permute(0, 2, 1)).squeeze(-1)

    def forward_feature(self, feature_1, feature_2):
        for cross_block in self.cross_blocks:
            feature_1, feature_2 = cross_block(feature_1, feature_2)
        return feature_1, feature_2

    def forward(self, feature_1, feature_2):
        feature_1, feature_2 = self.forward_feature(feature_1, feature_2)

        return self.mlp_head(torch.cat((torch.mean(feature_1, dim=1),
                                        torch.mean(feature_2, dim=1)), dim=1))

        # return self.mlp_head(torch.cat((self.block_feature(feature_1),
        #                                 self.block_feature(feature_2)), dim=1))

        # return self.mlp_head(torch.cat((feature_1[:, 0], feature_2[:, 0]), dim=1))


class CrossFeature(nn.Module):
    def __init__(self, feature_dim=768, num_heads=12, qkv_bias=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(feature_dim)
        self.attn1 = CrossAttention(feature_dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.mlp1 = nn.Sequential(nn.LayerNorm(feature_dim),
                                  nn.Linear(feature_dim, feature_dim * 2),
                                  nn.GELU(),
                                  nn.Linear(feature_dim * 2, feature_dim))

        self.norm2 = nn.LayerNorm(feature_dim)
        self.attn2 = CrossAttention(feature_dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.mlp2 = nn.Sequential(nn.LayerNorm(feature_dim),
                                  nn.Linear(feature_dim, feature_dim * 2),
                                  nn.GELU(),
                                  nn.Linear(feature_dim * 2, feature_dim))

    def forward(self, feat_1, feat_2):
        # cross attention for feat_1 and feat_2
        cal_qkv = torch.cat((feat_2[:, 0:1], feat_1), dim=1)
        cal_out = feat_1 + self.attn1(self.norm1(cal_qkv))
        feature_1 = cal_out + self.mlp1(cal_out)

        cal_qkv = torch.cat((feat_1[:, 0:1], feat_2), dim=1)
        cal_out = feat_2 + self.attn2(self.norm2(cal_qkv))
        feature_2 = cal_out + self.mlp2(cal_out)

        return feature_1, feature_2


class CrossAttention(nn.Module):
    def __init__(self, feature_dim=768, num_heads=12, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.wq = nn.Linear(feature_dim, feature_dim, bias=qkv_bias)
        self.wk = nn.Linear(feature_dim, feature_dim, bias=qkv_bias)
        self.wv = nn.Linear(feature_dim, feature_dim, bias=qkv_bias)
        self.linear = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        B, N, C = x.shape
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.wk(x[:, 1:, ...]).reshape(B, N - 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.wv(x[:, 1:, ...]).reshape(B, N - 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)  # BH1(C/H) @ BH(C/H)N -> BH1N
        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C

        return self.linear(x)


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print('Total params: ', total_num, '\nTrainable params: ', trainable_num)


if __name__ == '__main__':
    encoder_1 = EncoderViT(num_classes=256, feature_dim=768, encoder_backbone='vit_base_patch16_224')
    encoder_2 = CrossViT(num_classes=256, feature_dim=768, cross_heads=[12, 12, 12])
    get_parameter_number(encoder_1)
    get_parameter_number(encoder_2)
    # Total params: 87,423,541
    # Trainable params: 87,423,541
    # Total params: 38,279,168
    # Trainable params: 38,279,168

    img = torch.randn((1, 3, 224, 224))
    out1, out2 = encoder_1.forward_feature(img)
    out = encoder_2(out2, out2)

    print('Done !')
