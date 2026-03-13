import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class AnatomicalAttentionMIL(nn.Module):
    def __init__(self, num_classes=2, num_parts=8, clinical_dim=3, L=128, num_heads=8,
                 freeze_layers=["layer1", "layer2", "layer3"]):
        """
        freeze_layers: 列表，包含要冻结的层名。
                       可选值: "conv1", "bn1", "layer1", "layer2", "layer3", "layer4"
                       默认冻结深层 layer3, layer4，保留浅层学习超声纹理。
        """
        super(AnatomicalAttentionMIL, self).__init__()

        # 1. 加载并拆解 ResNet18 Backbone
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # 使用 ModuleDict 方便按名称访问和冻结
        self.backbone = nn.ModuleDict({
            'conv1': nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            'bn1': resnet.bn1,
            'layer1': resnet.layer1,
            'layer2': resnet.layer2,
            'layer3': resnet.layer3,
            'layer4': resnet.layer4
        })
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.avgpool = resnet.avgpool

        # --- 分层冻结逻辑 ---
        self._freeze_specific_layers(freeze_layers)

        self.img_feat_dim = 512
        self.part_embedding = nn.Embedding(num_parts, 128)

        # 2. 实例映射 (将图像和部位特征映射到 Transformer 维度 L)
        self.instance_mapping = nn.Sequential(
            nn.Linear(self.img_feat_dim + 128, L),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # 3. Transformer 多头注意力层
        self.global_token = nn.Parameter(torch.randn(1, 1, L))
        self.multihead_attn = nn.MultiheadAttention(embed_dim=L, num_heads=num_heads, batch_first=True)

        # 4. FiLM 线性调制生成器
        self.film_generator = nn.Sequential(
            nn.Linear(clinical_dim, 64),
            nn.ReLU(),
            nn.Linear(64, L * 2)
        )

        # 5. 分类器
        self.classifier = nn.Sequential(
            nn.Linear(L, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def _freeze_specific_layers(self, freeze_list):
        # 默认全部解冻
        for param in self.parameters():
            param.requires_grad = True

        # 根据传入列表冻结指定层
        if freeze_list:
            for name in freeze_list:
                if name in self.backbone:
                    for param in self.backbone[name].parameters():
                        param.requires_grad = False
            print(f"--> [Model Info] Frozen layers: {freeze_list}")

    def forward(self, bag, parts, clinical):
        # A. 分层特征提取
        x = self.backbone['conv1'](bag)
        x = self.backbone['bn1'](x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.backbone['layer1'](x)
        x = self.backbone['layer2'](x)
        x = self.backbone['layer3'](x)
        x = self.backbone['layer4'](x)

        x = self.avgpool(x)
        img_feats = x.view(x.size(0), -1)

        # B. 解剖部位拼接与映射
        p_emb = self.part_embedding(parts)
        h = self.instance_mapping(torch.cat([img_feats, p_emb], dim=1))

        # C. Transformer 聚合
        h = h.unsqueeze(0)
        query = self.global_token
        bag_feat, attn_weights = self.multihead_attn(query, h, h)
        bag_feat = bag_feat.squeeze(1)

        # D. FiLM 线性调制 (临床特征校准)
        film_params = self.film_generator(clinical)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)
        fused_feat = (1 + gamma) * bag_feat + beta

        # E. 分类
        logits = self.classifier(fused_feat)

        return logits, attn_weights, bag_feat, fused_feat