import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50
from network_vit import ViTWithResNet
from network import MGN

class CombinedModel(nn.Module):
    def __init__(self, num_classes=107, dim=2048, depth=4, heads=16, pool='mean', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super(CombinedModel, self).__init__()

        # Shared Backbone
        resnet = resnet50(pretrained=True)
        self.shared_backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3[0]
        )

        # ViT
        self.vit_model = ViTWithResNet(image_size=224, num_classes=num_classes, dim=dim, depth=depth, heads=heads, pool=pool, channels=channels, dim_head=dim_head, dropout=dropout, emb_dropout=emb_dropout)
        # Removing the backbone of ViT
        self.vit_model.backbone = nn.Identity()

        # MGN
        self.mgn_model = MGN()
        # Remove MGN backbone
        self.mgn_model.backbone = nn.Identity()

        # Final Classification Head
        self.classifier = nn.Linear(dim + 2048, num_classes)  # Modify 2048 based on the embeddings

    def forward(self, x):
        shared_features = self.shared_backbone(x)
        
        if self.training:
            logits, vit_embeddings = self.vit_model(shared_features)
        else: 
            vit_embeddings = self.vit_model(shared_features)

        mgn_outputs = self.mgn_model(shared_features)
        mgn_predict = mgn_outputs[0]
        
        # print(vit_embeddings.shape)
        # print(mgn_predict.shape)

        # Concatenating embeddings
        combined_embeddings = torch.cat([vit_embeddings, mgn_predict], dim=1)
        # add average method
        
        # Classification
        logits = self.classifier(combined_embeddings)

        if self.training:
            return logits, combined_embeddings, mgn_outputs
        else:
            return combined_embeddings
