import torch.nn as nn
from torchvision.models import resnet50
from transformers import ViTModel

# $env:PYTHONPATH = "C:/Users/22498/Desktop/feature-aggregation-master/feature aggregation/maskrcnn_benchmark"

class ResNet50ViT(nn.Module):
    def __init__(self, config, num_classes=1000):
        super(ResNet50ViT, self).__init__()
        
        resnet = resnet50(pretrained=True)

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # resnet_out_features = 2048
        
        self.vit = ViTModel(config)
    
        vit_out_features = config.hidden_size
        self.classifier = nn.Linear(vit_out_features, num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        
        x = x.flatten(2).transpose(1, 2)
        
        x = self.vit(x)

        x = x['last_hidden_state'][:, 0]
        
        x = self.classifier(x)
        
        return x
