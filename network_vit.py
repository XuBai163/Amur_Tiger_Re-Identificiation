import torch.nn as nn
from transformers import ViTForImageClassification
from torchvision.models.resnet import resnet50

num_classes = 107

class ViTNetwork(nn.Module):
    def __init__(self):
        super(ViTNetwork, self).__init__()

        # Backbone network: ResNet50
        resnet = resnet50(pretrained=True)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

        # Head network: ViT
        self.vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', add_pooling_layer=True)
        for param in self.vit.parameters():
            param.requires_grad = True 
        self.vit = nn.Sequential(*list(self.vit.children())[:-1]) # Remove the classification head
        self.fc = nn.Linear(768, num_classes)  # Adjust the size depending on the ViT variant's feature size

    def forward(self, x):
        # Extract features from resnet50
        x = self.backbone(x)

        # Flatten the features and pass to the ViT
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)  
        x = self.vit(x)
        
        x = self.fc(x)
        return x

