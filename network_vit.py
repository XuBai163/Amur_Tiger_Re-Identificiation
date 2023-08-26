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

        # Upsampling layer
        self.upsample = nn.Upsample((224, 224), mode='bilinear', align_corners=True)

        # Head network: ViT
        self.vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
        for param in self.vit.parameters():
            param.requires_grad = True 
        self.vit = nn.Sequential(*list(self.vit.children())[:-1])  # Remove head
        self.fc = nn.Linear(768, num_classes)  # Adjust the size 

    def forward(self, x):
        # Extract features from resnet50
        x = self.backbone(x)

        # Upsample the features
        x = self.upsample(x)

        # Flatten the features and pass to the ViT
        x = self.vit(x)
        x = self.fc(x)
        return x