import torch
import torch.nn as nn
from transformers import ViTConfig, ViTModel
from torchvision.models.resnet import resnet50

num_classes = 107

class CustomTransformer(nn.Module):
    def __init__(self, config):
        super(CustomTransformer, self).__init__()
        # Class token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.pos_embedding = nn.Parameter(torch.zeros(1, 37, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.transformer = ViTModel(config).encoder

    def forward(self, x):
        # Add class token
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embeddings and pass through transformer
        x = x + self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x)

        return x.last_hidden_state[:, 0]

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

        config = ViTConfig(
            hidden_size=2048,
            num_hidden_layers=12, 
            num_attention_heads=16,
            intermediate_size=8192,
            image_size=214, 
            patch_size=(4,9), 
            num_labels=num_classes
        )
        self.transformer = CustomTransformer(config)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        # Extract features from resnet50
        x = self.backbone(x)
        
        # Reshape the features
        x = x.view(x.size(0), 36, 2048)

        # Pass the reshaped features through the transformer
        x = self.transformer(x)

        # Classifier
        x = self.fc(x)
        return x
