import torch.nn as nn
from torchvision.models import resnet50

num_classes = 107

class SimpleResNet(nn.Module):
    def __init__(self):
        super(SimpleResNet, self).__init__()

        resnet = resnet50(pretrained=True)
        # only include layer 1-4

        for param in resnet.parameters():
            param.requires_grad = False

        modules = list(resnet.children())[:-1]  # Remove last FC layer
        self.resnet = nn.Sequential(*modules)

        # add pooling layer

        self.classifier = nn.Linear(resnet.fc.in_features, num_classes) # Add custom classifier
        
        self._init_fc(self.classifier)

    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        nn.init.constant_(fc.bias, 0.)

    def forward(self, x):
        features = self.resnet(x)
        # features.shape = [16, 2048, 1, 1]
        flattened_features = features.view(features.size(0), -1)
        # reshape to [16, 2048]
        
        class_scores = self.classifier(flattened_features)
        
        if self.training:
            return class_scores, flattened_features
        return flattened_features
