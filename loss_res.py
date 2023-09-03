import torch.nn as nn
from torch.nn import CrossEntropyLoss
from utils.triplet_loss import TripletLoss

class ResnetLoss(nn.Module):
    def __init__(self):
        super(ResnetLoss, self).__init__()
        self.cross_entropy_loss = CrossEntropyLoss()
        self.triplet_loss = TripletLoss(margin=1.2)

    def forward(self, scores, features, labels):        
        triplet_loss_value = self.triplet_loss(features, labels)

        ce_loss_value = self.cross_entropy_loss(scores, labels)

        total_loss = triplet_loss_value + 2 * ce_loss_value

        print('\rtotal loss:%.2f  Triplet_Loss:%.2f  CrossEntropy_Loss:%.2f' % (
            total_loss.data.cpu().numpy(),
            triplet_loss_value.data.cpu().numpy(),
            ce_loss_value.data.cpu().numpy()),
              end=' ')

        return total_loss