import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

class Loss(torch.nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.cross_entropy_loss = CrossEntropyLoss()

    def forward(self, outputs, labels):
        CrossEntropy_Loss = self.cross_entropy_loss(outputs, labels)

        print('\rCrossEntropy_Loss:%.2f' % (CrossEntropy_Loss.data.cpu().numpy()), end=' ')
        
        return CrossEntropy_Loss
