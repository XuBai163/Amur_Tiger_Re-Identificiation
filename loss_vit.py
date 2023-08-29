from torch.nn import CrossEntropyLoss
from torch.nn.modules import loss
from utils.triplet_loss import TripletLoss


class ViTLoss(loss._Loss):
    def __init__(self):
        super(ViTLoss, self).__init__()

    def forward(self, logits, embeddings, labels):
        logits, embeddings = logits, embeddings

        triplet_loss_fn = TripletLoss(margin=1.2)
        Triplet_Loss = triplet_loss_fn(embeddings, labels)

        cross_entropy_loss = CrossEntropyLoss()
        CrossEntropy_Loss = cross_entropy_loss(logits, labels)
        loss_sum = Triplet_Loss + 2 * CrossEntropy_Loss

        print('\rtotal loss:%.2f  Triplet_Loss:%.2f  CrossEntropy_Loss:%.2f' % (
            loss_sum.data.cpu().numpy(),
            Triplet_Loss.data.cpu().numpy(),
            CrossEntropy_Loss.data.cpu().numpy()),
            end=' ')
        return loss_sum




