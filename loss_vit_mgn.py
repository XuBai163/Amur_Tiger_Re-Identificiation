from torch.nn import CrossEntropyLoss
from torch.nn.modules import loss
from utils.triplet_loss import TripletLoss

    
class Loss(loss._Loss):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, combined_embeddings, logits, mgn_outputs, labels):
        cross_entropy_loss = CrossEntropyLoss()
        triplet_loss = TripletLoss(margin=1.2)

        # MGN part and global features loss
        global_features_loss = [triplet_loss(output, labels) for output in mgn_outputs[1:4]]
        global_features_loss = sum(global_features_loss) / len(global_features_loss)

        mgn_class_scores_loss = [cross_entropy_loss(output, labels) for output in mgn_outputs[4:]]
        mgn_class_scores_loss = sum(mgn_class_scores_loss) / len(mgn_class_scores_loss)
        
        # Combined embeddings and logits loss
        combined_embeddings_loss = triplet_loss(combined_embeddings, labels)
        logits_loss = cross_entropy_loss(logits, labels)

        # total loss
        total_loss = global_features_loss + mgn_class_scores_loss + combined_embeddings_loss + logits_loss

        print('\rtotal loss: %.2f  global_features_loss: %.2f  mgn_class_scores_loss: %.2f  combined_embeddings_loss: %.2f  logits_loss: %.2f' % (
            total_loss.data.cpu().numpy(),
            global_features_loss.data.cpu().numpy(),
            mgn_class_scores_loss.data.cpu().numpy(),
            combined_embeddings_loss.data.cpu().numpy(),
            logits_loss.data.cpu().numpy(),
            ), end=' ')
        
        return total_loss