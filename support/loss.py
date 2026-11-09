from fastai.vision.all import *
import config

class HybridTverskyCELoss(nn.Module):
    def __init__(self, alpha=config.TVERSKY_A, beta=config.TVERSKY_B, ce_weight=config.CE_WEIGHT, smooth = 1e-8):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce_weight = ce_weight
        self.ce = nn.CrossEntropyLoss()
        self.smooth = smooth

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        probs = nn.functional.softmax(inputs, dim=1)

        num_classes = inputs.shape[1]
        one_hot = nn.functional.one_hot(targets, num_classes=num_classes)
        one_hot = one_hot.permute(0, 3, 1, 2).float()

        tversky_loss = 0.0
        for c in range(1, num_classes):
            prob = probs[:, c, :, :]
            truth = one_hot[:, c,: , :]

            prob_flat = prob.contiguous().view(-1)
            truth_flat = truth.contiguous().view(-1)

            TP = (prob_flat * truth_flat).sum()
            FP = ((1-truth_flat) * prob_flat).sum()
            FN = (truth_flat * (1 - prob_flat)).sum()

            tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
            tversky_loss += 1.0 - tversky

        tversky_loss = tversky_loss / (num_classes - 1)
        loss = (1.0 - self.ce_weight) * tversky_loss + self.ce_weight * ce_loss

        return loss