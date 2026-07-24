import torch

class EMAStats:
    """
    Class-wise EMA mean & diagonal covariance for online learning.
    """
    def __init__(self, num_classes, feat_dim, alpha=0.01, device='cpu'):
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.alpha = alpha
        self.device = device

        # [C, D]
        self.means = torch.zeros(num_classes, feat_dim, device=device)
        # [C, D]
        self.vars = torch.zeros(num_classes, feat_dim, device=device)

        self.initialized = torch.zeros(num_classes, dtype=torch.bool, device=device)

    @torch.no_grad()
    def update(self, feats, labels):
        """
        feats: [B, 197, D] or [B, D] (ViT features)
        labels: [B]
        """
        # 取 class token
        if feats.dim() == 3:
            feats = feats[:, 0, :]   # [B, D]

        for c in torch.unique(labels):
            c = c.item()
            mask = (labels == c)
            x = feats[mask]  # [Nc, D]

            if x.numel() == 0:
                continue

            batch_mean = x.mean(dim=0)
            batch_var = ((x - batch_mean) ** 2).mean(dim=0)

            if not self.initialized[c]:
                self.means[c].copy_(batch_mean)
                self.vars[c].copy_(batch_var)
                self.initialized[c] = True
            else:
                self.means[c].mul_(1 - self.alpha).add_(self.alpha * batch_mean)
                self.vars[c].mul_(1 - self.alpha).add_(self.alpha * batch_var)

    def get_stats(self):
        """
        Returns:
            class_means: [C, D]
            class_covs : [C, D, D]  (diagonal covariance)
        """
        class_covs = torch.diag_embed(self.vars + 1e-6)
        return self.means, class_covs



