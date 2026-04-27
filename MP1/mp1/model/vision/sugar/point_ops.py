import torch
import torch.nn as nn
from pytorch3d.ops import sample_farthest_points


def fps(xyz, num_groups):
    """
    Farthest Point Sampling via pytorch3d.
    xyz: (B, N, 3)
    returns centers: (B, num_groups, 3)
    """
    centers, _ = sample_farthest_points(xyz.contiguous(), K=num_groups)
    return centers


def knn_point(k, xyz, query):
    """
    KNN via pairwise L2 distance (pure PyTorch).
    xyz:   (B, N, 3) — point cloud to search in
    query: (B, G, 3) — query centers
    returns idx: (B, G, k)
    """
    dist = torch.cdist(query, xyz)          # (B, G, N)
    idx = dist.topk(k, dim=-1, largest=False).indices  # (B, G, k)
    return idx


class PointGroup(nn.Module):
    """FPS + KNN grouping."""
    def __init__(self, num_groups, group_size, knn=True, radius=None):
        super().__init__()
        self.num_groups = num_groups
        self.group_size = group_size
        self.knn = knn
        self.radius = radius

    def forward(self, pc_fts):
        """
        pc_fts: (B, N, C) where C >= 3, first 3 dims are xyz
        returns:
            neighborhoods: (B, G, group_size, C)  relative xyz in first 3 dims
            centers:       (B, G, 3)
        """
        B, N, _ = pc_fts.shape
        xyz = pc_fts[..., :3].contiguous()

        centers = fps(xyz, self.num_groups)  # (B, G, 3)

        if self.knn:
            idx = knn_point(self.group_size, xyz, centers)  # (B, G, group_size)
        else:
            raise NotImplementedError("Ball-query grouping requires pointnet2_ops; use knn=True.")

        # Gather neighbors for all feature channels
        idx_base = torch.arange(B, device=xyz.device).view(-1, 1, 1) * N
        flat_idx = (idx + idx_base).view(-1)
        neighborhoods = pc_fts.reshape(B * N, -1)[flat_idx].reshape(B, self.num_groups, self.group_size, -1)

        # Normalize: subtract center xyz
        neighborhoods = neighborhoods.clone()
        neighborhoods[..., :3] = neighborhoods[..., :3] - centers.unsqueeze(2)
        return neighborhoods, centers
