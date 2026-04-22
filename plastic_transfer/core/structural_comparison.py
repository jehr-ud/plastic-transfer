import numpy as np
import torch


class StructuralComparison:
    def __init__(self, temperature=0.1, eps=1e-8):
        self.temperature = temperature
        self.eps = eps

    def compare(self, emb1, emb2):
        if isinstance(emb1, torch.Tensor):
            emb1 = emb1.detach().cpu().numpy()
        if isinstance(emb2, torch.Tensor):
            emb2 = emb2.detach().cpu().numpy()

        if emb1 is None or emb2 is None:
            return 0.0

        if emb1.shape != emb2.shape:
            return 0.0

        # -------------------------
        # 1. cosine similarity
        # -------------------------
        norm1 = np.linalg.norm(emb1) + self.eps
        norm2 = np.linalg.norm(emb2) + self.eps
        cos = np.dot(emb1, emb2) / (norm1 * norm2)

        # -------------------------
        # 2. overlap (structure)
        # -------------------------
        overlap = np.sum(np.minimum(emb1, emb2)) / (
            np.sum(np.maximum(emb1, emb2)) + self.eps
        )

        # -------------------------
        # 3. combination
        # -------------------------
        sim = 0.6 * cos + 0.4 * overlap

        # -------------------------
        # 4. temperature scaling
        # -------------------------
        sim = sim / self.temperature

        # -------------------------
        # 5. squash a [0,1]
        # -------------------------
        sim = 1 / (1 + np.exp(-sim))  # sigmoid

        return float(sim)