import numpy as np


class MemoryBank:
    def __init__(self, max_size=1000):
        self.data = []
        self.max_size = max_size

    # -------------------------------
    # ADD
    # -------------------------------
    def add(self, embedding, action=None, reward=0.0, skills=None, metadata=None):
        if embedding is None:
            return

        item = {
            "embedding": embedding,
            "action": action,
            "reward": reward,
            "skills": skills or [],
            "metadata": metadata or {}
        }

        self.data.append(item)

        # 🧹 control de tamaño (FIFO simple)
        if len(self.data) > self.max_size:
            self.data.pop(0)

    # -------------------------------
    # GET ALL
    # -------------------------------
    def get_all(self):
        return self.data

    # -------------------------------
    # QUERY (IMPORTANT)
    # -------------------------------
    def query(self, embedding, comparator, k=5, min_similarity=0.3):
        if embedding is None or not self.data:
            return []

        results = []

        for item in self.data:
            sim = comparator.compare(embedding, item["embedding"])

            if sim >= min_similarity:
                results.append({
                    "similarity": sim,
                    "embedding": item["embedding"],
                    "action": item["action"],
                    "reward": item["reward"],
                    "skills": item["skills"],
                    "metadata": item["metadata"]
                })

        # order by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)

        return results[:k]

    # -------------------------------
    # BEST MATCH
    # -------------------------------
    def find_best(self, embedding, comparator):
        best = None
        best_sim = -1

        for item in self.data:
            sim = comparator.compare(embedding, item["embedding"])
            if sim > best_sim:
                best_sim = sim
                best = item

        return best, best_sim

    # -------------------------------
    # CLEAR
    # -------------------------------
    def clear(self):
        self.data = []