import numpy as np

class ReusableAdaptation:
    def __init__(
        self,
        basal,
        comparator=None,
        k=5
    ):
        self.comparator = comparator
        self.basal = basal
        self.k = k

    # -------------------------------
    # MAIN ENTRY
    # -------------------------------
    def act(self, skills, obs, base_policy_fn, embedding=None, memory_bank=None):
        """
        Translate selected skills into final environment action.
        """

        action = self._compose_action(skills, obs, base_policy_fn)

        # memory refinement
        if embedding is not None and memory_bank is not None and self.comparator:
            action = self._adapt_with_memory_fast(
                embedding,
                action,
                memory_bank
            )

        return action

    # -------------------------------
    # SKILL COMPOSITION (FAST)
    # -------------------------------
    def _compose_action(self, skills, obs, base_policy_fn):
        if not skills and base_policy_fn:
            return base_policy_fn(obs)

        final_action = None
        total_weight = 0.0

        for s in skills:
            a = s.act(obs)

            if a is None and base_policy_fn:
                a = base_policy_fn(obs)

            w = max(self.basal.get_skill_score(s.name), 0.0)

            if final_action is None:
                final_action = np.zeros_like(a)

            final_action += w * a
            total_weight += w

        if total_weight == 0.0:
            return base_policy_fn(obs)

        return final_action / total_weight

    # -------------------------------
    # MEMORY ADAPTATION (FAST)
    # -------------------------------
    def _adapt_with_memory_fast(self, current_emb, base_action, memory_bank):
        """
        Faster memory adaptation:
        - avoids full sort
        - early filtering
        """

        memory = memory_bank.get_all()
        if not memory:
            return base_action

        best = []
        min_sim = -np.inf

        for item in memory:
            emb = item.get("embedding")
            action_k = item.get("action")

            if emb is None or action_k is None:
                continue

            sim = self.comparator.compare(current_emb, emb)

            if len(best) < self.k:
                best.append((sim, action_k))
                if sim < min_sim or min_sim == -np.inf:
                    min_sim = sim
            else:
                if sim > min_sim:
                    # replace worst
                    worst_idx = np.argmin([b[0] for b in best])
                    best[worst_idx] = (sim, action_k)
                    min_sim = min([b[0] for b in best])

        if not best:
            return base_action

        sims = np.array([s for s, _ in best])
        weights = sims / (sims.sum() + 1e-8)

        delta = np.zeros_like(base_action)

        for w, (_, a_k) in zip(weights, best):
            delta += w * (a_k - base_action)

        return base_action + delta