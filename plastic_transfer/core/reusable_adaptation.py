import numpy as np


class ReusableAdaptation:
    def __init__(self, basal, comparator=None, k=5):
        self.comparator = comparator
        self.basal = basal
        self.k = k

    # -------------------------------
    # MAIN ENTRY
    # -------------------------------
    def act(self, skills, obs, base_policy_fn, embedding=None, memory_bank=None):
        """
        skills: lista de skills seleccionadas
        obs: estado actual
        base_policy_fn: función que genera acción base
        embedding: opcional (para memoria)
        memory_bank: opcional
        """

        # 1. acción base por skills
        base_action = self._compose_action(skills, obs, base_policy_fn)

        # 2. refinamiento con memoria
        if embedding is not None and memory_bank is not None and self.comparator:
            base_action = self._adapt_with_memory(
                embedding,
                base_action,
                memory_bank
            )

        return base_action

    # -------------------------------
    # SKILL COMPOSITION
    # -------------------------------
    def _compose_action(self, skills, obs, base_policy_fn):
        if not skills:
            return base_policy_fn(obs)

        votes = []
        weights = []

        for sub_skill in skills:
            action = sub_skill.act(obs)

            if action is None:
                print("no action from skill calculating using policity")
                action = base_policy_fn(obs)

            weight = max(self._skill_weight(sub_skill), 0.0)
            votes.append(action)
            weights.append(weight)

        weights = np.array(weights)

        if weights.sum() == 0:
            return base_policy_fn(obs)

        weights = weights / weights.sum()

        final_action = np.zeros_like(votes[0])

        for w, a in zip(weights, votes):
            final_action += w * a

        return final_action

    def _skill_weight(self, skill):
        return self.basal.get_skill_score(skill.name)

    # -------------------------------
    # MEMORY ADAPTATION
    # -------------------------------
    def _adapt_with_memory(self, current_emb, base_action, memory_bank):
        memory = memory_bank.get_all()

        if not memory:
            return base_action

        sims = []

        for item in memory:
            emb = item.get("embedding")
            action_k = item.get("action")

            if emb is None or action_k is None:
                continue

            sim = self.comparator.compare(current_emb, emb)
            sims.append((sim, action_k))

        if not sims:
            return base_action

        sims.sort(key=lambda x: x[0], reverse=True)
        top_k = sims[:self.k]

        weights = np.array([s for s, _ in top_k])
        weights = weights / (weights.sum() + 1e-8)

        delta = 0
        for w, (_, action_k) in zip(weights, top_k):
            delta += w * (action_k - base_action)

        return base_action + delta