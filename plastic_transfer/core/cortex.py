import numpy as np


class Cortex:
    def __init__(self, comparator, skill_library, memory_bank, max_candidates=5, top_k_memory=3):
        self.max_candidates = max_candidates
        self.top_k_memory = top_k_memory
        self.comparator = comparator
        self.skill_library = skill_library
        self.memory_bank = memory_bank

    def propose(self, embedding, obs_dict):
        all_skills = self.skill_library.get_all()

        if not all_skills:
            return []

        # -----------------------------------
        # 1. contextual scoring de skills
        # -----------------------------------
        scored_skills = []

        for s in all_skills:
            score = self._score_skill_context(s, obs_dict)

            if score > 0.1:  # threshold low
                scored_skills.append((s, score))

        # order by relevance
        scored_skills.sort(key=lambda x: x[1], reverse=True)

        # only top skills
        active_skills = [s for s, _ in scored_skills[:10]]

        if not active_skills:
            return []

        # -----------------------------------
        # 2. memory
        # -----------------------------------
        similar = self.memory_bank.query(
            embedding,
            self.comparator,
            k=self.top_k_memory
        )

        candidates = []

        # -----------------------------------
        # 3. memory-based candidates
        # -----------------------------------
        for item in similar:
            past_skills = item.get("skills", [])

            combo = [s for s in past_skills if s in active_skills]

            if combo:
                score = self._score_memory(item, combo)
                candidates.append({
                    "skills": combo,
                    "score": score,
                    "source": "memory"
                })

        # -----------------------------------
        # 4. heuristic combinations
        # -----------------------------------
        heuristics = self._heuristic_combinations(active_skills, obs_dict)

        for combo in heuristics:
            score = self._score_heuristic(combo, obs_dict)
            candidates.append({
                "skills": combo,
                "score": score,
                "source": "heuristic"
            })

        # -----------------------------------
        # 5. order
        # -----------------------------------
        candidates.sort(key=lambda x: x["score"], reverse=True)

        return candidates[:self.max_candidates]

    def _heuristic_combinations(self, skills, obs_dict):
        combos = []

        perception = [s for s in skills if s.type == "perception"]
        planning = [s for s in skills if s.type == "planning"]
        control = [s for s in skills if s.type == "control"]

        # rule: perception + planning + control
        for p in perception:
            for pl in planning:
                for c in control:
                    combos.append([p, pl, c])

        return combos[:5]  # limit

    def _score_memory(self, memory_item, combo):
        # past reward + similarity
        sim = memory_item.get("similarity", 0.0)
        reward = memory_item.get("reward", 0.0)

        return 0.7 * sim + 0.3 * reward

    def _score_heuristic(self, combo, obs_dict):
        score = 0.0

        for skill in combo:
            for inp in skill.inputs:
                key = inp.get("key")
                weight = inp.get("score", 0.0)

                value = obs_dict.get(key, 0.0)

                # simple norm
                if isinstance(value, (list, np.ndarray)):
                    value = np.linalg.norm(value)

                score += value * weight * skill.order

        return score
    

    def _score_skill_context(self, skill, obs_dict):
        score = 0.0

        for inp in skill.inputs:
            key = inp.get("key")
            weight = inp.get("score", 0.0)

            value = obs_dict.get(key, 0.0)

            if isinstance(value, (list, np.ndarray)):
                value = np.linalg.norm(value)

            score += value * weight

        return score * skill.order