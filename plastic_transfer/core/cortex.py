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

            if score > 0.1:  # 🔥 threshold low
                scored_skills.append((s, score))

        # ordenar por relevancia
        scored_skills.sort(key=lambda x: x[1], reverse=True)

        # solo top relevantes
        active_skills = [s for s, _ in scored_skills[:10]]

        if not active_skills:
            return []

        # -----------------------------------
        # 2. memoria
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
        # 5. ordenar
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

        return combos[:5]  # limitar

    def _score_memory(self, memory_item, combo):
        # usa reward pasado + similitud
        sim = memory_item.get("similarity", 0.0)
        reward = memory_item.get("reward", 0.0)

        return 0.7 * sim + 0.3 * reward

    def _score_heuristic(self, combo, obs_dict):
        score = 0.0

        alignment = obs_dict.get("alignment", 0.0)
        dist = obs_dict.get("dist", 1.0)

        # favorece avanzar
        score += alignment * 0.5
        score += (1 - dist) * 0.3

        # penaliza demasiadas skills
        score -= 0.05 * len(combo)

        return score
    

    def _score_skill_context(self, skill, obs_dict):
        score = 0.0

        alignment = obs_dict.get("alignment", 0.0)
        dist = obs_dict.get("dist", 1.0)
        dist_norm = obs_dict.get("dist_norm", 1.0)

        # -----------------------------------
        # navegación
        # -----------------------------------
        if "goal" in skill.name:
            score += (1 - dist) * 0.5
            score += alignment * 0.5

        # -----------------------------------
        # evasión
        # -----------------------------------
        if "avoid" in skill.name:
            score += (1 - dist_norm)

        # -----------------------------------
        # estabilidad
        # -----------------------------------
        if "stabilize" in skill.name:
            rp = obs_dict.get("rp", [0, 0])
            score += abs(rp[0]) + abs(rp[1])

        return score