class Consolidation:
    def __init__(
        self,
        basal,
        skill_library,
        min_usage=5,
        min_score=-0.2,
        reinforce_factor=1.1,
        decay_factor=0.9,
        max_skills=50
    ):
        self.basal = basal
        self.skill_library = skill_library

        self.min_usage = min_usage
        self.min_score = min_score
        self.reinforce_factor = reinforce_factor
        self.decay_factor = decay_factor
        self.max_skills = max_skills

    # -------------------------------
    # MAIN CONSOLIDATION
    # -------------------------------
    def run(self):
        skills = self.skill_library.get_all()

        for skill in skills:
            name = skill.name

            stats = self.basal.skill_stats.get(name, {})

            usage = stats.get("usage", 0)
            avg_reward = (
                stats.get("reward", 0.0) / max(usage, 1)
            )

            # -------------------------
            # 1. REINFORCE
            # -------------------------
            if avg_reward > 0:
                self.basal.skill_weights[name] *= self.reinforce_factor

            # -------------------------
            # 2. DECAY
            # -------------------------
            else:
                self.basal.skill_weights[name] *= self.decay_factor

        # -------------------------
        # 3. PRUNING
        # -------------------------
        self._prune()

        # -------------------------
        # 4. NORMALIZE
        # -------------------------
        self._normalize()

    # -------------------------------
    def _prune(self):
        new_skills = []

        for skill in self.skill_library.get_all():
            name = skill.name
            score = self.basal.get_skill_score(name)
            usage = self.basal.skill_stats.get(name, {}).get("usage", 0)

            if usage < self.min_usage:
                continue

            if score < self.min_score:
                continue

            new_skills.append(skill)

        # size limit
        new_skills = sorted(
            new_skills,
            key=lambda s: self.basal.get_skill_score(s.name),
            reverse=True
        )[:self.max_skills]

        self.skill_library.skills = new_skills

    # -------------------------------
    def _normalize(self):
        weights = self.basal.skill_weights

        if not weights:
            return

        max_w = max(abs(w) for w in weights.values()) + 1e-8

        for k in weights:
            weights[k] /= max_w