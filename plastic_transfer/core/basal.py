import numpy as np


class BasalGanglia:
    def __init__(
        self,
        lr=0.01,
        gamma=0.99,
        entropy_beta=0.001,
        use_softmax=False
    ):
        """
        Basal Ganglia module (selection + learning core)

        Responsibilities:
        - Select skill combinations (gating)
        - Learn skill utility from reward signal
        - Maintain lightweight statistics

        Design goals:
        - Low computational cost per step
        - Decoupled from environment specifics
        - Incremental learning (no heavy recomputation)
        """

        self.lr = lr
        self.gamma = gamma
        self.entropy_beta = entropy_beta
        self.use_softmax = use_softmax

        # Learned parameters
        self.skill_weights = {}

        # Statistics (for consolidation / analysis)
        self.skill_stats = {}

        # Last selected candidate (for credit assignment)
        self.last = None

    # =========================================================
    # INIT HELPERS
    # =========================================================
    def _ensure_skill(self, name):
        """
        Ensure skill exists in internal structures.
        Avoids KeyError during training.
        """
        if name not in self.skill_weights:
            self.skill_weights[name] = 0.0

        if name not in self.skill_stats:
            self.skill_stats[name] = {
                "usage": 0,
                "reward": 0.0,
                "success": 0
            }

    # =========================================================
    # SELECTION
    # =========================================================
    def select(self, candidates):
        """
        Select best candidate based on learned weights.

        This function is intentionally lightweight:
        - No environment-dependent computation
        - No feature iteration
        - Only uses learned weights

        candidates: list of dicts
            {
                "skills": [Skill],
                "score": float (optional, from cortex)
            }
        """

        if not candidates:
            return None

        # Fast path: assume cortex already sorted candidates
        # (recommended for performance)
        best = candidates[0]

        # Optional: recompute score using learned weights
        # Uncomment if you want basal to override cortex
        """
        best = None
        best_score = -np.inf

        for c in candidates:
            score = self._score_skills(c["skills"])

            if score > best_score:
                best_score = score
                best = c
        """

        # Store for learning step
        self.last = {
            "skills": best["skills"],
            "probs": None  # placeholder for entropy if needed
        }

        return best

    def _score_skills(self, skills):
        """
        Compute score as sum of skill weights.

        This is O(n_skills) and extremely fast.
        """
        score = 0.0

        for s in skills:
            score += self.skill_weights.get(s.name, 0.0)

        return score

    # =========================================================
    # LEARNING
    # =========================================================
    def update(self, reward, done):
        if self.last is None:
            return

        skills = self.last["skills"]

        if not skills:
            self.last = None
            return

        # ---------------------------
        # Advantage = reward only
        # ---------------------------
        advantage = reward

        share = 1.0 / max(len(skills), 1)

        for s in skills:
            name = s.name

            self._ensure_skill(name)

            delta = self.lr * advantage * share
            self.skill_weights[name] += delta

            # stats
            self.skill_stats[name]["usage"] += 1
            self.skill_stats[name]["reward"] += reward

            if done and reward > 0:
                self.skill_stats[name]["success"] += 1

        # entropy (igual que antes)
        if self.last["probs"] is not None:
            entropy = -np.sum(
                self.last["probs"] * np.log(self.last["probs"] + 1e-8)
            )

            for s in skills:
                self.skill_weights[s.name] += self.entropy_beta * entropy

        self.last = None

    # =========================================================
    # ACCESSORS
    # =========================================================
    def get_skill_score(self, name):
        """
        Return learned weight of a skill.
        Used by adapter for voting.
        """
        return self.skill_weights.get(name, 0.0)

    def get_all_scores(self):
        """
        Return all skill weights.
        Useful for debugging / logging.
        """
        return self.skill_weights

    def reset(self):
        """
        Reset internal state (not learned weights).
        """
        self.last = None

    def register_skill(self, name):
        if name not in self.skill_weights:
            self.skill_weights[name] = 0.0

        if name not in self.skill_stats:
            self.skill_stats[name] = {
                "usage": 0,
                "reward": 0.0,
                "success": 0
            }