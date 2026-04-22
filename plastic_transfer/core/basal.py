import numpy as np


class BasalGanglia:
    def __init__(
        self,
        lr=0.05,
        gamma=0.99,
        entropy_beta=0.01,
        use_softmax=True
    ):
        self.lr = lr
        self.gamma = gamma
        self.entropy_beta = entropy_beta
        self.use_softmax = use_softmax

        # pesos por skill (gating)
        self.skill_weights = {}   # {skill.name: float}

        # stats por skill
        self.skill_stats = {}     # {skill.name: {...}}

        # último seleccionado (para update)
        self.last = None

    # -------------------------------
    # SELECTION
    # -------------------------------
    def select(self, candidates, obs_dict):
        if not candidates:
            return None

        scores = []

        for c in candidates:
            s = self._score_candidate(c, obs_dict)
            scores.append(s)

        scores = np.array(scores, dtype=np.float32)

        if self.use_softmax:
            probs = self._softmax(scores)
            idx = np.random.choice(len(candidates), p=probs)
        else:
            idx = int(np.argmax(scores))
            probs = None

        selected = candidates[idx]

        # guardar para aprendizaje
        self.last = {
            "candidate": selected,
            "obs": obs_dict,
            "probs": probs,
            "scores": scores,
            "idx": idx
        }

        return selected

    # -------------------------------
    def _score_candidate(self, candidate, obs_dict):
        base = candidate.get("score", 0.0)

        # contexto
        alignment = obs_dict.get("alignment", 0.0)
        dist = obs_dict.get("dist", 1.0)
        dist_norm = obs_dict.get("dist_norm", 1.0)  # obstáculo cercano

        # riesgo (más bajo = mejor)
        risk = 1.0 - dist_norm

        # pesos por skill
        weight_sum = 0.0
        for s in candidate["skills"]:
            w = self.skill_weights.get(s.name, 0.0)
            weight_sum += w

        # combinación
        score = (
            0.5 * base +
            0.2 * alignment +
            0.2 * (1 - dist) -
            0.2 * risk +
            0.3 * weight_sum
        )

        # penaliza combos grandes
        score -= 0.05 * len(candidate["skills"])

        return score

    # -------------------------------
    def _softmax(self, x):
        x = x - np.max(x)
        e = np.exp(x)
        return e / (np.sum(e) + 1e-8)

    # -------------------------------
    # LEARNING
    # -------------------------------
    def update(self, selected, reward, next_obs_dict, done):
        if self.last is None or selected is None:
            return

        skills = selected["skills"]

        # ---------------------------
        # señal de valor (simple TD)
        # ---------------------------
        alignment = next_obs_dict.get("alignment", 0.0)
        dist = next_obs_dict.get("dist", 1.0)

        value = 0.5 * alignment + 0.5 * (1 - dist)
        target = reward + self.gamma * value * (0.0 if done else 1.0)

        # baseline (opcional)
        baseline = 0.0
        advantage = target - baseline

        # ---------------------------
        # crédito por skill
        # ---------------------------
        share = 1.0 / max(len(skills), 1)

        for s in skills:
            # init
            if s.name not in self.skill_weights:
                self.skill_weights[s.name] = 0.0

            if s.name not in self.skill_stats:
                self.skill_stats[s.name] = {
                    "usage": 0,
                    "reward": 0.0,
                    "success": 0
                }

            # update weight
            delta = self.lr * advantage * share
            self.skill_weights[s.name] += delta

            # stats
            self.skill_stats[s.name]["usage"] += 1
            self.skill_stats[s.name]["reward"] += reward
            if done and reward > 0:
                self.skill_stats[s.name].setdefault("success", 0)
                self.skill_stats[s.name]["success"] += 1

        # ---------------------------
        # entropy regularization (exploración)
        # ---------------------------
        if self.last["probs"] is not None:
            entropy = -np.sum(self.last["probs"] * np.log(self.last["probs"] + 1e-8))
            for s in skills:
                self.skill_weights[s.name] += self.entropy_beta * entropy

        # limpiar
        self.last = None

    def register_skill(self, name):
        if name not in self.skill_weights:
            self.skill_weights[name] = 1.0

        if name not in self.skill_stats:
            self.skill_stats[name] = {
                "usage": 0,
                "reward": 0.0,
                "success": 0
            }

    # -------------------------------
    # UTIL
    # -------------------------------
    def get_skill_score(self, name):
        if name not in self.skill_weights:
            self.register_skill(name)

        return self.skill_weights[name]

    def summary(self):
        return {
            "weights": self.skill_weights,
            "stats": self.skill_stats
        }