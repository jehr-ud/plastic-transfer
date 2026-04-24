import numpy as np
import gymnasium as gym
from gymnasium import spaces

# =========================================================
# SKILL
# =========================================================
class Skill:
    def __init__(
        self,
        name,
        description,
        skill_type,
        inputs,
        outputs,
        trigger,
        objective,
        order=0,
        model_builder=None,
        env=None
    ):
        self.name = name
        self.description = description
        self.type = skill_type
        self.inputs = inputs
        self.outputs = outputs
        self.trigger_expr = trigger
        self.objective = objective
        self.order = order

        self.model = None

        if model_builder and env:
            wrapped_env = SkillEnvWrapper(env, self)
            self.model = model_builder(wrapped_env)

    # -------------------------------
    # BUILD INPUT VECTOR (AGNOSTIC)
    # -------------------------------
    def build_input_vector(self, obs_dict):
        values = []

        for inp in self.inputs:
            key = inp["key"]
            value = obs_dict.get(key, 0.0)

            if isinstance(value, (list, np.ndarray)):
                values.extend(value)
            else:
                values.append(value)

        return np.array(values, dtype=np.float32)

    # -------------------------------
    # ACT
    # -------------------------------
    def act(self, obs_dict):
        if not self.model:
            return None

        x = self.build_input_vector(obs_dict)
        action, _ = self.model.predict(x, deterministic=True)

        return self._format_output(action)

    # -------------------------------
    # OUTPUT MAPPING
    # -------------------------------
    def _format_output(self, action):
        output = {}

        for i, out in enumerate(self.outputs):
            key = out["key"]
            output[key] = float(action[i])

        return output

    def store(self, obs, action, reward):
        self.memory.append((obs, action, reward))

    def train(self, total_steps=10000):
        if not self.model:
            return

        self.model.learn(
            total_timesteps=total_steps,
            reset_num_timesteps=False,
            progress_bar=False
        )

# =========================================================
# SKILL ENV WRAPPER (AGNOSTIC)
# =========================================================
class SkillEnvWrapper(gym.Env):
    def __init__(self, base_env, skill):
        super().__init__()

        self.base_env = base_env
        self.skill = skill

        # -------------------------
        # OBS SPACE (from inputs)
        # -------------------------
        sample_obs = self.base_env.get_obs_dict()
        input_vector = self.skill.build_input_vector(sample_obs)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=input_vector.shape,
            dtype=np.float32
        )

        # -------------------------
        # ACTION SPACE (from outputs)
        # -------------------------
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(skill.outputs),),
            dtype=np.float32
        )

        # -------------------------
        # ACTION KEY MAPPING
        # -------------------------
        self.action_keys = self.base_env.action_keys
        self.key_to_idx = {
            key: i for i, key in enumerate(self.action_keys)
        }

    # -------------------------
    # RESET
    # -------------------------
    def reset(self, seed=None, options=None):
        obs, info = self.base_env.reset()

        obs_dict = self.base_env.get_obs_dict()

        return self.skill.build_input_vector(obs_dict), info

    # -------------------------
    # STEP
    # -------------------------
    def step(self, action):

        full_action = self._expand_action(action)

        obs, reward, terminated, truncated, info = self.base_env.step(full_action)

        obs_dict = self.base_env.get_obs_dict()

        return (
            self.skill.build_input_vector(obs_dict),
            reward,
            terminated,
            truncated,
            info
        )

    

    # -------------------------
    # EXPAND ACTION (AGNOSTIC)
    # -------------------------
    def _expand_action(self, action):
        full = np.zeros(len(self.action_keys))

        for i, out in enumerate(self.skill.outputs):
            key = out["key"]

            if key in self.key_to_idx:
                full[self.key_to_idx[key]] = action[i]

        return full