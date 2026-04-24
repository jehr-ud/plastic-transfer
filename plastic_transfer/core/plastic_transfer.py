import os
import json
from collections import deque

import torch
import numpy as np

from plastic_transfer.core.temporal_representation import TemporalRepresentation
from plastic_transfer.core.structural_comparison import StructuralComparison
from plastic_transfer.core.reusable_adaptation import ReusableAdaptation
from plastic_transfer.core.encoders.snn import SNNEncoder
from plastic_transfer.core.cortex import Cortex
from plastic_transfer.core.basal import BasalGanglia
from plastic_transfer.core.learning_definitions import LearningDefinitions
from plastic_transfer.core.consolidation import Consolidation
from plastic_transfer.core.builders.base_policy_builder import BasePolicyBuilder
from plastic_transfer.skills.skill import Skill
from plastic_transfer.skills.skill_library import SkillLibrary
from plastic_transfer.memory.memory_bank import MemoryBank
from plastic_transfer.utils.general import (
    validate_trigger,
    serialize_memory
)
from plastic_transfer.utils.step_logger import StepLogger
from plastic_transfer.utils.gym import get_action_dim, get_obs_dim


class PlasticTransfer:
    def __init__(
        self,
        env,
        model_builder,
        hidden_size,
        latent_size,
        learning_definitions,
        logger_path_file=None,
        obs_to_dict_fn=None,
        policy_config={},
        debug=False,
        skill_train_steps=20000,
    ):
        self.env = env
        self.model_builder = model_builder
        self.obs_to_dict_fn = obs_to_dict_fn

        self.policy_config = policy_config
        builder = BasePolicyBuilder(policy_config)
        self.base_policy_fn = builder.build()


        self.debug = debug

        # DIMENSIONS
        # -------------------------------
        obs_dim = get_obs_dim(env.observation_space)
        action_dim = get_action_dim(env.action_space)

        input_size = obs_dim + action_dim + 1

        print(f"[PT] obs_dim={obs_dim}, action_dim={action_dim}, input_size={input_size}")

        # -------------------------------
        # Core layers
        # -------------------------------
        encoder = SNNEncoder(
            input_size,
            hidden_size,
            latent_size
        )
        self.temporal = TemporalRepresentation(
            encoder=encoder
        )
        self.basal = BasalGanglia()
        self.structural_comparison = StructuralComparison()
        self.skill_library = SkillLibrary()
        self.memory_bank = MemoryBank()
        self.cortex = Cortex(
            comparator=self.structural_comparison,
            memory_bank=self.memory_bank,
            skill_library=self.skill_library

        )
        self.adapter = ReusableAdaptation(basal=self.basal, comparator=self.structural_comparison)
        self.consolidation = Consolidation(
            basal=self.basal,
            skill_library=self.skill_library
        )

        # -------------------------------
        # Logger
        # -------------------------------

        self.logger = None
        if logger_path_file:
            self.logger = StepLogger(name=logger_path_file)

        # -------------------------------
        # Skills firts creation
        # -------------------------------

        validate_trigger_fn = None

        if self.obs_to_dict_fn:
            sample_obs, _ = self.env.reset()
            sample_obs_dict = self.obs_to_dict_fn(sample_obs)
            validate_trigger_fn = validate_trigger

        self.learning_definitions = LearningDefinitions(
            learning_definitions,
            validate_trigger_fn=validate_trigger_fn,
            sample_obs_dict=sample_obs_dict
        )

        for skill in self.learning_definitions.get_skills():
            self.skill_library.add(skill)

            if skill.name not in self.basal.skill_weights:
                self.basal.register_skill(skill.name)

        print(f"[PT] Loaded {len(self.learning_definitions.get_skills())} skills")

        # -------------------------------
        # Stats
        self.skill_train_steps = skill_train_steps
        self.total_episodes = 0
        self.total_steps = 0
        self.recent_rewards = deque(maxlen=100)
        self.recent_success = deque(maxlen=100)

    # =========================================================
    # MAIN TRAIN LOOP
    # =========================================================
    def learn(self, total_steps=100_000):
        obs, _ = self.env.reset()

        accumulated_reward = 0
        total_steps_in_episode = 0
        self.total_steps = 0
        self.total_steps_with_skills = 0
        self.total_episodes = 0

        for step in range(total_steps):
            done = False

            # initial action
            last_action = self.env.action_space.sample()

            self._debug(f"\n=== Episode {self.total_episodes} ===")

            # -----------------------------------
            # 1. TEMPORAL ENCODING
            # -----------------------------------
            encoding_z = self.temporal.encode_current_step(
                obs, action=last_action, reward=0.0
            )

            if encoding_z is not None:
                self._debug(f"[ENC] z norm={np.linalg.norm(encoding_z):.3f}")
            else:
                self._debug("[ENC] z=None (warming buffer)")

            use_skills = False

            if encoding_z is not None and len(self.recent_rewards) >= 10:

                rewards = np.array(self.recent_rewards)

                stability = np.std(rewards)
                stability_norm = np.clip(stability / 1.0, 0.0, 1.0)

                trend = np.mean(rewards[-10:]) - np.mean(rewards[:10])
                trend_norm = (np.tanh(trend) + 1) / 2  # [0,1]

                skill_prob = 0.5 * (1 - stability_norm) + 0.5 * trend_norm
                skill_prob = np.clip(skill_prob, 0.2, 0.9)

                use_skills = np.random.rand() < skill_prob

            self._debug(f"[MODE] use_skills={use_skills}")

            selected = None

            # -----------------------------------
            # 2. ACTION SELECTION
            # -----------------------------------
            if use_skills:
                obs_dict = self.obs_to_dict_fn(obs)

                candidates = self.cortex.propose(
                    encoding_z, obs_dict
                )

                self._debug(f"[CORTEX] candidates={len(candidates)}")

                if candidates:
                    c = candidates[0]
                    self._debug(f"[CORTEX] best score={c['score']:.3f} skills={[s.name for s in c['skills']]}")

                selected = self.basal.select(candidates)

                if selected is not None:
                    names = [s.name for s in selected["skills"]]
                    self._debug(f"[BASAL] selected={names}")

                    action = self.adapter.act(
                        skills=selected["skills"],
                        obs=obs,
                        base_policy_fn=self.base_policy_fn,
                        embedding=encoding_z,
                        memory_bank=self.memory_bank
                    )

                    self.total_steps_with_skills += 1

                else:
                    self._debug("[BASAL] selected=None → random fallback")
                    action = self.base_policy_fn(obs)

            else:
                action = self.base_policy_fn(obs)

            self._debug(f"[ACT] action={np.round(action, 3)}")

            last_action = action

            assert action.shape[0] == self.env.action_space.shape[0], f"Invalid action dim: {action.shape}"

            # -----------------------------------
            # 3. ENV STEP
            # -----------------------------------
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            accumulated_reward += reward
            total_steps_in_episode += 1
            self.recent_rewards.append(reward)
            self.recent_success.append(1 if terminated else 0)

            self._debug(f"[REWARD] r={reward:.3f} acc={accumulated_reward:.3f}")


            # -----------------------------------
            # 4. TEMPORAL UPDATE
            # -----------------------------------
            self.temporal.add_step(obs, action, reward)

            # -----------------------------------
            # 5. BASAL UPDATE
            # -----------------------------------
            if selected is not None:
                self.basal.update(
                    reward,
                    done
                )
                    
                for skill in selected["skills"]:
                    skill.train(total_steps=self.skill_train_steps // len(selected["skills"]))

            obs = next_obs

            # -----------------------------------
            # 6. END EPISODE
            # -----------------------------------
            if done:
                self._debug("\n[EPISODE END]")
                self._debug(f"steps={total_steps_in_episode}")
                self._debug(f"total_reward={accumulated_reward:.3f}")
                self._debug(f"terminated={terminated}, truncated={truncated}")

                result = self.temporal.end_episode(terminated, truncated)

                if result:
                    embeddings = result["embeddings"]
                    metrics = result["metrics"]

                    self.memory_bank.add(
                        action=action,
                        embedding=embeddings[-1] if embeddings else None,
                        reward=metrics["total_reward"],
                        skills=selected["skills"] if selected else []
                    )

                    self._debug(
                        f"[MEMORY] stored | reward={metrics['total_reward']:.3f}"
                    )

                # -----------------------------------
                # CONSOLIDATION
                # -----------------------------------
                if self.total_episodes % (total_steps // 10) == 0:
                    self._debug("\n[CONSOLIDATION RUN]")

                    self.consolidation.run()

                obs, _ = self.env.reset()

                # -----------------------------------
                # LOGGING
                # -----------------------------------

                if self.logger:
                    self.logger.log(
                        steps=total_steps_in_episode,
                        episode=self.total_episodes,
                        reward=accumulated_reward,
                        terminated=terminated,
                        truncated=truncated,
                        total_steps_with_skills=self.total_steps_with_skills
                    )
            
                self.total_episodes += 1
                accumulated_reward = 0
                total_steps_in_episode = 0
                self.total_steps_with_skills = 0

        self.total_steps = total_steps

        self.logger.close()

    # =========================================================
    # ACTION
    # =========================================================
    def predict(self, obs):
        """
        Inference
        """
        # -----------------------------------
        # 1. encoding temporal
        # -----------------------------------
        z = self.temporal.encode_current_step(
            obs,
            action=np.zeros(self.env.action_space.shape[0]),
            reward=0.0
        )

        # -----------------------------------
        # 2. obs dict
        # -----------------------------------
        use_skills = z is not None

        selected = None

        # -----------------------------------
        # 3. cortex + basal
        # -----------------------------------
        if use_skills:
            candidates = self.cortex.propose(
                obs,
                z,
                self.skill_library,
                self.memory_bank
            )

            selected = self.basal.select(candidates)

        # -----------------------------------
        # 4. action
        # -----------------------------------
        if selected is not None:
            action = self.adapter.act(
                skills=selected["skills"],
                obs=obs,
                base_policy_fn=self.base_policy_fn,
                embedding=z,
                memory_bank=self.memory_bank
            )
        else:
            # fallback
            if self.base_policy_fn:
                action = self.base_policy_fn(obs)
            else:
                action = self.env.action_space.sample()

        return action

    def _debug(self, msg):
        if self.debug:
            print(msg)

    # =========================================================
    # INSPECTION
    # =========================================================
    def info(self):
        return {
            "num_skills": len(self.skill_library.skills),
            "total_steps": self.total_steps,
            "episodes": self.total_episodes
        }

    def save(self, path):
        os.makedirs(path, exist_ok=True)

        # -------------------------
        # 1. Encoder
        # -------------------------
        torch.save(
            self.temporal.encoder.state_dict(),
            f"{path}/encoder.pth"
        )

        # -------------------------
        # 2. Skills (model)
        # -------------------------
        skills_data = []

        for i, skill in enumerate(self.skill_library.get_all()):

            skill_path = f"{path}/skill_{i}.zip"

            if skill.model is not None:
                skill.model.save(skill_path)

            skills_data.append({
                "id": i,
                "name": skill.name,
                "type": skill.type,
                "inputs": skill.inputs,
                "outputs": skill.outputs,
                "trigger": skill.trigger_expr,
                "objective": skill.objective,
                "order": skill.order,
                "model_path": skill_path
            })

        with open(f"{path}/skills.json", "w") as f:
            json.dump(skills_data, f, indent=2)

        # -------------------------
        # 3. MemoryBank (only data)
        # -------------------------
        with open(f"{path}/memory.json", "w") as f:
            json.dump(serialize_memory(self.memory_bank.get_all()), f)

        # -------------------------
        # 4. Basal state
        # -------------------------
        basal_state = {
            "weights": self.basal.skill_weights,
            "stats": self.basal.skill_stats
        }

        with open(f"{path}/basal.json", "w") as f:
            json.dump(basal_state, f)

        # -------------------------
        # 5. Config
        # -------------------------
        config = {
            "total_steps": self.total_steps,
            "episode": self.total_episodes
        }

        with open(f"{path}/config.json", "w") as f:
            json.dump(config, f)

        # -------------------------
        # 5. funcition base 
        # -------------------------

        config = {
            "policy_config": self.policy_config,
        }

        with open(f"{path}/policy_base_config.json", "w") as f:
            json.dump(config, f)


        print(f"[PT] Model saved at {path}")

    def load(self, path):
        # -------------------------
        # 1. Encoder
        # -------------------------
        encoder_path = f"{path}/encoder.pth"
        if os.path.exists(encoder_path):
            self.temporal.encoder.load_state_dict(
                torch.load(encoder_path, map_location="cpu")
            )

        # -------------------------
        # 2. Skills
        # -------------------------
        with open(f"{path}/skills.json") as f:
            skills_data = json.load(f)

        self.skill_library.skills = []

        for s in skills_data:
            model = None

            if os.path.exists(s["model_path"]):
                model = self.model_builder(self.env)
                model = model.load(s["model_path"])

            skill = Skill(
                name=s["name"],
                description=s.get("description", ""),
                skill_type=s["type"],
                inputs=s.get("inputs", []),
                outputs=s.get("outputs", []),
                trigger=s.get("trigger", "always"),
                objective=s.get("objective", ""),
                order=s.get("order", 0),
                model_builder=None
            )

            skill.model = model

            self.skill_library.add(skill)

        # -------------------------
        # 3. Index
        # -------------------------
        if hasattr(self.skill_library, "build_index"):
            self.skill_library.build_index()

        # -------------------------
        # 4. Memory
        # -------------------------
        memory_path = f"{path}/memory.json"

        if os.path.exists(memory_path):
            with open(memory_path) as f:
                memory_data = json.load(f)

            self.memory_bank.data = []

            for item in memory_data:
                skills = []

                for name in item.get("skills", []):
                    s = self.skill_library.get_by_name(name)
                    if s is not None:
                        skills.append(s)
                    else:
                        print(f"[WARN] skill '{name}' not found during load")

                self.memory_bank.data.append({
                    "embedding": np.array(item["embedding"]) if item["embedding"] is not None else None,
                    "action": np.array(item["action"]) if item["action"] is not None else None,
                    "reward": item.get("reward", 0.0),
                    "skills": skills
                })

        # -------------------------
        # 5. Basal
        # -------------------------
        basal_path = f"{path}/basal.json"

        if os.path.exists(basal_path):
            with open(basal_path) as f:
                basal_state = json.load(f)

            self.basal.skill_weights = basal_state.get("weights", {})
            self.basal.skill_stats = basal_state.get("stats", {})

        # -------------------------
        # 6. Config
        # -------------------------
        config_path = f"{path}/config.json"

        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)

            self.total_steps = config.get("total_steps", 0)
            self.total_episodes = config.get("episode", 0)

        # -------------------------
        # 6. Policy base
        # -------------------------
        config_path = f"{path}/policy_base_config.json"

        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)

                builder = BasePolicyBuilder(config.get("policy_config"))
                self.base_policy_fn = builder.build()

        # -------------------------
        # 7. Debug
        # -------------------------
        print(f"[PT] Loaded {len(self.skill_library.get_all())} skills")
        print(f"[PT] Memory size: {len(self.memory_bank.get_all())}")   