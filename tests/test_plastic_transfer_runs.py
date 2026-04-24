import numpy as np

from plastic_transfer.core.plastic_transfer import PlasticTransfer


# ----------------------------------
# Mock ENV
# ----------------------------------
class MockEnv:
    def __init__(self):
        self.action_space = self

    def sample(self):
        return np.array([0.0])  # ✅ array

    def reset(self):
        return np.array([0.0]), {}  # ✅ array

    def step(self, action):
        obs = np.array([np.random.randn()])
        reward = float(np.random.randn())
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info


# ----------------------------------
# Mock PPO
# ----------------------------------
class MockModel:
    def __init__(self):
        self.counter = 0

    def predict(self, obs):
        self.counter += 1
        return np.array([0.5 + 0.1 * np.sin(self.counter)]), None

    def learn(self, total_timesteps):
        pass


def mock_model_builder(env):
    return MockModel()


# ==================================
# TEST
# ==================================

def test_plastic_transfer_runs():
    env = MockEnv()

    pt = PlasticTransfer(
        env=env,
        model_builder=mock_model_builder,
        input_size=3,
        hidden_size=8,
        latent_size=4,
        process_interval=10,
        skill_train_steps=10,
    )

    pt.learn(total_steps=50)

    info = pt.info()

    assert info["total_steps"] == 50


def test_creates_skill():
    env = MockEnv()

    pt = PlasticTransfer(
        env=env,
        model_builder=mock_model_builder,
        input_size=3,
        hidden_size=8,
        latent_size=4,
        process_interval=10,
        skill_train_steps=5,
    )

    pt.learn(total_steps=100)

    assert len(pt.skill_library.skills) > 0


def test_reuses_skill():
    env = MockEnv()

    pt = PlasticTransfer(
        env=env,
        model_builder=mock_model_builder,
        hidden_size=8,
        latent_size=4,
        process_interval=10,
        skill_train_steps=5
    )

    pt.learn(total_steps=100)

    assert len(pt.skill_library.skills) == 1


def test_memory_bank_updates():
    env = MockEnv()

    pt = PlasticTransfer(
        env=env,
        model_builder=mock_model_builder,
        hidden_size=8,
        latent_size=4,
        process_interval=10,
        skill_train_steps=5,
    )

    pt.learn(total_steps=50)

    assert len(pt.memory_bank.get_all()) > 0



def test_adaptation_changes_action():
    env = MockEnv()

    pt = PlasticTransfer(
        env=env,
        model_builder=mock_model_builder,
        input_size=3,
        hidden_size=8,
        latent_size=4,
        process_interval=5,
        skill_train_steps=5,
    )

    pt.learn(total_steps=100)

    obs, _ = env.reset()

    base_action = pt.current_skill.act(obs)
    adapted_action = pt.act(obs)

    assert not np.allclose(base_action, adapted_action)