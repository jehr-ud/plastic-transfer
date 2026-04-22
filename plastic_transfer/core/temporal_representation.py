import torch
import numpy as np

from plastic_transfer.core.encoders.base import Encoder


class TemporalRepresentation:

    def __init__(
        self,
        encoder,
        window_size=32,
        step_size=1,
        normalize=True,
        observations_keys=[]
    ):
        self.encoder: Encoder = encoder # SNN Encoder
        self.window_size = window_size
        self.step_size = step_size
        self.normalize = normalize
        self.observations_keys = observations_keys

        self.episode_buffer = []

    # -------------------------------
    # STEP
    # -------------------------------
    def add_step(self, obs, action, reward):
        x = self._preprocess(obs, action, reward)
        self.episode_buffer.append({
            "x": x,
            "reward": reward
        })

    # -------------------------------
    # EPISODE END
    # -------------------------------
    def end_episode(self, terminated, truncated):
        if len(self.episode_buffer) == 0:
            return None

        trajectory = [step["x"] for step in self.episode_buffer]
        rewards = [step["reward"] for step in self.episode_buffer]

        embeddings = self._encode_windows(trajectory)

        metrics = self._compute_metrics(
            rewards,
            terminated,
            truncated
        )

        result = {
            "embeddings": embeddings,
            "metrics": metrics
        }

        self.reset()
        return result

    # -------------------------------
    # WINDOW ENCODING
    # -------------------------------
    def _encode_windows(self, trajectory):
        windows = []

        T = len(trajectory)

        for i in range(0, T - self.window_size + 1, self.step_size):
            window = trajectory[i:i+self.window_size]

            window = np.array(window, dtype=np.float32)

            z = self._encode_window(window)
            windows.append(z)

        return windows

    def _encode_window(self, window):
        device = self._get_device()

        window = torch.from_numpy(window).to(device)

        with torch.no_grad():
            spikes = self.encoder.encode(window)

        if isinstance(spikes, torch.Tensor):
            spikes = spikes.cpu().numpy()

        return self._aggregate(spikes)

    # -------------------------------
    # METRICS
    # -------------------------------
    def _compute_metrics(self, rewards, terminated, truncated):
        rewards = np.array(rewards)

        total_reward = rewards.sum()
        mean_reward = rewards.mean()

        progress = rewards.sum()  # puedes mejorar luego

        return {
            "total_reward": float(total_reward),
            "mean_reward": float(mean_reward),
            "length": len(rewards),
            "terminated": terminated,
            "truncated": truncated,
            "success": bool(terminated),
            "progress": float(progress)
        }

    def encode_current_step(self, obs, action, reward=0.0):
        """
        Codifica usando una ventana deslizante del buffer actual.
        No rompe el episodio, solo usa los últimos window_size pasos.
        """

        # 1. add current step (sin modificar episodio aún)
        x = self._preprocess(obs, action, reward)

        temp_buffer = self.episode_buffer + [{"x": x}]

        if len(temp_buffer) < self.window_size:
            return None

        # 2. last window
        window = [step["x"] for step in temp_buffer[-self.window_size:]]

        window = np.array(window, dtype=np.float32)

        return self._encode_window(window)

    # -------------------------------
    # PREPROCESS
    # -------------------------------
    def _preprocess(self, obs, action, reward):
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        reward = np.asarray([reward], dtype=np.float32)

        x = np.concatenate([obs, action, reward])

        x = (x - x.mean()) / (x.std() + 1e-8)

        if not hasattr(self, "input_dim"):
            self.input_dim = x.shape[0]

        if x.shape[0] != self.input_dim:
            print(f"[WARN] fixing shape {x.shape[0]} -> {self.input_dim}")

            if x.shape[0] < self.input_dim:
                pad = np.zeros(self.input_dim - x.shape[0], dtype=np.float32)
                x = np.concatenate([x, pad])
            else:
                x = x[:self.input_dim]

        return x

    # -------------------------------
    def _aggregate(self, spikes):
        rho = spikes.mean(axis=0)

        if self.normalize:
            rho = rho / (np.linalg.norm(rho) + 1e-8)

        return rho

    def _get_device(self):
        try:
            return next(self.encoder.parameters()).device
        except:
            return torch.device("cpu")

    def reset(self):
        self.episode_buffer = []