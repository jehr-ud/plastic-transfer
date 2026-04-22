import numpy as np
import torch

from plastic_transfer.core.temporal_representation import TemporalRepresentation
from plastic_transfer.core.encoders.snn import SNNEncoder

class ConstantMockEncoder:
    def parameters(self):
        return iter([])  # simula nn.Module
    def encode(self, trajectory):
        T = trajectory.shape[0]
        N = 4

        if isinstance(trajectory, np.ndarray):
            return np.ones((T, N))
        else:
            return torch.ones((T, N), device=trajectory.device)
        
class DynamicMockEncoder:
    def encode(self, trajectory):

        if isinstance(trajectory, np.ndarray):
            val = trajectory.mean()
            return np.ones((trajectory.shape[0], 4)) * val

        else:
            val = trajectory.mean()
            return torch.ones((trajectory.shape[0], 4), device=trajectory.device) * val

class ShapeCheckingEncoder:
    def encode(self, trajectory):
        assert len(trajectory.shape) == 2
        return np.ones((trajectory.shape[0], 3))
    


def test_not_ready_returns_none():
    tr = TemporalRepresentation(
        encoder=ConstantMockEncoder(),
        window_size=5
    )

    for _ in range(4):  # menos que window
        tr.add_step(np.array([1, 2]), np.array([0]), 1.0)

    assert tr.encode() is None


def test_ready_returns_embedding():
    tr = TemporalRepresentation(
        encoder=ConstantMockEncoder(),
        window_size=5
    )

    for _ in range(5):
        tr.add_step(np.array([1, 2]), np.array([0]), 1.0)

    z = tr.encode()

    assert z is not None
    assert z.shape == (4,)  # 4 neuronas del mock


def test_firing_rate_mean():
    tr = TemporalRepresentation(
        encoder=ConstantMockEncoder(),
        window_size=5,
        normalize=False
    )

    for _ in range(5):
        tr.add_step(np.array([1, 2]), np.array([0]), 1.0)

    z = tr.encode()

    assert np.allclose(z, np.ones(4))


def test_normalization():
    tr = TemporalRepresentation(
        encoder=ConstantMockEncoder(),
        window_size=5,
        normalize=True
    )

    for _ in range(5):
        tr.add_step(np.array([1, 2]), np.array([0]), 1.0)

    z = tr.encode()

    norm = np.linalg.norm(z)
    assert np.isclose(norm, 1.0)


def test_step_size():
    tr = TemporalRepresentation(
        encoder=ConstantMockEncoder(),
        window_size=5,
        step_size=2
    )

    for _ in range(5):
        tr.add_step(np.array([1, 2]), np.array([0]), 1.0)

    # step impar → no codifica
    assert tr.encode(step=1) is None

    # step par → sí codifica
    assert tr.encode(step=2) is not None


def test_reset():
    tr = TemporalRepresentation(
        encoder=ConstantMockEncoder(),
        window_size=5
    )

    for _ in range(5):
        tr.add_step(np.array([1, 2]), np.array([0]), 1.0)

    assert tr.encode() is not None

    tr.reset()

    assert tr.encode() is None


### TEST FOR INPUT TO ENCODER

def test_encoder_input_shape():
    tr = TemporalRepresentation(
        encoder=ShapeCheckingEncoder(),
        window_size=5
    )

    for _ in range(5):
        tr.add_step(np.array([1, 2]), np.array([0]), 1.0)

    tr.encode()  # si falla, lanza assert


def test_temporal_consistency():
    tr = TemporalRepresentation(
        encoder=DynamicMockEncoder(),
        window_size=5
    )

    # trayectoria 1
    for i in range(5):
        tr.add_step(np.array([i, i]), np.array([0]), 1.0)

    z1 = tr.encode()

    tr.reset()

    # trayectoria 2 muy diferente
    for i in range(5):
        tr.add_step(np.array([100+i, 100+i]), np.array([0]), 1.0)

    z2 = tr.encode()

    assert not np.allclose(z1, z2)


### SNNEncoder test

def test_snn_output_shape():
    encoder = SNNEncoder(input_size=3, hidden_size=8, latent_size=4)

    trajectory = torch.randn(10, 3)  # [T, features]

    spikes = encoder.encode(trajectory)

    assert spikes.shape == (10, 4)



def test_spikes_binary():
    encoder = SNNEncoder(3, 8, 4)

    trajectory = torch.randn(10, 3)

    spikes = encoder.encode(trajectory)

    assert torch.all((spikes == 0) | (spikes == 1))


def test_no_nan():
    encoder = SNNEncoder(3, 8, 4)

    trajectory = torch.randn(20, 3)

    spikes = encoder.encode(trajectory)

    assert not torch.isnan(spikes).any()


def test_not_all_zero():
    encoder = SNNEncoder(3, 8, 4)

    trajectory = torch.randn(20, 3)

    spikes = encoder.encode(trajectory)

    assert spikes.sum() > 0


def test_different_inputs_produce_different_spikes():
    encoder = SNNEncoder(3, 8, 4)

    traj1 = torch.randn(20, 3)
    traj2 = torch.randn(20, 3)

    spikes1 = encoder.encode(traj1)
    spikes2 = encoder.encode(traj2)

    assert not torch.allclose(spikes1, spikes2)


def test_temporal_with_snn():
    encoder = SNNEncoder(3, 8, 4)

    tr = TemporalRepresentation(
        encoder=encoder,
        window_size=10
    )

    for _ in range(10):
        obs = np.random.randn(2)
        action = np.random.randn(0)  # o ajusta dims
        reward = np.random.randn()

        tr.add_step(obs, action, reward)

    z = tr.encode()

    assert z is not None
    assert z.shape == (4,)


def test_firing_rate_range():
    encoder = SNNEncoder(4, 8, 4)

    tr = TemporalRepresentation(
        encoder=encoder,
        window_size=10,
        normalize=False
    )

    for _ in range(10):
        tr.add_step(np.random.randn(2), np.array([0]), 1.0)

    z = tr.encode()

    assert np.all(z >= 0)
    assert np.all(z <= 1)