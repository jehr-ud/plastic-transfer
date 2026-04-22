import numpy as np
import torch

from plastic_transfer.core.structural_comparison import StructuralComparison
    

# sim(z,z)=1
def test_identical_embeddings():
    comp = StructuralComparison()

    z = np.array([1.0, 2.0, 3.0])

    sim = comp.compare(z, z)

    assert np.isclose(sim, 1.0)


# sim=0
def test_orthogonal_embeddings():
    comp = StructuralComparison()

    z1 = np.array([1.0, 0.0])
    z2 = np.array([0.0, 1.0])

    sim = comp.compare(z1, z2)

    assert np.isclose(sim, 0.0)

# sim=−1

def test_opposite_embeddings():
    comp = StructuralComparison()

    z1 = np.array([1.0, 2.0])
    z2 = np.array([-1.0, -2.0])

    sim = comp.compare(z1, z2)

    assert np.isclose(sim, -1.0)


# sim∈[−1,1]

def test_similarity_range():
    comp = StructuralComparison()

    z1 = np.random.randn(5)
    z2 = np.random.randn(5)

    sim = comp.compare(z1, z2)

    assert -1.0 <= sim <= 1.0

# sim(z,2z)=1 / Invarianza a escala

def test_scale_invariance():
    comp = StructuralComparison()

    z1 = np.array([1.0, 2.0, 3.0])
    z2 = 2 * z1

    sim = comp.compare(z1, z2)

    assert np.isclose(sim, 1.0)


def test_different_embeddings():
    comp = StructuralComparison()

    z1 = np.array([1.0, 0.0])
    z2 = np.array([1.0, 1.0])

    sim = comp.compare(z1, z2)

    assert sim > 0 and sim < 1


def test_zero_vector():
    comp = StructuralComparison()

    z1 = np.array([0.0, 0.0, 0.0])
    z2 = np.array([1.0, 2.0, 3.0])

    sim = comp.compare(z1, z2)

    assert not np.isnan(sim)
    assert sim == 0.0


def test_torch_inputs():
    comp = StructuralComparison()

    z1 = torch.tensor([1.0, 2.0, 3.0])
    z2 = torch.tensor([1.0, 2.0, 3.0])

    sim = comp.compare(z1, z2)

    assert np.isclose(sim, 1.0)

# Ranking correcto
def test_best_match():
    comp = StructuralComparison()

    query = np.array([1.0, 0.0])

    candidates = [
        np.array([0.0, 1.0]),  # ortogonal
        np.array([1.0, 0.0]),  # igual
        np.array([-1.0, 0.0])  # opuesto
    ]

    sims = [comp.compare(query, c) for c in candidates]

    best_idx = np.argmax(sims)

    assert best_idx == 1


# Consistencia con ruido

def test_noise_robustness():
    comp = StructuralComparison()

    z = np.array([1.0, 2.0, 3.0])
    z_noise = z + np.random.normal(0, 0.01, size=3)

    sim = comp.compare(z, z_noise)

    assert sim > 0.99