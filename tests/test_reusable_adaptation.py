import numpy as np
import pytest

from plastic_transfer.core.reusable_adaptation import ReusableAdaptation


# ----------------------------------
# Mock Comparator (cosine simple)
# ----------------------------------
class MockComparator:
    def compare(self, a, b):
        a = a / (np.linalg.norm(a) + 1e-8)
        b = b / (np.linalg.norm(b) + 1e-8)
        return np.dot(a, b)


# ----------------------------------
# Mock MemoryBank
# ----------------------------------
class MockMemoryBank:
    def __init__(self):
        self.data = []

    def add(self, emb, action):
        self.data.append((emb, action))

    def get_all(self):
        return self.data


# ==================================
# TESTS
# ==================================

# ✅ 1. memoria vacía → no cambia acción
def test_empty_memory_returns_base_action():
    adapter = ReusableAdaptation(MockComparator())
    memory = MockMemoryBank()

    emb = np.array([1.0, 0.0])
    base_action = np.array([0.5])

    out = adapter.adapt(emb, base_action, memory)

    assert np.allclose(out, base_action)


# ✅ 2. misma embedding → acción igual
def test_identical_embedding_returns_similar_action():
    adapter = ReusableAdaptation(MockComparator(), k=1)
    memory = MockMemoryBank()

    emb = np.array([1.0, 0.0])
    base_action = np.array([0.5])
    stored_action = np.array([0.8])

    memory.add(emb, stored_action)

    out = adapter.adapt(emb, base_action, memory)

    assert np.allclose(out, stored_action)


# ✅ 3. adaptación mueve la acción en dirección correcta
def test_adaptation_moves_action():
    adapter = ReusableAdaptation(MockComparator(), k=1)
    memory = MockMemoryBank()

    emb = np.array([1.0, 0.0])
    base_action = np.array([0.0])
    target_action = np.array([1.0])

    memory.add(emb, target_action)

    out = adapter.adapt(emb, base_action, memory)

    assert out > base_action
    assert out <= target_action


# ✅ 4. top-k funciona (solo usa los más similares)
def test_top_k_selection():
    adapter = ReusableAdaptation(MockComparator(), k=1)
    memory = MockMemoryBank()

    emb = np.array([1.0, 0.0])

    memory.add(np.array([1.0, 0.0]), np.array([1.0]))   # similar
    memory.add(np.array([0.0, 1.0]), np.array([10.0]))  # no similar

    base_action = np.array([0.0])

    out = adapter.adapt(emb, base_action, memory)

    # no debe irse hacia 10
    assert out < 5.0


# ✅ 5. múltiples vecinos → promedio ponderado
def test_weighted_average_behavior():
    adapter = ReusableAdaptation(MockComparator(), k=2)
    memory = MockMemoryBank()

    emb = np.array([1.0, 0.0])
    base_action = np.array([0.0])

    memory.add(np.array([1.0, 0.0]), np.array([1.0]))
    memory.add(np.array([0.8, 0.2]), np.array([0.5]))

    out = adapter.adapt(emb, base_action, memory)

    assert 0.5 <= out <= 1.0


# ✅ 6. no NaN con pesos pequeños
def test_no_nan_output():
    adapter = ReusableAdaptation(MockComparator(), k=3)
    memory = MockMemoryBank()

    emb = np.array([1.0, 0.0])
    base_action = np.array([0.0])

    # embeddings casi ortogonales → pesos pequeños
    memory.add(np.array([0.0, 1.0]), np.array([1.0]))
    memory.add(np.array([0.1, 0.9]), np.array([2.0]))

    out = adapter.adapt(emb, base_action, memory)

    assert not np.isnan(out).any()


# ✅ 7. estabilidad: pequeñas variaciones → salida similar
def test_stability_with_noise():
    adapter = ReusableAdaptation(MockComparator(), k=2)
    memory = MockMemoryBank()

    emb = np.array([1.0, 0.0])
    noisy_emb = emb + np.random.normal(0, 0.01, size=2)

    base_action = np.array([0.0])
    memory.add(emb, np.array([1.0]))

    out1 = adapter.adapt(emb, base_action, memory)
    out2 = adapter.adapt(noisy_emb, base_action, memory)

    assert np.allclose(out1, out2, atol=0.1)