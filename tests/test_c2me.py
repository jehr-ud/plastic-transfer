import uuid
from datetime import datetime
import random
import math


# ==============================
# Utils simples
# ==============================

def collect_trajectory():
    # trayectoria simulada (recompensas)
    return [random.uniform(0, 1) for _ in range(5)]

def compute_reward_score(trajectory):
    return sum(trajectory) / len(trajectory)

def similarity(a, b):
    # cosine similarity simple
    dot = sum(x*y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x*x for x in a))
    norm_b = math.sqrt(sum(x*x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot / (norm_a * norm_b)


# ==============================
# Fake models (simples)
# ==============================

class SimpleEmbeddingModel:
    def encode(self, data):
        # convierte trayectoria en vector (normalizado)
        return [x / sum(data) for x in data]

class SimpleConsolidationModel:
    def __call__(self, trajectory):
        # "modelo" = promedio → política simple
        avg = sum(trajectory) / len(trajectory)
        return {"policy": "slow_down" if avg > 0.5 else "explore", "value": avg}


embedding_model = SimpleEmbeddingModel()
learning_consolidation_model = SimpleConsolidationModel()


# ==============================
# Core classes
# ==============================

class KnowledgePackage:
    def __init__(self, k_type, content, embeddings, context, quality, source):
        self.type = k_type
        self.content = content
        self.embeddings = embeddings
        self.context = context
        self.quality = quality
        self.source = source


class C2MEMessage:
    def __init__(self, msg_type, payload, metadata=None, explanation=None):
        self.type = msg_type
        self.payload = payload
        self.metadata = metadata or self._default_metadata()
        self.explanation = explanation

    def _default_metadata(self):
        return {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "confidence": None,
            "task": None
        }


class C2MEProtocol:

    def submit_knowledge(self, sender, knowledge_package, explanation=None):
        return C2MEMessage(
            msg_type="SUBMISSION",
            payload=knowledge_package,
            metadata={
                "sender": sender,
                "confidence": knowledge_package.quality
            },
            explanation=explanation
        )


# ==============================
# Agents
# ==============================

class C2MEAgentExpert:

    def generate_embeddings(self, trajectory):
        return embedding_model.encode(trajectory)

    def consolidate(self, trajectory):
        return learning_consolidation_model(trajectory)

    def produce_knowledge(self, trajectory, context):
        embeddings = self.generate_embeddings(trajectory)
        consolidation = self.consolidate(trajectory)

        return KnowledgePackage(
            k_type="consolidation",
            content=consolidation,
            embeddings=embeddings,
            context=context,
            quality=self.evaluate(trajectory),
            source="expert_agent"
        )

    def evaluate(self, trajectory):
        return compute_reward_score(trajectory)


class C2MEAgentMentor:

    def __init__(self):
        self.short_memory = []
        self.long_memory = []

    def consolidate(self, message):
        pkg = message.payload
        confidence = message.metadata.get("confidence", 0)

        print(f"\n[MENTOR] Evaluating knowledge | quality={pkg.quality:.2f}")

        if pkg.quality > 0.6 and confidence > 0.6:
            print("[MENTOR] → Stored in LONG memory")
            self.long_memory.append(pkg)
        else:
            print("[MENTOR] → Stored in SHORT memory")
            self.short_memory.append(pkg)

    def retrieve(self, query_embedding):
        candidates = self.long_memory + self.short_memory

        if not candidates:
            return None

        best = max(
            candidates,
            key=lambda k: similarity(k.embeddings, query_embedding)
        )

        return best

    def act(self, query_embedding):
        knowledge = self.retrieve(query_embedding)

        if knowledge:
            print("[MENTOR] Providing knowledge from memory")
            return knowledge.content

        print("[MENTOR] No knowledge available")
        return None


class C2MEAdapter:

    def encode(self, input_data):
        return embedding_model.encode(input_data)

    def integrate(self, knowledge):
        print(f"[NOVICE] Integrating policy: {knowledge}")


class C2MEAgentNovice:

    def __init__(self, mentor, adapter):
        self.mentor = mentor
        self.adapter = adapter

    def explore(self, state):
        embedding = self.adapter.encode(state)

        knowledge = self.mentor.act(embedding)

        if knowledge:
            self.adapter.integrate(knowledge)

        return knowledge


# ==============================
# MAIN (PoC)
# ==============================

if __name__ == "__main__":

    print("=== C2ME Proof of Concept ===")

    # 1. Expert genera conocimiento
    trajectory = collect_trajectory()
    print("\n[EXPERT] Trajectory:", trajectory)

    expert = C2MEAgentExpert()
    knowledge = expert.produce_knowledge(trajectory, context="navigation")

    print("[EXPERT] Generated policy:", knowledge.content)

    # 2. Se envía al mentor
    protocol = C2MEProtocol()
    message = protocol.submit_knowledge(
        sender="expert_agent",
        knowledge_package=knowledge,
        explanation="Reducing speed improves reward under uncertainty"
    )

    mentor = C2MEAgentMentor()
    mentor.consolidate(message)

    # 3. Novato consulta
    novice = C2MEAgentNovice(mentor, C2MEAdapter())

    current_state = collect_trajectory()
    print("\n[NOVICE] Current state:", current_state)

    result = novice.explore(current_state)

    print("\nFinal result:", result)