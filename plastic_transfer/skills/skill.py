import numpy as np

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

        self.model = model_builder(env) if model_builder else None

        self.memory = []

    def act(self, obs):
        if self.model:
            action = self.model.predict(obs)
            return np.asarray(action).reshape(-1)

        return None

    def store(self, obs, action, reward):
        self.memory.append((obs, action, reward))

    def train(self, steps=1000):
        if not self.model or len(self.memory) < 10:
            return

        self.model.learn(self.memory, steps)

        self.memory = []