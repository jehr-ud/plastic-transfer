class TrajectoryStore:
    def __init__(self):
        self.data = []

    def add(self, obs, action, reward):
        self.data.append((obs, action, reward))

    def get_all(self):
        return self.data