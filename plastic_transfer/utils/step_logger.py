import csv
from datetime import datetime
import os

class StepLogger:
    def __init__(self, save_path="logs", name="plastic_transfer"):
        os.makedirs(save_path, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_path = f"{save_path}/training_steps_{timestamp}_{name}.csv"

        self.file = open(self.file_path, "w", newline="")
        self.writer = csv.writer(self.file)

        # columns
        self.writer.writerow([
            "step",
            "episode",
            "reward",
            "terminated",
            "truncated"
        ])

    def log(self, step, episode, reward, terminated, truncated):
        self.writer.writerow([
            step,
            episode,
            float(reward),
            int(terminated),
            int(truncated)
        ])

    def close(self):
        self.file.close()
        print(f"[LOGGER] Saved at {self.file_path}")