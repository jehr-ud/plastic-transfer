import numpy as np

class BasePolicyBuilder:

    def __init__(self, policy_config):
        self.config = policy_config

    def build(self):
        if not self.config:
            return None

        def policy(obs):
            ctx = {}

            # -----------------------
            # 1. inputs
            # -----------------------
            for inp in self.config["inputs"]:
                name = inp["name"]
                start, end = inp["indices"]
                ctx[name] = obs[start:end]

            # -----------------------
            # 2. intermediate ops
            # -----------------------
            for node in self.config.get("intermediate", []):
                name = node["name"]
                op = node["op"]
                inputs = [ctx[i] for i in node["inputs"]]
                params = node.get("params", {})

                if op == "sub":
                    ctx[name] = inputs[0] - inputs[1]

                elif op == "add":
                    ctx[name] = inputs[0] + inputs[1]

                elif op == "norm":
                    ctx[name] = np.linalg.norm(inputs[0])

                elif op == "clip":
                    ctx[name] = np.clip(
                        inputs[0],
                        params.get("min", -np.inf),
                        params.get("max", np.inf)
                    )

                elif op == "scale":
                    ctx[name] = inputs[0] * params.get("factor", 1.0)

                elif op == "dot":
                    ctx[name] = np.dot(inputs[0], inputs[1])

                else:
                    raise ValueError(f"Unknown op: {op}")

            # -----------------------
            # 3. output
            # -----------------------
            action = []

            for comp in self.config["outputs"][0]["components"]:
                source = comp["source"]
                idx = comp["index"]

                if idx is None:
                    action.append(ctx[source])
                else:
                    action.append(ctx[source][idx])

            return np.array(action, dtype=np.float32)

        return policy