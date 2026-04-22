import numpy as np

def validate_trigger(trigger, sample_obs_dict):
    if trigger == "always":
        return trigger

    try:
        result = eval(trigger, {}, sample_obs_dict)
        if isinstance(result, bool):
            return trigger
    except Exception as e:
        print(f"[Trigger INVALID] {trigger} -> {e}")

    return "always"


def serialize_memory(memory):
    serialized = []

    for item in memory:
        serialized.append({
            "embedding": item["embedding"].tolist() if item.get("embedding") is not None else None,
            "action": item["action"].tolist() if item.get("action") is not None else None,
            "reward": float(item.get("reward", 0.0)),
            "skills": [s.name for s in item.get("skills", [])]
        })

    return serialized
