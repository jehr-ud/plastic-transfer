import json

from plastic_transfer.skills.skill import Skill

class LearningDefinitions:
    def __init__(
            self,
            source, 
            validate_trigger_fn=None,
            sample_obs_dict=None
    ):
        self.raw = self._load(source)
        self.validate_trigger_fn = validate_trigger_fn
        self.sample_obs_dict = sample_obs_dict
        self.skills = self._build_skills(self.raw)

    def _load(self, source):
        if isinstance(source, dict):
            return source

        if isinstance(source, str):
            with open(source, "r", encoding="utf-8") as f:
                return json.load(f)

        raise ValueError("learning_definitions must be dict or path")

    def _build_skills(self, definitions):
        skills = []

        for s in definitions["skills"]:

            trigger = s["trigger"]

            if self.validate_trigger_fn:
                trigger = self.validate_trigger_fn(trigger, self.sample_obs_dict)

            skill = Skill(
                name=s["name"],
                description=s["description"],
                skill_type=s["type"],
                inputs=s["inputs"],
                outputs=s["outputs"],
                trigger=trigger,
                objective=s["objective"],
                order=s.get("order", 0)
            )

            skills.append(skill)

        skills.sort(key=lambda x: x.order)
        return skills

    def get_skills(self):
        return self.skills