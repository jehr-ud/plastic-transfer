class SkillLibrary:
    def __init__(self):
        self.skills = []

    def add(self, skill):
        self.skills.append(skill)

    def get_all(self):
        return self.skills

    def get_by_name(self, name):
        for s in self.skills:
            if s.name == name:
                return s
        return None

    def find_best(self, embedding, comparator, threshold=0.7):
        best_score = -1.0
        best_skill = None

        for skill in self.skills:
            score = comparator.compare(embedding, skill.embedding)

            if score > best_score:
                best_score = score
                best_skill = skill

        if best_score < threshold:
            return None, best_score

        return best_skill, best_score