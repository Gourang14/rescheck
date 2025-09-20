import numpy as np

class Scorer:
    def __init__(self, weights={'hard':0.6,'soft':0.4}):
        self.weights = weights

    def hard_score(self, resume_text, must_have, nice_to_have):
        score = 0
        for skill in must_have:
            if skill.lower() in resume_text.lower():
                score += 1
        for skill in nice_to_have:
            if skill.lower() in resume_text.lower():
                score += 0.5
        return (score / (len(must_have)+0.5*len(nice_to_have)+1e-6)) * 100

    def soft_score(self, jd_vec, resume_vec):
        return float(np.dot(jd_vec, resume_vec) / (np.linalg.norm(jd_vec)*np.linalg.norm(resume_vec))) * 100

    def final_score(self, hard, soft):
        return round(self.weights['hard']*hard + self.weights['soft']*soft, 2)

    def verdict(self, score):
        if score >= 70:
            return "High"
        elif score >= 40:
            return "Medium"
        return "Low"