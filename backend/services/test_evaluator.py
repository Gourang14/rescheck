from backend.services.evaluator import hard_match_score, soft_match_score, final_score
def test_hard_score_simple():
    resume = "Experienced in Python, SQL and AWS."
    must = ["Python","SQL"]
    good = ["AWS"]
    hard, missing_must, missing_good = hard_match_score(resume, must, good)
    assert hard >= 70
    assert missing_must == []