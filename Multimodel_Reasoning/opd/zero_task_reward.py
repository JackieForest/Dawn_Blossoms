def compute_score(data_source=None, solution_str=None, ground_truth=None, extra_info=None, **kwargs):
    """No-op task reward for OPD smoke tests.

    OPD uses teacher token logprobs through verl's distillation loss. Some
    trainer paths still call a reward function after rollout, so this returns
    a neutral score without adding any task-reward signal.
    """
    return {"score": 0.0, "acc": 0.0}
