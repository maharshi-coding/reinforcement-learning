"""AIRS baselines package."""
from src.baselines.no_op_policy import AlwaysNoopPolicy
from src.baselines.random_policy import RandomPolicy
from src.baselines.rule_based_defender import RuleBasedThresholdPolicy

BASELINE_REGISTRY = {
    "always_noop": AlwaysNoopPolicy,
    "random_policy": RandomPolicy,
    "rule_based_threshold": RuleBasedThresholdPolicy,
}


def get_baseline(name: str, **kwargs):
    """Instantiate a baseline policy by name."""
    if name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline '{name}'. Choose from {list(BASELINE_REGISTRY)}")
    return BASELINE_REGISTRY[name](**kwargs)


__all__ = [
    "AlwaysNoopPolicy",
    "RandomPolicy",
    "RuleBasedThresholdPolicy",
    "BASELINE_REGISTRY",
    "get_baseline",
]
