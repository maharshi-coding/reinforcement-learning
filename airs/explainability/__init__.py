"""
Explainability (XAI) module for AIRS.

Provides tools to understand *why* the RL agent chose a particular action:
  - Feature importance via observation perturbation
  - Q-value / action-probability decomposition
  - Human-readable textual explanations
  - Optional SHAP integration (if shap is installed)

Works with any Stable-Baselines3 model (DQN, PPO, A2C, RecurrentPPO).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger("airs.explainability")


# Feature names matching the 6-d observation vector
FEATURE_NAMES: list[str] = [
    "traffic_rate",
    "failed_logins",
    "cpu_usage",
    "memory_usage",
    "threat_level",
    "last_action",
]

ACTION_NAMES: dict[int, str] = {
    0: "no_op",
    1: "block_ip",
    2: "rate_limit",
    3: "isolate_service",
}


@dataclass
class Explanation:
    """Container for one decision explanation."""

    observation: np.ndarray
    chosen_action: int
    action_name: str
    # Per-action values (Q-values for DQN, log-probs for PPO/A2C)
    action_values: dict[int, float] = field(default_factory=dict)
    # Importance of each feature for the chosen action (higher = more important)
    feature_importance: dict[str, float] = field(default_factory=dict)
    # Human-readable summary
    summary: str = ""


class AIRSExplainer:
    """Explain AIRS agent decisions.

    Supports two methods:
      1. **Perturbation importance** (always available): perturb each feature
         independently and measure how the action distribution shifts.
      2. **SHAP** (optional): uses KernelSHAP on the model's predict function.

    Parameters
    ----------
    model : stable_baselines3 model
        A trained SB3 model with a ``predict`` method.
    num_perturbations : int
        Samples per feature for perturbation importance.
    use_shap : bool
        Whether to attempt SHAP-based explanation (requires ``shap`` package).
    """

    def __init__(
        self,
        model: Any,
        num_perturbations: int = 50,
        use_shap: bool = False,
    ):
        self._model = model
        self._n_perturb = num_perturbations
        self._use_shap = use_shap
        self._shap_explainer = None

        if use_shap:
            self._init_shap()

    # ------------------------------------------------------------------
    # Core public API
    # ------------------------------------------------------------------

    def explain(self, obs: np.ndarray) -> Explanation:
        """Produce a full explanation for a single observation.

        Args:
            obs: 1-D array of shape (6,) (or 6*N for temporal window).

        Returns:
            Explanation object with feature importance, action values, and summary.
        """
        obs = np.asarray(obs, dtype=np.float32).ravel()
        action = int(self._predict(obs))

        # Action values
        action_values = self._get_action_values(obs)

        # Feature importance
        if self._use_shap and self._shap_explainer is not None:
            importance = self._shap_importance(obs, action)
        else:
            importance = self._perturbation_importance(obs, action)

        summary = self._build_summary(obs, action, action_values, importance)

        return Explanation(
            observation=obs,
            chosen_action=action,
            action_name=ACTION_NAMES.get(action, f"action_{action}"),
            action_values=action_values,
            feature_importance=importance,
            summary=summary,
        )

    def explain_batch(self, observations: np.ndarray) -> list[Explanation]:
        """Explain multiple observations."""
        return [self.explain(obs) for obs in observations]

    # ------------------------------------------------------------------
    # Action value extraction
    # ------------------------------------------------------------------

    def _predict(self, obs: np.ndarray) -> int:
        action, _ = self._model.predict(obs, deterministic=True)
        return int(action)

    def _get_action_values(self, obs: np.ndarray) -> dict[int, float]:
        """Extract per-action values from the model.

        For DQN: Q-values.  For PPO/A2C: action log-probabilities.
        """
        import torch

        device = next(self._model.policy.parameters()).device
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

        # DQN: q_net gives Q-values directly
        if hasattr(self._model, "q_net"):
            with torch.no_grad():
                q_vals = self._model.q_net(obs_t).cpu().numpy().ravel()
            return {i: float(q_vals[i]) for i in range(len(q_vals))}

        # PPO / A2C: use the policy to get action logits
        if hasattr(self._model, "policy"):
            with torch.no_grad():
                dist = self._model.policy.get_distribution(obs_t)
                logits = dist.distribution.logits.cpu().numpy().ravel()
            return {i: float(logits[i]) for i in range(len(logits))}

        return {}

    # ------------------------------------------------------------------
    # Perturbation-based feature importance
    # ------------------------------------------------------------------

    def _perturbation_importance(
        self, obs: np.ndarray, chosen_action: int,
    ) -> dict[str, float]:
        """Measure how perturbing each feature changes the chosen-action score."""
        base_values = self._get_action_values(obs)
        if not base_values:
            return {name: 0.0 for name in FEATURE_NAMES}

        base_score = base_values.get(chosen_action, 0.0)
        n_features = min(len(obs), len(FEATURE_NAMES))
        importance = np.zeros(n_features)

        for feat_idx in range(n_features):
            deltas = []
            for _ in range(self._n_perturb):
                perturbed = obs.copy()
                perturbed[feat_idx] = np.random.uniform(0.0, 1.0)
                vals = self._get_action_values(perturbed)
                deltas.append(abs(base_score - vals.get(chosen_action, 0.0)))
            importance[feat_idx] = float(np.mean(deltas))

        # Normalise to sum=1
        total = importance.sum()
        if total > 0:
            importance = importance / total

        return {FEATURE_NAMES[i]: float(importance[i]) for i in range(n_features)}

    # ------------------------------------------------------------------
    # SHAP integration (optional)
    # ------------------------------------------------------------------

    def _init_shap(self):
        """Try to initialise a SHAP KernelExplainer."""
        try:
            import shap  # noqa: F401
            # Build background dataset (100 random observations)
            bg = np.random.uniform(0, 1, size=(100, 6)).astype(np.float32)
            self._shap_explainer = shap.KernelExplainer(
                self._shap_predict, bg,
            )
            logger.info("SHAP KernelExplainer initialised")
        except ImportError:
            logger.warning("shap package not installed — falling back to perturbation importance")
            self._use_shap = False
            self._shap_explainer = None

    def _shap_predict(self, obs_batch: np.ndarray) -> np.ndarray:
        """Wrapper for SHAP: returns chosen-action score per observation."""
        results = []
        for obs in obs_batch:
            vals = self._get_action_values(obs.astype(np.float32))
            action = self._predict(obs.astype(np.float32))
            results.append(vals.get(action, 0.0))
        return np.array(results)

    def _shap_importance(self, obs: np.ndarray, chosen_action: int) -> dict[str, float]:
        """Compute SHAP values for the chosen action."""
        shap_values = self._shap_explainer.shap_values(obs.reshape(1, -1))
        sv = np.abs(shap_values).ravel()
        total = sv.sum()
        if total > 0:
            sv = sv / total
        n = min(len(sv), len(FEATURE_NAMES))
        return {FEATURE_NAMES[i]: float(sv[i]) for i in range(n)}

    # ------------------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------------------

    def _build_summary(
        self,
        obs: np.ndarray,
        action: int,
        action_values: dict[int, float],
        importance: dict[str, float],
    ) -> str:
        """Build a concise textual explanation."""
        action_name = ACTION_NAMES.get(action, f"action_{action}")

        # Top 3 most important features
        sorted_feats = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
        top_feats = sorted_feats[:3]

        lines = [f"Action: {action_name}"]
        lines.append("Top influential features:")
        for fname, imp in top_feats:
            idx = FEATURE_NAMES.index(fname) if fname in FEATURE_NAMES else -1
            val = float(obs[idx]) if 0 <= idx < len(obs) else 0.0
            lines.append(f"  {fname}: value={val:.3f}, importance={imp:.1%}")

        if action_values:
            vals_str = ", ".join(f"{ACTION_NAMES.get(k, k)}={v:.3f}" for k, v in sorted(action_values.items()))
            lines.append(f"Action scores: [{vals_str}]")

        return "\n".join(lines)
