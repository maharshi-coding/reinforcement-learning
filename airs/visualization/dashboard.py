"""
AIRS Streamlit Dashboard – monitoring and visualisation.

Run:
    streamlit run src/visualization/dashboard.py
"""

import os
import sys
import glob
from pathlib import Path

import pandas as pd
import streamlit as st

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from airs.agent.rl_agent import AIRSAgent
from airs.environment.network_env import NetworkSecurityEnv as IntrusionEnv
from airs.evaluation import evaluate_policy

st.set_page_config(page_title="AIRS Dashboard", layout="wide")
st.title("AIRS – Autonomous Intrusion Response System")

# ─── Sidebar ───
st.sidebar.header("Configuration")
algorithm = st.sidebar.selectbox("Algorithm", ["dqn", "ppo"])
attack_mode = st.sidebar.selectbox("Attack Mode", ["brute_force", "flood", "adaptive", "multi_stage"])
intensity = st.sidebar.selectbox("Intensity", ["low", "medium", "high"])
episodes = st.sidebar.slider("Eval Episodes", 5, 100, 20)

model_dir = "models"
model_path = os.path.join(model_dir, f"{algorithm}_agent")

tab1, tab2, tab3, tab4 = st.tabs(["Evaluation", "Training Plots", "Results Table", "Episode Replay"])

# ─── Tab 1: Live evaluation ───
with tab1:
    st.subheader("Run Evaluation")
    if st.button("Evaluate Agent"):
        with st.spinner("Running evaluation..."):
            agent = AIRSAgent(
                algorithm=algorithm,
                attack_mode=attack_mode,
                intensity=intensity,
                model_path=model_path,
            )
            result = evaluate_policy(agent, f"{algorithm}_agent", attack_mode, intensity, episodes)

        col1, col2, col3 = st.columns(3)
        col1.metric("Mean Reward", f"{result.mean_reward:.2f}")
        col2.metric("Attack Success", f"{result.mean_attack_success*100:.1f}%")
        col3.metric("Detection Delay", f"{result.mean_detection_delay:.1f} steps")

        col4, col5, col6 = st.columns(3)
        col4.metric("False Positive Rate", f"{result.mean_fpr*100:.1f}%")
        col5.metric("Service Downtime", f"{result.mean_downtime:.2f}")
        col6.metric("Cost / Episode", f"{result.mean_cost:.2f}")

        import matplotlib.pyplot as plt

        st.subheader("Reward Distribution")
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(result.episode_rewards, bins=15, color="steelblue", edgecolor="white")
        ax.set_xlabel("Episode Reward")
        ax.set_ylabel("Count")
        ax.axvline(result.mean_reward, color="red", linestyle="--", label=f"Mean={result.mean_reward:.1f}")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

        if result.episode_metrics:
            ep = result.episode_metrics[0]
            st.subheader("Episode 1 – Threat Timeline")
            fig2, ax2 = plt.subplots(figsize=(10, 3))
            ax2.plot(ep.threat_levels, color="gray", label="Threat")
            ax2.set_ylabel("Threat Level")
            ax3 = ax2.twinx()
            ax3.scatter(range(len(ep.actions)), ep.actions, c=ep.actions, cmap="Set1", s=10, alpha=0.7)
            ax3.set_ylabel("Action")
            ax3.set_yticks([0, 1, 2, 3])
            ax3.set_yticklabels(["noop", "block", "rate_limit", "isolate"])
            ax2.legend(loc="upper left")
            st.pyplot(fig2)
            plt.close(fig2)

# ─── Tab 2: Saved plots ───
with tab2:
    st.subheader("Training & Evaluation Plots")
    plot_files = sorted(glob.glob("results/*.png") + glob.glob("plots/*.png"))
    if not plot_files:
        st.info("No plots found. Run training or evaluation first.")
    else:
        cols = st.columns(2)
        for i, pf in enumerate(plot_files):
            with cols[i % 2]:
                st.image(pf, caption=os.path.basename(pf), use_container_width=True)

# ─── Tab 3: Results table ───
with tab3:
    st.subheader("Evaluation Summary")
    results_csv = "results/eval_summary.csv"
    if os.path.exists(results_csv):
        df = pd.read_csv(results_csv)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No eval_summary.csv found. Run evaluation to generate it.")

# ─── Tab 4: Episode replay ───
with tab4:
    st.subheader("Episode Replay")
    st.info("Click 'Run Replay' to run a single episode and visualise step-by-step.")
    if st.button("Run Replay"):
        env = IntrusionEnv(attack_mode=attack_mode, intensity=intensity)
        agent = AIRSAgent(
            algorithm=algorithm,
            attack_mode=attack_mode,
            intensity=intensity,
            model_path=model_path,
        )
        obs, _ = env.reset()
        replay_data = []
        done = False
        while not done:
            action = agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            replay_data.append({
                "step": info["step"],
                "action": info["action_name"],
                "threat": round(info["threat_level"], 3),
                "reward": round(reward, 3),
                "service_cost": round(info["service_cost"], 4),
            })
        env.close()
        st.dataframe(pd.DataFrame(replay_data), use_container_width=True, height=400)
