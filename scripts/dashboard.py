"""
AIRS Streamlit Dashboard – Modern interactive monitoring and visualisation.

Run:
    streamlit run scripts/dashboard.py
"""

import os
import sys
import glob
import time

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from airs.agent.baselines import BASELINE_REGISTRY, get_baseline
from airs.agent.rl_agent import AIRSAgent
from airs.environment.network_env import NetworkSecurityEnv
from airs.evaluation import evaluate_policy

# ─── Page config ───
st.set_page_config(
    page_title="AIRS Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS for modern look ───
st.markdown("""
<style>
    /* Dark theme tweaks */
    .stMetric { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
                padding: 16px; border-radius: 12px; border: 1px solid #334155; }
    .stMetric label { color: #94a3b8 !important; font-size: 0.85rem !important; }
    .stMetric [data-testid="stMetricValue"] { font-size: 1.8rem !important; }
    div[data-testid="stHorizontalBlock"] > div { padding: 4px; }
    .scenario-tag { display: inline-block; padding: 2px 10px; border-radius: 12px;
                    font-size: 0.8rem; font-weight: 600; margin: 2px; }
    .tag-high { background: #ef4444; color: white; }
    .tag-medium { background: #f59e0b; color: white; }
    .tag-low { background: #22c55e; color: white; }
    .tag-dqn { background: #3b82f6; color: white; }
    .tag-ppo { background: #f97316; color: white; }
    h1 { color: #f8fafc !important; }
    .big-header { font-size: 2.2rem; font-weight: 700; margin-bottom: 0; }
    .sub-header { color: #94a3b8; font-size: 1rem; margin-top: 0; }
</style>
""", unsafe_allow_html=True)

# ─── Header ───
st.markdown('<p class="big-header">🛡️ AIRS – Autonomous Intrusion Response System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Real-time agent monitoring, evaluation, and comparison dashboard</p>', unsafe_allow_html=True)
st.divider()

# ─── Sidebar ───
with st.sidebar:
    st.header("⚙️ Configuration")

    algorithm = st.selectbox("Algorithm", ["dqn", "ppo"], format_func=str.upper)
    attack_mode = st.selectbox("Attack Mode",
                                ["brute_force", "flood", "adaptive", "multi_stage"])
    intensity = st.selectbox("Intensity", ["low", "medium", "high"])
    episodes = st.slider("Eval Episodes", 5, 100, 20, step=5)

    st.divider()
    st.markdown("**Quick Info**")
    st.caption(f"Algorithm: **{algorithm.upper()}**")
    st.caption(f"Scenario: **{attack_mode}** @ **{intensity}**")

    model_dir = "models"
    model_path = os.path.join(model_dir, f"{algorithm}_agent")
    model_exists = os.path.exists(model_path + ".zip") or os.path.exists(model_path)
    if model_exists:
        st.success(f"✅ Model found")
    else:
        st.error(f"❌ No model at {model_path}")

# ─── Tabs ───
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Live Evaluation",
    "📈 Training Plots",
    "🔬 All Scenarios",
    "⚔️ DQN vs PPO",
    "📋 Results Table",
    "🎬 Episode Replay",
])

# ═══════════════════════════════════════════════════════════════
# Tab 1: Live Evaluation
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Run Live Evaluation")
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_eval = st.button("🚀 Evaluate Agent", use_container_width=True, type="primary")
    with col_info:
        st.caption(f"Will run **{episodes}** episodes of **{algorithm.upper()}** "
                   f"on **{attack_mode}** / **{intensity}**")

    if run_eval:
        if not model_exists:
            st.error("Model not found! Train the agent first.")
        else:
            progress = st.progress(0, text="Initializing...")
            progress.progress(10, text="Loading model...")
            agent = AIRSAgent(
                algorithm=algorithm, attack_mode=attack_mode,
                intensity=intensity, model_path=model_path,
            )
            progress.progress(30, text="Running evaluation...")
            result = evaluate_policy(
                agent, f"{algorithm}_agent", attack_mode, intensity, episodes,
            )
            progress.progress(100, text="Done!")
            time.sleep(0.3)
            progress.empty()

            # Metrics row
            st.divider()
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Mean Reward", f"{result.mean_reward:.1f}",
                       delta=f"±{result.std_reward:.1f}")
            c2.metric("Attack Success", f"{result.mean_attack_success*100:.1f}%",
                       delta="lower is better", delta_color="inverse")
            c3.metric("Detection Delay", f"{result.mean_detection_delay:.1f}",
                       delta="steps")
            c4.metric("False Positive Rate", f"{result.mean_fpr*100:.1f}%",
                       delta="lower is better", delta_color="inverse")
            c5.metric("Service Downtime", f"{result.mean_downtime:.1f}")
            c6.metric("Avg Cost", f"{result.mean_cost:.1f}")

            st.divider()

            # Charts
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.markdown("**Reward Distribution**")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(result.episode_rewards, bins=20, color="#3b82f6",
                        edgecolor="#1e293b", alpha=0.85)
                ax.axvline(result.mean_reward, color="#ef4444", linestyle="--",
                           linewidth=2, label=f"Mean = {result.mean_reward:.1f}")
                ax.set_xlabel("Episode Reward", fontsize=11)
                ax.set_ylabel("Count", fontsize=11)
                ax.legend(fontsize=10)
                ax.set_facecolor("#0f172a")
                fig.patch.set_facecolor("#0f172a")
                ax.tick_params(colors="#94a3b8")
                ax.xaxis.label.set_color("#94a3b8")
                ax.yaxis.label.set_color("#94a3b8")
                for spine in ax.spines.values():
                    spine.set_color("#334155")
                st.pyplot(fig)
                plt.close(fig)

            with chart_col2:
                st.markdown("**Action Distribution**")
                action_names = list(result.action_counts.keys())
                action_vals = list(result.action_counts.values())
                colours = ["#64748b", "#ef4444", "#f59e0b", "#8b5cf6"]
                fig2, ax2 = plt.subplots(figsize=(8, 4))
                bars = ax2.barh(action_names, action_vals, color=colours[:len(action_names)])
                for bar, val in zip(bars, action_vals):
                    ax2.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                             str(val), va="center", fontsize=10, color="#94a3b8")
                ax2.set_xlabel("Count", fontsize=11)
                ax2.set_facecolor("#0f172a")
                fig2.patch.set_facecolor("#0f172a")
                ax2.tick_params(colors="#94a3b8")
                ax2.xaxis.label.set_color("#94a3b8")
                for spine in ax2.spines.values():
                    spine.set_color("#334155")
                st.pyplot(fig2)
                plt.close(fig2)

            # Threat timeline for first episode
            if result.episode_metrics:
                ep = result.episode_metrics[0]
                st.markdown("**Episode 1 – Threat Level & Agent Actions**")
                fig3, ax3 = plt.subplots(figsize=(14, 4))

                steps = range(len(ep.threat_levels))
                ax3.fill_between(steps, ep.threat_levels, alpha=0.3, color="#ef4444")
                ax3.plot(steps, ep.threat_levels, color="#ef4444", linewidth=1.5,
                         label="Threat Level")

                # Thresholds
                ax3.axhline(0.6, color="#f59e0b", linestyle=":", alpha=0.5, label="High threshold")
                ax3.axhline(0.2, color="#22c55e", linestyle=":", alpha=0.5, label="Low threshold")

                ax3.set_ylabel("Threat Level", color="#94a3b8", fontsize=11)
                ax3.set_xlabel("Step", color="#94a3b8", fontsize=11)

                # Actions on secondary axis
                ax4 = ax3.twinx()
                action_colours = {0: "#64748b", 1: "#ef4444", 2: "#f59e0b", 3: "#8b5cf6"}
                action_labels = {0: "Observe", 1: "Block", 2: "Rate Limit", 3: "Isolate"}
                for a_val, a_label in action_labels.items():
                    mask = [i for i, a in enumerate(ep.actions) if a == a_val]
                    if mask:
                        ax4.scatter(mask, [a_val] * len(mask), c=action_colours[a_val],
                                    s=15, alpha=0.8, label=a_label, zorder=5)
                ax4.set_ylabel("Action", color="#94a3b8", fontsize=11)
                ax4.set_yticks([0, 1, 2, 3])
                ax4.set_yticklabels(["Observe", "Block", "Rate Limit", "Isolate"])
                ax4.tick_params(colors="#94a3b8")

                ax3.set_facecolor("#0f172a")
                fig3.patch.set_facecolor("#0f172a")
                ax3.tick_params(colors="#94a3b8")
                for spine in ax3.spines.values():
                    spine.set_color("#334155")
                for spine in ax4.spines.values():
                    spine.set_color("#334155")

                lines1, labels1 = ax3.get_legend_handles_labels()
                lines2, labels2 = ax4.get_legend_handles_labels()
                ax3.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
                           fontsize=8, facecolor="#1e293b", edgecolor="#334155",
                           labelcolor="#94a3b8")
                fig3.tight_layout()
                st.pyplot(fig3)
                plt.close(fig3)

# ═══════════════════════════════════════════════════════════════
# Tab 2: Saved Training Plots
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Training & Evaluation Plots")
    plot_files = sorted(glob.glob("results/*.png"))
    if not plot_files:
        st.info("No plots found in `results/`. Run training or evaluation first.")
    else:
        # Filter bar
        filter_text = st.text_input("🔍 Filter plots", placeholder="e.g. heatmap, dqn, reward...")
        if filter_text:
            plot_files = [p for p in plot_files if filter_text.lower() in p.lower()]

        cols = st.columns(2)
        for i, pf in enumerate(plot_files):
            with cols[i % 2]:
                st.image(pf, caption=os.path.basename(pf), use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# Tab 3: All Scenarios Overview
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.subheader("All Scenarios Evaluation")
    csv_path = "results/eval_all_scenarios.csv"

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        # Summary metrics
        rl_df = df[df["algorithm"].isin(["DQN", "PPO"])]
        if not rl_df.empty:
            st.markdown("**Overall Performance Summary**")
            c1, c2, c3, c4 = st.columns(4)
            for algo, col in zip(["DQN", "PPO"], [c1, c2]):
                sub = rl_df[rl_df["algorithm"] == algo]
                if not sub.empty:
                    with col:
                        st.metric(f"{algo} Avg Reward",
                                  f"{sub['mean_reward'].mean():.1f}",
                                  delta=f"min={sub['mean_reward'].min():.0f}")
            baseline_df = df[~df["algorithm"].isin(["DQN", "PPO"])]
            if not baseline_df.empty:
                best_bl = baseline_df.groupby("algorithm")["mean_reward"].mean().idxmax()
                best_bl_val = baseline_df.groupby("algorithm")["mean_reward"].mean().max()
                c3.metric("Best Baseline", best_bl, delta=f"{best_bl_val:.0f}")
                c4.metric("RL Advantage",
                          f"+{rl_df['mean_reward'].mean() - best_bl_val:.0f}",
                          delta="vs best baseline")

            st.divider()

            # Heatmaps
            st.markdown("**Reward Heatmaps**")
            hm_cols = st.columns(2)
            for algo, col in zip(["DQN", "PPO"], hm_cols):
                hm_path = f"results/heatmap_{algo.lower()}.png"
                if os.path.exists(hm_path):
                    with col:
                        st.image(hm_path, caption=f"{algo} Reward Heatmap",
                                 use_container_width=True)

            # Comparison chart
            comp_path = "results/dqn_vs_ppo_comparison.png"
            if os.path.exists(comp_path):
                st.markdown("**Algorithm Comparison**")
                st.image(comp_path, use_container_width=True)

        # Full data table
        st.divider()
        st.markdown("**Full Results Table**")
        st.dataframe(
            df.style.format({
                "mean_reward": "{:.1f}",
                "std_reward": "{:.1f}",
                "mean_fpr": "{:.3f}",
                "mean_detection_delay": "{:.1f}",
                "mean_cost": "{:.1f}",
                "mean_downtime": "{:.1f}",
                "mean_attack_success": "{:.3f}",
            }).background_gradient(subset=["mean_reward"], cmap="RdYlGn"),
            use_container_width=True,
            height=500,
        )
    else:
        st.info("Run `python scripts/evaluate_all.py` to generate comprehensive results.")

# ═══════════════════════════════════════════════════════════════
# Tab 4: DQN vs PPO Head-to-Head
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.subheader("⚔️ DQN vs PPO Head-to-Head")

    csv_path = "results/eval_all_scenarios.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        rl_df = df[df["algorithm"].isin(["DQN", "PPO"])]

        if len(rl_df["algorithm"].unique()) == 2:
            # Pivot: reward by scenario
            pivot = rl_df.pivot_table(values="mean_reward",
                                       index=["attack_mode", "intensity"],
                                       columns="algorithm", aggfunc="mean").reset_index()

            st.markdown("**Winner per scenario**")
            results_data = []
            dqn_wins = ppo_wins = ties = 0
            for _, row in pivot.iterrows():
                dqn_r = row.get("DQN", 0)
                ppo_r = row.get("PPO", 0)
                diff = dqn_r - ppo_r
                if abs(diff) < 5:
                    winner = "Tie"
                    ties += 1
                elif diff > 0:
                    winner = "DQN"
                    dqn_wins += 1
                else:
                    winner = "PPO"
                    ppo_wins += 1
                results_data.append({
                    "Attack Mode": row["attack_mode"],
                    "Intensity": row["intensity"],
                    "DQN": f"{dqn_r:.1f}",
                    "PPO": f"{ppo_r:.1f}",
                    "Δ (DQN-PPO)": f"{diff:+.1f}",
                    "Winner": winner,
                })
            st.dataframe(pd.DataFrame(results_data), use_container_width=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("DQN Wins", dqn_wins)
            c2.metric("PPO Wins", ppo_wins)
            c3.metric("Ties (< 5 pts)", ties)

            # By intensity chart
            st.divider()
            int_path = "results/reward_by_intensity.png"
            mode_path = "results/reward_by_attack_mode.png"
            if os.path.exists(int_path) and os.path.exists(mode_path):
                c1, c2 = st.columns(2)
                with c1:
                    st.image(int_path, caption="By Intensity", use_container_width=True)
                with c2:
                    st.image(mode_path, caption="By Attack Mode", use_container_width=True)
        else:
            st.info("Need both DQN and PPO results. Run evaluate_all.py first.")
    else:
        st.info("Run `python scripts/evaluate_all.py` to generate comparison data.")

# ═══════════════════════════════════════════════════════════════
# Tab 5: Results Table
# ═══════════════════════════════════════════════════════════════
with tab5:
    st.subheader("📋 Raw Results Table")

    # Try multiple CSV sources
    csv_files = sorted(glob.glob("results/*.csv"))
    if csv_files:
        selected_csv = st.selectbox("Select CSV file", csv_files)
        df = pd.read_csv(selected_csv)
        st.dataframe(df, use_container_width=True, height=500)

        # Download button
        csv_data = df.to_csv(index=False)
        st.download_button("📥 Download CSV", csv_data,
                           file_name=os.path.basename(selected_csv),
                           mime="text/csv")
    else:
        st.info("No CSV results found. Run evaluation first.")

# ═══════════════════════════════════════════════════════════════
# Tab 6: Episode Replay
# ═══════════════════════════════════════════════════════════════
with tab6:
    st.subheader("🎬 Episode Replay")
    st.caption("Watch step-by-step how the agent responds to an attack episode.")

    replay_col1, replay_col2 = st.columns([1, 3])
    with replay_col1:
        run_replay = st.button("▶️ Run Replay", type="primary", use_container_width=True)

    if run_replay:
        if not model_exists:
            st.error("Model not found!")
        else:
            with st.spinner("Running episode..."):
                env = NetworkSecurityEnv(attack_mode=attack_mode, intensity=intensity)
                agent = AIRSAgent(
                    algorithm=algorithm, attack_mode=attack_mode,
                    intensity=intensity, model_path=model_path,
                )
                obs, _ = env.reset()
                replay_data = []
                ep_reward = 0.0
                done = False
                while not done:
                    action = agent.predict(obs)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    ep_reward += reward
                    replay_data.append({
                        "Step": info["step"],
                        "Action": info["action_name"],
                        "Threat": round(info["threat_level"], 3),
                        "Reward": round(reward, 2),
                        "Cumulative": round(ep_reward, 1),
                        "Phase": info.get("phase", ""),
                        "Service Cost": round(info["service_cost"], 4),
                    })
                env.close()

            # Metrics
            rdf = pd.DataFrame(replay_data)
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Reward", f"{ep_reward:.1f}")
            c2.metric("Steps", len(replay_data))
            c3.metric("Actions Used", int(info.get("actions_used", 0)))

            # Replay chart
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                                            gridspec_kw={"height_ratios": [2, 1]})
            steps = rdf["Step"].values
            threats = rdf["Threat"].values
            rewards = rdf["Cumulative"].values

            ax1.fill_between(steps, threats, alpha=0.3, color="#ef4444")
            ax1.plot(steps, threats, color="#ef4444", linewidth=1.5, label="Threat")
            ax1_r = ax1.twinx()
            ax1_r.plot(steps, rewards, color="#22c55e", linewidth=1.5, label="Cumul. Reward")
            ax1.set_ylabel("Threat", color="#94a3b8")
            ax1_r.set_ylabel("Cumulative Reward", color="#94a3b8")
            ax1.set_facecolor("#0f172a")
            fig.patch.set_facecolor("#0f172a")
            ax1.tick_params(colors="#94a3b8")
            ax1_r.tick_params(colors="#94a3b8")
            for s in ax1.spines.values(): s.set_color("#334155")
            for s in ax1_r.spines.values(): s.set_color("#334155")

            # Actions bar
            action_map = {"no_op": 0, "block_ip": 1, "rate_limit": 2, "isolate_service": 3}
            action_cols = {0: "#64748b", 1: "#ef4444", 2: "#f59e0b", 3: "#8b5cf6"}
            action_ints = [action_map.get(a, 0) for a in rdf["Action"]]
            bar_colours = [action_cols.get(a, "#64748b") for a in action_ints]
            ax2.bar(steps, [1]*len(steps), color=bar_colours, width=1.0)
            ax2.set_ylabel("Action", color="#94a3b8")
            ax2.set_xlabel("Step", color="#94a3b8")
            ax2.set_yticks([])
            ax2.set_facecolor("#0f172a")
            ax2.tick_params(colors="#94a3b8")
            for s in ax2.spines.values(): s.set_color("#334155")

            # Legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=c, label=n) for n, c in
                               [("Observe","#64748b"),("Block","#ef4444"),
                                ("Rate Limit","#f59e0b"),("Isolate","#8b5cf6")]]
            ax2.legend(handles=legend_elements, loc="upper right", ncol=4,
                       fontsize=8, facecolor="#1e293b", edgecolor="#334155",
                       labelcolor="#94a3b8")

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Data table
            with st.expander("📋 Show step-by-step data"):
                st.dataframe(rdf, use_container_width=True, height=400)
