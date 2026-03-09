"""
Pygame-based pixel renderer for AIRS – Autonomous Intrusion Response System.

Enhanced visual features:
  - Animated network topology with threat-reactive glow
  - Particle system for attack beams and defense pulses
  - Smooth colour transitions and pulsing effects
  - Real-time HUD with gauges, bars and sparklines
  - Action timeline with colour-coded history
  - Status banner with scenario info
  - Reward sparkline chart
  - Phase indicator
"""

import math
import pygame
import numpy as np
from collections import deque


# ── Colours ──────────────────────────────────────────────────────────
BLACK = (15, 15, 25)
BG_DARK = (10, 12, 20)
WHITE = (240, 240, 245)
DARK_GREY = (40, 42, 54)
GRID_GREY = (22, 24, 36)
PANEL_BG = (20, 22, 34)
CYAN = (80, 250, 250)
GREEN = (80, 250, 123)
RED = (255, 85, 85)
ORANGE = (255, 183, 77)
YELLOW = (241, 250, 140)
PURPLE = (189, 147, 249)
PINK = (255, 121, 198)
BLUE = (98, 114, 250)
TEAL = (45, 212, 191)
DIM_RED = (120, 40, 40)
DIM_GREEN = (40, 120, 60)
DIM_CYAN = (30, 90, 90)
DIM_BLUE = (40, 50, 120)
STEEL = (100, 116, 139)

# ── Action metadata ──────────────────────────────────────────────────
ACTION_INFO = {
    0: {"name": "OBSERVE", "colour": DIM_CYAN, "icon": "eye", "short": "OBS"},
    1: {"name": "BLOCK IP", "colour": RED, "icon": "shield", "short": "BLK"},
    2: {"name": "RATE LIMIT", "colour": ORANGE, "icon": "throttle", "short": "RLM"},
    3: {"name": "ISOLATE", "colour": PURPLE, "icon": "wall", "short": "ISO"},
}


class AIRSRenderer:
    """Pygame pixel renderer for the AIRS environment."""

    WIDTH = 1100
    HEIGHT = 700
    FPS = 8  # steps per second

    # Node positions (x_frac, y_frac) relative to the network area
    _NODES = {
        "server": (0.5, 0.22),
        "firewall": (0.22, 0.52),
        "database": (0.78, 0.52),
    }
    _ATTACKER_POS = (0.5, 0.82)

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("AIRS – Network Defense Visualizer")
        self._screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self._clock = pygame.time.Clock()
        self._font_lg = pygame.font.SysFont("monospace", 22, bold=True)
        self._font_md = pygame.font.SysFont("monospace", 16)
        self._font_sm = pygame.font.SysFont("monospace", 13)
        self._font_xs = pygame.font.SysFont("monospace", 11)
        self._tick = 0
        self._particles: list[dict] = []
        self._action_flash = 0
        self._history: list[dict] = []
        self._reward_sparkline: deque[float] = deque(maxlen=80)
        self._threat_sparkline: deque[float] = deque(maxlen=80)
        self._paused = False
        self._scenario_label = ""

    # ── Public API ───────────────────────────────────────────────────

    def render_frame(self, state: dict) -> bool:
        """Draw one frame. Returns False if the user closed the window."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False

        self._tick += 1
        self._history.append(state)
        self._reward_sparkline.append(state.get("episode_reward", 0.0))
        self._threat_sparkline.append(state.get("threat_level", 0.0))

        self._screen.fill(BG_DARK)
        self._draw_grid()
        self._draw_status_banner(state)
        self._draw_network(state)
        self._draw_attacker(state)
        self._draw_action_effect(state)
        self._draw_connections(state)
        self._draw_particles(state)
        self._draw_hud(state)
        self._draw_threat_gauge(state)
        self._draw_reward_bar(state)
        self._draw_sparklines(state)
        self._draw_action_history()
        self._draw_controls_hint()

        pygame.display.flip()
        self._clock.tick(self.FPS)
        return True

    def close(self):
        pygame.quit()

    def set_fps(self, fps: int):
        self.FPS = max(1, fps)

    def set_scenario(self, attack_mode: str, intensity: str):
        self._scenario_label = f"{attack_mode.upper()} | {intensity.upper()}"

    # ── Drawing helpers ──────────────────────────────────────────────

    def _net_x(self, frac: float) -> int:
        """Convert fractional x to pixel x within the network area (left 65%)."""
        return int(40 + frac * (self.WIDTH * 0.58))

    def _net_y(self, frac: float) -> int:
        return int(40 + frac * (self.HEIGHT - 80))

    def _draw_grid(self):
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self._screen, GRID_GREY, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self._screen, GRID_GREY, (0, y), (self.WIDTH, y))

    def _draw_network(self, state: dict):
        threat = state.get("threat_level", 0.0)
        action = state.get("action", 0)

        for name, (fx, fy) in self._NODES.items():
            cx, cy = self._net_x(fx), self._net_y(fy)
            # Glow based on threat
            glow_r = 30 + int(threat * 25)
            glow_col = self._lerp_colour(GREEN, RED, threat)
            glow_alpha = 60 + int(threat * 100)
            glow_surf = pygame.Surface((glow_r * 2, glow_r * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*glow_col, glow_alpha), (glow_r, glow_r), glow_r)
            self._screen.blit(glow_surf, (cx - glow_r, cy - glow_r))

            # Node body
            node_col = self._lerp_colour(CYAN, RED, threat * 0.6)
            if action == 3:  # isolate -> purple shield
                node_col = PURPLE
            pygame.draw.circle(self._screen, node_col, (cx, cy), 22)
            pygame.draw.circle(self._screen, WHITE, (cx, cy), 22, 2)

            # Label
            label = self._font_sm.render(name.upper(), True, WHITE)
            self._screen.blit(label, (cx - label.get_width() // 2, cy - 36))

            # Icon
            if name == "server":
                self._draw_server_icon(cx, cy)
            elif name == "firewall":
                self._draw_firewall_icon(cx, cy, action)
            elif name == "database":
                self._draw_db_icon(cx, cy)

    def _draw_server_icon(self, cx, cy):
        # Simple box icon
        pygame.draw.rect(self._screen, BLACK, (cx - 8, cy - 8, 16, 16), 0, 2)
        pygame.draw.rect(self._screen, WHITE, (cx - 8, cy - 8, 16, 16), 1, 2)
        pygame.draw.line(self._screen, WHITE, (cx - 5, cy - 3), (cx + 5, cy - 3))
        pygame.draw.line(self._screen, WHITE, (cx - 5, cy + 2), (cx + 5, cy + 2))

    def _draw_firewall_icon(self, cx, cy, action):
        # Shield shape when defending
        if action in (1, 2, 3):
            col = ACTION_INFO[action]["colour"]
            pts = [(cx, cy - 10), (cx + 10, cy - 4), (cx + 7, cy + 8),
                   (cx, cy + 12), (cx - 7, cy + 8), (cx - 10, cy - 4)]
            pygame.draw.polygon(self._screen, col, pts)
            pygame.draw.polygon(self._screen, WHITE, pts, 1)
        else:
            pygame.draw.rect(self._screen, BLACK, (cx - 8, cy - 8, 16, 16), 0, 2)
            for i in range(3):
                y = cy - 5 + i * 5
                pygame.draw.line(self._screen, WHITE, (cx - 5, y), (cx + 5, y))

    def _draw_db_icon(self, cx, cy):
        pygame.draw.ellipse(self._screen, BLACK, (cx - 10, cy - 6, 20, 10))
        pygame.draw.ellipse(self._screen, WHITE, (cx - 10, cy - 6, 20, 10), 1)
        pygame.draw.rect(self._screen, BLACK, (cx - 10, cy - 1, 20, 12))
        pygame.draw.line(self._screen, WHITE, (cx - 10, cy - 1), (cx - 10, cy + 11))
        pygame.draw.line(self._screen, WHITE, (cx + 9, cy - 1), (cx + 9, cy + 11))
        pygame.draw.ellipse(self._screen, WHITE, (cx - 10, cy + 5, 20, 10), 1)

    def _draw_attacker(self, state: dict):
        threat = state.get("threat_level", 0.0)
        fx, fy = self._ATTACKER_POS
        cx, cy = self._net_x(fx), self._net_y(fy)

        # Pulsing red glow
        pulse = 0.5 + 0.5 * math.sin(self._tick * 0.5)
        glow_r = int(20 + threat * 30 * pulse)
        glow_surf = pygame.Surface((glow_r * 2, glow_r * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (255, 60, 60, int(80 * threat)), (glow_r, glow_r), glow_r)
        self._screen.blit(glow_surf, (cx - glow_r, cy - glow_r))

        # Skull-like attacker icon
        pygame.draw.circle(self._screen, RED, (cx, cy), 18)
        pygame.draw.circle(self._screen, BLACK, (cx, cy), 16)
        # Eyes
        pygame.draw.circle(self._screen, RED, (cx - 5, cy - 3), 3)
        pygame.draw.circle(self._screen, RED, (cx + 5, cy - 3), 3)
        # Mouth
        pygame.draw.line(self._screen, RED, (cx - 4, cy + 5), (cx + 4, cy + 5), 2)

        label = self._font_sm.render("ATTACKER", True, RED)
        self._screen.blit(label, (cx - label.get_width() // 2, cy + 24))

        phase = state.get("phase", "")
        if phase:
            phase_label = self._font_sm.render(phase.upper(), True, ORANGE)
            self._screen.blit(phase_label, (cx - phase_label.get_width() // 2, cy + 38))

    def _draw_connections(self, state: dict):
        threat = state.get("threat_level", 0.0)
        action = state.get("action", 0)

        # Connections between nodes
        pairs = [("server", "firewall"), ("server", "database"), ("firewall", "database")]
        for n1, n2 in pairs:
            p1 = (self._net_x(self._NODES[n1][0]), self._net_y(self._NODES[n1][1]))
            p2 = (self._net_x(self._NODES[n2][0]), self._net_y(self._NODES[n2][1]))
            col = self._lerp_colour(DIM_CYAN, DIM_RED, threat)
            pygame.draw.line(self._screen, col, p1, p2, 2)

        # Attack line from attacker to firewall
        ax, ay = self._net_x(self._ATTACKER_POS[0]), self._net_y(self._ATTACKER_POS[1])
        fx, fy = self._net_x(self._NODES["firewall"][0]), self._net_y(self._NODES["firewall"][1])

        if threat > 0.1:
            # Animated dashed attack line
            num_dashes = 12
            for i in range(num_dashes):
                t1 = i / num_dashes
                t2 = (i + 0.5) / num_dashes
                if (i + self._tick) % 3 == 0:
                    continue
                x1 = int(ax + (fx - ax) * t1)
                y1 = int(ay + (fy - ay) * t1)
                x2 = int(ax + (fx - ax) * t2)
                y2 = int(ay + (fy - ay) * t2)
                intensity = min(255, int(threat * 400))
                col = (intensity, 40, 40)
                pygame.draw.line(self._screen, col, (x1, y1), (x2, y2), 2)

            # Spawn attack particles
            if self._tick % 2 == 0:
                self._particles.append({
                    "x": float(ax), "y": float(ay),
                    "dx": (fx - ax) / 30.0, "dy": (fy - ay) / 30.0,
                    "life": 30, "colour": RED,
                })

    def _draw_particles(self, state: dict):
        alive = []
        for p in self._particles:
            p["x"] += p["dx"]
            p["y"] += p["dy"]
            p["life"] -= 1
            if p["life"] > 0:
                alpha = min(255, p["life"] * 8)
                r = max(1, p["life"] // 8)
                surf = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
                col = p["colour"]
                pygame.draw.circle(surf, (*col, alpha), (r, r), r)
                self._screen.blit(surf, (int(p["x"]) - r, int(p["y"]) - r))
                alive.append(p)
        self._particles = alive

    def _draw_action_effect(self, state: dict):
        action = state.get("action", 0)
        if action == 0:
            return

        fx, fy = self._NODES["firewall"]
        cx, cy = self._net_x(fx), self._net_y(fy)

        info = ACTION_INFO[action]
        # Flash ring
        ring_r = 30 + int(10 * math.sin(self._tick * 0.8))
        pygame.draw.circle(self._screen, info["colour"], (cx, cy), ring_r, 3)

        # Action text
        text = self._font_md.render(info["name"], True, info["colour"])
        self._screen.blit(text, (cx - text.get_width() // 2, cy + 32))

        # Spawn defense particles
        if self._tick % 3 == 0:
            angle = self._tick * 0.3
            self._particles.append({
                "x": float(cx), "y": float(cy),
                "dx": math.cos(angle) * 3, "dy": math.sin(angle) * 3,
                "life": 15, "colour": info["colour"],
            })

    def _draw_hud(self, state: dict):
        """Draw the right-side HUD panel with metrics."""
        panel_x = self.WIDTH - 360
        panel_w = 350
        # Semi-transparent background
        panel_surf = pygame.Surface((panel_w, self.HEIGHT - 20), pygame.SRCALPHA)
        panel_surf.fill((20, 22, 34, 200))
        self._screen.blit(panel_surf, (panel_x, 10))

        # Title
        title = self._font_lg.render("AIRS Monitor", True, CYAN)
        self._screen.blit(title, (panel_x + 15, 20))
        pygame.draw.line(self._screen, CYAN, (panel_x + 15, 48), (panel_x + panel_w - 15, 48), 1)

        y = 60
        step = state.get("step", 0)
        reward = state.get("reward", 0.0)
        ep_reward = state.get("episode_reward", 0.0)
        threat = state.get("threat_level", 0.0)
        action = state.get("action", 0)
        action_name = ACTION_INFO.get(action, {}).get("name", "?")
        action_col = ACTION_INFO.get(action, {}).get("colour", WHITE)
        cost = state.get("service_cost", 0.0)
        logins = state.get("failed_logins", 0.0)
        traffic = state.get("traffic_rate", 0.0)
        cpu = state.get("cpu", 0.0)
        mem = state.get("memory", 0.0)

        metrics = [
            ("Step", f"{step} / 200", WHITE),
            ("Action", action_name, action_col),
            ("Threat", f"{threat:.3f}", self._lerp_colour(GREEN, RED, threat)),
            ("Step Reward", f"{reward:+.2f}", GREEN if reward > 0 else RED),
            ("Episode Total", f"{ep_reward:+.1f}", GREEN if ep_reward > 0 else RED),
            ("Service Cost", f"{cost:.3f}", ORANGE if cost > 0.1 else WHITE),
            ("", "", BLACK),
            ("Traffic Rate", f"{traffic:.1f}", WHITE),
            ("Failed Logins", f"{logins:.1f}", WHITE),
            ("CPU Usage", f"{cpu:.1%}", WHITE),
            ("Memory", f"{mem:.1%}", WHITE),
        ]

        for label, value, col in metrics:
            if not label:
                y += 8
                continue
            lbl_surf = self._font_sm.render(f"{label}:", True, (160, 160, 180))
            val_surf = self._font_md.render(value, True, col)
            self._screen.blit(lbl_surf, (panel_x + 20, y))
            self._screen.blit(val_surf, (panel_x + 20, y + 15))
            y += 36

    def _draw_threat_gauge(self, state: dict):
        """Draw a vertical threat gauge on the right panel."""
        threat = state.get("threat_level", 0.0)
        panel_x = self.WIDTH - 50
        gauge_y = 70
        gauge_h = 200

        # Background
        pygame.draw.rect(self._screen, DARK_GREY, (panel_x, gauge_y, 20, gauge_h), 0, 3)

        # Fill
        fill_h = int(threat * gauge_h)
        fill_col = self._lerp_colour(GREEN, RED, threat)
        if fill_h > 0:
            pygame.draw.rect(self._screen, fill_col,
                             (panel_x, gauge_y + gauge_h - fill_h, 20, fill_h), 0, 3)

        # Border
        pygame.draw.rect(self._screen, WHITE, (panel_x, gauge_y, 20, gauge_h), 1, 3)

        # Threshold marks
        for thresh, label in [(0.2, "LOW"), (0.6, "HIGH")]:
            ty = gauge_y + gauge_h - int(thresh * gauge_h)
            pygame.draw.line(self._screen, YELLOW, (panel_x - 5, ty), (panel_x + 25, ty), 1)

        lbl = self._font_sm.render("THREAT", True, WHITE)
        lbl = pygame.transform.rotate(lbl, 90)
        self._screen.blit(lbl, (panel_x - 18, gauge_y + gauge_h // 2 - lbl.get_height() // 2))

    def _draw_reward_bar(self, state: dict):
        """Draw a horizontal reward indicator bar."""
        panel_x = self.WIDTH - 350
        y = self.HEIGHT - 110
        w = 280
        h = 12

        ep_reward = state.get("episode_reward", 0.0)
        # Map reward to bar position: center = 0, left = -600, right = +600
        max_r = 600.0
        frac = max(-1.0, min(1.0, ep_reward / max_r))
        center = panel_x + w // 2

        # Background
        pygame.draw.rect(self._screen, DARK_GREY, (panel_x, y, w, h), 0, 3)
        # Center line
        pygame.draw.line(self._screen, WHITE, (center, y - 2), (center, y + h + 2), 1)

        # Fill bar
        if frac >= 0:
            bar_w = int(frac * (w // 2))
            pygame.draw.rect(self._screen, GREEN, (center, y, bar_w, h), 0, 3)
        else:
            bar_w = int(-frac * (w // 2))
            pygame.draw.rect(self._screen, RED, (center - bar_w, y, bar_w, h), 0, 3)

        label = self._font_sm.render(f"Episode Reward: {ep_reward:+.0f}", True, WHITE)
        self._screen.blit(label, (panel_x, y - 18))

    def _draw_action_history(self):
        """Draw a timeline of recent actions as coloured dots."""
        panel_x = self.WIDTH - 350
        y = self.HEIGHT - 35
        n = min(50, len(self._history))

        label = self._font_sm.render("Action Timeline:", True, (160, 160, 180))
        self._screen.blit(label, (panel_x, y - 15))

        for i in range(n):
            s = self._history[-(n - i)]
            a = s.get("action", 0)
            col = ACTION_INFO.get(a, {}).get("colour", DARK_GREY)
            x = panel_x + i * 5 + 5
            pygame.draw.rect(self._screen, col, (x, y, 4, 10), 0, 1)

    # ── Utilities ────────────────────────────────────────────────────

    def _draw_status_banner(self, state: dict):
        """Draw a top status banner with scenario info and step counter."""
        # Background bar
        banner_h = 32
        banner_surf = pygame.Surface((self.WIDTH, banner_h), pygame.SRCALPHA)
        banner_surf.fill((15, 23, 42, 220))
        self._screen.blit(banner_surf, (0, 0))

        # Left: scenario label
        if self._scenario_label:
            lbl = self._font_sm.render(f"SCENARIO: {self._scenario_label}", True, TEAL)
            self._screen.blit(lbl, (12, 8))

        # Center: step / max
        step = state.get("step", 0)
        step_text = self._font_sm.render(f"STEP {step:3d} / 200", True, WHITE)
        self._screen.blit(step_text, (self.WIDTH // 2 - step_text.get_width() // 2, 8))

        # Right: FPS and phase
        phase = state.get("phase", "")
        fps_text = self._font_xs.render(f"FPS {self.FPS}  |  {phase.upper()}", True, STEEL)
        self._screen.blit(fps_text, (self.WIDTH - fps_text.get_width() - 12, 10))

        # Progress bar
        progress = step / 200.0
        bar_y = banner_h - 3
        pygame.draw.rect(self._screen, DARK_GREY, (0, bar_y, self.WIDTH, 3))
        bar_col = self._lerp_colour(TEAL, ORANGE, progress)
        pygame.draw.rect(self._screen, bar_col, (0, bar_y, int(self.WIDTH * progress), 3))

    def _draw_sparklines(self, state: dict):
        """Draw mini sparkline charts for reward and threat history."""
        panel_x = self.WIDTH - 350
        y_base = self.HEIGHT - 80

        # Reward sparkline
        self._draw_one_sparkline(panel_x, y_base, 160, 30,
                                  list(self._reward_sparkline), GREEN, "Reward Trend")

        # Threat sparkline
        self._draw_one_sparkline(panel_x + 170, y_base, 160, 30,
                                  list(self._threat_sparkline), RED, "Threat Trend")

    def _draw_one_sparkline(self, x, y, w, h, data, colour, label):
        """Draw a mini sparkline chart."""
        # Background
        pygame.draw.rect(self._screen, DARK_GREY, (x, y, w, h), 0, 3)

        if len(data) < 2:
            return

        # Label
        lbl = self._font_xs.render(label, True, STEEL)
        self._screen.blit(lbl, (x + 2, y - 12))

        # Normalize
        mn, mx = min(data), max(data)
        rng = mx - mn if mx != mn else 1.0

        points = []
        for i, val in enumerate(data):
            px = x + int(i * w / max(len(data) - 1, 1))
            py = y + h - int((val - mn) / rng * (h - 4)) - 2
            points.append((px, py))

        if len(points) >= 2:
            pygame.draw.lines(self._screen, colour, False, points, 2)

            # Fill under line
            fill_points = points + [(points[-1][0], y + h), (points[0][0], y + h)]
            fill_surf = pygame.Surface((w, h), pygame.SRCALPHA)
            shifted = [(px - x, py - y) for px, py in fill_points]
            if len(shifted) >= 3:
                pygame.draw.polygon(fill_surf, (*colour, 40), shifted)
                self._screen.blit(fill_surf, (x, y))

    def _draw_controls_hint(self):
        """Draw control hints at the bottom left."""
        hints = "SPACE: Pause  |  UP/DOWN: Speed  |  R: Restart  |  ESC: Quit"
        lbl = self._font_xs.render(hints, True, (80, 80, 100))
        self._screen.blit(lbl, (12, self.HEIGHT - 18))

    @staticmethod
    def _lerp_colour(c1, c2, t):
        t = max(0.0, min(1.0, t))
        return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))
