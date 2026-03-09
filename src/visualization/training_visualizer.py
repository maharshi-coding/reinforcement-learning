"""
Pixel-art real-time training visualizer for AIRS.

Displays training results *slowly* so a viewer can follow along,
while the actual training runs at full speed in the background.

Controls:
  SPACE  - Pause / Resume display
  +/-    - Faster / Slower playback  (1x … 20x)
  ESC    - Quit
"""

from __future__ import annotations

import math
import random as _rand
from collections import deque

import numpy as np
import pygame

from src.visualization.training_state import TrainingState, StepData, EpisodeData

# ── Colour Palette ───────────────────────────────────────────────────
BG            = (10, 10, 22)
BG_ARENA      = (12, 14, 30)
PANEL_BG      = (14, 16, 32)
PANEL_BORDER  = (38, 42, 72)
WHITE         = (235, 235, 240)
DIM           = (90, 92, 115)
BRIGHT_DIM    = (140, 142, 165)
CYAN          = (0, 230, 255)
NEON_GREEN    = (50, 230, 20)
NEON_PINK     = (255, 30, 200)
NEON_BLUE     = (30, 140, 255)
RED           = (230, 55, 55)
ORANGE        = (240, 150, 40)
YELLOW        = (245, 220, 50)
GOLD          = (255, 200, 60)
PURPLE        = (170, 80, 240)
GREEN         = (50, 200, 50)
TEAL          = (40, 190, 170)
STEEL         = (75, 85, 105)
DARK_GREY     = (28, 30, 44)
MID_GREY      = (50, 54, 72)
SHIELD_BLUE   = (90, 170, 245)
FIRE_ORANGE   = (255, 110, 20)
XP_GOLD       = (220, 180, 40)
XP_BG         = (35, 30, 12)
NARR_BG       = (18, 22, 44)

ACTION_COLOURS = {0: TEAL, 1: SHIELD_BLUE, 2: ORANGE, 3: PURPLE}
ACTION_NAMES   = {0: "OBSERVE", 1: "BLOCK IP", 2: "RATE LIMIT", 3: "ISOLATE"}

# Narration templates – plain-english explanations
_NARRATIONS = {
    (0, "low"):  "The agent is monitoring the network. Threat is low — observing is the right call.",
    (0, "mid"):  "The agent watches carefully as threat rises. It's gathering information.",
    (0, "high"): "Threat is high but the agent chose to observe. Risky — it may lose reward.",
    (1, "low"):  "The agent blocks an IP address. Proactive defense against low threat.",
    (1, "mid"):  "The agent blocks a suspicious IP! Good defense at medium threat.",
    (1, "high"): "Under heavy attack, the agent blocks the attacker's IP. Strong move!",
    (2, "low"):  "The agent rate-limits traffic. It's being cautious even at low threat.",
    (2, "mid"):  "Traffic is throttled to slow down the attacker. Smart at medium threat.",
    (2, "high"): "Heavy attack incoming — the agent throttles traffic to buy time!",
    (3, "low"):  "The agent isolates the system. Drastic for low threat — costly move.",
    (3, "mid"):  "System isolated! The agent cuts all connections to stop the attack.",
    (3, "high"): "FULL ISOLATION! Maximum defense against critical threat level.",
}


def _lerp(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def _threat_band(threat: float) -> str:
    if threat < 0.35:
        return "low"
    elif threat < 0.65:
        return "mid"
    return "high"


# ── Pixel Sprites ────────────────────────────────────────────────────
# Each character maps to a colour.  ' ' is transparent.

DEFENDER_SPRITE = [
    "     cc     ",
    "    cCCc    ",
    "    cWCc    ",
    "     cc     ",
    "    BBBB    ",
    "   SsBBsS   ",
    "  SS BB SS  ",
    "  S  BB  S  ",
    "     BB     ",
    "    B  B    ",
    "   BB  BB   ",
    "   BB  BB   ",
]

HACKER_SPRITE = [
    "    gggg    ",
    "   gGGGGg   ",
    "   gRRGGg   ",
    "   gGGGGg   ",
    "    gggg    ",
    "   dDDDDd   ",
    "  dd DD dd  ",
    "  d  DD  d  ",
    "     DD     ",
    "    D  D    ",
    "   DD  DD   ",
    "   DD  DD   ",
]

SERVER_SPRITE = [
    "   aAAAAa   ",
    "  AAAAAAAA  ",
    "  A .gg. A  ",
    "  A .gg. A  ",
    "  AAAAAAAA  ",
    "  A gGGg A  ",
    "  A gGGg A  ",
    "  AAAAAAAA  ",
    "  A .gg. A  ",
    "  AAAAAAAA  ",
    "   aAAAAa   ",
]

FIREWALL_SPRITE = [
    "   oOOOOo   ",
    "  OOOOOOOO  ",
    "  O RRRR O  ",
    "  O R  R O  ",
    "  O RRRR O  ",
    "  OOOOOOOO  ",
    "  O fFFf O  ",
    "  O fFFf O  ",
    "  OOOOOOOO  ",
    "   oOOOOo   ",
]

DATABASE_SPRITE = [
    "   pPPPPp   ",
    "  PPPPPPPP  ",
    "  PP    PP  ",
    "  PPPPPPPP  ",
    "  PP    PP  ",
    "  PPPPPPPP  ",
    "  PP    PP  ",
    "  PPPPPPPP  ",
    "  PP    PP  ",
    "  PPPPPPPP  ",
    "   pPPPPp   ",
]

_PAL = {
    # Defender
    'c': (0, 160, 180),  'C': CYAN,   'W': WHITE,  'B': NEON_BLUE,
    'S': SHIELD_BLUE,    's': (60, 120, 180),
    # Hacker
    'g': (30, 140, 15),  'G': NEON_GREEN, 'R': RED,
    'd': (22, 24, 36),   'D': (40, 42, 60),
    # Server / buildings
    'a': (55, 60, 78),   'A': STEEL,  '.': (50, 60, 70),
    # Firewall
    'o': (180, 110, 30), 'O': ORANGE, 'F': FIRE_ORANGE, 'f': (180, 90, 20),
    # Database
    'p': (120, 55, 170), 'P': PURPLE,
}


def _draw_sprite(surf, data, x, y, scale=5, flash=None):
    for ri, row in enumerate(data):
        for ci, ch in enumerate(row):
            if ch == ' ':
                continue
            col = flash if flash else _PAL.get(ch, WHITE)
            surf.fill(col, (x + ci * scale, y + ri * scale, scale, scale))


def _sprite_px(data, scale=5):
    w = max(len(r) for r in data) * scale
    h = len(data) * scale
    return w, h


# ── Floating Text ────────────────────────────────────────────────────

class _Float:
    __slots__ = ("text", "x", "y", "col", "life", "max_life", "vy")

    def __init__(self, text, x, y, col, life=55):
        self.text, self.x, self.y = text, x, y
        self.col, self.life, self.max_life = col, life, life
        self.vy = -1.0

    def tick(self):
        self.y += self.vy
        self.vy *= 0.96
        self.life -= 1

    def draw(self, surf, font):
        if self.life <= 0:
            return
        a = max(0, int(255 * self.life / self.max_life))
        r = font.render(self.text, True, self.col)
        r.set_alpha(a)
        surf.blit(r, (int(self.x), int(self.y)))


# ── Particle ─────────────────────────────────────────────────────────

class _Particle:
    __slots__ = ("x", "y", "vx", "vy", "col", "life", "mlife", "sz")

    def __init__(self, x, y, vx, vy, col, life=24, sz=3):
        self.x, self.y, self.vx, self.vy = x, y, vx, vy
        self.col, self.life, self.mlife, self.sz = col, life, life, sz

    def tick(self):
        self.x += self.vx;  self.y += self.vy
        self.vy += 0.06;    self.life -= 1

    def draw(self, surf):
        if self.life <= 0:
            return
        a = max(40, int(200 * self.life / self.mlife))
        s = max(1, int(self.sz * self.life / self.mlife))
        ps = pygame.Surface((s * 2, s * 2), pygame.SRCALPHA)
        pygame.draw.circle(ps, (*self.col, a), (s, s), s)
        surf.blit(ps, (int(self.x) - s, int(self.y) - s))


# ═════════════════════════════════════════════════════════════════════

class TrainingVisualizer:
    """Pixel-art training visualizer with slow playback for presentations."""

    W = 1340
    H = 820
    FPS = 30

    # Playback: how many render-frames between consuming a step
    _SPEED_TABLE = [15, 10, 6, 4, 3, 2, 1]   # index 0 = slowest
    _SPEED_LABELS = ["0.2x", "0.3x", "0.5x", "1x", "1.5x", "2x", "5x"]
    _DEFAULT_SPEED = 2   # start at 0.5x  (1 step every 6 frames = ~5 steps/sec)

    def __init__(self, state: TrainingState):
        self._state = state
        pygame.init()
        pygame.display.set_caption(
            f"AIRS Cyber Arena  —  {state.algorithm.upper()} Training")
        self._scr = pygame.display.set_mode((self.W, self.H))
        self._clock = pygame.time.Clock()

        # Fonts
        self._f_title = pygame.font.SysFont("monospace", 22, bold=True)
        self._f_lg    = pygame.font.SysFont("monospace", 18, bold=True)
        self._f_md    = pygame.font.SysFont("monospace", 14, bold=True)
        self._f_sm    = pygame.font.SysFont("monospace", 13)
        self._f_xs    = pygame.font.SysFont("monospace", 11)
        self._f_dmg   = pygame.font.SysFont("monospace", 18, bold=True)
        self._f_big   = pygame.font.SysFont("monospace", 30, bold=True)
        self._f_narr  = pygame.font.SysFont("monospace", 14)

        # State
        self._tick = 0
        self._paused = False
        self._speed_idx = self._DEFAULT_SPEED
        self._frames_since_step = 0

        self._current: StepData = StepData()
        self._prev_action = -1

        self._episode_rewards: deque[float] = deque(maxlen=300)
        self._threat_history: deque[float] = deque(maxlen=200)
        self._reward_history: deque[float] = deque(maxlen=200)
        self._action_counts = [0, 0, 0, 0]
        self._total_steps = 0
        self._total_episodes = 0
        self._best_reward = float("-inf")
        self._recent_actions: deque[int] = deque(maxlen=60)

        # Internal buffer — training dumps fast, we drain slowly
        self._step_buffer: deque[StepData] = deque(maxlen=50_000)
        self._ep_buffer: deque[EpisodeData] = deque(maxlen=5_000)
        self._train_steps_collected = 0  # how many training pushed total

        # Layout
        self._arena_y = 50
        self._arena_h = 380
        self._narr_y = self._arena_y + self._arena_h + 2
        self._narr_h = 60
        self._panel_y = self._narr_y + self._narr_h + 2
        self._panel_h = self.H - self._panel_y - 24

        # Effects
        self._floats: list[_Float] = []
        self._particles: list[_Particle] = []
        self._log: deque[tuple[str, tuple]] = deque(maxlen=14)
        self._stars = self._mk_stars(90)
        self._def_flash = 0
        self._hkr_flash = 0
        self._shake = 0.0
        self._level = 1
        self._xp = 0.0
        self._xp_next = 100.0
        self._combo = 0
        self._combo_timer = 0
        self._narration = "Waiting for training data…"
        self._narr_col = DIM

        self._scanlines = self._mk_scanlines()

    # ── Setup helpers ────────────────────────────────────────────────

    def _mk_stars(self, n):
        return [
            {"x": _rand.randint(0, self.W),
             "y": _rand.randint(self._arena_y + 4, self._arena_y + self._arena_h - 4),
             "b": _rand.randint(30, 130),
             "spd": _rand.uniform(0.08, 0.35),
             "sz": _rand.choice([1, 1, 2])}
            for _ in range(n)
        ]

    def _mk_scanlines(self):
        s = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        for y in range(0, self.H, 4):
            pygame.draw.line(s, (0, 0, 0, 14), (0, y), (self.W, y))
        return s

    def _particles_burst(self, x, y, col, n=10, spd=2.5, life=22, sz=3):
        for _ in range(n):
            a = _rand.uniform(0, math.tau)
            v = _rand.uniform(0.3, spd)
            self._particles.append(_Particle(
                x, y, math.cos(a) * v, math.sin(a) * v,
                col, _rand.randint(int(life * 0.6), life), sz))

    def _float(self, text, x, y, col, life=50):
        self._floats.append(_Float(text, x + _rand.randint(-6, 6), y, col, life))

    def _add_log(self, msg, col=DIM):
        self._log.appendleft((msg, col))

    # ── Main Loop ────────────────────────────────────────────────────

    def run(self) -> None:
        running = True
        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        running = False
                    elif ev.key == pygame.K_SPACE:
                        self._paused = not self._paused
                    elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        self._speed_idx = min(self._speed_idx + 1, len(self._SPEED_TABLE) - 1)
                    elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        self._speed_idx = max(self._speed_idx - 1, 0)

            # Always drain training queues into our local buffer (fast)
            self._drain_queues()

            # Auto-speed-up once training is done to drain remaining buffer
            if self._state.is_training_done() and self._step_buffer:
                # Drain at ~200 steps/frame so the buffer clears quickly
                if not self._paused:
                    for _ in range(200):
                        if not self._step_buffer:
                            break
                        self._display_one_step()
                        self._display_pending_episodes()
            elif self._state.is_training_done() and not self._step_buffer:
                pass  # buffer empty, just idle showing final state
            elif not self._paused:
                # Normal slow playback during training
                self._frames_since_step += 1
                interval = self._SPEED_TABLE[self._speed_idx]
                if self._frames_since_step >= interval:
                    self._frames_since_step = 0
                    self._display_one_step()
                    self._display_pending_episodes()

            self._tick += 1
            self._update_fx()
            self._render()
            self._clock.tick(self.FPS)

        self._state.request_stop()
        pygame.quit()

    # ── Buffer management ────────────────────────────────────────────

    def _drain_queues(self):
        """Move everything from the thread-safe queues into local deques."""
        steps = self._state.get_pending_steps(max_items=500)
        for s in steps:
            self._step_buffer.append(s)
            self._train_steps_collected += 1
        eps = self._state.get_pending_episodes(max_items=100)
        for e in eps:
            self._ep_buffer.append(e)

    def _display_one_step(self):
        """Pop ONE step from local buffer and process it visually."""
        if not self._step_buffer:
            return
        s = self._step_buffer.popleft()
        self._current = s
        self._total_steps += 1
        self._action_counts[s.action] += 1
        self._recent_actions.append(s.action)
        self._threat_history.append(s.threat_level)
        self._reward_history.append(s.reward)

        threat = s.threat_level
        action = s.action
        band = _threat_band(threat)

        # ── Narration ────────────────────────────────────────────────
        key = (action, band)
        self._narration = _NARRATIONS.get(key, f"Agent chose {ACTION_NAMES[action]}.")
        self._narr_col = ACTION_COLOURS.get(action, WHITE)

        # ── Effects (restrained) ─────────────────────────────────────
        def_cx = self.W - 220 + 30
        def_cy = self._arena_y + self._arena_h - 90

        if action != self._prev_action and action > 0:
            self._def_flash = 12
            col = ACTION_COLOURS[action]
            self._particles_burst(def_cx, def_cy, col, n=8, spd=2.0, life=18, sz=3)
            labels = {1: "BLOCK!", 2: "THROTTLE!", 3: "ISOLATE!"}
            self._float(labels[action], def_cx - 30, def_cy - 50, col, life=45)
            self._add_log(f"  Agent {ACTION_NAMES[action]}s  (threat={threat:.2f})", col)

            if action == 3:
                self._shake = 4.0

            # Combo (capped at 10)
            if self._combo_timer > 0 and self._combo < 10:
                self._combo += 1
            else:
                self._combo = 1
            self._combo_timer = 50
        elif action == 0 and self._prev_action != 0:
            self._add_log(f"  Agent observes  (threat={threat:.2f})", DIM)
            self._combo = 0

        if threat > 0.3 and self._tick % 3 == 0:
            hk_cx = 130 + 30
            hk_cy = self._arena_y + self._arena_h - 90
            self._particles_burst(hk_cx, hk_cy, _lerp(ORANGE, RED, threat),
                                  n=max(1, int(threat * 3)), spd=1.8, life=14, sz=2)

        if threat > 0.5 and action == 0:
            self._hkr_flash = 8

        # Reward floater — only for notable rewards
        if abs(s.reward) > 1.5:
            rx = self.W // 2 + _rand.randint(-20, 20)
            ry = self._arena_y + 80
            txt = f"+{s.reward:.1f}" if s.reward > 0 else f"{s.reward:.1f}"
            self._float(txt, rx, ry, NEON_GREEN if s.reward > 0 else RED, life=40)

        # XP
        if s.reward > 0:
            self._xp += s.reward * 0.25
            if self._xp >= self._xp_next:
                self._level += 1
                self._xp -= self._xp_next
                self._xp_next *= 1.4
                self._float(f"LEVEL {self._level}!", self.W // 2 - 40,
                            self._arena_y + 50, GOLD, life=55)
                self._particles_burst(self.W // 2, self._arena_y + 80,
                                      GOLD, 14, 3.0, 28, 4)
                self._add_log(f"  ** LEVEL UP → Lv.{self._level}!", GOLD)

        self._prev_action = action

    def _display_pending_episodes(self):
        """Pop episodes whose timestep we've now passed."""
        while self._ep_buffer:
            ep = self._ep_buffer[0]
            # only show episodes we've "caught up" to
            self._ep_buffer.popleft()
            self._total_episodes += 1
            self._episode_rewards.append(ep.total_reward)
            if ep.total_reward > self._best_reward:
                self._best_reward = ep.total_reward
                self._add_log(f"  ★ NEW BEST: {ep.total_reward:.0f}!", GOLD)
                self._particles_burst(self.W // 2, self._arena_y + 100,
                                      GOLD, 12, 2.5, 25, 3)
            self._add_log(
                f"  Episode {ep.episode} done  →  reward: {ep.total_reward:.0f}", TEAL)
            break  # max 1 episode per display tick for readability

    # ── Effect update ────────────────────────────────────────────────

    def _update_fx(self):
        for s in self._stars:
            s["x"] -= s["spd"]
            if s["x"] < 0:
                s["x"] = self.W
                s["y"] = _rand.randint(self._arena_y + 4, self._arena_y + self._arena_h - 4)

        for p in self._particles:
            p.tick()
        self._particles = [p for p in self._particles if p.life > 0][:200]

        for f in self._floats:
            f.tick()
        self._floats = [f for f in self._floats if f.life > 0]

        if self._def_flash > 0:
            self._def_flash -= 1
        if self._hkr_flash > 0:
            self._hkr_flash -= 1
        if self._combo_timer > 0:
            self._combo_timer -= 1
        else:
            self._combo = 0
        if self._shake > 0:
            self._shake *= 0.82

    # ── Rendering ────────────────────────────────────────────────────

    def _render(self):
        sx = int(_rand.uniform(-self._shake, self._shake)) if self._shake > 0.4 else 0
        sy = int(_rand.uniform(-self._shake, self._shake)) if self._shake > 0.4 else 0

        self._scr.fill(BG)
        if sx or sy:
            self._scr.scroll(sx, sy)

        self._r_stars()
        self._r_topbar()
        self._r_arena()
        self._r_narration()
        self._r_panels()
        self._r_fx()
        self._scr.blit(self._scanlines, (0, 0))
        self._r_statusbar()

        pygame.display.flip()

    # ── Top Bar ──────────────────────────────────────────────────────

    def _r_topbar(self):
        h = 46
        # gradient
        for i in range(h):
            c = _lerp((22, 16, 42), (10, 10, 24), i / h)
            pygame.draw.line(self._scr, c, (0, i), (self.W, i))
        pygame.draw.line(self._scr, NEON_PINK, (0, h), (self.W, h), 2)

        algo = self._state.algorithm.upper() or "RL"
        title = f"AIRS CYBER ARENA   ─   {algo} AGENT TRAINING"
        # shadow text
        ts = self._f_title.render(title, True, (80, 20, 100))
        self._scr.blit(ts, (18, 9))
        t = self._f_title.render(title, True, WHITE)
        self._scr.blit(t, (16, 8))

        # ── XP bar ───────────────────────────────
        xp_x, xp_w, xp_h = 520, 200, 14
        xp_y = 5
        lv = self._f_md.render(f"Lv.{self._level}", True, GOLD)
        self._scr.blit(lv, (xp_x - 44, xp_y))
        pygame.draw.rect(self._scr, XP_BG, (xp_x, xp_y, xp_w, xp_h), border_radius=3)
        frac = min(self._xp / max(self._xp_next, 1), 1.0)
        fw = int(xp_w * frac)
        if fw > 0:
            c = _lerp(XP_GOLD, YELLOW, 0.5 + 0.5 * math.sin(self._tick * 0.08))
            pygame.draw.rect(self._scr, c, (xp_x, xp_y, fw, xp_h), border_radius=3)
        pygame.draw.rect(self._scr, GOLD, (xp_x, xp_y, xp_w, xp_h), 1, border_radius=3)
        xt = self._f_xs.render(f"{self._xp:.0f}/{self._xp_next:.0f}", True, WHITE)
        self._scr.blit(xt, (xp_x + xp_w // 2 - xt.get_width() // 2, xp_y + 1))

        # ── Progress bar ─────────────────────────
        pg_x, pg_w = 770, 320
        total = max(self._state.total_timesteps, 1)
        prog = min(self._current.timestep / total, 1.0)
        pygame.draw.rect(self._scr, DARK_GREY, (pg_x, xp_y, pg_w, xp_h), border_radius=3)
        pw = int(pg_w * prog)
        if pw > 0:
            pc = _lerp(NEON_BLUE, NEON_GREEN, prog)
            pygame.draw.rect(self._scr, pc, (pg_x, xp_y, pw, xp_h), border_radius=3)
        pygame.draw.rect(self._scr, NEON_BLUE, (pg_x, xp_y, pg_w, xp_h), 1, border_radius=3)
        pt = self._f_xs.render(f"TRAINING  {prog*100:.1f}%  ({self._current.timestep:,})", True, WHITE)
        self._scr.blit(pt, (pg_x + pg_w // 2 - pt.get_width() // 2, xp_y + 1))

        # ── Speed indicator ──────────────────────
        spd_lbl = self._SPEED_LABELS[self._speed_idx]
        sp = self._f_md.render(f"Speed: {spd_lbl}", True, YELLOW)
        self._scr.blit(sp, (pg_x + pg_w + 16, xp_y))

        # ── Episode/Step info line ───────────────
        ep = self._current
        info = f"Episode {ep.episode}  │  Step {ep.step_in_episode}/200  │  Displayed: {self._total_steps:,}  │  Trained: {self._train_steps_collected:,}"
        self._scr.blit(self._f_xs.render(info, True, DIM), (16, 28))

        buf_n = len(self._step_buffer)
        if buf_n > 50:
            bt = self._f_xs.render(f"Buffer: {buf_n:,} steps ahead", True, BRIGHT_DIM)
            self._scr.blit(bt, (self.W - bt.get_width() - 16, 28))

    # ── Stars ────────────────────────────────────────────────────────

    def _r_stars(self):
        for s in self._stars:
            b = int(s["b"] * (0.5 + 0.5 * math.sin(self._tick * 0.04 + s["x"])))
            self._scr.fill((b, b, min(255, b + 20)),
                           (int(s["x"]), int(s["y"]), s["sz"], s["sz"]))

    # ── Arena ────────────────────────────────────────────────────────

    def _r_arena(self):
        ay, ah = self._arena_y, self._arena_h
        # subtle arena gradient
        for i in range(0, ah, 2):
            c = _lerp(BG, BG_ARENA, i / ah)
            pygame.draw.line(self._scr, c, (0, ay + i), (self.W, ay + i))
            pygame.draw.line(self._scr, c, (0, ay + i + 1), (self.W, ay + i + 1))

        ground_y = ay + ah - 55
        # ground line
        pygame.draw.line(self._scr, MID_GREY, (0, ground_y), (self.W, ground_y), 2)
        # subtle checkerboard ground
        for gy in range(ground_y + 2, ay + ah, 6):
            for gx in range(0, self.W, 6):
                if (gx // 6 + gy // 6) % 2 == 0:
                    self._scr.fill((16, 20, 32), (gx, gy, 3, 3))

        threat = self._current.threat_level
        action = self._current.action
        scale = 5

        # ── Buildings (centre) ───────────────────
        sw, sh = _sprite_px(SERVER_SPRITE, scale)
        bld_base = ground_y - 8

        srv_x = self.W // 2 - sw // 2
        srv_y = bld_base - sh
        _draw_sprite(self._scr, SERVER_SPRITE, srv_x, srv_y, scale,
                     flash=RED if threat > 0.7 and self._tick % 10 < 5 else None)
        lbl = self._f_xs.render("SERVER", True, STEEL)
        self._scr.blit(lbl, (srv_x + sw // 2 - lbl.get_width() // 2, bld_base + 4))

        fw_x = self.W // 2 - 180
        fws, fwh = _sprite_px(FIREWALL_SPRITE, scale)
        _draw_sprite(self._scr, FIREWALL_SPRITE, fw_x, bld_base - fwh, scale,
                     flash=SHIELD_BLUE if action == 1 and self._def_flash > 0 else None)
        lbl = self._f_xs.render("FIREWALL", True, ORANGE)
        self._scr.blit(lbl, (fw_x + fws // 2 - lbl.get_width() // 2, bld_base + 4))

        db_x = self.W // 2 + 130
        dbs, dbh = _sprite_px(DATABASE_SPRITE, scale)
        _draw_sprite(self._scr, DATABASE_SPRITE, db_x, bld_base - dbh, scale,
                     flash=PURPLE if action == 3 and self._def_flash > 0 else None)
        lbl = self._f_xs.render("DATABASE", True, PURPLE)
        self._scr.blit(lbl, (db_x + dbs // 2 - lbl.get_width() // 2, bld_base + 4))

        # ── Hacker (left) ────────────────────────
        hk_x = 100
        hkw, hkh = _sprite_px(HACKER_SPRITE, scale)
        hk_y = ground_y - hkh - 4
        bob = int(2.5 * math.sin(self._tick * 0.06))

        # threat aura
        if threat > 0.25:
            r = 36 + int(threat * 20)
            aura = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            ac = _lerp(FIRE_ORANGE, RED, threat)
            pygame.draw.circle(aura, (*ac, int(30 + threat * 45)), (r, r), r)
            self._scr.blit(aura, (hk_x + hkw // 2 - r, hk_y + bob + hkh // 2 - r))

        flash_h = RED if self._hkr_flash > 0 else None
        _draw_sprite(self._scr, HACKER_SPRITE, hk_x, hk_y + bob, scale, flash=flash_h)

        # labels
        nm = self._f_sm.render("ATTACKER", True, NEON_GREEN)
        self._scr.blit(nm, (hk_x + hkw // 2 - nm.get_width() // 2, hk_y - 20))
        self._draw_bar(hk_x, hk_y - 8, hkw, 6, threat, RED, "THR")

        # ── Attack beam ──────────────────────────
        if threat > 0.15:
            bx0 = hk_x + hkw + 4
            by0 = hk_y + bob + hkh // 2
            bx1 = fw_x
            by1 = bld_base - fwh // 2
            segs = 18
            pts = []
            for i in range(segs + 1):
                t = i / segs
                px = bx0 + (bx1 - bx0) * t
                py = by0 + (by1 - by0) * t + math.sin(self._tick * 0.2 + i * 0.7) * (5 * threat)
                pts.append((int(px), int(py)))
            if len(pts) >= 2:
                bc = _lerp(ORANGE, RED, threat)
                thick = max(1, int(1.5 + threat * 2.5))
                pygame.draw.lines(self._scr, bc, False, pts, thick)
                # glow highlight on even ticks
                if self._tick % 2 == 0 and thick > 1:
                    gc = _lerp(YELLOW, RED, threat)
                    pygame.draw.lines(self._scr, gc, False, pts, 1)

        # ── Defender (right) ─────────────────────
        def_x = self.W - 220
        dfw, dfh = _sprite_px(DEFENDER_SPRITE, scale)
        def_y = ground_y - dfh - 4
        bob2 = int(2 * math.sin(self._tick * 0.07 + 1.5))

        # shield aura
        if action > 0 and self._def_flash > 0:
            r = 40 + int(6 * math.sin(self._tick * 0.15))
            sh = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            sc = ACTION_COLOURS.get(action, SHIELD_BLUE)
            pygame.draw.circle(sh, (*sc, 45), (r, r), r)
            pygame.draw.circle(sh, (*sc, 80), (r, r), r, 2)
            self._scr.blit(sh, (def_x + dfw // 2 - r, def_y + bob2 + dfh // 2 - r))

        d_flash = ACTION_COLOURS.get(action, None) if self._def_flash > 0 else None
        _draw_sprite(self._scr, DEFENDER_SPRITE, def_x, def_y + bob2, scale, flash=d_flash)

        health = max(0.0, 1.0 - threat * 0.5)
        nm = self._f_sm.render(f"{self._state.algorithm.upper()} AGENT", True, CYAN)
        self._scr.blit(nm, (def_x + dfw // 2 - nm.get_width() // 2, def_y - 20))
        self._draw_bar(def_x, def_y - 8, dfw, 6, health, GREEN, "HP")

        act_tag = ACTION_NAMES.get(action, "?")
        act_col = ACTION_COLOURS.get(action, WHITE)
        tag = self._f_md.render(f"▸ {act_tag}", True, act_col)
        self._scr.blit(tag, (def_x, def_y + dfh + bob2 + 8))

        # ── VS badge ────────────────────────────
        vx = self.W // 2
        vy = ground_y + 2
        pulse = 0.6 + 0.4 * math.sin(self._tick * 0.05)
        vc = _lerp(RED, NEON_PINK, pulse)
        vs = self._f_big.render("VS", True, vc)
        self._scr.blit(vs, (vx - vs.get_width() // 2, vy - vs.get_height()))

        # ── Episode reward ───────────────────────
        rew = self._current.episode_reward
        rc = NEON_GREEN if rew >= 0 else RED
        rt = self._f_lg.render(f"Episode Reward: {rew:+.1f}", True, rc)
        self._scr.blit(rt, (self.W // 2 - rt.get_width() // 2, ay + 6))

        if self._combo >= 3 and self._combo_timer > 0:
            cc = GOLD if self._combo < 6 else NEON_PINK
            cs = self._f_md.render(f"COMBO x{self._combo}", True, cc)
            self._scr.blit(cs, (self.W // 2 - cs.get_width() // 2, ay + 26))

    # ── HP/Threat bar helper ─────────────────────────────────────────

    def _draw_bar(self, x, y, w, h, frac, col, label):
        pygame.draw.rect(self._scr, DARK_GREY, (x, y, w, h), border_radius=2)
        fw = int(w * max(0.0, min(frac, 1.0)))
        if fw > 0:
            pygame.draw.rect(self._scr, col, (x, y, fw, h), border_radius=2)
        pygame.draw.rect(self._scr, WHITE, (x, y, w, h), 1, border_radius=2)
        ls = self._f_xs.render(label, True, WHITE)
        self._scr.blit(ls, (x + w + 4, y - 2))

    # ── Narration Panel ──────────────────────────────────────────────

    def _r_narration(self):
        ny, nh = self._narr_y, self._narr_h
        pygame.draw.rect(self._scr, NARR_BG, (0, ny, self.W, nh))
        pygame.draw.line(self._scr, PANEL_BORDER, (0, ny), (self.W, ny), 1)
        pygame.draw.line(self._scr, PANEL_BORDER, (0, ny + nh - 1), (self.W, ny + nh - 1), 1)

        # icon
        icon = self._f_md.render("💡 WHAT'S HAPPENING:", True, YELLOW)
        self._scr.blit(icon, (16, ny + 8))

        # narration text with typing effect
        txt = self._narration
        n = self._f_narr.render(txt, True, self._narr_col)
        self._scr.blit(n, (230, ny + 8))

        # second line: context
        s = self._current
        ctx_parts = []
        if s.attack_mode:
            ctx_parts.append(f"Attack: {s.attack_mode.upper()}")
        if s.intensity:
            ctx_parts.append(f"Intensity: {s.intensity.upper()}")
        ctx_parts.append(f"Threat: {s.threat_level:.2f}")
        ctx_parts.append(f"Reward: {s.reward:+.2f}")
        if s.phase:
            ctx_parts.append(f"Phase: {s.phase}")
        ctx = "   │   ".join(ctx_parts)
        cs = self._f_xs.render(ctx, True, BRIGHT_DIM)
        self._scr.blit(cs, (230, ny + 30))

    # ── Bottom Panels ────────────────────────────────────────────────

    def _r_panels(self):
        py, ph = self._panel_y, self._panel_h
        pygame.draw.rect(self._scr, PANEL_BG, (0, py, self.W, ph))
        pygame.draw.line(self._scr, PANEL_BORDER, (0, py), (self.W, py), 2)

        cw = self.W // 4
        self._r_stats(6, py + 6, cw - 12, ph - 12)
        self._r_actions(cw + 6, py + 6, cw - 12, ph - 12)
        self._r_rewards(cw * 2 + 6, py + 6, cw - 12, ph - 12)
        self._r_combat_log(cw * 3 + 6, py + 6, cw - 12, ph - 12)

        for i in range(1, 4):
            pygame.draw.line(self._scr, PANEL_BORDER, (cw * i, py + 4), (cw * i, py + ph - 4), 1)

    def _r_stats(self, x, y, w, h):
        self._scr.blit(self._f_md.render("BATTLE STATS", True, NEON_PINK), (x + 4, y))
        cy = y + 22
        s = self._current
        avg = np.mean(list(self._episode_rewards)) if self._episode_rewards else 0.0
        avg20 = np.mean(list(self._episode_rewards)[-20:]) if self._episode_rewards else 0.0

        rows = [
            ("Episodes",    f"{self._total_episodes}",    YELLOW),
            ("Displayed",   f"{self._total_steps:,}",     DIM),
            ("Trained",     f"{self._train_steps_collected:,}", BRIGHT_DIM),
            ("Threat",      f"{s.threat_level:.3f}",      _lerp(GREEN, RED, s.threat_level)),
            ("Step Reward", f"{s.reward:+.2f}",           NEON_GREEN if s.reward >= 0 else RED),
            ("Avg Reward",  f"{avg:.1f}",                 NEON_GREEN if avg > 0 else RED),
            ("Avg (20)",    f"{avg20:.1f}",               NEON_GREEN if avg20 > 0 else RED),
            ("Best",        f"{self._best_reward:.0f}" if self._best_reward > float("-inf") else "--", GOLD),
            ("Level",       f"{self._level}",             GOLD),
        ]
        for lbl, val, col in rows:
            ls = self._f_xs.render(f"{lbl}:", True, DIM)
            vs = self._f_xs.render(f" {val}", True, col)
            self._scr.blit(ls, (x + 6, cy))
            self._scr.blit(vs, (x + 6 + ls.get_width(), cy))
            cy += 15

        cy += 6
        bars = [
            ("Traffic", s.traffic,       _lerp(GREEN, RED, s.traffic)),
            ("CPU",     s.cpu,           _lerp(GREEN, ORANGE, s.cpu)),
            ("Memory",  s.memory,        _lerp(GREEN, ORANGE, s.memory)),
            ("Logins",  s.failed_logins, _lerp(GREEN, RED, s.failed_logins)),
        ]
        for lbl, val, col in bars:
            ls = self._f_xs.render(lbl, True, DIM)
            self._scr.blit(ls, (x + 6, cy))
            bx = x + 62
            bw = w - 95
            pygame.draw.rect(self._scr, DARK_GREY, (bx, cy + 2, bw, 7), border_radius=2)
            fw = int(bw * min(val, 1.0))
            if fw > 0:
                pygame.draw.rect(self._scr, col, (bx, cy + 2, fw, 7), border_radius=2)
            vs = self._f_xs.render(f"{val:.2f}", True, WHITE)
            self._scr.blit(vs, (bx + bw + 4, cy))
            cy += 14

    def _r_actions(self, x, y, w, h):
        self._scr.blit(self._f_md.render("ACTIONS", True, CYAN), (x + 4, y))
        cy = y + 24
        total = max(1, sum(self._action_counts))
        for i in range(4):
            cnt = self._action_counts[i]
            frac = cnt / total
            col = ACTION_COLOURS[i]
            self._scr.blit(self._f_xs.render(ACTION_NAMES[i], True, col), (x + 6, cy))
            cy += 13
            bw = w - 70
            pygame.draw.rect(self._scr, DARK_GREY, (x + 8, cy, bw, 10), border_radius=3)
            fw = int(bw * frac)
            if fw > 0:
                pygame.draw.rect(self._scr, col, (x + 8, cy, fw, 10), border_radius=3)
            pt = self._f_xs.render(f"{frac*100:.0f}% ({cnt})", True, WHITE)
            self._scr.blit(pt, (x + bw + 16, cy))
            cy += 16

        cy += 10
        self._scr.blit(self._f_xs.render("TIMELINE:", True, DIM), (x + 6, cy))
        cy += 13
        tx = x + 6
        for act in list(self._recent_actions)[-40:]:
            pygame.draw.rect(self._scr, ACTION_COLOURS.get(act, DIM), (tx, cy, 5, 8))
            tx += 6
            if tx > x + w - 10:
                tx = x + 6
                cy += 10

    def _r_rewards(self, x, y, w, h):
        self._scr.blit(self._f_md.render("REWARD HISTORY", True, NEON_GREEN), (x + 4, y))

        data = list(self._episode_rewards)
        cy = y + 22
        ch = h - 55
        cw = w - 14

        pygame.draw.rect(self._scr, (14, 16, 28), (x + 4, cy, cw, ch), border_radius=3)

        if len(data) < 2:
            wt = self._f_xs.render("Waiting for episodes…", True, STEEL)
            self._scr.blit(wt, (x + 20, cy + ch // 2 - 5))
        else:
            dmin, dmax = min(data), max(data)
            if dmax == dmin:
                dmax = dmin + 1.0
            rng = dmax - dmin
            n = len(data)
            sx = cw / max(n - 1, 1)

            pts = []
            for i, v in enumerate(data):
                px = x + 4 + int(i * sx)
                py = cy + ch - int(((v - dmin) / rng) * (ch - 8)) - 4
                pts.append((px, py))

            if len(pts) >= 3:
                fill = pts + [(pts[-1][0], cy + ch), (pts[0][0], cy + ch)]
                fs = pygame.Surface((cw, ch), pygame.SRCALPHA)
                shifted = [(px - x - 4, py - cy) for px, py in fill]
                pygame.draw.polygon(fs, (*NEON_GREEN, 20), shifted)
                self._scr.blit(fs, (x + 4, cy))

            if len(pts) >= 2:
                pygame.draw.lines(self._scr, NEON_GREEN, False, pts, 2)

            self._scr.blit(self._f_xs.render(f"{dmax:.0f}", True, DIM), (x + 6, cy + 2))
            self._scr.blit(self._f_xs.render(f"{dmin:.0f}", True, DIM), (x + 6, cy + ch - 12))

        # Threat sparkline
        ty = cy + ch + 6
        th = h - ch - 30
        self._scr.blit(self._f_xs.render("THREAT:", True, RED), (x + 4, ty))
        td = list(self._threat_history)
        if len(td) >= 2 and th > 6:
            pygame.draw.rect(self._scr, (18, 14, 14), (x + 52, ty, cw - 52, th), border_radius=2)
            sx = (cw - 52) / max(len(td) - 1, 1)
            pts = [(x + 52 + int(i * sx), ty + th - int(v * (th - 4)) - 2)
                   for i, v in enumerate(td)]
            if len(pts) >= 2:
                pygame.draw.lines(self._scr, RED, False, pts, 1)

    def _r_combat_log(self, x, y, w, h):
        self._scr.blit(self._f_md.render("COMBAT LOG", True, YELLOW), (x + 4, y))
        cy = y + 22
        for msg, col in self._log:
            if cy > y + h - 8:
                break
            self._scr.blit(self._f_xs.render(msg[:52], True, col), (x + 4, cy))
            cy += 14

    # ── FX overlay ───────────────────────────────────────────────────

    def _r_fx(self):
        for p in self._particles:
            p.draw(self._scr)
        for f in self._floats:
            f.draw(self._scr, self._f_dmg)

    # ── Status Bar ───────────────────────────────────────────────────

    def _r_statusbar(self):
        by = self.H - 22
        pygame.draw.rect(self._scr, (8, 8, 16), (0, by, self.W, 22))
        pygame.draw.line(self._scr, PANEL_BORDER, (0, by), (self.W, by), 1)

        status = "⏸ PAUSED" if self._paused else "▶ LIVE"
        speed = self._SPEED_LABELS[self._speed_idx]
        done_draining = self._state.is_training_done() and self._step_buffer
        done_idle = self._state.is_training_done() and not self._step_buffer
        scene = ""
        if self._current.attack_mode:
            scene = f"  │  {self._current.attack_mode.upper()} / {self._current.intensity.upper()}"

        if done_draining:
            txt = f"  ⏩ CATCHING UP...  ({len(self._step_buffer):,} steps remaining)   [ESC] Quit{scene}"
        elif done_idle:
            txt = f"  ★ TRAINING COMPLETE ★   All {self._total_steps:,} steps displayed.   [ESC] Quit{scene}"
        else:
            txt = f"  {status}   [SPACE] Pause   [+/-] Speed: {speed}   [ESC] Quit{scene}"
        self._scr.blit(self._f_xs.render(txt, True, DIM), (8, by + 5))

        fps = self._clock.get_fps()
        self._scr.blit(self._f_xs.render(f"FPS:{fps:.0f}", True, STEEL),
                       (self.W - 56, by + 5))
