import streamlit as st
import numpy as np
import cv2
import math
import time
import os
import tempfile
from collections import deque

st.set_page_config(
    page_title="Two-Hand + Torso Fusion Lift Controller",
    layout="wide"
)

try:
    import imageio
    HAS_IMAGEIO = True
except Exception:
    HAS_IMAGEIO = False


FRAME_W = 1120
FRAME_H = 520
PANEL_W = FRAME_W // 2

LIVE_INTERVAL = 0.12
REC_FPS = 8
REC_SECONDS = 60

MIN_BOX_HEIGHT = 0.30
MAX_BOX_HEIGHT = 1.25


def clamp(x, lo, hi):
    return float(max(lo, min(hi, x)))


def smoothstep(a, b, x):
    if b <= a:
        return 1.0 if x >= b else 0.0
    t = clamp((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def wrap_text(text, max_chars=86):
    text = str(text).strip()
    if not text:
        return [""]
    words = text.split()
    lines = []
    cur = ""
    for w in words:
        candidate = (cur + " " + w).strip()
        if len(candidate) <= max_chars:
            cur = candidate
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


def put_lines(img, lines, x, y, scale=0.45, color=(230, 230, 230), thickness=1, gap=18):
    for i, line in enumerate(lines):
        cv2.putText(
            img,
            line,
            (x, y + i * gap),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA
        )


def draw_series(img, series, x0, y0, x1, y1, vmin, vmax, color, thickness=2):
    vals = list(series)[-180:]
    if len(vals) < 2:
        return

    pts = []
    denom = max(1, len(vals) - 1)
    for i, v in enumerate(vals):
        x = int(x0 + (x1 - x0) * i / denom)
        vv = clamp((v - vmin) / (vmax - vmin), 0.0, 1.0)
        y = int(y1 - (y1 - y0) * vv)
        pts.append((x, y))

    cv2.polylines(
        img,
        [np.array(pts, dtype=np.int32)],
        False,
        color,
        thickness,
        cv2.LINE_AA
    )


class EMA:
    def __init__(self, alpha=0.30, init=0.0):
        self.alpha = float(alpha)
        self.value = float(init)
        self.ready = False

    def update(self, x):
        x = float(x)
        if not self.ready:
            self.value = x
            self.ready = True
        else:
            self.value += self.alpha * (x - self.value)
        return self.value


class SymmetryDetector:
    def __init__(self, maxlen=18):
        self.lv = deque(maxlen=maxlen)
        self.rv = deque(maxlen=maxlen)
        self.prev_l = None
        self.prev_r = None
        self.prev_t = None

    def update(self, l, r, t):
        l = float(l)
        r = float(r)
        t = float(t)

        if self.prev_l is not None:
            dt = max(t - self.prev_t, 1e-3)
            self.lv.append((l - self.prev_l) / dt)
            self.rv.append((r - self.prev_r) / dt)

        self.prev_l = l
        self.prev_r = r
        self.prev_t = t

        if len(self.lv) < 5:
            return 0.0

        a = np.array(self.lv, dtype=np.float64)
        b = np.array(self.rv, dtype=np.float64)

        # If both wrists are nearly static, treat them as synchronized.
        if np.std(a) < 0.02 and np.std(b) < 0.02:
            return 1.0

        if np.std(a) < 1e-5 or np.std(b) < 1e-5:
            return 0.0

        corr = np.corrcoef(a, b)[0, 1]
        if np.isnan(corr):
            corr = 0.0

        return float(clamp(corr, -1.0, 1.0))


class WholeBodyLiftSim:
    def __init__(self):
        self.filters = {
            "lw": EMA(0.35, 0.45),
            "rw": EMA(0.35, 0.45),
            "torso": EMA(0.25, 4.0),
        }
        self.detector = SymmetryDetector()

        self.height = MIN_BOX_HEIGHT
        self.legs_pct = 0.60
        self.arms_pct = 0.40
        self.com_x = 0.0
        self.com_margin = 1.0
        self.support_half = 0.21
        self.box_x = 0.45

        self.message = "Initializing"
        self.phase = "idle"
        self.symmetric = False
        self.corr = 0.0

        self.hist_lw = deque(maxlen=240)
        self.hist_rw = deque(maxlen=240)
        self.hist_torso = deque(maxlen=240)
        self.hist_legs = deque(maxlen=240)
        self.hist_margin = deque(maxlen=240)

        self.trail = deque(maxlen=30)
        self.last_t = None

    def update(self, t, s):
        dt = 0.05 if self.last_t is None else clamp(t - self.last_t, 0.005, 0.25)
        self.last_t = float(t)

        lw = self.filters["lw"].update(s["lw"])
        rw = self.filters["rw"].update(s["rw"])
        torso = self.filters["torso"].update(s["torso"])

        pinch_l = bool(s.get("pinch_l", False))
        pinch_r = bool(s.get("pinch_r", False))
        mass = float(s.get("mass", 18.0))
        stance = float(s.get("stance", 0.42))

        self.corr = self.detector.update(lw, rw, t)
        wrist_diff = abs(lw - rw)
        self.symmetric = (wrist_diff < 0.15) and (self.corr > 0.8)

        both_closed = pinch_l and pinch_r
        one_closed = pinch_l != pinch_r

        avg_wrist = 0.5 * (lw + rw)
        target = clamp((avg_wrist - 0.25) / (1.45 - 0.25), 0.0, 1.0)
        target = MIN_BOX_HEIGHT + target * (MAX_BOX_HEIGHT - MIN_BOX_HEIGHT)

        # Grip synchronization and symmetry gating.
        if not both_closed:
            target = self.height
            self.phase = "grip_wait"
            if one_closed:
                self.message = "Grip sync: waiting for second hand pinch."
            else:
                self.message = "Close both pinch gestures to start lift."
        elif not self.symmetric:
            target = self.height
            self.phase = "independent"
            self.message = "Independent arm motion detected. Align both wrists for symmetric lift."
        else:
            self.message = ""

        height_norm = clamp(
            (self.height - MIN_BOX_HEIGHT) / (MAX_BOX_HEIGHT - MIN_BOX_HEIGHT),
            0.0,
            1.0
        )

        torso_assist = clamp(torso / 25.0, -0.5, 0.8)

        # Load-sharing model:
        # Heavier box => more legs.
        # Lower phase of lift => more legs.
        # Slight backward lean helps stabilize and slightly increases leg contribution.
        self.legs_pct = clamp(
            0.28 + 0.011 * mass + 0.22 * (1.0 - height_norm) + 0.10 * torso_assist,
            0.12,
            0.93
        )
        self.arms_pct = 1.0 - self.legs_pct

        # Balance / support polygon check.
        self.support_half = max(0.05, stance / 2.0)

        # Box forward moment arm decreases slightly with backward torso lean
        # and with stronger leg-driven posture.
        self.box_x = 0.62 - 0.12 * torso_assist - 0.08 * self.legs_pct
        robot_x = -0.04 - 0.08 * torso_assist + 0.03 * self.arms_pct
        robot_mass = 55.0

        self.com_x = (robot_mass * robot_x + mass * self.box_x) / (robot_mass + mass)
        self.com_margin = self.support_half - abs(self.com_x)

        cap_msg = ""
        if self.com_margin < 0.0:
            cap_norm = clamp(
                0.65
                + 2.0 * (self.support_half - 0.21)
                - 0.006 * (mass - 20.0)
                + 0.05 * torso_assist,
                0.05,
                0.95
            )
            cap_height = MIN_BOX_HEIGHT + cap_norm * (MAX_BOX_HEIGHT - MIN_BOX_HEIGHT)

            if target > cap_height:
                target = cap_height

            if self.height > cap_height:
                target = min(target, cap_height)

            cap_msg = "Load too heavy for current stance. Widen stance."

        # Final phase classification after balance cap.
        if both_closed and self.symmetric:
            if target > self.height + 0.02:
                self.phase = "lift"
            elif target < self.height - 0.02:
                self.phase = "place"
            else:
                self.phase = "hold"

        max_speed = 0.32 if mass < 15.0 else (0.22 if mass < 30.0 else 0.16)
        delta = clamp(target - self.height, -max_speed * dt, max_speed * dt)
        self.height = clamp(self.height + delta, MIN_BOX_HEIGHT, MAX_BOX_HEIGHT)

        # Final message priority.
        if not both_closed:
            pass
        elif cap_msg:
            self.message = cap_msg
        elif not self.symmetric:
            self.message = "Independent arm motion detected. Align both wrists for symmetric lift."
        elif self.phase == "lift":
            self.message = "Symmetric lift: load shared between legs and arms."
        elif self.phase == "place":
            self.message = "Placing: mirrored lowering with grip sync."
        elif self.phase == "hold":
            self.message = "Holding stable. Torso lean counterbalances box."
        else:
            self.message = "Ready."

        self.hist_lw.append(lw)
        self.hist_rw.append(rw)
        self.hist_torso.append(torso)
        self.hist_legs.append(self.legs_pct)
        self.hist_margin.append(self.com_margin)

        self.trail.append({
            "height": self.height,
            "box_x": self.box_x,
            "torso": torso,
        })

        return {
            "t": t,
            "lw": lw,
            "rw": rw,
            "torso": torso,
            "corr": self.corr,
            "symmetric": self.symmetric,
            "both_closed": both_closed,
            "one_closed": one_closed,
            "pinch_l": pinch_l,
            "pinch_r": pinch_r,
            "mass": mass,
            "stance": stance,
            "height": self.height,
            "target": target,
            "legs_pct": self.legs_pct,
            "arms_pct": self.arms_pct,
            "com_x": self.com_x,
            "com_margin": self.com_margin,
            "phase": self.phase,
            "message": self.message,
            "box_x": self.box_x,
            "support_half": self.support_half,
        }


def demo_signals(t):
    t = float(t) % 60.0

    mass = 22.0
    stance = 0.42
    pinch_l = False
    pinch_r = False
    torso = 3.0
    h = 0.45

    if t < 1.2:
        s = smoothstep(0.0, 1.2, t)
        h = 0.60 - 0.20 * s
        torso = 2.0 + 4.0 * s
    elif t < 1.7:
        h = 0.40
        torso = 6.0
        pinch_l = True
        pinch_r = False
    elif t < 2.3:
        h = 0.39
        torso = 6.5
        pinch_l = True
        pinch_r = True
    elif t < 10.0:
        s = smoothstep(2.3, 10.0, t)
        h = 0.39 + 0.77 * s
        torso = 6.5 + 7.0 * s
        pinch_l = True
        pinch_r = True
    elif t < 14.0:
        h = 1.16 + 0.015 * math.sin(2.0 * math.pi * 0.6 * t)
        torso = 13.0 + 1.2 * math.sin(2.0 * math.pi * 0.3 * t)
        pinch_l = True
        pinch_r = True
    elif t < 26.0:
        h = 1.12 + 0.035 * math.sin(2.0 * math.pi * 0.18 * (t - 14.0))
        torso = 10.0 + 2.0 * math.sin(2.0 * math.pi * 0.12 * (t - 14.0))
        pinch_l = True
        pinch_r = True
    elif t < 28.0:
        s = smoothstep(26.0, 28.0, t)
        h = 1.12 - 0.50 * s
        torso = 10.0 - 3.0 * s
        pinch_l = True
        pinch_r = True
    elif t < 38.0:
        s = smoothstep(28.0, 36.0, t)
        h = 0.62 + 0.48 * s
        torso = 7.0 + 5.0 * s
        mass = 12.0
        pinch_l = True
        pinch_r = True
    elif t < 40.0:
        h = 1.10
        torso = 12.0
        mass = 12.0
        pinch_l = True
        pinch_r = True
    elif t < 43.0:
        s = smoothstep(40.0, 43.0, t)
        mass = 12.0 + 23.0 * s
        stance = 0.42 - 0.18 * s
        h = 1.10 - 0.18 * s
        torso = 12.0 - 2.0 * s
        pinch_l = True
        pinch_r = True
    elif t < 47.0:
        mass = 35.0
        stance = 0.24
        s = smoothstep(43.0, 47.0, t)
        h = 0.92 + 0.15 * s
        torso = 10.0 + 4.0 * s
        pinch_l = True
        pinch_r = True
    elif t < 49.0:
        mass = 35.0
        s = smoothstep(47.0, 49.0, t)
        stance = 0.24 + 0.31 * s
        h = 0.95
        torso = 13.0
        pinch_l = True
        pinch_r = True
    elif t < 52.0:
        mass = 35.0
        stance = 0.55
        h = 0.95
        torso = 13.0
        pinch_l = True
        pinch_r = True
    elif t < 58.0:
        s = smoothstep(52.0, 58.0, t)
        mass = 35.0
        stance = 0.55
        h = 0.95 - 0.52 * s
        torso = 13.0 - 8.0 * s
        pinch_l = True
        pinch_r = True
    else:
        mass = 35.0
        stance = 0.55
        h = 0.43
        torso = 5.0
        pinch_l = pinch_r = t < 58.6

    common = (
        0.006 * math.sin(2.0 * math.pi * 0.7 * t)
        + 0.004 * math.sin(2.0 * math.pi * 1.7 * t + 0.35)
    )

    lw = h + common + 0.002 * math.sin(3.1 * t)
    rw = h + common + 0.002 * math.sin(3.1 * t + 0.08)
    torso += 0.25 * math.sin(0.9 * t)

    return {
        "lw": lw,
        "rw": rw,
        "torso": torso,
        "pinch_l": pinch_l,
        "pinch_r": pinch_r,
        "mass": mass,
        "stance": stance,
    }


def manual_signals(t):
    lw = st.session_state.get("manual_lw", 0.55)
    rw = st.session_state.get("manual_rw", 0.55)
    torso = st.session_state.get("manual_torso", 6.0)
    mass = st.session_state.get("manual_mass", 18.0)
    stance = st.session_state.get("manual_stance", 0.42)
    pinch_l = st.session_state.get("manual_pinch_l", True)
    pinch_r = st.session_state.get("manual_pinch_r", True)

    n1 = 0.0025 * math.sin(2.0 * t)
    n2 = 0.0025 * math.sin(2.2 * t + 0.7)

    return {
        "lw": float(lw) + n1,
        "rw": float(rw) + n2,
        "torso": float(torso) + 0.12 * math.sin(0.8 * t),
        "pinch_l": bool(pinch_l),
        "pinch_r": bool(pinch_r),
        "mass": float(mass),
        "stance": float(stance),
    }


def draw_perception(frame, ctrl, sim):
    cv2.rectangle(frame, (0, 0), (PANEL_W, FRAME_H), (24, 26, 32), -1)

    cv2.putText(
        frame,
        "PERCEPTION: WRIST + TORSO CHANNELS",
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (240, 240, 245),
        1,
        cv2.LINE_AA
    )

    l_color = (80, 220, 120) if ctrl["pinch_l"] else (95, 95, 105)
    r_color = (80, 220, 120) if ctrl["pinch_r"] else (95, 95, 105)

    cv2.circle(frame, (PANEL_W - 115, 26), 8, l_color, -1, cv2.LINE_AA)
    cv2.putText(frame, "L", (PANEL_W - 120, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 225), 1, cv2.LINE_AA)

    cv2.circle(frame, (PANEL_W - 55, 26), 8, r_color, -1, cv2.LINE_AA)
    cv2.putText(frame, "R", (PANEL_W - 60, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 225), 1, cv2.LINE_AA)

    cv2.putText(frame, "Wrist height sync (m)", (18, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 205, 215), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (24, 72), (PANEL_W - 24, 182), (45, 48, 58), 1, cv2.LINE_AA)

    draw_series(frame, sim.hist_lw, 24, 72, PANEL_W - 24, 182, 0.2, 1.6, (80, 220, 120), 2)
    draw_series(frame, sim.hist_rw, 24, 72, PANEL_W - 24, 182, 0.2, 1.6, (255, 180, 80), 2)

    cv2.putText(frame, "Left wrist", (30, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 220, 120), 1, cv2.LINE_AA)
    cv2.putText(frame, "Right wrist", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 180, 80), 1, cv2.LINE_AA)

    cv2.putText(frame, "Torso pitch (deg)", (18, 206), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 205, 215), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (24, 216), (PANEL_W - 24, 268), (45, 48, 58), 1, cv2.LINE_AA)
    draw_series(frame, sim.hist_torso, 24, 216, PANEL_W - 24, 268, -15.0, 25.0, (170, 120, 255), 2)

    bar_x = 24
    bar_y = 292
    bar_w = PANEL_W - 48
    bar_h = 24

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 62, 72), -1)
    legs_w = int(bar_w * ctrl["legs_pct"])
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + legs_w, bar_y + bar_h), (80, 200, 120), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (95, 98, 108), 1)

    cv2.putText(
        frame,
        f"Load split: Legs {ctrl['legs_pct'] * 100:.0f}%  |  Arms {ctrl['arms_pct'] * 100:.0f}%",
        (bar_x, bar_y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (225, 228, 235),
        1,
        cv2.LINE_AA
    )

    margin = ctrl["com_margin"]
    margin_norm = clamp((margin + 0.10) / 0.40, 0.0, 1.0)
    margin_color = (80, 220, 120) if margin >= 0.0 else (80, 80, 255)

    cv2.rectangle(frame, (bar_x, bar_y + 56), (bar_x + bar_w, bar_y + 56 + bar_h), (60, 62, 72), -1)
    cv2.rectangle(frame, (bar_x, bar_y + 56), (bar_x + int(bar_w * margin_norm), bar_y + 56 + bar_h), margin_color, -1)
    cv2.rectangle(frame, (bar_x, bar_y + 56), (bar_x + bar_w, bar_y + 56 + bar_h), (95, 98, 108), 1)

    cv2.putText(
        frame,
        f"CoM margin: {margin * 100:.0f} cm",
        (bar_x, bar_y + 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (225, 228, 235),
        1,
        cv2.LINE_AA
    )

    status = [
        f"Phase: {ctrl['phase']}",
        f"Symmetric: {'YES' if ctrl['symmetric'] else 'NO'}   Velocity corr: {ctrl['corr']:.2f}",
        f"Wrist diff: {abs(ctrl['lw'] - ctrl['rw']) * 100:.1f} cm",
        f"Target height: {ctrl['target']:.2f} m   Current height: {ctrl['height']:.2f} m",
    ]
    put_lines(frame, status, 24, 402, 0.42, (225, 228, 235), 1, 18)

    y = 482
    for line in wrap_text(ctrl["message"], 58):
        cv2.putText(frame, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 220, 255), 1, cv2.LINE_AA)
        y += 16


def draw_physics(frame, ctrl, sim):
    x0 = PANEL_W
    cv2.rectangle(frame, (x0, 0), (FRAME_W, FRAME_H), (14, 18, 24), -1)

    cv2.putText(
        frame,
        "WHOLE-BODY PHYSICS TWIN",
        (x0 + 18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (240, 240, 245),
        1,
        cv2.LINE_AA
    )

    ground_y = 430
    px = 220.0
    hip_x = x0 + 185

    cv2.line(frame, (x0 + 20, ground_y), (FRAME_W - 20, ground_y), (80, 90, 100), 2, cv2.LINE_AA)

    support_half_px = ctrl["support_half"] * px
    base_color = (80, 220, 120) if ctrl["com_margin"] >= 0.0 else (80, 80, 255)

    cv2.line(
        frame,
        (int(hip_x - support_half_px), ground_y + 16),
        (int(hip_x + support_half_px), ground_y + 16),
        base_color,
        6,
        cv2.LINE_AA
    )
    cv2.putText(
        frame,
        "support polygon",
        (int(hip_x - support_half_px), ground_y + 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (180, 190, 200),
        1,
        cv2.LINE_AA
    )

    trail = list(sim.trail)
    for i, item in enumerate(trail):
        alpha = (i + 1) / max(1, len(trail))
        bx = int(hip_x + item["box_x"] * px)
        by = int(ground_y - item["height"] * px)
        col = (
            int(35 + 70 * alpha),
            int(55 + 90 * alpha),
            int(90 + 110 * alpha),
        )
        cv2.circle(frame, (bx, by), 3, col, -1, cv2.LINE_AA)

    box_px = int(hip_x + ctrl["box_x"] * px)
    box_py = int(ground_y - ctrl["height"] * px)
    box_w, box_h = 88, 60

    box_color = (0, 140, 255) if ctrl["com_margin"] >= 0.0 else (0, 80, 255)
    cv2.rectangle(
        frame,
        (box_px - box_w // 2, box_py - box_h // 2),
        (box_px + box_w // 2, box_py + box_h // 2),
        box_color,
        -1,
        cv2.LINE_AA
    )
    cv2.rectangle(
        frame,
        (box_px - box_w // 2, box_py - box_h // 2),
        (box_px + box_w // 2, box_py + box_h // 2),
        (235, 235, 240),
        2,
        cv2.LINE_AA
    )
    cv2.putText(
        frame,
        f"{ctrl['mass']:.0f} kg",
        (box_px - 24, box_py + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (20, 20, 25),
        1,
        cv2.LINE_AA
    )

    height_norm = clamp(
        (ctrl["height"] - MIN_BOX_HEIGHT) / (MAX_BOX_HEIGHT - MIN_BOX_HEIGHT),
        0.0,
        1.0
    )

    extension = clamp(height_norm * ctrl["legs_pct"] * 1.35, 0.0, 1.0)
    crouch = 1.0 - extension
    hip_height = 0.42 + 0.50 * extension
    hip_y = int(ground_y - hip_height * px)

    torso_rad = math.radians(ctrl["torso"])
    torso_len = 0.36 * px

    shoulder_x = int(hip_x - math.sin(torso_rad) * torso_len)
    shoulder_y = int(hip_y - math.cos(torso_rad) * torso_len)

    for dx, leg_color in [(-14, (190, 195, 210)), (16, (220, 225, 235))]:
        knee_x = int(hip_x + dx + 8 + 16 * crouch)
        knee_y = int(hip_y + (ground_y - hip_y) * 0.55)
        foot_x = int(hip_x + dx + 22)

        cv2.line(frame, (hip_x + dx, hip_y), (knee_x, knee_y), leg_color, 5, cv2.LINE_AA)
        cv2.line(frame, (knee_x, knee_y), (foot_x, ground_y), leg_color, 5, cv2.LINE_AA)

    cv2.line(frame, (hip_x, hip_y), (shoulder_x, shoulder_y), (230, 230, 235), 6, cv2.LINE_AA)

    head_x = int(shoulder_x - math.sin(torso_rad) * 20)
    head_y = int(shoulder_y - math.cos(torso_rad) * 20)
    cv2.circle(frame, (head_x, head_y), 12, (230, 230, 235), -1, cv2.LINE_AA)

    hand1 = (box_px - 26, box_py - 2)
    hand2 = (box_px + 26, box_py + 8)

    arm_color = (120, 200, 255)
    cv2.line(frame, (shoulder_x, shoulder_y), hand1, arm_color, 4, cv2.LINE_AA)
    cv2.line(frame, (shoulder_x, shoulder_y), hand2, arm_color, 4, cv2.LINE_AA)

    hand_color = (80, 220, 120) if ctrl["both_closed"] else (80, 80, 255)
    cv2.circle(frame, hand1, 6, hand_color, -1, cv2.LINE_AA)
    cv2.circle(frame, hand2, 6, hand_color, -1, cv2.LINE_AA)

    com_px = int(hip_x + ctrl["com_x"] * px)
    com_color = (80, 220, 120) if ctrl["com_margin"] >= 0.0 else (60, 60, 255)

    cv2.line(frame, (com_px, ground_y), (com_px, ground_y - 70), com_color, 2, cv2.LINE_AA)
    cv2.circle(frame, (com_px, ground_y - 70), 7, com_color, -1, cv2.LINE_AA)
    cv2.putText(frame, "CoM", (com_px + 8, ground_y - 68), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 225, 230), 1, cv2.LINE_AA)

    lines = [
        f"Phase: {ctrl['phase']}",
        f"Box mass: {ctrl['mass']:.1f} kg",
        f"Stance width: {ctrl['stance']:.2f} m",
        f"Legs {ctrl['legs_pct'] * 100:.0f}% / Arms {ctrl['arms_pct'] * 100:.0f}%",
        f"Torso pitch: {ctrl['torso']:.1f} deg",
        f"CoM margin: {ctrl['com_margin'] * 100:.0f} cm",
    ]
    put_lines(frame, lines, x0 + 18, 58, 0.43, (220, 225, 235), 1, 17)

    msg_color = (120, 220, 160) if ctrl["com_margin"] >= 0.0 else (80, 80, 255)
    y = FRAME_H - 58
    for line in wrap_text(ctrl["message"], 74):
        cv2.putText(
            frame,
            line,
            (x0 + 18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            msg_color,
            1,
            cv2.LINE_AA
        )
        y += 18


def render_frame(ctrl, sim):
    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    frame[:] = (18, 18, 24)

    draw_perception(frame, ctrl, sim)
    draw_physics(frame, ctrl, sim)

    return frame


def encode_video_bytes(frame_generator_factory, fps, size):
    if HAS_IMAGEIO:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.close()

        try:
            writer = imageio.get_writer(
                tmp.name,
                fps=fps,
                codec="libx264",
                quality=8,
                macro_block_size=1,
                output_params=["-pix_fmt", "yuv420p"]
            )

            for frame in frame_generator_factory():
                if frame.shape[1] != size[0] or frame.shape[0] != size[1]:
                    frame = cv2.resize(frame, size)
                writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            writer.close()

            with open(tmp.name, "rb") as f:
                data = f.read()

            os.unlink(tmp.name)
            return data, "video/mp4"

        except Exception:
            if os.path.exists(tmp.name):
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".avi")
    tmp.close()

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(tmp.name, fourcc, fps, size)

    if not writer.isOpened():
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
        return None, None

    for frame in frame_generator_factory():
        if frame.shape[1] != size[0] or frame.shape[0] != size[1]:
            frame = cv2.resize(frame, size)
        writer.write(frame)

    writer.release()

    with open(tmp.name, "rb") as f:
        data = f.read()

    os.unlink(tmp.name)
    return data, "video/avi"


def scripted_frame_factory(seconds, fps, progress=None):
    def factory():
        sim = WholeBodyLiftSim()
        total = int(seconds * fps)

        for i in range(total):
            t = i / fps
            sig = demo_signals(t)
            ctrl = sim.update(t, sig)
            frame = render_frame(ctrl, sim)

            if progress is not None and total > 0:
                progress.progress((i + 1) / total)

            yield frame

    return factory


def live_ring_factory(ring):
    def factory():
        for jpg in list(ring):
            arr = np.frombuffer(jpg, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if img is None:
                continue

            if img.shape[1] != FRAME_W or img.shape[0] != FRAME_H:
                img = cv2.resize(img, (FRAME_W, FRAME_H))

            yield img

    return factory


def reset_simulation():
    st.session_state.sim = WholeBodyLiftSim()
    st.session_state.start_time = time.time()
    st.session_state.frame_ring.clear()
    st.session_state.last_ring_time = 0.0


if "sim" not in st.session_state:
    st.session_state.sim = WholeBodyLiftSim()

if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()

if "frame_ring" not in st.session_state:
    st.session_state.frame_ring = deque(maxlen=int(REC_SECONDS * REC_FPS))

if "last_ring_time" not in st.session_state:
    st.session_state.last_ring_time = 0.0

if "live_video_bytes" not in st.session_state:
    st.session_state.live_video_bytes = None
    st.session_state.live_video_mime = "video/mp4"

if "scripted_video_bytes" not in st.session_state:
    st.session_state.scripted_video_bytes = None
    st.session_state.scripted_video_mime = "video/mp4"


st.sidebar.title("Whole-body lift control")

mode_options = [
    "Auto Demo (hands-free)",
    "Manual Gesture Simulator",
]

st.sidebar.selectbox(
    "Mode",
    mode_options,
    key="mode"
)

st.sidebar.button(
    "Restart demo / reset simulation",
    on_click=reset_simulation
)

st.sidebar.markdown("### Live recording")

st.sidebar.checkbox(
    "Keep rolling 60s buffer",
    value=True,
    key="record_live"
)

if st.sidebar.button("Download live recording (last up to 60s)"):
    if len(st.session_state.frame_ring) > 8:
        with st.spinner("Encoding live frames..."):
            data, mime = encode_video_bytes(
                live_ring_factory(st.session_state.frame_ring),
                REC_FPS,
                (FRAME_W, FRAME_H)
            )

        if data is not None:
            st.session_state.live_video_bytes = data
            st.session_state.live_video_mime = mime
        else:
            st.sidebar.error("Could not encode live video.")
    else:
        st.sidebar.warning("Not enough live frames yet. Let the demo run a few seconds.")

if st.session_state.live_video_bytes is not None:
    live_ext = "mp4" if st.session_state.live_video_mime == "video/mp4" else "avi"
    st.sidebar.download_button(
        "Save live recording",
        data=st.session_state.live_video_bytes,
        file_name=f"whole_body_lift_live.{live_ext}",
        mime=st.session_state.live_video_mime
    )

st.sidebar.markdown("### Scripted 60s export")

if st.sidebar.button("Render 60s demo video"):
    prog = st.sidebar.progress(0.0)

    with st.spinner("Rendering 60-second demo..."):
        data, mime = encode_video_bytes(
            scripted_frame_factory(60, 12, progress=prog),
            12,
            (FRAME_W, FRAME_H)
        )

    prog.empty()

    if data is not None:
        st.session_state.scripted_video_bytes = data
        st.session_state.scripted_video_mime = mime
    else:
        st.sidebar.error("Could not render video.")

if st.session_state.scripted_video_bytes is not None:
    scripted_ext = "mp4" if st.session_state.scripted_video_mime == "video/mp4" else "avi"
    st.sidebar.download_button(
        "Save 60s scripted demo",
        data=st.session_state.scripted_video_bytes,
        file_name=f"whole_body_lift_demo_60s.{scripted_ext}",
        mime=st.session_state.scripted_video_mime
    )

current_mode = st.session_state.get("mode", mode_options[0])

if current_mode.startswith("Manual"):
    st.sidebar.markdown("### Manual gesture channels")

    st.sidebar.slider(
        "Box mass (kg)",
        0.0,
        60.0,
        18.0,
        0.5,
        key="manual_mass"
    )

    st.sidebar.slider(
        "Stance width (m)",
        0.15,
        0.75,
        0.42,
        0.01,
        key="manual_stance"
    )

    st.sidebar.slider(
        "Left wrist height (m)",
        0.20,
        1.60,
        0.55,
        0.01,
        key="manual_lw"
    )

    st.sidebar.slider(
        "Right wrist height (m)",
        0.20,
        1.60,
        0.55,
        0.01,
        key="manual_rw"
    )

    st.sidebar.slider(
        "Torso pitch (deg, + lean back)",
        -15.0,
        25.0,
        6.0,
        0.5,
        key="manual_torso"
    )

    st.sidebar.checkbox("Left pinch closed", value=True, key="manual_pinch_l")
    st.sidebar.checkbox("Right pinch closed", value=True, key="manual_pinch_r")

else:
    st.sidebar.info(
        "Auto demo runs a 60-second scripted heavy-box lift, load-sharing visualization, "
        "stance warning, and place phase. Switch to Manual to interact directly."
    )


st.title("Two-Hand + Torso Fusion Lift Controller")
st.caption(
    "Symmetric whole-body lifting with wrist-height sync, torso-lean counterbalance, "
    "load sharing between legs and arms, grip synchronization, and CoM balance checking."
)


def update_live_area():
    elapsed = time.time() - st.session_state.start_time
    mode = st.session_state.get("mode", "Auto Demo (hands-free)")

    if mode.startswith("Auto"):
        sig = demo_signals(elapsed)
    else:
        sig = manual_signals(elapsed)

    ctrl = st.session_state.sim.update(elapsed, sig)
    frame = render_frame(ctrl, st.session_state.sim)

    if st.session_state.get("record_live", True):
        now = time.time()
        if now - st.session_state.get("last_ring_time", 0.0) >= 1.0 / REC_FPS:
            ok, jpg = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 62]
            )
            if ok:
                st.session_state.frame_ring.append(jpg.tobytes())
                st.session_state.last_ring_time = now

    st.image(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        use_container_width=True
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Legs share", f"{ctrl['legs_pct'] * 100:.0f}%")
    c2.metric("Arms share", f"{ctrl['arms_pct'] * 100:.0f}%")
    c3.metric("CoM margin", f"{ctrl['com_margin'] * 100:.0f} cm")
    c4.metric("Phase", ctrl["phase"])


def safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


try:
    if hasattr(st, "fragment"):
        live_panel = st.fragment(run_every=LIVE_INTERVAL)(update_live_area)
        live_panel()
    else:
        raise AttributeError("No fragment support")
except Exception:
    update_live_area()
    time.sleep(LIVE_INTERVAL)
    safe_rerun()