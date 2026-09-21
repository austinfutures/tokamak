"""Renders tokamak_field.gif: poloidal flux surfaces from the GS solve,
with a rotating m=2 mode overlay (visualization only, not self-consistent MHD)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
import matplotlib.patheffects as pe
import imageio.v2 as imageio

from tokamak_sim import run_simulation

ROOT = os.path.dirname(os.path.abspath(__file__))
M_MODE = 2           # poloidal mode number of the overlay
ROT_PER_FRAME = 0.32 # radians of mode rotation per frame


def perturbed_psi(s, psi_ref):
    """psi + m=2 helical perturbation localized around the q=2-ish edge region."""
    R, Z = s["R"], s["Z"]
    R0, a = s["R0"], s["a"]
    Rg, Zg = np.meshgrid(R, Z, indexing="ij")
    rr = np.sqrt((Rg - R0) ** 2 + Zg ** 2)
    th = np.arctan2(Zg, Rg - R0)

    x = rr / a
    radial = np.exp(-((x - 0.75) / 0.28) ** 2)          # peaks near the edge
    phase = M_MODE * th - ROT_PER_FRAME * s["frame"]
    dpsi = s["mode_amp"] * psi_ref * radial * np.cos(phase)
    return np.abs(s["psi"]) + dpsi


def draw_frame(ax, s, stars, psi_ref):
    ax.clear()
    ax.set_facecolor("black")
    ax.set_aspect("equal")
    ax.axis("off")
    R, Z = s["R"], s["Z"]
    R0, a = s["R0"], s["a"]
    ax.set_xlim(R[0], R[-1])
    ax.set_ylim(Z[0], Z[-1])
    fi = s["frame"]
    sd = s["since_disrupt"]

    # starfield
    sx, sy, ss, sa, sc = stars
    ax.scatter(sx, sy, s=ss, c=sc, alpha=sa * 0.55, zorder=0, linewidths=0)

    # plasma core glow (fades during disruption)
    fade = 1.0 if sd < 0 else max(0.0, 1.0 - sd / 10.0)
    for i in range(16):
        f = i / 16
        size = (1 - f) * a * 1.15
        ax.add_patch(Ellipse((R0, 0), 2 * size, 2 * size,
                             facecolor=plt.cm.hot(0.95 - f * 0.6),
                             alpha=(0.02 + 0.13 * f ** 2) * 0.5 * fade,
                             zorder=1, edgecolor="none"))

    # flux surfaces: fixed levels against the final-frame peak so you
    # actually see the field grow as Ip ramps
    psi_p = perturbed_psi(s, psi_ref)
    levels = psi_ref * np.linspace(0.04, 0.98, 16)
    color_map = "autumn" if (s["events"] or sd >= 0) else "cool"
    lw_scale = fade if sd >= 0 else 1.0
    ax.contour(R, Z, psi_p.T, levels=levels, cmap=color_map,
               linewidths=5.5, alpha=0.10 * lw_scale + 0.02, zorder=3)
    ax.contour(R, Z, psi_p.T, levels=levels, cmap=color_map,
               linewidths=1.4, alpha=0.85 * lw_scale + 0.10, zorder=4)

    # vessel wall (flashes red on disruption)
    wall_col = "red" if sd >= 0 else "gray"
    wall_a = 0.35 if sd < 0 else 0.35 + 0.5 * abs(np.sin(0.9 * sd))
    ax.plot([R[0], R[-1], R[-1], R[0], R[0]],
            [Z[0], Z[0], Z[-1], Z[-1], Z[0]],
            color=wall_col, lw=0.8 if sd < 0 else 2.0, alpha=wall_a, zorder=2)

    # 8 coil nodes around the vessel
    coil_r = a + 0.22
    for th in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        Rc, Zc = R0 + coil_r * np.cos(th), coil_r * np.sin(th)
        for g in range(5, 0, -1):
            ax.plot(Rc, Zc, "o", color="deepskyblue",
                    markersize=3 + g * 2.5, alpha=0.08, zorder=9)
        ax.plot(Rc, Zc, "o", color="white", markersize=3.5, zorder=10)
        fl = 0.04
        ax.plot([Rc - fl, Rc + fl], [Zc, Zc], color="cyan", lw=0.7, alpha=0.6, zorder=10)
        ax.plot([Rc, Rc], [Zc - fl, Zc + fl], color="cyan", lw=0.7, alpha=0.6, zorder=10)

    # sawtooth: expanding shockwave ring from the q=1 surface
    if s["since_crash"] < 8 and s["r_q1"] is not None:
        k = s["since_crash"]
        ax.add_patch(Circle((R0, 0), s["r_q1"] + 0.03 * k, facecolor="none",
                            edgecolor="yellow", lw=2.5 * (1 - k / 8) + 0.3,
                            alpha=0.7 * (1 - k / 8), zorder=12))

    # disruption sparks
    if sd >= 0:
        rng = np.random.default_rng(fi)
        n = 40
        ang = rng.uniform(0, 2 * np.pi, n)
        rad = rng.uniform(0.1, 0.7, n)
        for a_, r_ in zip(ang, rad):
            Rc, Zc = R0 + r_ * np.cos(a_), r_ * np.sin(a_)
            ln = 0.05
            ax.plot([Rc - ln, Rc + ln], [Zc - ln * 0.5, Zc + ln * 0.5],
                    color="red", lw=0.9, alpha=0.6, zorder=11)

    # HUD
    ax.text(0.03, 0.97, "TOKAMAK EQUILIBRIUM", transform=ax.transAxes,
            color="white", fontsize=13, fontweight="bold", va="top",
            family="monospace",
            path_effects=[pe.withStroke(linewidth=4, foreground="deepskyblue")])
    ax.text(0.03, 0.93, f"Ip = {s['Ip'] / 1e6:.2f} MA    t = {s['t']:.2f} s",
            transform=ax.transAxes, color="lightcyan", fontsize=9,
            va="top", family="monospace")
    ax.text(0.03, 0.90,
            f"q0 = {s['q0']:.2f}   q95 = {s['q95']:.2f}   beta_N = {s['beta_N']:.2f}",
            transform=ax.transAxes, color="lightcyan", fontsize=9,
            va="top", family="monospace")
    ax.text(0.03, 0.87, f"m=2 mode: overlay  amp = {s['mode_amp']:.2f}",
            transform=ax.transAxes, color="lightsteelblue", fontsize=8,
            va="top", family="monospace")

    if sd >= 0:
        status, status_col = "DISRUPTION  (TROYON LIMIT)", "red"
    elif "sawtooth" in s["events"]:
        status, status_col = "SAWTOOTH CRASH", "yellow"
    elif "kink" in s["events"]:
        status, status_col = "EXTERNAL KINK", "red"
    elif s["q95"] < 2.4 or s["beta_N"] > 2.4:
        status, status_col = "MARGINAL", "orange"
    else:
        status, status_col = "STABLE", "lime"

    ax.text(0.97, 0.97, f">> {status}", transform=ax.transAxes,
            color=status_col, fontsize=12, fontweight="bold", va="top", ha="right",
            family="monospace",
            path_effects=[pe.withStroke(linewidth=3, foreground="black")])
    ax.text(0.5, 0.02,
            "poloidal cross-section  |  Grad-Shafranov equilibrium",
            transform=ax.transAxes, color="gray", fontsize=8,
            ha="center", family="monospace")


def main():
    print("running simulation...")
    history = run_simulation(n_frames=120, dt=0.02)

    # fixed reference so surfaces visibly strengthen as Ip ramps
    psi_ref = max(np.abs(h["psi"]).max() for h in history)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=90, facecolor="black")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    R, Z = history[0]["R"], history[0]["Z"]
    rng = np.random.default_rng(7)
    stars = (rng.uniform(R[0], R[-1], 350),
             rng.uniform(Z[0], Z[-1], 350),
             rng.uniform(0.2, 2.0, 350),
             rng.uniform(0.05, 0.85, 350),
             rng.choice(["white", "lightblue", "lightyellow"], 350))

    print("rendering frames...")
    frames = []
    for i, s in enumerate(history):
        draw_frame(ax, s, stars, psi_ref)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        if i % 20 == 0:
            print(f"  {i}/{len(history)}")

    out = os.path.join(ROOT, "tokamak_field.gif")
    imageio.mimsave(out, frames, duration=0.08, loop=0)
    print("-> wrote", out)


if __name__ == "__main__":
    main()