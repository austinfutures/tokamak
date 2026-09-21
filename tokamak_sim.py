"""Time evolution of the 1-D profiles + equilibrium re-solve each frame."""
import numpy as np
from tokamak_physics import Tokamak, _trapz


def diffuse_cylindrical(J, r, dt, D):
    """Explicit scheme for dJ/dt = (1/r) d/dr ( r D dJ/dr ), vectorized."""
    Jn = J.copy()
    dr_p = r[2:] - r[1:-1]
    dr_m = r[1:-1] - r[:-2]
    rp = 0.5 * (r[1:-1] + r[2:])
    rm = 0.5 * (r[1:-1] + r[:-2])
    flux_p = rp * D * (J[2:] - J[1:-1]) / dr_p
    flux_m = rm * D * (J[1:-1] - J[:-2]) / dr_m
    Jn[1:-1] = J[1:-1] + dt / r[1:-1] * (flux_p - flux_m) / (0.5 * (dr_p + dr_m))
    Jn[0] = Jn[1]        # dJ/dr = 0 on axis
    Jn[-1] = 0.0         # scrape-off layer
    return Jn


def run_simulation(n_frames=120, dt=0.02):
    tok = Tokamak()
    tok.set_profiles(Ip=0.35e6, alpha_J=1.8, alpha_p=2.0, p0=8e4)
    tok.solve_equilibrium()
    tok.compute_q()

    history = []
    Ip_target = 0.85e6
    n_sub = 25
    disrupt_frame = None
    since_crash = 999

    for frame in range(n_frames):
        t = frame * dt

        # ramp plasma current and heat
        tok.Ip += (Ip_target - tok.Ip) * 0.03
        tok.p *= 1.012

        # resistive current diffusion (substepped for stability)
        for _ in range(n_sub):
            tok.J = diffuse_cylindrical(tok.J, tok.r, dt / n_sub, D=8e-4)
        Ip_now = _trapz(tok.J * 2 * np.pi * tok.r, tok.r)
        tok.J *= tok.Ip / max(Ip_now, 1e-9)

        tok.solve_equilibrium()
        tok.compute_q()

        events = tok.stability_events()
        crashed = False
        r_q1 = None
        if "sawtooth" in events:
            below = np.where(tok.q >= 1.0)[0]
            if len(below) and below[0] >= 3:
                r_q1 = float(tok.r[below[0]])
            tok.sawtooth_crash()
            tok.solve_equilibrium()
            tok.compute_q()
            events = tok.stability_events()
            crashed = True
            since_crash = 0
        else:
            since_crash += 1

        if "troyon" in events and disrupt_frame is None:
            disrupt_frame = frame
        since_disrupt = (frame - disrupt_frame) if disrupt_frame is not None else -1

        # amplitude of the m=2 mode overlay: grows as q95 -> 2 and beta_N -> 3
        s_q = float(np.clip((4.5 - tok.q95) / 2.5, 0.0, 1.0))
        s_b = float(np.clip((tok.beta_N - 1.5) / 1.5, 0.0, 1.0))
        mode_amp = 0.02 + 0.10 * s_q ** 2 + 0.10 * s_b ** 2
        if since_disrupt >= 0:
            mode_amp += 0.08 * (1 + since_disrupt)     # tearing runs away

        history.append(dict(
            t=t, frame=frame,
            R=tok.solver.R.copy(), Z=tok.solver.Z.copy(),
            psi=tok.psi.copy(),
            r=tok.r.copy(), J=tok.J.copy(), p=tok.p.copy(), q=tok.q.copy(),
            Ip=tok.Ip, q0=tok.q0, q95=tok.q95, beta_N=tok.beta_N,
            events=events, crashed=crashed, r_q1=r_q1, since_crash=since_crash,
            mode_amp=mode_amp, since_disrupt=since_disrupt,
            R0=tok.R0, a=tok.a,
        ))
        if frame % 15 == 0:
            print(f"  sim {frame}/{n_frames}  q0={tok.q0:.2f} q95={tok.q95:.2f} "
                  f"beta_N={tok.beta_N:.2f} {events or 'stable'}")
    return history