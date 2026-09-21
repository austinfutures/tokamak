"""Axisymmetric tokamak equilibrium via the Grad-Shafranov equation.

Solves   Delta* psi = -mu0 R J_phi
on a rectangular (R, Z) grid with psi = 0 on the walls, where
    Delta* = R d/dR (1/R dpsi/dR) + d2psi/dZ2

Safety factor (large-aspect-ratio, circular cross-section):
    q(r) = 2 pi r^2 B0 / (mu0 R0 Ip(r)),   Ip(r) = int_0^r J 2 pi r' dr'

Stability criteria:
    q0     < 1  -> sawtooth (Kadomtsev reconnection)
    q95    < 2  -> external kink
    beta_N > 3  -> Troyon pressure limit   (beta_N in % m T / MA)

Simplifications: the current profile is prescribed rather than solved
self-consistently from p'(psi) and FF'(psi), and the plasma cross-section
is circular.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

MU0 = 4.0 * np.pi * 1e-7
_trapz = getattr(np, "trapezoid", None) or np.trapz


class GradShafranovSolver:
    def __init__(self, R0=1.0, a=0.35, wall=0.30, nR=81, nZ=81):
        self.R0, self.a = R0, a
        self.R = np.linspace(R0 - a - wall, R0 + a + wall, nR)
        self.Z = np.linspace(-(a + wall), a + wall, nZ)
        self.nR, self.nZ = nR, nZ
        self.dR = self.R[1] - self.R[0]
        self.dZ = self.Z[1] - self.Z[0]
        self.Rg, self.Zg = np.meshgrid(self.R, self.Z, indexing="ij")
        self._build_operator()

    def _idx(self, i, j):
        return i * self.nZ + j

    def _build_operator(self):
        nR, nZ = self.nR, self.nZ
        N = nR * nZ
        rows, cols, vals = [], [], []
        dR2, dZ2 = self.dR ** 2, self.dZ ** 2

        for i in range(nR):
            for j in range(nZ):
                k = self._idx(i, j)
                if i in (0, nR - 1) or j in (0, nZ - 1):    # Dirichlet psi = 0
                    rows.append(k); cols.append(k); vals.append(1.0)
                    continue
                R = self.R[i]
                Rp, Rm = R + self.dR / 2, R - self.dR / 2
                aE = R / (dR2 * Rp)
                aW = R / (dR2 * Rm)
                aN = aS = 1.0 / dZ2
                aC = -(aE + aW + aN + aS)
                rows += [k, k, k, k, k]
                cols += [self._idx(i + 1, j), self._idx(i - 1, j),
                         self._idx(i, j + 1), self._idx(i, j - 1), k]
                vals += [aE, aW, aN, aS, aC]

        A = coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsc()
        self.lu = splu(A)          # factorize once, reuse every frame

    def solve(self, Jphi_2d):
        rhs = -MU0 * self.Rg * Jphi_2d
        rhs[0, :] = rhs[-1, :] = 0.0
        rhs[:, 0] = rhs[:, -1] = 0.0
        return self.lu.solve(rhs.ravel()).reshape(self.nR, self.nZ)


class Tokamak:
    """State: 1-D toroidal current density J(r) and pressure p(r)."""

    def __init__(self, R0=1.0, a=0.35, B0=3.0, wall=0.30):
        self.R0, self.a, self.B0 = R0, a, B0
        self.solver = GradShafranovSolver(R0, a, wall)
        self.r = np.linspace(1e-3, a, 80)
        self.J = np.zeros_like(self.r)
        self.p = np.zeros_like(self.r)
        self.Ip = 0.0
        self.psi = None
        self.q = None
        self.q0, self.q95, self.beta_N = 1.5, 3.0, 0.0

    # ---------------------------------------------------------- profiles
    def set_profiles(self, Ip, alpha_J=1.8, alpha_p=2.0, p0=8e4):
        x = self.r / self.a
        Jshape = np.clip(1 - x ** 2, 0, None) ** alpha_J
        area = _trapz(Jshape * 2 * np.pi * self.r, self.r)
        self.J = (Ip / max(area, 1e-9)) * Jshape
        self.Ip = Ip
        self.p = p0 * np.clip(1 - x ** 2, 0, None) ** alpha_p

    # ---------------------------------------------------------- physics
    def solve_equilibrium(self):
        r_cyl = np.sqrt((self.solver.Rg - self.R0) ** 2 + self.solver.Zg ** 2)
        J2D = np.interp(r_cyl, self.r, self.J, left=self.J[0], right=0.0)
        self.psi = self.solver.solve(J2D)

    def compute_q(self):
        integrand = self.J * 2 * np.pi * self.r
        Ip_r = np.concatenate(([0.0], np.cumsum(
            0.5 * (integrand[1:] + integrand[:-1]) * np.diff(self.r))))
        q = 2 * np.pi * self.r ** 2 * self.B0 / (MU0 * self.R0 * np.maximum(Ip_r, 1e-9))
        q[0] = q[1]
        self.q = q
        self.q0 = float(q[1])
        i95 = int(np.argmin(np.abs(self.r - 0.95 * self.a)))
        self.q95 = float(q[i95])

        # Troyon normalized beta: beta_N = beta[%] * a[m] * B0[T] / Ip[MA]
        p_avg = _trapz(self.p * 2 * np.pi * self.r, self.r) / (np.pi * self.a ** 2)
        beta_pct = 100.0 * p_avg / (self.B0 ** 2 / (2 * MU0))
        self.beta_N = beta_pct * self.a * self.B0 / max(self.Ip / 1e6, 1e-9)

    def stability_events(self):
        ev = []
        if self.q0 < 1.0:
            ev.append("sawtooth")
        if self.q95 < 2.0:
            ev.append("kink")
        if self.beta_N > 3.0:
            ev.append("troyon")
        return ev

    def sawtooth_crash(self):
        """Kadomtsev-style relaxation: flatten J inside the q=1 surface while
        conserving the enclosed current."""
        below = np.where(self.q >= 1.0)[0]
        if len(below) == 0:
            return
        i_q1 = int(below[0])
        if i_q1 < 3:
            return
        r_q1 = self.r[i_q1]
        Ip_in = _trapz(self.J[: i_q1 + 1] * 2 * np.pi * self.r[: i_q1 + 1],
                       self.r[: i_q1 + 1])
        self.J[: i_q1 + 1] = Ip_in / (np.pi * r_q1 ** 2)