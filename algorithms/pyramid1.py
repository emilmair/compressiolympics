"""
Self-Taught Pyramid (STP) -- a decoder-side-learned image codec.

Core idea
---------
Nothing about the predictors is ever transmitted.  Both encoder and decoder
build a decimation pyramid of the image and decode it coarse -> fine.  Before
each finer level is decoded, *both sides* fit (by ridge-regularised least
squares) a bank of 32 class-conditional linear interpolators on the image's
OWN already-decoded coarser scales ("zero-shot super-resolution as a
predictor"), under the assumption that natural images are roughly scale
invariant.  The learned interpolators predict the next finer level; only the
quantised innovation is coded.

  * Pyramid           : pure 2x decimation (critically sampled, no overhead).
  * Level refinement  : 3 phases per level (diagonal / horizontal / vertical
                        samples), i.e. square lattice -> quincunx -> full grid.
  * Predictors        : 16 (20 for the last phase) neighbour features, centred
                        on the local mean, classified by a 1-bit-ADRC pattern
                        x activity, trained on the decoded image itself at the
                        previous two scales, shrunk towards a pooled regressor.
  * Closed loop       : prediction is always made from *reconstructed* data.
  * Quantiser         : dead-zone uniform, step shrinks slightly at coarse
                        levels (they are references for many samples).
  * Colour            : per-image KLT (9 int16 in the header).
  * Entropy coder     : 48-bit binary range coder, unary/Exp-Golomb binarisation,
                        4 context models mixed with a logistic mixer.  The
                        contexts use information the decoder gets for free:
                        local prediction-neighbourhood activity, residual
                        magnitudes of already-decoded neighbours (same level,
                        coarser level, other colour planes).

``compression_amount`` in (0, 1]: higher = more compression (larger step).
Set HIGHER_IS_MORE_COMPRESSION = False to invert the meaning.
"""
from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np
from PIL import Image

# ----------------------------------------------------------------------------
# Tunables
# ----------------------------------------------------------------------------
HIGHER_IS_MORE_COMPRESSION = True
DELTA_MIN = 1.0          # quantiser step at the "quality" end
DELTA_MAX = 96.0         # quantiser step at the "compression" end
LEVEL_GAMMA = 0.62       # step multiplier per coarser level
LEVEL_MIN_FACTOR = 0.12
THETA = 0.15             # dead-zone widening at the finest level
THETA_COARSE = 0.0       # ... and at coarser levels
RHO = 0.0                # reconstruction offset for non-zero bins (in steps)
NMAX = 150_000           # max training samples per regression
PRIOR_G = 4.0            # pseudo-samples of shrinkage to bilinear (global fit)
PRIOR_C = 40.0           # pseudo-samples of shrinkage to global (class fits)
PREV_W = 1.0             # training weight of level l+1
PREV2_W = 0.4            # training weight of level l+2
SAME_LEVEL_W = 1.0       # training weight of rotated same-level data (phase B)
RDO_COARSE_MULT = 0.5    # lambda multiplier for levels > 0
RDO_LAMBDA = 0.07        # Lagrangian (step^2 per bit) for RDO quantisation; 0 = off
FILT_ON = True
FILT_LAMBDA = 0.08       # D-vs-bytes slope for enabling a filter
FILT_T1, FILT_T2 = 0.25, 0.8  # activity thresholds (in steps) separating filter classes
STOP_SIZE = 12           # stop decimating when min(h, w) <= STOP_SIZE

MAGIC = 0xB7
PAD = 3
NB = 8                   # distinct "bit index" slots in context models
MAXU = 20                # unary limit before Exp-Golomb escape
LR = 0.015               # mixer learning rate
CLIM = 60                # counter adaptation limit

# ----------------------------------------------------------------------------
# Range coder (48-bit, LZMA-style carry handling, 16-bit probabilities)
# ----------------------------------------------------------------------------
_TOP = 1 << 40
_MASK40 = (1 << 40) - 1


class _Enc:
    def __init__(self):
        self.low = 0
        self.rng = (1 << 48) - 1
        self.cache = 0
        self.cache_size = 1
        self.out = bytearray()

    def code(self, bit, p0):
        bound = (self.rng >> 16) * p0
        if bit:
            self.low += bound
            self.rng -= bound
        else:
            self.rng = bound
        while self.rng < _TOP:
            self.rng <<= 8
            self._shift()
        return bit

    def _shift(self):
        low = self.low
        if low < 0xFF0000000000 or low >= (1 << 48):
            carry = low >> 48
            temp = self.cache
            while True:
                self.out.append((temp + carry) & 0xFF)
                temp = 0xFF
                self.cache_size -= 1
                if self.cache_size == 0:
                    break
            self.cache = (low >> 40) & 0xFF
        self.cache_size += 1
        self.low = (low & _MASK40) << 8

    def finish(self):
        # pick the value inside [low, low+rng) with the most trailing zeros
        for k in range(48, -1, -1):
            v = ((self.low + (1 << k) - 1) >> k) << k
            if v < self.low + self.rng:
                self.low = v
                break
        for _ in range(7):
            self._shift()
        data = bytes(self.out[1:])          # first byte is always 0
        return data.rstrip(b"\x00")


class _Dec:
    def __init__(self, data, pos=0):
        self.data = data
        self.pos = pos
        self.n = len(data)
        self.rng = (1 << 48) - 1
        code = 0
        for _ in range(6):
            code = (code << 8) | self._next()
        self.code_ = code

    def _next(self):
        p = self.pos
        self.pos = p + 1
        return self.data[p] if p < self.n else 0

    def code(self, _bit, p0):
        bound = (self.rng >> 16) * p0
        if self.code_ < bound:
            self.rng = bound
            bit = 0
        else:
            self.code_ -= bound
            self.rng -= bound
            bit = 1
        while self.rng < _TOP:
            self.rng <<= 8
            self.code_ = (self.code_ << 8) | self._next()
        return bit


# ----------------------------------------------------------------------------
# Context-mixing residual model (shared by encoder and decoder)
# ----------------------------------------------------------------------------
_N1 = 8 * 7 * 4
_N2 = 8 * 7 * 4
_N3 = 8 * 4 * 4 * 4
_N4 = 9 * 8 * 4 * 4
_O2 = _N1 * NB
_O3 = _O2 + _N2 * NB
_O4 = _O3 + _N3 * NB
_NTOT = _O4 + _N4 * NB
_RATE = [1.0 / (n + 1.6) for n in range(260)]
_QM = [0, 1, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6] + [7] * 48
_PMIN, _PMAX = 1e-4, 1.0 - 1e-4
_exp, _log = math.exp, math.log
_INV_LN2 = 1.0 / math.log(2.0)


class _Model:
    def __init__(self, coder):
        self.coder = coder
        self.P = [0.5] * _NTOT
        self.C = [0] * _NTOT
        self.W = [0.3, 0.3, 0.3, 0.3, 0.0] * (NB * 4)
        self.S = [0.5] * 9
        self.SC = [0] * 9

    def _bit(self, b, i1, i2, i3, i4, wi):
        P = self.P
        C = self.C
        W = self.W
        p1 = P[i1]
        p2 = P[i2]
        p3 = P[i3]
        p4 = P[i4]
        s1 = _log(p1 / (1.0 - p1))
        s2 = _log(p2 / (1.0 - p2))
        s3 = _log(p3 / (1.0 - p3))
        s4 = _log(p4 / (1.0 - p4))
        j = wi * 5
        dot = W[j] * s1 + W[j + 1] * s2 + W[j + 2] * s3 + W[j + 3] * s4 + W[j + 4] * 0.5
        if dot > 11.0:
            dot = 11.0
        elif dot < -11.0:
            dot = -11.0
        pm = 1.0 / (1.0 + _exp(-dot))
        p0 = int((1.0 - pm) * 65536.0)
        if p0 < 8:
            p0 = 8
        elif p0 > 65528:
            p0 = 65528
        bit = self.coder.code(b, p0)
        e = (bit - pm) * LR
        W[j] += e * s1
        W[j + 1] += e * s2
        W[j + 2] += e * s3
        W[j + 3] += e * s4
        W[j + 4] += e * 0.5
        n = C[i1]
        v = p1 + (bit - p1) * _RATE[n]
        P[i1] = _PMIN if v < _PMIN else (_PMAX if v > _PMAX else v)
        if n < CLIM:
            C[i1] = n + 1
        n = C[i2]
        v = p2 + (bit - p2) * _RATE[n]
        P[i2] = _PMIN if v < _PMIN else (_PMAX if v > _PMAX else v)
        if n < CLIM:
            C[i2] = n + 1
        n = C[i3]
        v = p3 + (bit - p3) * _RATE[n]
        P[i3] = _PMIN if v < _PMIN else (_PMAX if v > _PMAX else v)
        if n < CLIM:
            C[i3] = n + 1
        n = C[i4]
        v = p4 + (bit - p4) * _RATE[n]
        P[i4] = _PMIN if v < _PMIN else (_PMAX if v > _PMAX else v)
        if n < CLIM:
            C[i4] = n + 1
        return bit

    def _p(self, i1, i2, i3, i4, wi):
        P = self.P
        W = self.W
        p1 = P[i1]
        p2 = P[i2]
        p3 = P[i3]
        p4 = P[i4]
        j = wi * 5
        dot = (W[j] * _log(p1 / (1.0 - p1)) + W[j + 1] * _log(p2 / (1.0 - p2))
               + W[j + 2] * _log(p3 / (1.0 - p3)) + W[j + 3] * _log(p4 / (1.0 - p4)) + W[j + 4] * 0.5)
        if dot > 11.0:
            dot = 11.0
        elif dot < -11.0:
            dot = -11.0
        pm = 1.0 / (1.0 + _exp(-dot))
        return 1e-5 if pm < 1e-5 else (1.0 - 1e-5 if pm > 1.0 - 1e-5 else pm)

    def rdo(self, m, a, b1, b2, b3, b4, pl, lam):
        """Choose between magnitude m (nearest) and m-1 by D + lam*R (units: step^2)."""
        b2 += _O2
        b3 += _O3
        b4 += _O4
        pp = self._p
        if m == 1:
            pnz = pp(b1, b2, b3, b4, pl)
            pm1 = pp(b1 + 1, b2 + 1, b3 + 1, b4 + 1, 4 + pl)
            dbits = -_log(pnz) * _INV_LN2 + 1.0 - _log(1.0 - pm1) * _INV_LN2 + _log(1.0 - pnz) * _INV_LN2
            dd = (a - 1.0 - RHO) ** 2 - a * a
            return 0 if dd + lam * dbits > 0.0 else 1
        k = m - 1
        kb = k if k < NB else NB - 1
        pk = pp(b1 + kb, b2 + kb, b3 + kb, b4 + kb, kb * 4 + pl)
        kb = m if m < NB else NB - 1
        pm_ = pp(b1 + kb, b2 + kb, b3 + kb, b4 + kb, kb * 4 + pl)
        dbits = (-_log(pk) - _log(1.0 - pm_) + _log(1.0 - pk)) * _INV_LN2
        dd = (a - m - RHO) ** 2 - (a - k - RHO) ** 2
        return k if dd + lam * dbits > 0.0 else m

    def sym(self, q, b1, b2, b3, b4, pl, sctx):
        """Code one signed integer.  q is None when decoding."""
        enc = q is not None
        m = abs(q) if enc else 0
        bit = self._bit
        b2 += _O2
        b3 += _O3
        b4 += _O4
        if not bit((1 if m else 0), b1, b2, b3, b4, pl):
            return 0
        # sign
        ps = self.S[sctx]
        p0 = int((1.0 - ps) * 65536.0)
        p0 = 64 if p0 < 64 else (65472 if p0 > 65472 else p0)
        sgn = self.coder.code((1 if (enc and q < 0) else 0), p0)
        n = self.SC[sctx]
        self.S[sctx] = ps + (sgn - ps) * _RATE[n]
        if n < 250:
            self.SC[sctx] = n + 1
        # magnitude (unary with escape)
        k = 1
        while k <= MAXU:
            kb = k if k < NB else NB - 1
            if not bit((1 if m > k else 0), b1 + kb, b2 + kb, b3 + kb, b4 + kb, kb * 4 + pl):
                break
            k += 1
        else:
            e = m - MAXU - 1
            v_in = e + 1
            nb = v_in.bit_length() - 1 if enc else 0
            n = 0
            coder = self.coder
            while coder.code((1 if (enc and n < nb) else 0), 32768):
                n += 1
            v = 1
            for i in range(n - 1, -1, -1):
                v = (v << 1) | coder.code(((v_in >> i) & 1) if enc else 0, 32768)
            k = MAXU + v          # v = e + 1  ->  m = MAXU + 1 + e
        return -k if sgn else k


# ----------------------------------------------------------------------------
# Geometry of the three refinement phases
# ----------------------------------------------------------------------------
_A_OFFS = [(dy, dx) for dy in (-3, -1, 1, 3) for dx in (-3, -1, 1, 3)]
_A_CEN = [_A_OFFS.index(o) for o in ((-1, -1), (-1, 1), (1, -1), (1, 1))]
_BC_OFFS = [(-1, 0), (0, -1), (0, 1), (1, 0),
            (-1, -2), (-1, 2), (1, -2), (1, 2), (-2, -1), (-2, 1), (2, -1), (2, 1),
            (-3, 0), (3, 0), (0, -3), (0, 3)]
_C_OFFS = _BC_OFFS + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
_PHASES = [  # offsets, centre-feature indices, (r0, c0)
    (_A_OFFS, _A_CEN, (1, 1)),
    (_BC_OFFS, [0, 1, 2, 3], (0, 1)),
    (_C_OFFS, [0, 1, 2, 3], (1, 0)),
]
_NCLS = 32
_EDGES_A = np.array([0.15, 0.35, 0.7, 1.2, 2.0, 3.5, 6.0])
_EDGES_S = np.array([1, 2, 3, 5, 9, 17])
_EDGES_W = np.array([0.3, 0.9, 1.8, 3.0, 5.0, 8.0, 13.0, 22.0])
_EDGES_AC = np.array([0.15, 0.35, 0.7, 1.2, 2.0, 3.5, 6.0])


def _classify(xs, cidx, thr):
    c0, c1, c2, c3 = (xs[i] for i in cidx)
    m = (c0 + c1 + c2 + c3) * 0.25
    pat = ((c0 > m).astype(np.int64) | ((c1 > m).astype(np.int64) << 1)
           | ((c2 > m).astype(np.int64) << 2) | ((c3 > m).astype(np.int64) << 3))
    rng = np.maximum(np.maximum(c0, c1), np.maximum(c2, c3)) - \
        np.minimum(np.minimum(c0, c1), np.minimum(c2, c3))
    return m, pat * 2 + (rng >= thr)


def _quant(r, delta, theta):
    q = np.floor(np.abs(r) / delta + (0.5 - theta)).astype(np.int64)
    return np.where(r < 0, -q, q)


def _deq(q, delta):
    a = np.abs(q)
    v = np.where(a > 0, a + RHO, 0.0) * delta
    return np.where(q < 0, -v, v)


# ----------------------------------------------------------------------------
# In-loop restoration filter (encoder fits, decoder applies)
# ----------------------------------------------------------------------------
_FOFFS = [(0, 1), (1, 0), (1, 1), (1, -1), (0, 2), (2, 0)]
_NF = len(_FOFFS)


def _filter_feats(R, delta):
    H, W = R.shape
    Rp = np.pad(R, 2, mode="reflect")
    c = Rp[2:2 + H, 2:2 + W]
    F = np.stack([Rp[2 + dy:2 + dy + H, 2 + dx:2 + dx + W] + Rp[2 - dy:2 - dy + H, 2 - dx:2 - dx + W] - 2.0 * c
                  for dy, dx in _FOFFS], axis=-1)
    act = (np.abs(Rp[2:2 + H, 3:3 + W] - c) + np.abs(Rp[2:2 + H, 1:1 + W] - c)
           + np.abs(Rp[3:3 + H, 2:2 + W] - c) + np.abs(Rp[1:1 + H, 2:2 + W] - c)) * 0.25
    return F, act


def _filter_classes(act, delta, ncls):
    if ncls == 1:
        return np.zeros(act.shape, dtype=np.int64)
    return np.searchsorted(np.array([FILT_T1 * delta, FILT_T2 * delta]), act, side="right")


def _apply_filter(R, delta, w):
    ncls = w.shape[0]
    F, act = _filter_feats(R, delta)
    cls = _filter_classes(act, delta, ncls)
    wf = w.astype(np.float64) / 128.0
    return R + np.einsum("ijf,ijf->ij", F, wf[cls])


def _fit_filter(R, O, delta, ncls, delta0):
    F, act = _filter_feats(R, delta)
    cls = _filter_classes(act, delta, ncls)
    t = O - R
    w = np.zeros((ncls, _NF), dtype=np.int8)
    for k in range(ncls):
        m = cls == k
        if m.sum() < 64:
            continue
        X = F[m]
        y = t[m]
        sol = np.linalg.solve(X.T @ X + 1e-6 * np.eye(_NF), X.T @ y)
        w[k] = np.clip(np.round(sol * 128.0), -127, 127).astype(np.int8)
    out = R + np.einsum("ijf,ijf->ij", F, (w.astype(np.float64) / 128.0)[cls])
    gain = float(np.sum(t * t) - np.sum((O - out) ** 2))
    cost = FILT_LAMBDA * delta0 * delta0 * 8.0 * (ncls * _NF + 0.25)
    return (w if gain > cost else None)

# ----------------------------------------------------------------------------
# The codec proper (same code path for encode and decode)
# ----------------------------------------------------------------------------
class _Codec:
    def __init__(self, coder, enc, shape, nplanes, delta, planes=None, filt_in=None):
        self.enc = enc
        self.filt_in = filt_in or {}
        self.filt_out = {}
        self.model = _Model(coder)
        self.np_ = nplanes
        shapes = [tuple(shape)]
        while min(shapes[-1]) > STOP_SIZE:
            shapes.append(((shapes[-1][0] + 1) // 2, (shapes[-1][1] + 1) // 2))
        self.shapes = shapes
        self.L = len(shapes) - 1
        if enc:
            self.orig = [planes]
            for _ in range(self.L):
                self.orig.append([p[::2, ::2] for p in self.orig[-1]])
        self.recon = [None] * (self.L + 1)
        self.qabs = [None] * (self.L + 1)
        self.deltas = [delta * max(LEVEL_MIN_FACTOR, LEVEL_GAMMA ** l) for l in range(self.L + 1)]
        self.thetas = [THETA if l == 0 else THETA_COARSE for l in range(self.L + 1)]

    # ---- coarsest level: raster scan with MED prediction ---------------------
    def _coarsest(self):
        L = self.L
        h, w = self.shapes[L]
        d = self.deltas[L]
        th = 0.0 if RDO_LAMBDA > 0 else self.thetas[L]
        model = self.model
        sym = model.sym
        recs, qas = [], []
        for c in range(self.np_):
            R = [[0.0] * w for _ in range(h)]
            QA = [[0] * w for _ in range(h)]
            pl = min(c, 3)
            if self.enc:
                O = self.orig[L][c].tolist()
            for y in range(h):
                Ry = R[y]
                Rn = R[y - 1] if y else None
                for x in range(w):
                    a = Ry[x - 1] if x else (Rn[x] if y else 0.0)
                    b = Rn[x] if y else a
                    cc = Rn[x - 1] if (x and y) else (b if x == 0 else a)
                    if cc >= max(a, b):
                        p = min(a, b)
                    elif cc <= min(a, b):
                        p = max(a, b)
                    else:
                        p = a + b - cc
                    p = round(p * 16.0) / 16.0
                    act = (abs(a - cc) + abs(b - cc)) / d
                    ca = int(np.searchsorted(_EDGES_A, act, side="right"))
                    cp = 0
                    for cc2 in range(c):
                        cp += qas[cc2][y][x]
                    cp = cp if cp < 3 else 3
                    lq = QA[y][x - 1] if x else 0
                    uq = QA[y - 1][x] if y else 0
                    ql = lq + uq
                    cl = _QM[(2 * ql) >> 1 if ql < 64 else 63]
                    b1 = ((ca * 7) * 4 + pl) * NB
                    b2 = ((cl * 7) * 4 + pl) * NB
                    b3 = (((ca * 4 + cp) * 4 + 3) * 4 + pl) * NB
                    b4 = ((((0 * 8) * 8 + cl) * 4 + 3) * 4 + pl) * NB
                    if self.enc:
                        r = O[y][x] - p
                        av = abs(r) / d
                        qi = int(math.floor(av + (0.5 - th)))
                        if RDO_LAMBDA > 0 and qi > 0:
                            qi = model.rdo(qi, av, b1, b2, b3, b4, pl, RDO_LAMBDA * RDO_COARSE_MULT)
                        qi = -qi if r < 0 else qi
                    else:
                        qi = None
                    q = sym(qi, b1, b2, b3, b4, pl, 0)
                    aq = abs(q)
                    v = (aq + RHO) * d if aq else 0.0
                    Ry[x] = p + (-v if q < 0 else v)
                    QA[y][x] = aq
            recs.append(np.array(R))
            qas.append(QA)
        self.recon[L] = recs
        self.qabs[L] = [np.array(q, dtype=np.int64) for q in qas]

    # ---- learned predictors ---------------------------------------------------
    def _train_sources(self, l, ph, c, d):
        """Yield (X, y, cls, weight) training sets built from decoded data."""
        offs, cidx, _ = _PHASES[ph]
        # (a) same-level, rotated: the known quincunx of level l is a 45-degree
        #     rotated, sqrt(2)-scaled copy of a square lattice, so it can teach
        #     the phase-B interpolator at a scale much closer than level l+1.
        if ph == 1 and SAME_LEVEL_W > 0:
            R = self.recon[l][c]
            h, w = R.shape
            if h > 2 * PAD + 2 and w > 2 * PAD + 2:
                rot = [(dy + dx, dy - dx) for dy, dx in offs]
                par = (np.add.outer(np.arange(PAD, h - PAD), np.arange(PAD, w - PAD)) % 2 == 0).ravel()
                idx = np.nonzero(par)[0]
                step = max(1, int(math.ceil(len(idx) / NMAX)))
                idx = idx[::step]
                xs = [R[PAD + dy:h - PAD + dy, PAD + dx:w - PAD + dx].ravel()[idx] for dy, dx in rot]
                y = R[PAD:h - PAD, PAD:w - PAD].ravel()[idx]
                m, cls = _classify(xs, cidx, 1.5 * d)
                yield (np.stack(xs, axis=1) - m[:, None]), y - m, cls, SAME_LEVEL_W
        # (b) previous scales
        for lev, wt in ((l + 1, PREV_W), (l + 2, PREV2_W)):
            if lev > self.L:
                continue
            R1 = self.recon[lev][c]
            h, w = R1.shape
            if h <= 2 * PAD + 2 or w <= 2 * PAD + 2:
                continue
            n = (h - 2 * PAD) * (w - 2 * PAD)
            s = max(1, int(math.ceil(math.sqrt(n / NMAX))))
            xs = [R1[PAD + dy:h - PAD + dy:s, PAD + dx:w - PAD + dx:s].ravel() for dy, dx in offs]
            y = R1[PAD:h - PAD:s, PAD:w - PAD:s].ravel()
            m, cls = _classify(xs, cidx, 1.5 * d)
            yield (np.stack(xs, axis=1) - m[:, None]), y - m, cls, wt

    def _weights(self, l, ph, c, d):
        offs, cidx, _ = _PHASES[ph]
        F = len(offs)
        Xs, ys, cls_l = [], [], []
        for X, y, cls, wt in self._train_sources(l, ph, c, d):
            sw = math.sqrt(wt)
            Xs.append(X * sw)
            ys.append(y * sw)
            cls_l.append(cls)
        Wt = np.zeros((_NCLS, F))
        if not Xs:
            return Wt
        X = np.concatenate(Xs)
        y = np.concatenate(ys)
        cls = np.concatenate(cls_l)
        N = len(y)
        if N < 24:
            return Wt
        XtX = X.T @ X
        sigma2 = max(np.trace(XtX) / (N * F), 1e-6)
        I = np.eye(F)
        wg = np.linalg.solve(XtX + PRIOR_G * sigma2 * I, X.T @ y)
        Wt[:] = wg
        lam = PRIOR_C * sigma2
        for k in range(_NCLS):
            idx = cls == k
            if idx.sum() < 4:
                continue
            Xk = X[idx]
            Wt[k] = np.linalg.solve(Xk.T @ Xk + lam * I, Xk.T @ y[idx] + lam * wg)
        return Wt

    # ---- one refinement phase of one plane -----------------------------------
    def _phase(self, l, ph, c):
        H, W = self.shapes[l]
        d = self.deltas[l]
        offs, cidx, (r0, c0) = _PHASES[ph]
        R = self.recon[l][c]
        Rp = np.pad(R, PAD, mode="reflect")
        xs = [Rp[PAD + r0 + dy:PAD + H + dy:2, PAD + c0 + dx:PAD + W + dx:2] for dy, dx in offs]
        nr, nc = xs[0].shape
        m, cls = _classify(xs, cidx, 1.5 * d)
        Wt = self._weights(l, ph, c, d)
        Xc = np.stack(xs) - m
        pred = m + np.einsum("fij,ijf->ij", Xc, Wt[cls])
        pred = np.rint(pred * 16.0) / 16.0

        a = np.abs(Xc).mean(axis=0) / d
        ca = np.searchsorted(_EDGES_A, a, side="right")
        Qp = np.pad(self.qabs[l][c], PAD, mode="reflect")
        S = 0
        for i in cidx:
            dy, dx = offs[i]
            S = S + Qp[PAD + r0 + dy:PAD + H + dy:2, PAD + c0 + dx:PAD + W + dx:2]
        cs = np.searchsorted(_EDGES_S, S, side="right")
        SW = 0
        for dy, dx in offs:
            SW = SW + (4.0 / (1.0 + dy * dy + dx * dx)) * \
                Qp[PAD + r0 + dy:PAD + H + dy:2, PAD + c0 + dx:PAD + W + dx:2]
        cw = np.searchsorted(_EDGES_W, SW, side="right")
        cpv = np.zeros((nr, nc), dtype=np.int64)
        for cc in range(c):
            cpv += self.qabs[l][cc][r0::2, c0::2]
        cp = np.minimum(cpv, 3)

        if self.enc:
            r = self.orig[l][c][r0::2, c0::2] - pred
            th = 0.0 if RDO_LAMBDA > 0 else self.thetas[l]
            qin = _quant(r, d, th).ravel().tolist()
            av = (np.abs(r) / d).ravel().tolist() if RDO_LAMBDA > 0 else None
        else:
            qin = None
            av = None
        qlist = self._symbols(qin, av, ca.ravel().tolist(), cs.ravel().tolist(),
                              cp.ravel().tolist(), cw.ravel().tolist(), nc, c, ph, l)
        q = np.array(qlist, dtype=np.int64).reshape(nr, nc)
        R[r0::2, c0::2] = pred + _deq(q, d)
        self.qabs[l][c][r0::2, c0::2] = np.abs(q)

    def _symbols(self, qin, av, ca, cs, cp, cw, nc, c, ph, l):
        sym = self.model.sym
        rdo = self.model.rdo
        lam = RDO_LAMBDA * (1.0 if l == 0 else RDO_COARSE_MULT)
        n = len(ca)
        ab = [0] * n
        sg = [0] * n
        out = [0] * n
        pl = min(c, 3)
        lv = min(l, 7)
        col = 0
        enc = qin is not None
        for k in range(n):
            if col:
                lq = ab[k - 1]
                ls = sg[k - 1]
            else:
                lq = 0
                ls = 0
            if k >= nc:
                uq = ab[k - nc]
                us = sg[k - nc]
            else:
                uq = 0
                us = 0
            ul = ab[k - nc - 1] if (col and k > nc) else 0
            ur = ab[k - nc + 1] if (col < nc - 1 and k >= nc) else 0
            ll = ab[k - 2] if col > 1 else 0
            mm = (2 * (lq + uq) + ul + ur + ll + 1) >> 1
            cl = _QM[mm if mm < 64 else 63]
            s_ = ca[k]
            t_ = cs[k]
            b1 = ((s_ * 7 + t_) * 4 + pl) * NB
            b2 = ((cl * 7 + t_) * 4 + pl) * NB
            b3 = (((s_ * 4 + cp[k]) * 4 + ph) * 4 + pl) * NB
            b4 = (((cw[k] * 8 + cl) * 4 + ph) * 4 + pl) * NB
            if enc:
                qk = qin[k]
                if lam > 0.0 and qk:
                    if qk > 0:
                        qk = rdo(qk, av[k], b1, b2, b3, b4, pl, lam)
                    else:
                        qk = -rdo(-qk, av[k], b1, b2, b3, b4, pl, lam)
                q = sym(qk, b1, b2, b3, b4, pl, ls * 3 + us)
            else:
                q = sym(None, b1, b2, b3, b4, pl, ls * 3 + us)
            out[k] = q
            if q:
                ab[k] = q if q > 0 else -q
                sg[k] = 1 if q > 0 else 2
            col += 1
            if col == nc:
                col = 0
        return out

    def _filter_level(self, l):
        if not FILT_ON:
            return
        d = self.deltas[l]
        for c in range(self.np_):
            ncls = 3 if c == 0 else 1
            if self.enc:
                w = _fit_filter(self.recon[l][c], self.orig[l][c], d, ncls, self.deltas[0])
                self.filt_out[(l, c)] = w
            else:
                w = self.filt_in.get((l, c))
            if w is not None:
                self.recon[l][c] = _apply_filter(self.recon[l][c], d, w)

    def run(self):
        self._coarsest()
        for l in range(self.L - 1, -1, -1):
            H, W = self.shapes[l]
            rec = [np.zeros((H, W)) for _ in range(self.np_)]
            qa = [np.zeros((H, W), dtype=np.int64) for _ in range(self.np_)]
            for c in range(self.np_):
                rec[c][::2, ::2] = self.recon[l + 1][c]
                qa[c][::2, ::2] = self.qabs[l + 1][c]
            self.recon[l] = rec
            self.qabs[l] = qa
            for ph in range(3):
                for c in range(self.np_):
                    self._phase(l, ph, c)
            self._filter_level(l)
            # free memory of levels no longer needed for training
            if l + 3 <= self.L:
                self.recon[l + 3] = None
        return self.recon[0]


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def _pack_filters(filt, L, nplanes):
    out = bytearray()
    for l in range(L):
        mask = 0
        for c in range(nplanes):
            if filt.get((l, c)) is not None:
                mask |= 1 << c
        out.append(mask)
        for c in range(nplanes):
            w = filt.get((l, c))
            if w is not None:
                out += w.astype(np.int8).tobytes()
    return bytes(out)


def _unpack_filters(data, pos, L, nplanes):
    filt = {}
    for l in range(L):
        mask = data[pos]
        pos += 1
        for c in range(nplanes):
            if mask >> c & 1:
                ncls = 3 if c == 0 else 1
                n = ncls * _NF
                filt[(l, c)] = np.frombuffer(data[pos:pos + n], dtype=np.int8).reshape(ncls, _NF).copy()
                pos += n
    return filt, pos

_LAST = None  # debugging hook: last encoder-side reconstruction
_HDR = struct.Struct("<BIIBBf9h")


def _load(image_path):
    im = Image.open(image_path)
    has_alpha = im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info
    im = im.convert("RGBA" if has_alpha else "RGB")
    return np.asarray(im, dtype=np.float64), has_alpha


def compress(image_path: str | Path, output_path: str | Path, compression_amount: float) -> None:
    arr, has_alpha = _load(image_path)
    H, W = arr.shape[:2]
    amt = float(min(max(compression_amount, 1e-6), 1.0))
    if not HIGHER_IS_MORE_COMPRESSION:
        amt = 1.0 - amt
    delta = DELTA_MIN * (DELTA_MAX / DELTA_MIN) ** amt
    delta = float(np.float32(delta))

    rgb = arr[..., :3] - 128.0
    flat = rgb.reshape(-1, 3)
    cov = np.cov(flat.T) if flat.shape[0] > 1 else np.eye(3)
    cov = np.nan_to_num(cov) + 1e-9 * np.eye(3)
    ev, evec = np.linalg.eigh(cov)
    M = evec[:, ::-1].T                              # rows = principal axes
    for i in range(3):
        if M[i].sum() < 0:
            M[i] = -M[i]
    Mq = np.round(M * 32767.0).astype(np.int16)
    Mf = Mq.astype(np.float64) / 32767.0
    planes = [(flat @ Mf.T)[:, i].reshape(H, W) for i in range(3)]

    mode, aconst = 0, 255
    if has_alpha:
        alpha = arr[..., 3]
        if alpha.min() == alpha.max():
            mode, aconst = 1, int(alpha.flat[0])
        else:
            mode = 2
            planes.append(alpha - 128.0)

    enc = _Enc()
    global _LAST
    codec = _Codec(enc, True, (H, W), len(planes), delta, planes)
    _LAST = (codec.run(), Mf, mode, aconst)
    body = enc.finish()
    hdr = _HDR.pack(MAGIC, W, H, mode, aconst, delta, *Mq.ravel().tolist())
    fb = _pack_filters(codec.filt_out, codec.L, len(planes)) if FILT_ON else b""
    Path(output_path).write_bytes(hdr + fb + body)


def decompress(compressed_path: str | Path, output_path: str | Path) -> None:
    data = Path(compressed_path).read_bytes()
    magic, W, H, mode, aconst, delta, *mq = _HDR.unpack_from(data, 0)
    if magic != MAGIC:
        raise ValueError("not a self_taught_pyramid file")
    Mf = np.array(mq, dtype=np.float64).reshape(3, 3) / 32767.0
    nplanes = 4 if mode == 2 else 3
    shapes = [(H, W)]
    while min(shapes[-1]) > STOP_SIZE:
        shapes.append(((shapes[-1][0] + 1) // 2, (shapes[-1][1] + 1) // 2))
    pos = _HDR.size
    filt = {}
    if FILT_ON:
        filt, pos = _unpack_filters(data, pos, len(shapes) - 1, nplanes)
    dec = _Dec(data, pos)
    planes = _Codec(dec, False, (H, W), nplanes, delta, filt_in=filt).run()
    P = np.stack([planes[i].ravel() for i in range(3)], axis=1)
    rgb = (P @ np.linalg.inv(Mf).T).reshape(H, W, 3) + 128.0
    out = np.clip(np.rint(rgb), 0, 255)
    if mode == 1:
        out = np.concatenate([out, np.full((H, W, 1), float(aconst))], axis=2)
    elif mode == 2:
        out = np.concatenate([out, np.clip(np.rint(planes[3] + 128.0), 0, 255)[..., None]], axis=2)
    Image.fromarray(out.astype(np.uint8)).save(output_path, format="PNG")
