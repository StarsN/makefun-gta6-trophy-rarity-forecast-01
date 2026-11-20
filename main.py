"""gta6-trophy-rarity-forecast-01: forecast every GTA VI trophy's Exophase earn rate at the case's target.

Model, per case. A read of the page is a pair (players tracked N, earners per trophy E = rate * N / 100);
the earn rate is the ratio E / N, so the forecast models the numerator and the denominator separately
and divides, which is what lets it see dilution (a fast-growing tracked population dragging rates down
even as earners grow). On top of that structural median it blends a direct fit of the rate's logit
against ln(1 + hours since release) -- the scale on which earn rates grow like a power law and bend
toward 100% -- with both slopes shrunk toward the cross-trophy typical slope and damped over the long
extrapolation. The spread combines the fits' residual scale, the slope uncertainty over the
extrapolation distance, and a horizon term, so it widens with the horizon and with how jumpy the
page has been.

One regime is special: a history that is a single frozen page (every read identical, the page never
refreshed during the window) holds stale rates from a much smaller tracked population. Around a game's
launch the next refresh multiplies the tracked count with newer, less-progressed players while the
earner counts barely move, so rates fall hard in logit terms; the frozen branch therefore shifts the
median down in logit by FROZEN_SHIFT and carries a wide, honest spread instead of the narrow band a
persistence forecast would write.

Python 3 standard library only, no network, deterministic: the same case always writes the same bytes.
Run as the grader runs it: `python3 -P -s /bot/main.py` with the case in /input (MAKEFUN_INPUT_DIR and
MAKEFUN_OUTPUT_DIR stand in for /input and /output outside the grader).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# --- tunables (set from the kit's published practice facts and synthetic launch-week backtests) ---
FROZEN_SHIFT = 1.5        # logit drop of a frozen launch page's rates at the next refresh
FROZEN_SHIFT_OFFLAUNCH = 0.5
FROZEN_SIGMA = 0.55       # spread in the frozen regime
HALF_LIFE_H = 14.0        # recency weighting half-life of the fits, hours
FIT_WINDOW_H = 36.0       # fits use only page states from the last this-many hours
DAMP_N = 0.75             # damping of the players-tracked extrapolation
DAMP_E = 0.65             # damping of the earners / logit extrapolations
EN_WEIGHT = 0.35          # weight of the earners/players median in the median blend
SIGMA_A = 1.0             # weight of the logit fit's residual variance in the spread
SIGMA_B = 1.0             # weight of the slope-uncertainty variance in the spread
SIGMA_C = 0.06            # horizon variance per day (refresh / random-walk risk)
SIGMA_D = 0.02            # variance floor
SIGMA_MIN = 0.18
SIGMA_MAX = 2.2


def clamp_rate(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def inv_logit(l: float) -> float:
    if l >= 0:
        return 100.0 / (1.0 + math.exp(-l))
    e = math.exp(l)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's Math.round(x * 10000) / 10000 rounds it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def read_time(read: dict) -> datetime:
    return parse_ts(read.get("slot") or read["at"])


def quantiles(median_logit: float, sigma: float) -> list:
    return [round4(inv_logit(median_logit + z * sigma)) for z in Z]


class Fit:
    __slots__ = ("a", "b", "resid", "se_b", "x_last", "n")

    def __init__(self, a, b, resid, se_b, x_last, n):
        self.a = a
        self.b = b
        self.resid = resid
        self.se_b = se_b
        self.x_last = x_last
        self.n = n

    def at(self, x):
        return self.a + self.b * x


def wls(xs, ys, ws):
    """Weighted least squares y ~ x. Returns a Fit or None when the fit is meaningless."""
    n = len(xs)
    if n < 3:
        return None
    sw = sum(ws)
    if sw <= 0:
        return None
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = my - b * mx
    resid2 = sum(w * (y - (a + b * x)) ** 2 for x, y, w in zip(xs, ys, ws)) / sw
    resid = math.sqrt(max(resid2, 0.0))
    se_b = resid / math.sqrt(sxx)
    return Fit(a, b, resid, se_b, xs[-1], n)


def weighted_median(values, weights):
    pairs = sorted(zip(values, weights))
    total = sum(weights)
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= total / 2:
            return v
    return pairs[-1][0]


def shrink_slopes(fits):
    """Empirical-Bayes shrinkage of per-trophy slopes toward their precision-weighted centre."""
    idx = [i for i, f in enumerate(fits) if f is not None]
    if not idx:
        return {}
    slopes = [fits[i].b for i in idx]
    ses = [max(fits[i].se_b, 1e-6) for i in idx]
    centre = weighted_median(slopes, [1.0 / (s * s) for s in ses])
    mean_se2 = sum(s * s for s in ses) / len(ses)
    var_b = sum((s - sum(slopes) / len(slopes)) ** 2 for s in slopes) / len(slopes)
    tau2 = max(0.0, var_b - mean_se2)
    out = {}
    for k, i in enumerate(idx):
        se2 = ses[k] * ses[k]
        k_i = tau2 / (tau2 + se2) if tau2 + se2 > 0 else 0.0
        out[i] = centre + k_i * (slopes[k] - centre)
    return out


def forecast_frozen(case, hist, release_dt, target_dt):
    """Every read identical: a stale page. Expect the refresh's dilution (see the module docstring)."""
    last = hist[-1]
    first_t = read_time(hist[0])
    launch = first_t <= release_dt <= target_dt
    shift = FROZEN_SHIFT if launch else FROZEN_SHIFT_OFFLAUNCH
    horizon_days = case["horizon_minutes"] / 1440.0
    sigma = FROZEN_SIGMA + 0.1 * math.sqrt(horizon_days)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        r = last["rates"][i]
        if r is None:
            rows.append({"id": trophy["id"], "q": quantiles(logit(0.05), 1.3)})
        elif r >= 99.9:
            rows.append({"id": trophy["id"], "q": quantiles(logit(r), 0.35)})
        elif r <= 0.01:
            rows.append({"id": trophy["id"], "q": quantiles(logit(0.04), 1.3)})
        else:
            rows.append({"id": trophy["id"], "q": quantiles(logit(r) - shift, sigma)})
    return {"trophies": rows}


def forecast_general(case, hist, release_dt, target_dt):
    n_tr = len(case["trophies"])
    times = [read_time(r) for r in hist]
    t_last = times[-1]
    hours = [(t - release_dt).total_seconds() / 3600.0 for t in times]
    xs_all = [math.log(1.0 + max(h, -0.95)) for h in hours]
    x_target = math.log(1.0 + max((target_dt - release_dt).total_seconds() / 3600.0, -0.95))
    horizon_days = case["horizon_minutes"] / 1440.0

    # Compress to page states: keep the last read of each run of identical pages, plus the final read.
    keep = []
    for j in range(len(hist)):
        nxt = hist[j + 1] if j + 1 < len(hist) else None
        if nxt is None or nxt["players_tracked"] != hist[j]["players_tracked"] or nxt["rates"] != hist[j]["rates"]:
            keep.append(j)
    # Restrict the fits to the recent window (always keeping at least the last five states).
    if len(keep) > 5:
        within = [j for j in keep if hours[-1] - hours[j] <= FIT_WINDOW_H]
        start = min(within[0], keep[-5]) if within else keep[-5]
        keep = [j for j in keep if j >= start]
    xs = [xs_all[j] for j in keep]
    ws = [0.5 ** ((hours[-1] - hours[j]) / HALF_LIFE_H) for j in keep]
    dx = x_target - xs[-1]

    # Players tracked: fit ln N, damp the extrapolation.
    ln_n = [math.log(max(hist[j]["players_tracked"], 1)) for j in keep]
    fit_n = wls(xs, ln_n, ws)
    n_last = hist[-1]["players_tracked"]
    if fit_n is not None:
        ln_n_hat = fit_n.at(xs[-1]) + DAMP_N * fit_n.b * dx
        n_hat = math.exp(ln_n_hat)
        n_hat = min(max(n_hat, 0.95 * n_last), 25.0 * n_last)
        var_ln_n = fit_n.resid ** 2 + (fit_n.se_b * dx) ** 2
    else:
        n_hat = float(n_last)
        var_ln_n = 0.04

    # Per-trophy fits: earners (ln (E + 0.5)) and the rate's logit directly.
    earn_fits = [None] * n_tr
    logit_fits = [None] * n_tr
    last_rate = [None] * n_tr
    last_earn = [None] * n_tr
    # Build the per-trophy series over kept reads.
    series = [[] for _ in range(n_tr)]  # per trophy: list of (x, w, rate, earners)
    for pos, j in enumerate(keep):
        read = hist[j]
        for i in range(n_tr):
            r = read["rates"][i]
            if r is not None:
                series[i].append((xs[pos], ws[pos], r, r / 100.0 * read["players_tracked"]))
    for i in range(n_tr):
        pts = series[i]
        if not pts:
            continue
        last_rate[i] = pts[-1][2]
        last_earn[i] = pts[-1][3]
        if len(pts) >= 3:
            earn_fits[i] = wls([p[0] for p in pts], [math.log(p[3] + 0.5) for p in pts], [p[1] for p in pts])
            logit_fits[i] = wls([p[0] for p in pts], [logit(p[2]) for p in pts], [p[1] for p in pts])
    shrunk_e = shrink_slopes(earn_fits)
    shrunk_l = shrink_slopes(logit_fits)
    resids = [f.resid for f in logit_fits if f is not None]
    pooled_resid = sorted(resids)[len(resids) // 2] if resids else 0.3

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        if last_rate[i] is None:
            rows.append({"id": trophy["id"], "q": quantiles(logit(0.05), 1.3)})
            continue
        medians = []  # (value, weight)
        fl = logit_fits[i]
        if fl is not None:
            b = shrunk_l.get(i, fl.b)
            medians.append((fl.at(xs[-1]) + DAMP_E * b * dx, 1.0 - EN_WEIGHT))
        fe = earn_fits[i]
        if fe is not None:
            b = shrunk_e.get(i, fe.b)
            ln_e_hat = fe.at(xs[-1]) + DAMP_E * b * dx
            e_hat = math.exp(ln_e_hat) - 0.5
            e_hat = max(e_hat, 0.85 * last_earn[i])
            e_hat = min(e_hat, 0.999 * n_hat)
            medians.append((logit(100.0 * e_hat / n_hat), EN_WEIGHT))
        if medians:
            median_logit = sum(v * w for v, w in medians) / sum(w for _, w in medians)
        else:
            median_logit = logit(last_rate[i])
        if fl is not None:
            var = SIGMA_A * fl.resid ** 2 + SIGMA_B * (fl.se_b * dx) ** 2 + SIGMA_C * horizon_days + SIGMA_D
        else:
            var = SIGMA_A * pooled_resid ** 2 + SIGMA_C * horizon_days + SIGMA_D + 0.09
        sigma = min(max(math.sqrt(var), SIGMA_MIN), SIGMA_MAX)
        rows.append({"id": trophy["id"], "q": quantiles(median_logit, sigma)})
    return {"trophies": rows}


def forecast_case(case, name):
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    hist = case["history"]
    release_dt = datetime.fromisoformat(case["game"]["release_date"]).replace(tzinfo=timezone.utc)
    target_dt = parse_ts(case["target"])
    first = hist[0]
    frozen = all(
        r["players_tracked"] == first["players_tracked"] and r["rates"] == first["rates"] for r in hist
    )
    if frozen:
        return forecast_frozen(case, hist, release_dt, target_dt)
    return forecast_general(case, hist, release_dt, target_dt)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
