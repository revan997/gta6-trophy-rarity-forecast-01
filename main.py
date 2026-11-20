"""gta6-trophy-rarity-forecast-01: forecast each GTA VI PS5 trophy's Exophase earn
rate at the case's target as five quantiles, scored by pinball loss on the logit
scale.

Model (computed from the case input at run time, nothing hardcoded):
  A trophy's earn rate is r = H / N, holders over players tracked.  Both grow,
  but at very different speeds: N (tracked players) surges as more players sync
  profiles, while H (holders of one trophy) grows only as players actually reach
  that trophy.  The rates therefore mostly FALL after launch, fast for late-game
  trophies, barely at all for the near-universal ones -- exactly what the launch
  night's own recording shows (players tracked 638 -> 3,868 at the first refresh
  while every rate but the 100% one fell).

  So the program models H and N separately, in log time since release:
    log N(t)     = a_N + b_N * log(1 + t)
    log(H_i(t))  = a_i + b_i * log(1 + t)      (per trophy, shrunk toward the
                                               cross-trophy median slope)
  and forecasts r_i(target) = H_i(target) / N(target).  Slopes are fitted on the
  last 24 h of distinct reads blended with the whole history, because growth
  decelerates as the launch surge passes.  Reads are step functions (Exophase
  refreshes every few hours), so consecutive identical reads collapse to one
  snapshot before fitting.

  The spread is an honest logit-scale sigma combining slope standard errors
  (which grow with the extrapolation distance), the fits' residual scatter, a
  model-misspecification term for growth that bends, and a refresh-lottery term
  for the chance that a refresh lands inside a short horizon.  Near-universal
  trophies (last rate >= 99.9%) get a tight band around ~99.99%.

Standard library only. Deterministic: same case in, same bytes out.
Run:  mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Release day begins 2026-11-19T00:00:00Z; log time is measured from there.
RELEASE_START = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# --- Priors (generic, documented in the rules / launch-night recording) ---
PRIOR_B_N = 1.0        # log N slope when history cannot identify it
PRIOR_B_H = 0.5        # log H slope when history cannot identify it
PRIOR_SLOPE_SE = 0.5   # slope standard error when no fit is available
PRIOR_RESID = 0.35     # residual scatter (logit) when no fit is available
MISS_K = 0.5           # extra logit uncertainty per unit of log-time extrapolation
LOTTERY_D = 1.6        # typical per-refresh logit drop of a rate
REFRESH_DRIFT_J = 1.0  # expected logit drop when one refresh lands (net of new holders)
REFRESH_GAP_H = 4.5    # typical hours between Exophase refreshes (launch night)
SIGMA_FLOOR = 0.8      # minimum logit sigma for an ordinary trophy
SIGMA_CAP = 2.5        # maximum logit sigma
UNIVERSAL_RATE = 99.9  # a trophy (nearly) every tracked player holds
UNIVERSAL_SIGMA = 0.3


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up (matches the starter's rounding)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def ols(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b, se_b, resid_sd, n) or None."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    if n > 2:
        s2 = sum(r * r for r in resid) / (n - 2)
        se_b = math.sqrt(s2 / sxx) if sxx > 0 else PRIOR_SLOPE_SE
        resid_sd = math.sqrt(sum(r * r for r in resid) / n)
    else:
        se_b = PRIOR_SLOPE_SE
        resid_sd = 0.0
    return (a, b, se_b, resid_sd, n)


def median(vals):
    v = sorted(vals)
    n = len(v)
    if n == 0:
        return 0.0
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError("unexpected levels %r" % (case["levels"],))
    trophies = case["trophies"]
    history = case["history"]
    if not history:
        raise ValueError("case has no history")
    n_tr = len(trophies)
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_h = case["horizon_minutes"] / 60.0

    # --- Distinct snapshots: collapse consecutive identical reads. ---
    # A read is a step function (Exophase refreshes every few hours), so a run
    # of identical reads is one state, stamped at the time it was LAST
    # confirmed -- that is the anchor the forecast extrapolates from.
    snaps = []  # (t_hours, N, rates tuple)
    prev_key = None
    for h in history:
        key = (h["players_tracked"], tuple(h["rates"]))
        t_h = (parse_time(h["at"]) - RELEASE_START).total_seconds() / 3600.0
        if key == prev_key:
            snaps[-1] = (t_h, snaps[-1][1], snaps[-1][2])
            continue
        prev_key = key
        snaps.append((t_h, h["players_tracked"], h["rates"]))
    m = len(snaps)
    x_last = math.log1p(max(snaps[-1][0], -0.9))
    x_target = math.log1p(max((target - RELEASE_START).total_seconds() / 3600.0, -0.9))
    dx = max(x_target - x_last, 0.0)
    xs_all = [math.log1p(max(t, -0.9)) for t, _, _ in snaps]

    # --- Denominator model: log N(t). ---
    log_n = [math.log(max(N, 1)) for _, N, _ in snaps]
    fit_n_all = ols(xs_all, log_n)
    t_last = snaps[-1][0]
    idx24 = [j for j, (t, _, _) in enumerate(snaps) if t >= t_last - 24.0]
    fit_n_24 = ols([xs_all[j] for j in idx24], [log_n[j] for j in idx24]) \
        if len(idx24) >= 3 else None
    if fit_n_all is None:
        b_n, se_b_n = PRIOR_B_N, PRIOR_SLOPE_SE
        log_n_target = math.log(max(snaps[-1][1], 1)) + b_n * dx
    else:
        b_n_all = fit_n_all[1]
        b_n = 0.5 * b_n_all + 0.5 * fit_n_24[1] if fit_n_24 else b_n_all
        b_n = min(max(b_n, 0.0), 2.5)
        se_b_n = fit_n_24[2] if fit_n_24 else fit_n_all[2]
        log_n_target = math.log(max(snaps[-1][1], 1)) + b_n * dx
    n_target = math.exp(log_n_target)

    # --- Per-trophy holder fits. ---
    last_rates = snaps[-1][2]
    # Global fallback rate (median of the last read's non-null rates).
    seen_rates = [r for r in last_rates if r is not None]
    global_rate = median(seen_rates) if seen_rates else 1.0

    fits = []  # per trophy: dict with b, se_b, resid, h_last
    b_raw = []
    for i in range(n_tr):
        pts = []
        for j, (t, N, rates) in enumerate(snaps):
            r = rates[i]
            if r is None:
                continue
            pts.append((xs_all[j], math.log(r * N / 100.0 + 0.5)))
        fit = ols([p[0] for p in pts], [p[1] for p in pts]) if len(pts) >= 2 else None
        fit24 = None
        if len(pts) >= 2 and idx24:
            x24_start = xs_all[idx24[0]]
            pts24 = [p for p in pts if p[0] >= x24_start]
            if len(pts24) >= 3:
                fit24 = ols([p[0] for p in pts24], [p[1] for p in pts24])
        if fit is not None:
            b = 0.5 * fit[1] + 0.5 * fit24[1] if fit24 else fit[1]
            fits.append({"b": b, "se": fit24[2] if fit24 else fit[2],
                         "resid": fit[3], "n": fit[4]})
            b_raw.append(b)
        else:
            fits.append(None)
    b_bar = median(b_raw) if b_raw else PRIOR_B_H
    b_bar = min(max(b_bar, 0.0), 2.5)
    resid_pool = median([f["resid"] for f in fits if f and f["resid"] > 0]) \
        if any(f and f["resid"] > 0 for f in fits) else PRIOR_RESID

    # Refresh-lottery spread: only matters when few refreshes fit in the horizon.
    gaps = [snaps[j + 1][0] - snaps[j][0] for j in range(m - 1)]
    gap_h = median(gaps) if gaps else REFRESH_GAP_H
    gap_h = min(max(gap_h, 0.5), 24.0)
    p_refresh = min(max(horizon_h / gap_h, 0.0), 1.0)
    lottery_sd = LOTTERY_D * math.sqrt(p_refresh * (1.0 - p_refresh))
    # When the history shows no refresh at all, the smooth trend cannot know
    # one is coming: add the expected drop of the next refresh (capped at half
    # a refresh's worth for long horizons, where the trend already covers it).
    # The weight fades as the history itself shows refreshes.
    refresh_evidence = min(1.0, max(m - 1, 0) / 3.0)
    extra_drift = -(1.0 - refresh_evidence) * min(p_refresh, 0.5) * REFRESH_DRIFT_J

    rows = []
    for i, trophy in enumerate(trophies):
        r_last = last_rates[i]
        if r_last is None:
            r_last = global_rate
        # Near-universal trophy: (nearly) every tracked player holds it; it
        # stays near 100% whatever the denominator does.
        if r_last >= UNIVERSAL_RATE:
            med = min(99.99, r_last)
            sigma = UNIVERSAL_SIGMA
            rows.append({"id": trophy["id"], "q": quantiles(med, sigma)})
            continue
        f = fits[i]
        h_last = r_last * snaps[-1][1] / 100.0
        if f is not None:
            w = f["n"] / (f["n"] + 3.0)  # shrink weak fits toward the group
            b_i = w * f["b"] + (1.0 - w) * b_bar
            se_b_i = f["se"]
        else:
            b_i, se_b_i = b_bar, PRIOR_SLOPE_SE
        b_i = min(max(b_i, 0.0), 2.5)
        log_h_target = math.log(h_last + 0.5) + b_i * dx
        h_target = max(math.exp(log_h_target) - 0.5, 0.0)
        med = clamp_pct(100.0 * h_target / n_target)
        med = clamp_pct(inv_logit(logit(med) + extra_drift))
        sigma = math.sqrt((se_b_i * dx) ** 2 + (se_b_n * dx) ** 2
                          + resid_pool ** 2 + (MISS_K * dx) ** 2
                          + lottery_sd ** 2 + 0.3 ** 2)
        sigma = min(max(sigma, SIGMA_FLOOR), SIGMA_CAP)
        rows.append({"id": trophy["id"], "q": quantiles(med, sigma)})
    return {"trophies": rows}


def quantiles(med_pct, sigma):
    lg = logit(med_pct)
    qs = []
    prev = None
    for z in Z:
        q = round4(inv_logit(lg + z * sigma))
        q = min(max(q, 0.01), 99.99)
        if prev is not None and q < prev:
            q = prev
        qs.append(q)
        prev = q
    return qs


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
