# GTA VI trophy rarity forecast 01

Forecast, for each of Grand Theft Auto VI's 55 PlayStation 5 trophies, the share
of Exophase-tracked players holding it at 2026-11-26T00:00:00Z, as five
quantiles (5%, 25%, 50%, 75%, 95%), scored by pinball loss on the logit scale.

## The model

A trophy's earn rate is `r = H / N`: holders over players tracked. Both grow,
but at very different speeds. `N` (tracked players) surges as more players sync
profiles; `H` (holders of one trophy) grows only as players actually reach that
trophy. Rates therefore mostly **fall** after launch — fast for late-game
trophies, barely at all for near-universal ones. Launch night's own recording
shows it: players tracked 638 → 3,868 at the first refresh while every rate but
the 100% one fell, and by 2026-11-20 ~01:00 UTC the page showed ~30,000 players
tracked with most rates down 10–50× from launch night.

So the program (`main.py`, Python 3 stdlib only, no network) models `H` and `N`
separately, in log time since release:

- `log N(t) = a_N + b_N · log(1 + t)`
- `log H_i(t) = a_i + b_i · log(1 + t)` per trophy, shrunk toward the
  cross-trophy median slope (weak fits shrink most)

Slopes are fitted on the last 24 h of distinct reads blended with the whole
history, because growth decelerates as the launch surge passes. Reads are step
functions (Exophase refreshes every few hours), so consecutive identical reads
collapse to one snapshot, stamped at the time it was last confirmed.

The forecast is `r_i(target) = H_i(target) / N(target)`. The spread is an honest
logit-scale sigma combining slope standard errors (growing with extrapolation
distance), the fits' residual scatter, a model-misspecification term for growth
that bends, and a refresh-lottery term for a refresh landing inside a short
horizon. When the history shows no refresh at all, an explicit expected refresh
drop is added to the median. Near-universal trophies (last rate ≥ 99.9%) get a
tight band around ~99.99%.

## Backtests (see `../backtest.py` in the build directory)

| Test | Model loss | House persistence | House curve |
| --- | --- | --- | --- |
| Practice p00/p01 vs reconstructed 04:30Z refresh (central case) | **0.40** | 1.22 | 1.22 |
| Practice, pessimistic reconstruction (new players hold ~nothing) | **0.55** | 1.47 | 1.47 |
| Flat history → live 2026-11-20 page (N = 30,000, 22 h horizon) | **0.72** | 2.29 | 2.29 |
| Simulated final, 10 seeds (power-law truth + noise) | 0.20 | 0.41 | 0.20 |

The practice answers are not in the kit, so they are reconstructed from the
documented refresh (N 638 → 3,868, trophies earned 943 → 4,276) under a
new-cohort assumption, with sensitivity shown. The simulated final favours the
curve baseline's exact functional form; the model matches it there while
beating persistence 2×.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input dir, writes `<id>.out` per case. Deterministic:
two runs on the same case write identical bytes. Runs in ~0.05 s per case.

## License

MIT. Entry for make.fun competition `gta6-trophy-rarity-forecast-01`.
