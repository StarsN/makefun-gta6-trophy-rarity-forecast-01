# GTA VI Trophy Rarity Forecaster

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/StarsN/makefun-gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/StarsN/makefun-gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for the make.fun bounty **gta6-trophy-rarity-forecast-01**: given every read the
house made of Grand Theft Auto VI's Exophase trophy page up to a case's checkpoint, it forecasts
each trophy's earn rate (the percent of tracked players who have earned it) at the case's target,
as five quantiles (5%, 25%, 50%, 75%, 95%), scored by pinball loss on the logit scale.

## Approach

An earn rate is a ratio — earners over players tracked — so the program models the numerator and
the denominator separately and divides:

- **Earners and players, separately.** From each read it recovers the earner counts
  (rate × players tracked) and fits, per trophy, a recency-weighted least-squares line of
  ln(earners) against ln(1 + hours since release), and one for ln(players tracked). Both
  extrapolations are damped, the per-trophy slopes are shrunk toward the cross-trophy typical
  slope (empirical Bayes, weighted by fit precision), and the median rate is the forecast
  earners over the forecast players. Modelling the ratio's parts is what lets the forecast see
  dilution: a fast-growing tracked population drags rates down even while earners grow.
- **A direct blend.** The structural median is blended with a direct fit of the rate's logit on
  the same log-time axis (the scale on which earn rates grow like a power law and bend toward
  100%), so neither model's failure mode dominates.
- **An honest spread.** Each trophy's spread combines its fit's residual scale, the slope
  uncertainty over the extrapolation distance, and a term that grows with the horizon, so the
  band widens exactly when the forecast is extrapolating far or the page has been jumpy.
- **The frozen-page regime.** If a case's history is a single frozen page — every read identical,
  the page never refreshed during the window — the rates are stale values from a much smaller
  tracked population. Around a launch, the next refresh multiplies the tracked count with newer,
  less-progressed players while earner counts barely move, so rates fall hard in logit terms. For
  that regime the program shifts the median down in logit and carries a wide spread, instead of
  the narrow band a persistence forecast would write. (On the kit's practice cases, cut across
  Exophase's first refresh, the house persistence baseline scores a loss of about 1.51; this
  forecaster's design targets exactly that failure.)

Only the last 36 hours of page states feed the fits: earn-rate dynamics are local, and older
reads describe a younger game.

## Running it

Python 3 standard library only, no network, deterministic — the same case always writes the
same bytes, in well under a second.

```
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

It reads every `<id>.in` in the input directory and writes `<id>.out` per case:
`{"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}`, one row per trophy.
