# ai-intel-daily

Daily data collector for two workstreams:

1. AI tool intel - Hacker News "Show HN" plus the Product Hunt daily leaderboard.
2. FLUX prompt packs - trending Civitai images with full generation metadata.

`producthunt.com` and `civitai.com` are not reachable from the mainland
network, so this repository is the overseas collector: a GitHub Actions job runs
every day at 00:00 UTC (= 08:00 Asia/Shanghai) and commits the raw bundle under
`data/`. The local machine then pulls `data/latest.json` through a mirror.

Hacker News uses the public Algolia API and works everywhere.

## Layout

```
scrapers/common.py         shared HTTP helpers and heuristics
scrapers/fetch_hn.py       Show HN, last 48h
scrapers/fetch_ph.py       Product Hunt daily leaderboard
scrapers/fetch_civitai.py  trending FLUX images + metadata + licence
scrapers/run_all.py        orchestrator, writes data/YYYY-MM-DD.json
data/                      generated output
civitai/images/<date>/     downloaded reference images
```

## Local usage

```
python scrapers/run_all.py --out data            # all sources
python scrapers/run_all.py --out data --only hn  # single source
```

Every source fails independently, so a blocked or changed upstream never wipes
out the whole run.

## Licence note

The Civitai collector records `allowCommercialUse`, `allowDerivatives` and
`allowNoCredit` for every picked image. Do not redistribute model weights or
artwork that the author has not licensed for reuse.
