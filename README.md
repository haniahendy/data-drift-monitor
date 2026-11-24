# data-drift-monitor

Compare a reference dataset against a current dataset and get a per-column
drift report: Population Stability Index (PSI), a none/moderate/large flag,
and mean/std shift summaries.

## What it does

- `drift.population_stability_index` computes PSI between two numeric
  samples, binning on the reference distribution.
- `drift.drift_report` runs PSI plus shift stats across every shared numeric
  column of two DataFrames.
- `run.py` is a CLI wrapper comparing two CSV files.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python run.py reference.csv current.csv --bins 10
```

PSI rule of thumb used for the flag: < 0.1 none, 0.1-0.25 moderate,
> 0.25 large. These are common heuristics, not statistical guarantees.

## Tests

```bash
pytest -q
```

## Honest limits

- Numeric columns only; categorical drift (e.g. chi-square) is not
  implemented.
- PSI depends on bin count and sample size; small samples give noisy
  values. There is no significance testing.
