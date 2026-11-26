# Maritime news and oil-market uncertainty

Does news about maritime threats and actual shipping disruptions improve forecasts
of oil-market uncertainty beyond market history and the amount of news?

This academic research project studies the Strait of Hormuz and the Red Sea /
Bab el-Mandeb corridor over 2019–2025. It combines news classification with
chronological forecasts of the next week's observed CBOE Crude Oil Volatility
Index (OVX) mean. The implementation and working reference labels are AI-assisted.

**Main finding:** event indicators produce a small final-test improvement in mean
absolute error, but the uncertainty interval includes zero. The study does not
establish a reliable incremental forecasting advantage, causality, or trading profits.

This repository shares the code, frozen experiment settings, annotation method,
and synthetic tests. **No research datasets are distributed.** The result summaries
below describe the completed local experiment; reproducing them requires its
private inputs and saved artifacts.

## What we built

1. **News ingestion and quality checks:** Guardian API retrieval and local ProQuest
   text import, preserving source versions, metadata, and duplicate groups.
2. **Article classification:** relevance, prospective threat, and actual operational
   disruption labels; comparison of keyword rules, TF–IDF with logistic regression,
   and frozen sentence embeddings with logistic regression.
3. **Weekly features:** news volume and threat/disruption shares aligned with market
   information available at each forecast cutoff.
4. **Forecast evaluation:** persistence and nested ridge models, chronological
   validation, expanding training windows, and predefined source/timing checks.

The primary text corpus contains 1,549 FT/Guardian articles. NLP development uses
138 primary articles; the separate 2023 evaluation uses 120 articles. The financial
dataset contains 204 training weeks (2019–2022), 52 validation weeks (2023), and
104 final-test weeks (2024–2025).

## Results and interpretation

On the 120-article NLP holdout, mean positive-class F1 across the three labels is
0.803 for keyword rules, 0.497 for TF–IDF, and 0.661 for embeddings. These scores
measure agreement with AI reference labels. Removing the largest related-report
component reduces keyword mean F1 to 0.584; all methods miss the three remaining
disruption positives.

The financial comparison uses the same 104 final-test weeks for every model:

| Model | Predictors | MAE, OVX points | RMSE, OVX points |
|---|---|---:|---:|
| M0: persistence | Latest available weekly OVX | 2.8118 | 4.4567 |
| M1: market history | Latest OVX and trailing four-week mean | 2.9606 | 4.3467 |
| M2: news volume | M1 plus log(1 + relevant article count) | 2.7469 | 4.2669 |
| M3: event indicators | M2 plus threat and disruption shares | 2.6861 | 4.2668 |

M3 reduces MAE by 2.2% relative to M2. The paired difference is −0.0608 OVX points,
with a 95% four-week block-bootstrap interval of [−0.1490, +0.0227]. RMSE is almost
identical. On 2023 validation, both news models performed worse than persistence;
the final-test results should be read alongside that earlier evidence.

Limitations include AI reference labels, concentration in related news reports,
FT coverage gaps in 2019, 2020 and 2024, unverified historical article availability,
market-calendar gaps, and only two final-test years. The final test has now been
examined; later outcome-guided revisions require new evaluation observations.

## Explore the implementation

| Area | Starting points |
|---|---|
| Retrieval and import | [retrieve_guardian.py](scripts/retrieve_guardian.py), [import_proquest.py](scripts/import_proquest.py) |
| Corpus quality and grouping | [analyze_news_corpus.py](scripts/analyze_news_corpus.py) |
| Label definitions | [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md) |
| NLP comparison and holdout | [compare_nlp.py](scripts/compare_nlp.py), [evaluate_nlp_holdout.py](scripts/evaluate_nlp_holdout.py) |
| Weekly alignment | [build_weekly_finance.py](scripts/build_weekly_finance.py) |
| Financial models | [financial_baselines.py](scripts/financial_baselines.py), [financial_news.py](scripts/financial_news.py) |
| Final evaluation | [evaluate_finance.py](scripts/evaluate_finance.py) |
| Frozen policies and tests | [config/](config/), [tests/](tests/) |

The original annotation guide and research source files are preserved unchanged.
Some refer to local audit reports, annotation batches, and manifests that are
intentionally absent from the public repository. Configuration hashes identify
those frozen private inputs; they do not make the inputs available.

## Run the tests without research data

Use Python 3.13 and `uv`. From the repository root:

```sh
uv sync --locked
uv run --locked python -m unittest discover -s tests -v
```

The 148 tests use synthetic fixtures and temporary directories. They cover parsing,
duplicate handling, annotation integrity, chronological split protection,
training-only preprocessing, forecast timing, and evaluation calculations. They
do not require article downloads, a Guardian API key, or an encoder download.
Installing the pinned dependencies requires network access on first setup.

## Data and reproducibility

The original research used Guardian API responses, licensed ProQuest exports
(principally Financial Times), and OVX observations obtained through FRED.
Readers must obtain their own source access to run ingestion. For Guardian
retrieval, copy `.env.example` to `.env` and set `GUARDIAN_API_KEY` locally.

Raw and processed data, annotations and evidence spans, trained models, experiment
outputs, executed notebooks, generated charts, PDFs, presentations, working logs,
and credentials are excluded from Git. The `.gitignore` allows only the public
source files and documentation; new publication material needs an explicit review
and an appropriate exception. Do not force-add private files.

The complete pipeline and `scripts/reproduce_results.py` expect the original
private data, annotation snapshots, encoder files, and saved manifests/artifacts.
A new download may differ from that archive and will not necessarily satisfy its
hash checks. The public tests validate implementation behavior; they do not
independently reproduce the reported empirical scores.

Reporting code is retained for inspection. Python figure/PDF generation uses the
separate `requirements-reporting.txt` environment. The presentation builder also
depends on a local OpenAI artifact runtime and presentation helpers, configured
through `ARTIFACT_RUNTIME_ROOT` and `PRESENTATIONS_SKILL_DIR`; these are not bundled
or installed by `uv sync`.
