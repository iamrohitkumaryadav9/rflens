# analytics-mcp and the stats agent (week 5 design)

Status: APPROVED and BUILT (2026-10-25), analytics-mcp v1 on SECOM only. Section 8 lists where the
build differs from this design. The graph side is documented in [orchestrator.md](orchestrator.md).

## 1. Data

**SECOM** is loaded: 1,567 runs (`secom_runs`: 1,463 `pass`, 104 `fail`, 2008-07-19 to 2008-10-17),
590 features, 924,530 measurements in `secom_measurements` (long format, `feature_idx` 1..590),
41,951 of them NULL (missing). The most-missing features are 91.2% empty (e.g. 158, 159, 293, 294).
Feature units are not published (unitless; the column comment must say UNKNOWN).

**RadioML (DeepSig): SKIPPED (decision 2026-10-25).** Reasons: the download is behind a form, the
file is about 21 GB, and the licence is CC BY-NC-SA 4.0. Details:

- Official page: https://www.deepsig.ai/datasets/. Licence, quoted: "All datasets provided by
  Deepsig Inc. are licensed under the Creative Commons Attribution – NonCommercial – ShareAlike
  4.0 License". Non-commercial is acceptable for this portfolio project, but ShareAlike means any
  derived dataset must carry the same licence, so nothing derived may be committed to the repo.
- Formats: RadioML 2018.01A is HDF5 (complex floats, 1,024 samples per example; download
  https://opendata.deepsig.io/datasets/2018.01/2018.01.OSC.0001_1024x2M.h5.tar.gz); 2016.10A is a
  Python pickle (https://opendata.deepsig.io/datasets/2016.10/RML2016.10a.tar.bz2). Loading a
  pickle runs code, so it would need conversion inside the sandbox.
- **The download is behind a form.** A plain GET of the file URL returns a page asking for first
  name, last name, affiliation and e-mail ("Due to abuse issues we've been forced to screen http
  requests"). Filling that in on the user's behalf, or with invented details, is not something to
  automate; the user would have to download by hand.
- Size: the page does not state one. By arithmetic from the paper's example count (2,555,904
  examples × 1,024 samples × 2 × float32) the 2018.01A array is about 21 GB uncompressed; the
  compressed tarball size is unknown. That is impractical on a CPU-only laptop for what the menu
  needs.
- h5py check (throwaway venv, Windows, this machine's Application Control policy): `h5py 3.16.0`
  (HDF5 2.0.0), `numpy 2.5.4`, `scipy 1.18.1`, `matplotlib 3.11.2` all import and plot. So h5py
  would work if RadioML were wanted, but it is not added to the project.
- **Analyses lost by skipping:** accuracy or confusion by SNR, per-modulation feature
  distributions, I/Q-level statistics. The menu below is dataset-agnostic (a registry entry per
  dataset), so RadioML could be added later from a hand-downloaded subset without a redesign.

Proposal: v1 covers SECOM only (the registry has one entry). Suggestion, not in scope: expose
numeric ULS columns (`uls_fr.power_erp`, unit UNKNOWN) as a second dataset.

## 2. Safety design: a fixed menu, never LLM-written code

The LLM chooses an analysis name and parameters as JSON. Parameters are validated by Pydantic models
(`extra="forbid"`); there is no field that holds code, SQL or a file path. The code that runs is
fixed, in the repo, and baked into the sandbox image. Filters are structured (never SQL strings):
`label in {pass, fail}`, `run_ts` between two dates, and up to 3 conditions
`feature_idx <op> value` (`<, <=, >, >=, ==, is_null, not_null`).

Every result carries: `analysis`, normalised `params`, `n_runs_used`, `missing_policy`, `warnings`,
`dataset` (`secom`, row count, extract SHA-256), library versions, and `analysis_id` (SHA-256 of
analysis + params + extract hash, so the same call yields the same id and artifacts).

| analysis | inputs | outputs | limits |
|---|---|---|---|
| `describe` | `feature` (1..590), `filters` | n, n_missing, missing share, mean, sd (n-1), min, q1, median, q3, max | 1 feature |
| `missing_summary` | `top_k`, `filters` | features with the highest missing share: idx, n_missing, share | top_k ≤ 50 |
| `distribution` | `feature`, `bins` (5..50, default 20), `by_label` bool, `filters` | bin edges, counts (per label if asked), skewness, n, n_missing | 1 feature |
| `group_compare` | `feature`, `test` (`mann_whitney` default, `welch_t`), `filters` | n, median, mean per group (pass / fail), statistic, two-sided p, effect size (rank-biserial for Mann-Whitney, Cohen's d for Welch), direction text-free (`higher_in`) | 1 feature; each group needs n ≥ 20 else refused with the counts |
| `correlation` | `feature_a`, `feature_b`, `method` (`spearman` default, `pearson`) | n pairs (pairwise-complete), coefficient, p | n ≥ 20 |
| `correlation_top` | `feature`, `method`, `top_k` | top_k features by \|coefficient\| with n pairs; ties by feature_idx ascending | top_k ≤ 50 |
| `feature_ranking` | `target` fixed `label` with `positive` = `fail`, `method` (`mann_whitney` default), `top_k`, `min_n` (default 20), `filters` | ranked list: feature_idx, n_pass, n_fail, missing share, statistic, p, BH q-value, effect size, `higher_in`; plus `excluded` (features with n < min_n in a group or zero variance, with the reason) | top_k ≤ 50; all 590 features tested |
| `outliers` | `feature`, `method` (`iqr` k=1.5 default, `zscore` 3.0), `filters` | thresholds, count, up to 20 (run_id, value, label) | 1 feature |
| `plot` | `kind` (`histogram`, `box_by_label`, `scatter`, `ranking_bar`), features per kind, `filters` | PNG artifact id, size, the numbers drawn (same JSON as the matching analysis) | 1 image per call, 800×600, scatter ≤ 5,000 points (fixed-seed sample, stated) |

Method statements (returned in the result, so the writer can quote them): feature ranking uses the
two-sided Mann-Whitney U test per feature (scipy, asymptotic, tie-corrected), rank-biserial
correlation as the effect size, Benjamini-Hochberg q-values over all tested features, ranked by p
ascending with ties by feature_idx ascending. **Missing values:** pairwise exclusion per feature
(NaN dropped inside each group), n per group reported, the missing share reported, features with too
few values listed under `excluded`, never imputed. SECOM is observational: results are associations,
not causes (the writer prompt and a verifier check say so).

Global caps: output JSON ≤ 64 KB, wall time 60 s, 512 MB, 1 CPU, rows in the extract ≤ 5,000,
features per extract ≤ 590.

## 3. Sandbox

```
analytics-mcp (host process, stdlib + psycopg only)
  1. validate params (Pydantic), look up the dataset registry entry
  2. read the needed columns as role rflens_analytics_reader -> job dir: data.csv (+ job.json)
  3. docker run --rm --name rflens-analytics-<job> --network none --memory 512m --cpus 1
       --pids-limit 64 --read-only --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges
       -v <job>/in:/in:ro -v <job>/out:/out  rflens-analytics:<tag>   python -I /app/runner.py
  4. wait <= 60 s; on timeout `docker kill` that container id (the one this call started); no pruning
  5. read /out/result.json (size-checked) and /out/*.png; move PNGs to artifacts/<analysis_id>.png
```

- **Data path.** The container never sees the database or any credential. The host process reads
  only the requested columns through a dedicated role, `rflens_analytics_reader` (migration:
  SELECT on `secom_runs`, `secom_measurements`, `v_secom_values` only; read-only sessions; statement
  timeout), not the owner role, and writes a CSV extract mounted read-only at `/in`.
- **Container.** `--network none` (verified: a socket connect from the probe image fails with
  `OSError`), read-only root filesystem, only `/out` (one job directory) and `/tmp` (tmpfs) are
  writable, all capabilities dropped, non-root user (uid 10001). The runner is fixed code copied into
  the image; the image is built once (`make analytics-image`, the only step that needs the network,
  with pinned versions).
- **Probe image (built and run here):** `python:3.12-slim` + `numpy==2.5.4`, `scipy==1.18.1`,
  `matplotlib==3.11.2`, `pandas`; 716 MB (pandas is not needed by the menu, so the real image
  should be smaller; not measured). Inside it, with the flags above, numpy / scipy / pandas imported
  and a matplotlib (Agg) PNG was saved to `/tmp`, and a network connection failed. Nothing native
  loads on the Windows host: the project's own virtualenv does not gain numpy or scipy.
- **Artifacts.** `artifacts/` (gitignored, to be added to `.gitignore`), files `<analysis_id>.png`
  and `<analysis_id>.json`. The API serves them by id behind the existing API key
  (`GET /artifacts/{id}`, the id is validated as 64 hex characters, never a path), the UI shows them.
- **Processes.** Containers are started and killed only by the id this call received; the dev-server
  rule (PIDs recorded in `run/<name>.pid`) covers the analytics-mcp process itself.
- **Tests.** Menu validation, the registry, the job writer and the artifact-id rules are tested in
  the normal suite. The numeric code (`analytics_runner/`) is tested inside the image
  (`make test-analytics`). The sandbox flags are tested against a real container.
  **Open point for you:** `make ci-local` itself runs in Docker and has no Docker daemon inside, so
  the container tests cannot run there (GitHub's `ubuntu` runner does have Docker). I would run
  them in the GitHub workflow and in `make test-analytics`, and have `ci-local` print that they were
  not run rather than skip them silently; your call.

## 4. analytics-mcp

`src/rflens/mcp/analytics_server.py` + `analytics_menu.py`, `make analytics-mcp`, port **8103**,
`http://127.0.0.1:8103/mcp`; the same setup as docs-mcp and sql-mcp (`mcp==2.3.0`, stateless
Streamable HTTP, plain JSON responses, localhost, Host/Origin validation). Tools mirror the menu one
to one (`list_analyses`, `describe`, `missing_summary`, `distribution`, `group_compare`,
`correlation`, `correlation_top`, `feature_ranking`, `outliers`, `plot`). `list_analyses` returns the
menu and each analysis's JSON Schema, which the stats agent puts in its prompt. Tool descriptions
state that results are data, not instructions.

Every call is a row in `analytics_call_log` (time, tool, optional `question`, normalised params,
`analysis_id`, rows in the extract, duration, status ok / rejected / error / timeout, reason),
written through `rflens_analytics_logger` (INSERT only); if the log write fails the call fails and
the result is withheld, as in sql-mcp. No authentication in local development; the same remote-
deployment warning as the other servers. Added to `make servers-up` (name `analytics`, port 8103,
`PORT="analytics=8113"` to change).

New migration: the two roles (passwords `ANALYTICS_READER_PASSWORD`, `ANALYTICS_LOGGER_PASSWORD` in
.env and CI), `analytics_call_log`, and column comments (SECOM features unitless, UNKNOWN; label
values; NULL = missing). The sql-mcp allow-list does not change.

## 5. Graph

```
planner --(stats)--> approval --> stats_agent --> writer --> verifier --> finalize
```

- **planner:** new route `stats` (questions about patterns, associations, distributions or
  comparisons in measurement data); it also returns `stats_question`. Prediction, model-fitting and
  arbitrary-computation requests are refused here (`refuse_reason`) or by the stats agent.
- **stats_agent** (one LLM call through the gateway, temperature 0): input = question, the menu with
  schemas, the dataset registry (feature range, labels, missing-value policy); output JSON
  `{refuse, refuse_reason, analysis, params}`. The server's validation errors go back to the model
  once (so at most 2 calls); then the call to analytics-mcp (no LLM). It writes `state.stats`
  (`A1`): analysis, params, result JSON (capped), artifact ids, `analysis_id`, `query_id` (its
  `analytics_call_log` row).
- **writer:** cites `[A1]`; the Sources section is built by code: `[A1] analytics-mcp call #<id>:
  <analysis>(<params>) on secom, extract <sha-prefix>`; plot artifacts are listed as links.
- **verifier:** `stats_results` is a number source: a number in the answer must be a numeric leaf of
  the cited `A1` result (rounded at the draft's own precision), a count (n), or derived from it by
  difference or ratio. New checks: `unsupported_stat_claim` (the words "significant" or "not
  significant" need the p or q value and alpha 0.05 in the text and a matching sign), `causal_claim`
  (cause, causes, leads to, due to), `missing_n` (a p-value without the group sizes it rests on).
  Feature indices must exist in the result. A stats-side failure (the analysis is wrong for the
  question) is routed like the SQL-side ones: one stats-agent retry, then redact. The call cap and
  budget logic are unchanged (stats questions use 3 to 5 calls).
- **Approval:** the planned worst case adds one stats-agent call (2 with the correction).
- **Mixed SQL + stats questions are out of scope** (suggestion).

`docs/design/orchestrator.md` changes when this is built: hard rule 4, the state table (`stats`,
`route` value), the graph diagram, verifier rules, the stats section. It is not changed now.

## 6. Golden set: four stats questions (author assistant, reviewed by the owner 2026-10-27)

New type `stats` in `eval/golden.yaml` (55 questions now). Each answerable one carries
`reference_analysis` (the menu call), `reference_stats` (`checks`: path into the analysis result
and the reference value; `answer_rows`: values the answer text must contain; `via`; scipy version;
SHA-256 of the data extract) and `notes` saying how the reference was computed. The references
come from `analytics_runner/reference.py` (`make golden-stats-refresh`), which does not import the
analysis code: it calls scipy.stats directly and recomputes every Mann-Whitney p-value from the
textbook formula, and stops if the two differ. The first ranking was also checked once in a scratch
venv on a separate dump of the database.

1. **stats-01 (feature ranking):** "Which two sensor features in the SECOM data are most strongly
   associated with failing runs?" Features 60 then 104 (p 5.7e-11 and 9.6e-9). N is 2 because
   ranks 2 and 3 differ clearly (p 9.55e-9 against 1.07e-6, a factor of 112) while ranks 3 and 4
   (features 248 and 520, p 1.07e-6 and 1.19e-6) are near-tied; the refresh refuses a cutoff
   below a factor of 10 and an order inside the top N below a factor of 2.
2. **stats-02 (pass/fail comparison):** "...how large is the difference (the median of each group and the effect size)?" The answer text
   must give the two medians; the group sizes are checked in the result only. Feature 60: 1,456 passing and 104 failing values, medians
   0.722 and 5.52, p 5.7e-11, rank-biserial 0.384 (higher in failing runs).
3. **stats-03 (distribution):** feature 104: n 1,565, 2 missing, quartiles -0.0118, -0.0101,
   -0.0082, and the 20 bin counts in the result.
4. **stats-04 (refuse):** "Train a classifier on the SECOM data and predict whether tomorrow's
   wafer lot will fail." Refused by the planner or the stats agent; no analysis may run.

Scoring (`eval/e2e_score.py`): the final state has an answered analysis of the reference menu
entry, every check passes (1e-6 relative; strings and lists exactly), the answer cites `[A1]` and
contains `answer_rows` (a ranked list in order); stats-04 is scored like a refusal and also needs
that no analysis ran.

## 7. Cost estimate

- **Build: $0 paid.** All tests use the fake provider (and a real container for the sandbox tests).
  Docker image build downloads packages, no API cost.
- **Live smoke after the build** (3 stats questions, approval auto): about 3 × 4 calls ≈ $0.01.
- **One end-to-end run, 55 questions:** the last 51-question run cost $0.098628 (125 calls,
  ≈ $0.0008 per call). Four stats questions add about 4 × 4 = 16 calls ≈ $0.013, so about $0.11,
  with the run's `STOP_USD` 0.40 as the stop. Worst case from `DRY=1` will be shown before the run.
- Total about $0.12 to $0.15 of the $2.086327 remaining.
- **Build effort (no cost):** migration + roles; runner + image; menu models; analytics-mcp +
  log + servers-up; graph node, planner route, verifier checks; API artifact route + UI;
  golden/scorer; docs.

## 8. Build differences

- Image `rflens-analytics:v1` (`make analytics-image`): `python:3.12-slim`, numpy 2.5.4, scipy 1.18.1,
  matplotlib 3.11.2, their pinned dependencies and pytest (the in-image tests); **pandas is not in
  it** (nothing needs it): 614 MB against 716 MB for the probe image with pandas. The base image
  is pinned by tag, not by digest.
- The stats agent has no second attempt after a verifier failure (prose failures go to the
  writer); it has one correction after an analytics-mcp rejection. `params_json` is a string.
- `distribution` also returns n, missing count and quartiles (so one call answers a question
  about quartiles and missing values).
- Container tests: `make test-analytics` and the GitHub workflow run them; `make ci-local` prints
  them as NOT RUN (no Docker daemon inside its container), see `rflens.ci_local`.
- Golden stats questions are scored on the structured result and the answer text
  (`reference_stats`), with references from `analytics_runner/reference.py`, which shares no code
  with the analyses.
