# Resume bullet options for RFLens

Every number below is in [eval/RESULTS.md](../eval/RESULTS.md), in the repository, or was produced by a
command on 2026-10-27 (pytest collection counts, database counts, the spend ledger). Evaluation
figures are single runs per question at temperature 0 on `gemini-3.1-flash-lite`; keep the
denominator and "single run" if you shorten a bullet. Pick 4 or 5; do not mix a 55-question total with
a 51-question total in the same line.

## Option set A: what it is and how well it did

- Built RFLens, an agentic Q&A system over public RF data (FCC licences, 3GPP specifications, the
  UCI SECOM dataset): a LangGraph orchestrator (planner, SQL, document and statistics agents, writer,
  verifier) that reaches data only through three MCP servers over Postgres with pgvector.
- Built a 55-question golden set (30 owner-written, 25 AI-drafted, all owner-reviewed) and an
  end-to-end evaluation harness; the final run answered 48 of 55 correctly (single run, temperature 0),
  up from 37 of 51 at the first full baseline.

## Option set B: evaluation discipline (use if you want to show honesty about the numbers)

- Raised the end-to-end score on the original 51 questions from 37/51 to 44/51 through a diagnosed
  improvement round (temperature 0, clause lookup, variant-clause down-ranking), then added 4 statistics
  questions: the latest run scores 48/55, and equals 44/51 on the original 51 (single runs; totals
  moved between 41 and 44 on those 51 across four later runs while prompts changed).
- Documented every eval run, scorer change and remaining miss in a results log: 7 of 55 questions still
  miss in the last run (2 answer-wording misses, 1 routing refusal, 4 retrieval misses), each with a
  recorded cause.

## Option set C: engineering for safe, bounded model use

- Put every model call behind one gateway with a hard spend ledger (budget cap $2.50 over recorded call
  costs, worst-case reservation before each paid call), call caps per run, question and minute, bounded
  retries and repeated-prompt detection; a full 55-question evaluation run cost $0.100019
  ($0.001819 per question, 135 calls).
- Restricted data access with seven narrow Postgres roles and a SQL guard (single SELECT over an
  allow-list, row, cost and time caps, every call logged), covered by 113 SQL-guard tests and 59 sql-mcp
  tests.
- Wrote a deterministic verifier (no model call) that traces each number, unit, quotation and citation
  of an answer to a SQL cell, a cited specification chunk or an analysis result, covered by 78 tests
  (39 for the SQL and document rules, 39 for the statistics rules).

## Option set D: data engineering

- Loaded and modelled four public sources behind 15 Alembic migrations: 236,958 FCC ULS frequency rows
  and 5,464 licence/lease header rows, 50,153 equipment grantees, 1,567 SECOM runs x 590 sensor
  features (924,530 measurements), and 6,861 chunks of two 3GPP specifications embedded for retrieval.
- Found that 3,174 of the 5,464 ULS header rows are spectrum leases rather than licences, and parsed the
  specifications' operating-band tables into 141 rows that the SQL joins and the answer cites, instead
  of typing band edges.

## Option set E: statistics without model-written code

- Designed a fixed menu of nine parameterized statistical analyses (Mann-Whitney tests, effect sizes,
  Benjamini-Hochberg q-values, histograms, correlations, outliers, plots) that run in a Docker container
  with no network, a read-only root file system and dropped capabilities; the model only chooses an
  analysis and its parameters. Reference values for the golden stats questions are computed by an
  independent script that shares no code with the analyses.
- Tested the project with 970 pytest tests, 21 tests inside the sandbox image and 10 container
  isolation tests that run against real Docker containers.

## Source of each number

| number | source |
|---|---|
| 48/55, 44/51, 37/51, 46/55, 44/55, 41 to 44 on the original 51 | eval/RESULTS.md, "End-to-end runs" and "Week 5 close-out" |
| 55 questions; 30 human / 25 assistant; all reviewed | eval/RESULTS.md, golden-counts block |
| 7 remaining misses and their causes | eval/RESULTS.md, "Week 5 close-out" |
| $0.100019, 135 calls, $0.001819 per question | eval/RESULTS.md run eval-e2e-d7222d07; $0.100019 / 55 |
| $2.50 cap; call caps; retries | `LlmSettings` in src/rflens/config.py; CLAUDE.md |
| 113 SQL-guard tests; 59 sql-mcp tests | `pytest tests/test_sql_guard.py --collect-only`; `tests/test_sql_mcp.py` |
| 39 + 39 verifier tests | `pytest --collect-only` of tests/test_graph_verifier.py and tests/test_stats_verifier.py |
| 970 / 21 / 10 tests | `pytest tests --collect-only`; the same inside the image; `pytest tests_docker --collect-only` |
| seven roles | `pg_roles` rows named rflens_* in the development database |
| 15 migrations | files in migrations/versions |
| row counts, 6,861 chunks (4,359 + 2,502), 141 band rows | development database counts |
| 3,174 of 5,464 leases | CLAUDE.md open question (j) |
| nine analyses | `ANALYSES` in src/rflens/analytics/menu.py |
