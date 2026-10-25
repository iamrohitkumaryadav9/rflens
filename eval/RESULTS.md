# Eval results

Rows are appended automatically, newest last (`NOTE=` / `--note` fills the last column). The golden
set grew between runs, so compare rows through their question counts. "commit" is HEAD when the run
started, "+dirty" if the working tree had uncommitted changes.

<!-- golden-counts:start (generated: make results-header) -->
**Golden set** (eval/golden.yaml): 51 questions. author = who wrote the question (human: edited and reviewed by the project owner).

| type | human | assistant | total | draft | reviewed |
|---|---|---|---|---|---|
| sql | 17 | 2 | 19 | 0 | 19 |
| doc | 10 | 12 | 22 | 0 | 22 |
| mixed | 0 | 4 | 4 | 0 | 4 |
| refuse | 3 | 3 | 6 | 0 | 6 |
| total | 30 | 21 | 51 | 0 | 51 |
<!-- golden-counts:end -->

## SQL agent runs (make eval-sql)

| date (UTC) | run_id | commit | model / tier | golden set | execution accuracy | refusal correctness | calls | cost | note |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-24 17:18 | eval-sql-9cf4bce9 | 1db62b1+dirty (recorded in c853514) | gemini-3.1-flash-lite / standard | 17 (14 sql + 3 refuse) | 14/14 | 3/3 | 18 | $0.024675 | First run. BEFORE the lease fix: references counted spectrum leases as licences (sql-05 3,861, sql-06 AST & Science 1,439, sql-10 28 were later found to include leases), so this 14/14 overstates. |
| 2026-10-24 17:39 | eval-sql-6d590871 | c853514+dirty (recorded in 7a4a833) | gemini-3.1-flash-lite / standard | 18 (15 sql + 3 refuse) | 14/15 | 3/3 | 18 | $0.026761 | After the lease fix (is_lease; sql-05/06/10 redefined, sql-15 added). Miss sql-06: grouped by licensee_frn AND licensee_name, which splits one FRN into several spellings (Cellco 418 vs 422). |
| 2026-10-25 02:09 | eval-sql-861790a2 | 7a4a833+dirty | gemini-3.1-flash-lite / standard | 20 (17 sql + 3 refuse) | 17/17 | 3/3 | 21 | $0.031850 | After FRN guidance in column comments (migration 0011). sql-06 motivated the fix; sql-16 and sql-17 are held-out. Generation cap 700 tokens to keep the worst case under the 0.25 stop. |
| 2026-10-25 09:50 | eval-sql-fd8d24a7 | 8e77472+dirty | gemini-3.1-flash-lite / standard | 19 (19 sql + 0 refuse) | 19/19 | 0/0 | 20 | $0.038069 | variance check, temperature 0 run 1/2, temperature 0.0 |
| 2026-10-25 09:52 | eval-sql-732227ac | 8e77472+dirty | gemini-3.1-flash-lite / standard | 12 (12 sql + 0 refuse) | 12/12 | 0/0 | 12 | $0.022538 | variance check, temperature 0 run 2/2, temperature 0.0 STOPPED EARLY at cost stop $0.06 reached before sql-13 |

## Retrieval runs (make eval-docs)

| date (UTC) | run_id | commit | questions | vector (no filter / spec filter) | keyword (no filter / spec filter) | hybrid (no filter / spec filter) | note |
|---|---|---|---|---|---|---|---|
| 2026-10-25 02:23 | eval-docs-6be0f708 | bf74e19+dirty | human doc questions (10) | 8/10 / 9/10 | 0/10 / 2/10 | 6/10 / 7/10 | 132 searches, free. First run: 10 human (former smoke set) and 12 assistant drafts; k=5; dev corpus fully embedded. |
| 2026-10-25 02:23 | eval-docs-6be0f708 | bf74e19+dirty | assistant doc questions (12) | 5/12 / 7/12 | 0/12 / 0/12 | 2/12 / 3/12 | 132 searches, free. First run: 10 human (former smoke set) and 12 assistant drafts; k=5; dev corpus fully embedded. |
| 2026-10-25 09:49 | eval-docs-142b06d0 | 8e77472+dirty | human doc questions (10) | 9/10 / 9/10 | 0/10 / 2/10 | 7/10 / 9/10 | 132 searches, free. improvement round: clause lookup and variant down-rank on; doc-03/04 reworded to NR |
| 2026-10-25 09:49 | eval-docs-142b06d0 | 8e77472+dirty | assistant doc questions (12) | 8/12 / 8/12 | 1/12 / 1/12 | 6/12 / 7/12 | 132 searches, free. improvement round: clause lookup and variant down-rank on; doc-03/04 reworded to NR |

## End-to-end runs (make eval-e2e)

| date (UTC) | run_id | commit | model / tier | questions run | sql | doc (answer / evidence recall) | mixed | refuse | human | assistant | calls | cost | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-25 08:42 | eval-e2e-613631e1 | 9d15098+dirty | gemini-3.1-flash-lite / standard | 19/51 | 14/19 | 0/0 / 0/0 | 0/0 | 0/0 | 13/17 | 1/2 | 58 | $0.044240 | first full end-to-end run, approval auto, diagnosis only STOPPED EARLY: doc-01: LoopDetected: run eval-e2e-613631e1 stopped: the same prompt (hash 828d9e2f3abc) was sent 3 times |
| 2026-10-25 09:09 | eval-e2e-fc0ac320 | 9d15098+dirty | gemini-3.1-flash-lite / standard | 51/51 | 15/19 | 13/22 / 15/22 | 3/4 | 6/6 | 23/30 | 14/21 | 132 | $0.088257 | second full run: after writer/loop fixes and scorer v2; baseline for sql-12 and sql-14 unchanged |
| 2026-10-25 09:59 | eval-e2e-559aacfa | 8e77472+dirty | gemini-3.1-flash-lite / standard | 51/51 | 17/19 | 18/22 / 19/22 | 3/4 | 6/6 | 28/30 | 16/21 | 125 | $0.098628 | improvement round: temperature 0, clause lookup, variant down-rank, writer depth 5, NULL keys, missing_band_citation; doc-03/04 reworded to NR |

## Scorer changes, rescores and open agent errors

**End-to-end scorer v2 (2026-10-25).** The first end-to-end run (eval-e2e-613631e1, stopped at
doc-01, 19 sql questions scored) was scored with scorer v1. Two general rules changed (the row of
that run above is kept as it was recorded):

- Rule 5: for a ranked question the order of ROWS is checked, never the order of values within one
  row (sql-07 answered "KNKN259, 1564 records ... licensee UNION TELEPHONE COMPANY").
- Rule 6: rows tied on the ranking value form an unordered group; order is checked between groups,
  not within them (sql-03 listed four features tied at 1429, then four tied at 1341).
- Unchanged: `must_mention` stays strict (sql-19 paraphrased "2021" and stays a miss).

Rescoring the 19 saved threads of that run with scorer v2 (no LLM call, `eval-e2e --rescore`):
old 14/19, new 16/19. sql-03 and sql-07 changed from miss to correct; no other question changed.

**Writer and loop fixes before the second run.** The first run stopped because a long answer was
cut off at the writer's 700-token output limit, the empty draft produced a retry prompt identical to
the first, and the third identical prompt stopped the whole run. Fixed: a MAX_TOKENS finish or an
unparseable output is the failure type `writer_truncated` (the retry says to shorten the answer);
`ORCH_WRITER_MAX_TOKENS` is 1500; every writer retry carries the failure feedback and its retry
number, so no retry prompt repeats an earlier one; the raw output of each writer call is kept in
`draft_history`; a repeated prompt or error within one question fails that question
(`loop_detected`) and the run goes on, while the run caps and the budget still stop the run.

**Open agent errors** (genuine misses, deliberately not fixed so that they stay in the baseline):

- sql-12: "How many grantees and how many accredited test firms are in Japan?" The agent filtered
  `fcc_grantees.country = 'JAPAN'` (upper case) and answered 0 grantees; the data stores the
  country in another case (reference: 1,093 grantees, 71 test firms).
- sql-14: "When was the ULS cellular snapshot taken, and under what licence is the data published?"
  The planner routed it to refuse ("the system does not hold information about its own versioning"),
  although `data_sources` holds both facts (reference: 2026-10-18, US Government Work).

**Second full run (eval-e2e-fc0ac320, scorer v2, writer and loop fixes in place).** 37/51 correct:
sql 15/19, doc 13/22 (evidence recall: search results 15/22, chunks the writer saw 14/22), mixed
3/4, refuse 6/6; human 23/30, assistant 14/21. The agent is not deterministic: sql-12 (an open
error above) answered correctly this time, while sql-13 (the placeholder country 'Unknown' was not
excluded) and sql-17 (the NULL-FRN group counted as a licensee) were wrong although they were
correct in the first run. sql-14 and sql-19 missed again. One run per question: differences of one
or two questions between runs are within this variance.

**Improvement round (eval-e2e-559aacfa, temperature 0).** 44/51 correct against 37/51 in the
baseline run eval-e2e-fc0ac320 (+7): sql 17/19, doc 18/22, mixed 3/4, refuse 6/6; human 23/30 ->
28/30 (+5), assistant 14/21 -> 16/21 (+2). Doc evidence recall: search results 15/22 -> 19/22,
chunks the writer saw 14/22 -> 19/22. Per question: 9 fixed (sql-13, sql-17, doc-03, doc-04, doc-05,
doc-06, doc-18, doc-20, doc-21), 2 broke (doc-01, doc-19), 40 unchanged (35 correct in both; wrong
in both: sql-14, sql-19, doc-12, doc-22, mixed-04).

Changes kept: temperature 0 for eval runs (the SQL agent gave identical SQL and results for 12 of 12
questions in two runs); the variant-clause down-rank (eval-docs vector recall@5 without / with the
spec filter: human 8/10, 9/10 -> 9/10, 9/10; assistant 5/12, 7/12 -> 8/12, 8/12); clause lookup
(assistant +1 without the filter, human unchanged: no human question names a clause; kept on the
reading "no group worse, one better"); writer depth 5 (offline, base configuration, vector with the
spec filter: recall@3 -> @5 human 8/10 -> 9/10, assistant 4/12 -> 7/12; no extra gain once the
variant rule is on); NULL-key comments and prompt rule; `missing_band_citation`; doc-03 and doc-04
reworded to NR (no change in eval-docs recall; doc-04 still misses there). The end-to-end run
cannot separate the changes: it measures them together, with one run per question, and every change
was designed from baseline misses, so neither author group is untouched.

Misses after the round (7): sql-14 (open agent error: the planner refuses a provenance question),
sql-19 and mixed-04 (the answer omits a fact `must_mention` requires: the 2021 cutoff, the ERP unit
being unknown), doc-12 (the planner's spec hint is TS 36.101 for an NR identifier), doc-22 (retrieval:
the adjacent-channel clause 7.5.1 is not in the top 5), doc-19 (broke: the planner wrote "UE minimum
output power" where it wrote "minimum output power" before, and clause 6.3.2.1 left the top 5), doc-01
(broke: the writer's list of NR bands was cut off at 1500 tokens twice and redacted to `partial`).

