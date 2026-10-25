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

## Retrieval runs (make eval-docs)

| date (UTC) | run_id | commit | questions | vector (no filter / spec filter) | keyword (no filter / spec filter) | hybrid (no filter / spec filter) | note |
|---|---|---|---|---|---|---|---|
| 2026-10-25 02:23 | eval-docs-6be0f708 | bf74e19+dirty | human doc questions (10) | 8/10 / 9/10 | 0/10 / 2/10 | 6/10 / 7/10 | 132 searches, free. First run: 10 human (former smoke set) and 12 assistant drafts; k=5; dev corpus fully embedded. |
| 2026-10-25 02:23 | eval-docs-6be0f708 | bf74e19+dirty | assistant doc questions (12) | 5/12 / 7/12 | 0/12 / 0/12 | 2/12 / 3/12 | 132 searches, free. First run: 10 human (former smoke set) and 12 assistant drafts; k=5; dev corpus fully embedded. |
