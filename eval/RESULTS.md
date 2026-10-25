# Eval results

One row per `make eval-sql` run, newest last; rows are appended automatically (`--note` / `NOTE=`
adds the last column). Execution accuracy and refusal correctness are matches over the questions
of that run; the golden set changed between runs, so compare rows with their "golden set" column.
"commit" is HEAD when the run started, "+dirty" if the working tree had uncommitted changes.

| date (UTC) | run_id | commit | model / tier | golden set | execution accuracy | refusal correctness | calls | cost | note |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-24 17:18 | eval-sql-9cf4bce9 | 1db62b1+dirty (recorded in c853514) | gemini-3.1-flash-lite / standard | 17 (14 sql + 3 refuse) | 14/14 | 3/3 | 18 | $0.024675 | First run. BEFORE the lease fix: references counted spectrum leases as licences (sql-05 3,861, sql-06 AST & Science 1,439, sql-10 28 were later found to include leases), so this 14/14 overstates. |
| 2026-10-24 17:39 | eval-sql-6d590871 | c853514+dirty (recorded in 7a4a833) | gemini-3.1-flash-lite / standard | 18 (15 sql + 3 refuse) | 14/15 | 3/3 | 18 | $0.026761 | After the lease fix (is_lease; sql-05/06/10 redefined, sql-15 added). Miss sql-06: grouped by licensee_frn AND licensee_name, which splits one FRN into several spellings (Cellco 418 vs 422). |
| 2026-10-25 02:09 | eval-sql-861790a2 | 7a4a833+dirty | gemini-3.1-flash-lite / standard | 20 (17 sql + 3 refuse) | 17/17 | 3/3 | 21 | $0.031850 | After FRN guidance in column comments (migration 0011). sql-06 motivated the fix; sql-16 and sql-17 are held-out. Generation cap 700 tokens to keep the worst case under the 0.25 stop. |
