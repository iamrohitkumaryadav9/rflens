# Vendor RF module datasheets: extraction design (week 5)

**Status: NOT PURSUED** (decision 2026-10-25; see [report-extraction.md](report-extraction.md)).
Kept as a record of the source survey and of a design that could be revived. Nothing was built and
no paid calls were made. Investigated 2026-10-25.

## 0. Finding that blocks the plan as written

Under the rule "no login, no browser spoofing, no mirrors", **no vendor's datasheet with per-band
RF tables could be fetched from this machine.** One file was downloaded (Telit LE910Cx product
brief, 2 pages); it has no per-band tables (see section 2). The approval decision is in section 9.

## 1. Sources (robots.txt, terms, direct fetch)

Method: Windows PowerShell `Invoke-WebRequest` with its default client (no spoofed browser), one
request per URL, 0.6 to 12 s apart. Nothing was logged into and no mirror was used.

| Vendor | robots.txt | Terms about automated download | Datasheet fetch from here | Verdict |
|---|---|---|---|---|
| u-blox | www.u-blox.com disallows `/en/docs/*`; content.u-blox.com (PDF host) does not disallow PDFs | No terms page found (6 guessed legal URLs returned 404; none invented as a source) | Product pages: 403 (SARA-R5) / 404. The official data sheet URL was not found; search only returns third-party mirrors (SparkFun, Core Electronics, alldatasheet), which are excluded | **Skip** (blocked page, no official URL) |
| Quectel | fetchable | https://www.quectel.com/terms-of-use: "the information on the Quectel website and any part thereof may not be redacted, reproduced, copied, duplicated, or used for any commercial purpose not permitted by Quectel" | Product page loads, but datasheet links go to `/login/` and `/download-zone/` | **Skip** (login, and terms restrict copying) |
| Telit Cinterion | www.telit.com: `Allow: /`, `Crawl-delay: 10`, disallows only `/wp-admin/` and some query strings | https://www.telit.com/terms-conditions/ (4,453 chars): no crawling or download clause found | Product briefs (2 pages, `wp-content/uploads/...`) fetch fine. Hardware design guides are on `dz.telit.com/file/download/<id>` linked from the public download-zone page; from here that host fails TLS trust ("Could not establish trust relationship"), robots.txt of dz.telit.com unreadable. Not bypassed | **Partial**: briefs yes (no RF tables), design guides not reachable |
| Sierra Wireless / Semtech | semtech.com fetchable; source.sierrawireless.com/robots.txt is 404 | https://www.semtech.com/legal and /terms-conditions: no crawling clause found | Spec PDF link redirects (307) to `source.sierrawireless.com/login?...` | **Skip** (login) |
| SIMCom | www.simcom.com: `Disallow: /Uploads/` (where its files live); `/bocadmin/`, `/ThinkPHP/` | No terms page found | Product pages list "Click to download" via `/login.html`, `/download.html` | **Skip** (login, and `/Uploads/` disallowed) |
| Fibocom | fetchable; allows named AI crawlers, no PDF rule | Terms URLs guessed returned the home page (SPA), so no terms found | Product page HTML contains no datasheet link (button is script-driven, likely a form) | **Skip** (no direct URL, no terms) |
| Nordic | robots.txt returns 403 on all three hosts | not read | not attempted | **Skip** (blocked) |

Bottom line: 6 of 7 vendors are skipped by your rule, Telit is partial. Telit's design guides (for
example LE910Cx hardware design guide, listed on its download zone) are the best candidate, but only
reachable if the TLS problem is on this machine's trust store, which should be checked by you (a
browser on your machine, not a download by me), not worked around.

## 2. Candidate set and the one download

Required: ~15 datasheets from at least 3 vendors. **Cannot be met with fetchable official files
today.** Verified candidates by name only (existence of the document, not a fetched file):

- Telit LE910Cx (Cat-4/Cat-1 family), ME910C1 (Cat-M/NB), LE910C1 mini PCIe, LE910Cx ThreadX:
  hardware design guides (r46 and r17 per the download zones). Not fetched.
- u-blox SARA-R5 (UBX-19016638), SARA-R52 (UBX-22038918), LARA-R6, SARA-R4: not fetched.
- Quectel EC25, BG95; SIMCom SIM7600 family; Fibocom L716, NL668; Semtech HL78xx: gated, skipped.

Downloaded (1 file, gitignored): `data/raw/datasheets/telit/TC_LE910Cx_Linux_Product_Brief.pdf`
from https://www.telit.com/wp-content/uploads/2025/07/TC_LE910Cx_Linux_Product_Brief.pdf
(1.86 MB, HTTP 200). Analysed with pdfplumber 0.11.10:

- 2 pages; both have a text layer (3,253 and 2,311 chars) plus images (logos/diagrams); not scanned.
- 1 table detected (page 2). No per-band conducted power table and no sensitivity table: the only
  RF figures are "Class 3 (0.2 W, 23 dBm) @ LTE" and a band list. The word "sensitiv" appears 0 times.
- Units: dBm only. FCC ID: not stated ("FCC/ISED" appears as a certification list only).
- Producer Adobe PDF Library 15.0, created 2017-06-01 metadata; no revision field on the brief.

So the brief is useless for the target; the per-band tables live in the design guides. Only one
vendor (not three) was downloaded because only one vendor allowed any fetch.

## 3. Schema (proposal, migration 0014 + 0015, not written)

`modules`: `module_id` PK, `vendor`, `model`, `variant` (nullable), `fcc_id` (nullable, only if the
datasheet states it), `grantee_code` (nullable; set only if the first 3 or 5 characters of
`fcc_id` exist in `fcc_grantees`), `source_url`, `source_sha256`, `datasheet_revision`,
`datasheet_date`, `fetched_at`; unique `(vendor, model, source_sha256)`.

`module_rf_specs`: `spec_id` PK, `module_id` FK, `technology` (`LTE`, `NR`, `LTE-M`, `NB-IoT`),
`band` (integer, validated against `spec_operating_bands`), `parameter` (`tx_power` |
`rx_sensitivity`), `value` numeric, `unit` (`dBm` after normalisation; raw unit kept in
`unit_raw`), `tolerance_db` (nullable), `channel_bw_mhz` (nullable), `modulation` (nullable),
`conditions` text, `page`, `table_ref`, `source_snippet` (the table row text), `check_flags` text[],
`extraction_run_id`. Datasheet text is vendor copyright: `source_snippet` is a single table row,
nothing is redistributed in the repo, and the PDFs stay under gitignored `data/`.

Pydantic model (`DatasheetRow`) mirrors these columns; `value` is `float | None` and the LLM is told to
return `null` rather than guess.

## 4. Margin against 3GPP

Sensitivity margin (dB) = reference-sensitivity requirement − datasheet sensitivity (both dBm,
negative numbers; positive = better than required). Requirement source: 36.101 Table 7.3.1-1
(by band and channel bandwidth) and the 38.101-1 REFSENS tables in clause 7.3, read from the
existing `chunks` rows. Rules:

1. Match on technology, band and channel bandwidth. A requirement needs a table cell with that
   exact (band, bandwidth); duplex-gap / MSD / CA rows are never used.
2. If the datasheet gives no bandwidth, or one the table lacks, the row gets
   `margin = NULL` and flag `no_matching_requirement` (never nearest-neighbour).
3. The matched chunk id is stored (`requirement_chunk_id`) and returned as the citation.
4. Margin is computed in SQL view `v_module_margin` (not stored), so a corrected requirement
   changes it. Table parsing for the requirement uses the same strict-parser pattern as
   `ingest/spec_bands.py`: if a table does not parse completely the import fails.

Tx power has no equal margin; it is stored and compared to the class power (23 dBm for Class 3)
only as a flag.

## 5. Extraction design

1. `pdfplumber` per page: text plus `extract_tables()`. Scanned pages (<80 chars and a page-sized
   image) are skipped and listed, not OCR'd.
2. Pages whose text mentions band tokens with `dBm` are windowed (≤ ~12k tokens) and sent through
   the gateway with the Pydantic JSON schema, temperature 0, question id = datasheet id, per-document
   call cap, budget-capped (existing ledger).
3. Code checks after the model: unit in {dBm, dB, W, mW} normalised to dBm (W/mW converted with
   arithmetic, raw kept); band must exist in `spec_operating_bands` for the technology; sensitivity
   must be −130 to −80 dBm, tx power −10 to 35 dBm; band may not repeat for the same
   (parameter, bandwidth). Failures set `check_flags`, they never overwrite the value.
4. Each row's `source_snippet` must appear in the page text (string match after whitespace
   normalisation); otherwise flag `snippet_not_found` and drop the row from the margin view.

## 6. Accuracy plan (blind spot-check)

- Sample 35 extracted values with a fixed seed (stratified: at least 8 per vendor layout, at least
  10 sensitivity and 10 tx power).
- The checker sees `vendor, model, page, table_ref, band` and the PDF page, not the extracted value.
  They type the value they read; a script compares (exact for dBm to 0.1).
- Time: ~30 s per value, ~20 to 30 min total.
- Report field-level accuracy (value, unit, band, bandwidth) per vendor layout, with n; with n≈8 per
  layout a single error moves the figure 12 points, so say so in the report. Gate: no vendor used in
  an answer below 90% value accuracy.

## 7. Agent wiring

- Migration: tables `modules`, `module_rf_specs`, view `v_module_margin`, column and table
  comments (units, "datasheet typical vs guaranteed unknown unless stated", 3GPP citation), grants
  to `rflens_sql_reader`.
- `sql_guard.py`: add the two tables and the view to the allow-list (tests assert counts: 10→12
  tables, 3→4 views; update those assertions in the same change).
- Golden questions (human review required before they count):
  - SQL: "Which modules in the database list a band 4 LTE receiver sensitivity better than −100 dBm
    at 10 MHz?" (reference SQL on `module_rf_specs`).
  - Mixed: "For module X, by how many dB does its LTE band 4 10 MHz sensitivity beat the 3GPP
    reference sensitivity?" (SQL for the datasheet value + docs for Table 7.3.1-1 citation).
  Reference results are computed only after extraction and verified by the blind check.

## 8. Cost estimate (paid; not spent)

Per datasheet ~100 pages; RF pages ~6 to 10, windowed into ~3 calls of ~12k input tokens and ~2k
output. At `gemini-3.1-flash-lite` prices in `config/llm_prices.yaml` the worst case is about
3 × (12k in + 2k out) per datasheet; for 15 datasheets ≈ 540k input + 90k output tokens, well under
$0.25 on the existing price table (to be recomputed with `DRY=1` before any run). Budget remaining:
$2.086327.

## 9. Decision needed

Options: (a) you check in your browser whether the Telit design-guide download works and tell me
the file ids/URLs; I then fetch only those (Telit's robots.txt allows it, 10 s delay); (b) you hand-
download datasheets from any vendors into `data/raw/datasheets/<vendor>/` and I do the rest; (c)
drop vendor datasheets and go back to the FCC reports by hand. Option (a)+(b) both fit this design.

## FCC test reports (superseded target)

FCC exhibits cannot be fetched automatically (fcc.gov returned 403 to default clients). No
test-report extraction code exists; only the grantee and test-firm loaders (`ingest/fcc_eas.py`)
were built, and they stay. The earlier test-report design applies only to reports someone
downloads by hand.
