# Report extraction: considered, not pursued

Decision 2026-10-25: RFLens does not extract values from FCC test reports or vendor datasheets.
Nothing was built for it (no code, tables or dependencies); only the grantee and test-firm loaders
(`ingest/fcc_eas.py`), which have other uses, remain.

Reasons:

1. **FCC blocks automated access.** fcc.gov and the equipment-authorization pages answered the
   default HTTP clients with 403. We do not spoof a browser.
2. **Manual downloads are not a pipeline.** Fetching each exhibit by hand does not scale and cannot
   be reproduced from the repository.
3. **Many reports are plot-only.** Spectrum masks and emissions appear as images, with no text or
   table to extract; OCR of plots would be guessing.
4. **Some PDFs disallow content copying.** We do not work around such restrictions.
5. **Vendor datasheets need a login** (Quectel, Sierra Wireless/Semtech, SIMCom), block automated
   access (u-blox product pages, Nordic robots.txt) or were unreachable (Telit design guides on a
   host that failed TLS). The one public file fetched, a Telit product brief, has no per-band RF
   tables. Survey and design: [datasheets.md](datasheets.md) (not pursued).

The document-to-SQL piece of the project is the 3GPP operating-band extraction
(`spec_operating_bands`, migration 0013, with a citation to the source chunk for every row).

## Future work

- Report or datasheet value extraction (design in `datasheets.md`) if a legitimate machine-readable
  source appears, for example a vendor API or a licensed dataset.
