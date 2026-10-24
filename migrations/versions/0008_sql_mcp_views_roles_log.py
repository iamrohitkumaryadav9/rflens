"""sql-mcp: curated views, column comments, two database roles and the query log

Views (v_uls_licences, v_uls_frequencies, v_secom_values) and COMMENT ON for every allow-listed
table, view and column: describe_table returns these comments to the text-to-SQL agent.

rflens_sql_reader: login role that can SELECT from the allow-listed tables and views only. It has
NO access to uls_en (contact persons' names; reachable only through v_uls_licences, licensee entity
only), chunks, documents, alembic_version or sql_query_log, and cannot write. Defence in depth: its
sessions default to read-only transactions and have a statement timeout.

rflens_sql_logger: login role with INSERT on sql_query_log and nothing else. The server writes
every call (accepted or rejected) to sql_query_log through this role, so the reader role cannot
read or alter the log.

Passwords come from SQL_READER_PASSWORD and SQL_LOGGER_PASSWORD (environment or .env), passed as
bind parameters of set_config() because a DO block body cannot take bind parameters, and quoted by
format('%L') inside the database. To rotate: ALTER ROLE <role> PASSWORD '...'.

Revision ID: 0008
Revises: 0007
Create Date: 2026-10-24
"""

import sqlalchemy as sa
from alembic import op

from rflens.config import get_sql_mcp_settings

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None

READER_TABLES = (
    "secom_runs, secom_measurements, uls_hd, uls_fr, uls_lo, uls_file_counts, "
    "fcc_grantees, fcc_test_firms, data_sources, "
    "v_uls_licences, v_uls_frequencies, v_secom_values"
)
COMMENTED_RELATIONS = (
    "data_sources", "secom_runs", "secom_measurements", "uls_hd", "uls_en", "uls_fr", "uls_lo",
    "uls_file_counts", "fcc_grantees", "fcc_test_firms",
)  # fmt: skip
VIEWS = ("v_uls_frequencies", "v_uls_licences", "v_secom_values")

# ruff: noqa: E501
# (the COMMENT ON strings below are text for describe_table; they are not wrapped)

# Views first (they read uls_en as their owner), then comments. Single source for both.
VIEWS_AND_COMMENTS = r"""
-- ---------------------------------------------------------------------------------------------
-- Part 1: views
-- ---------------------------------------------------------------------------------------------

-- One row per licence with its licensee. uls_en is unique on (service, usi, entity_type), so the
-- 'L' join cannot multiply rows; LEFT JOIN keeps licences that have no 'L' row.
CREATE VIEW v_uls_licences AS
SELECT h.service,
       h.usi,
       h.call_sign,
       h.license_status,
       h.radio_service_code,
       h.grant_date,
       h.expired_date,
       h.cancellation_date,
       h.effective_date,
       h.last_action_date,
       h.auction_id,
       e.entity_name AS licensee_name,
       e.licensee_id,
       e.frn         AS licensee_frn,
       e.city        AS licensee_city,
       e.state       AS licensee_state
FROM uls_hd h
LEFT JOIN uls_en e
       ON e.service = h.service AND e.usi = h.usi AND e.entity_type = 'L';

-- Frequency rows with their licence: call sign, status, licensee. Every uls_fr row has a licence
-- (foreign key), so the inner join keeps all frequency rows.
CREATE VIEW v_uls_frequencies AS
SELECT f.service,
       f.usi,
       l.call_sign,
       l.license_status,
       l.licensee_name,
       f.location_number,
       f.antenna_number,
       f.freq_seq_id,
       f.class_station_code,
       f.frequency_assigned,
       f.frequency_upper_band,
       f.power_output,
       f.power_erp,
       f.eirp
FROM uls_fr f
JOIN v_uls_licences l ON l.service = f.service AND l.usi = f.usi;

-- SECOM readings with their run's label and time: "feature X across failing runs" without
-- writing the join. One row per (run, feature); value is NULL where the source had NaN.
CREATE VIEW v_secom_values AS
SELECT r.run_id,
       r.run_ts,
       r.label,
       m.feature_idx,
       m.value
FROM secom_measurements m
JOIN secom_runs r ON r.run_id = m.run_id;

-- ---------------------------------------------------------------------------------------------
-- Part 2: comments
-- ---------------------------------------------------------------------------------------------

-- data_sources ----------------------------------------------------------------------------------
COMMENT ON TABLE data_sources IS 'Data card: one row per loaded source (licence, URL, access date, size, notes).';
COMMENT ON COLUMN data_sources.id IS 'Surrogate key.';
COMMENT ON COLUMN data_sources.name IS 'Source name, unique: uci_secom, uls_cell, fcc_eas_grantees, fcc_eas_test_firms.';
COMMENT ON COLUMN data_sources.licence IS 'Licence of the source data as recorded when it was loaded.';
COMMENT ON COLUMN data_sources.url IS 'Where the data comes from (dataset page or download URL).';
COMMENT ON COLUMN data_sources.access_date IS 'Date the data was downloaded and loaded.';
COMMENT ON COLUMN data_sources.row_count IS 'Count of the source''s primary entity, not of all rows: runs for uci_secom, licences (uls_hd rows) for uls_cell, grantees / test firms for fcc_eas_*.';
COMMENT ON COLUMN data_sources.notes IS 'Provenance notes: citation, snapshot date and hash, known data problems.';
COMMENT ON COLUMN data_sources.created_at IS 'When the row was first created.';

-- SECOM (UCI id 179: a semiconductor fab, not RF) -----------------------------------------------
COMMENT ON TABLE secom_runs IS 'UCI SECOM production runs (semiconductor manufacturing, not RF): 1,567 runs, 104 fails. Join to secom_measurements on run_id, or use v_secom_values.';
COMMENT ON COLUMN secom_runs.run_id IS '1-based row order in the SECOM file; the source has no run id.';
COMMENT ON COLUMN secom_runs.label IS '''pass'' or ''fail''. Source encoding: class -1 = pass (1,463 runs), 1 = fail (104 runs).';
COMMENT ON COLUMN secom_runs.run_ts IS 'Test timestamp, no time zone (2008-07-19 to 2008-10-17). The source mixes two formats, parsed day-first; 33 timestamps repeat, so it is not a key.';
COMMENT ON TABLE secom_measurements IS 'SECOM sensor readings in long format: one row per (run, feature), 924,530 rows (1,567 runs x 590 features).';
COMMENT ON COLUMN secom_measurements.run_id IS 'References secom_runs.run_id.';
COMMENT ON COLUMN secom_measurements.feature_idx IS 'Sensor number 1..590 (source column ''Attribute N''). Sensors are anonymised: no physical name or unit is known.';
COMMENT ON COLUMN secom_measurements.value IS 'Sensor reading, unit unknown. NULL where the source has NaN (41,951 of 924,530 cells); the source metadata wrongly says there are no missing values.';

-- FCC ULS (service: cell = Cellular, complete/l_cell.zip, snapshot 2026-10-18) --------------------
COMMENT ON TABLE uls_hd IS 'FCC ULS licence header (HD record): one row per licence. Loaded service: cell (Cellular), 5,464 licences. Key (service, usi).';
COMMENT ON COLUMN uls_hd.service IS 'ULS service file the row came from; ''cell'' = Cellular. Part of every ULS key: always join on (service, usi).';
COMMENT ON COLUMN uls_hd.usi IS 'FCC unique system identifier of the licence. Joins uls_en, uls_fr and uls_lo together with service.';
COMMENT ON COLUMN uls_hd.line_no IS '1-based line number in the source HD.dat file (row provenance).';
COMMENT ON COLUMN uls_hd.call_sign IS 'Licence call sign; unique within the cell service.';
COMMENT ON COLUMN uls_hd.license_status IS 'One-letter status code: A 3,861, C 1,122, E 371, T 110. The FCC layout document does not define the codes; meanings are unconfirmed (open question b).';
COMMENT ON COLUMN uls_hd.radio_service_code IS 'FCC radio service code; ''CL'' for every row of the cell service.';
COMMENT ON COLUMN uls_hd.grant_date IS 'Date the licence was granted. NULL for 206 licences.';
COMMENT ON COLUMN uls_hd.expired_date IS 'Licence expiration date. NULL for 206 licences.';
COMMENT ON COLUMN uls_hd.cancellation_date IS 'Date the licence was cancelled; NULL on 4,034 of 5,464 licences.';
COMMENT ON COLUMN uls_hd.effective_date IS 'Effective date of the licence record. NULL for 206 licences.';
COMMENT ON COLUMN uls_hd.last_action_date IS 'Date of the last action on the licence (present for every row).';
COMMENT ON COLUMN uls_hd.auction_id IS 'FCC auction identifier; NULL for all but 11 licences.';

COMMENT ON TABLE uls_en IS 'FCC ULS entities (EN record): up to three rows per licence (entity types L, CL, O). Key (service, usi, entity_type). Phone, email and street addresses are not loaded.';
COMMENT ON COLUMN uls_en.service IS 'ULS service file the row came from (''cell'').';
COMMENT ON COLUMN uls_en.usi IS 'FCC unique system identifier; references uls_hd (service, usi).';
COMMENT ON COLUMN uls_en.line_no IS '1-based line number in the source EN.dat file.';
COMMENT ON COLUMN uls_en.entity_type IS 'Entity role code: L (5,464 rows, exactly one per licence), CL (5,241), O (3,287). The layout document does not define them; meanings are unconfirmed (open question b). v_uls_licences uses ''L'' as the licensee.';
COMMENT ON COLUMN uls_en.licensee_id IS 'FCC licensee identifier; NULL on 5,476 of 13,992 rows.';
COMMENT ON COLUMN uls_en.entity_name IS 'Company or organisation name. NULL for one row.';
COMMENT ON COLUMN uls_en.first_name IS 'First name when the entity is an individual (4,628 rows have a first and last name).';
COMMENT ON COLUMN uls_en.last_name IS 'Last name when the entity is an individual.';
COMMENT ON COLUMN uls_en.city IS 'Entity city, as entered in the source.';
COMMENT ON COLUMN uls_en.state IS 'Entity state (two letters).';
COMMENT ON COLUMN uls_en.zip_code IS 'Entity ZIP code (5 or 9 digits, as text).';
COMMENT ON COLUMN uls_en.frn IS 'FCC registration number of the entity; NULL on 8,818 of 13,992 rows.';

COMMENT ON TABLE uls_fr IS 'FCC ULS frequency records (FR record), one row per licence / location / antenna / frequency, 236,958 rows. Not unique on its natural key (usi, location, antenna, freq_seq_id): two groups repeat for usi 13401 (open question e); the primary key is (service, line_no). Units are NOT stated in the FCC layout document (open question a).';
COMMENT ON COLUMN uls_fr.service IS 'ULS service file the row came from (''cell'').';
COMMENT ON COLUMN uls_fr.line_no IS '1-based line number in the source FR.dat file; with service, the primary key.';
COMMENT ON COLUMN uls_fr.usi IS 'FCC unique system identifier; references uls_hd (service, usi).';
COMMENT ON COLUMN uls_fr.location_number IS 'Location number; joins uls_lo.location_number (with service, usi).';
COMMENT ON COLUMN uls_fr.antenna_number IS 'Antenna number at the location.';
COMMENT ON COLUMN uls_fr.freq_seq_id IS 'Frequency sequence number within the licence (the layout document calls it Frequency Number).';
COMMENT ON COLUMN uls_fr.class_station_code IS 'Class of station code; empty on all but 32 rows of the cell service. Meanings unconfirmed (open question b).';
COMMENT ON COLUMN uls_fr.frequency_assigned IS 'Assigned frequency (lower edge). Unit not stated by the layout document; ASSUMED MHz (cell values run 824-892). NULL on 30 rows.';
COMMENT ON COLUMN uls_fr.frequency_upper_band IS 'Upper edge of the assigned band, same ASSUMED unit as frequency_assigned (MHz). NULL on 30 rows.';
COMMENT ON COLUMN uls_fr.power_output IS 'Power output. Unit UNKNOWN (open question a). Filled in one row only.';
COMMENT ON COLUMN uls_fr.power_erp IS 'Effective radiated power. Unit UNKNOWN (open question a). 380 rows are <= 0 (120 are 0, the rest negative down to -1035) and are loaded as in the source (open question c).';
COMMENT ON COLUMN uls_fr.eirp IS 'EIRP. Unit UNKNOWN (open question a). Empty in every row of the cell service.';

COMMENT ON TABLE uls_lo IS 'FCC ULS locations (LO record), 25,145 rows. Key (service, usi, location_number). Street addresses are not loaded. Some coordinates are suspect (open question d).';
COMMENT ON COLUMN uls_lo.service IS 'ULS service file the row came from (''cell'').';
COMMENT ON COLUMN uls_lo.usi IS 'FCC unique system identifier; references uls_hd (service, usi).';
COMMENT ON COLUMN uls_lo.line_no IS '1-based line number in the source LO.dat file.';
COMMENT ON COLUMN uls_lo.location_number IS 'Location number within the licence; uls_fr.location_number points here.';
COMMENT ON COLUMN uls_lo.location_type_code IS 'Location type code; ''F'' for every cell row. Meaning unconfirmed.';
COMMENT ON COLUMN uls_lo.location_class_code IS 'Location class code; ''T'' for every cell row. Meaning unconfirmed.';
COMMENT ON COLUMN uls_lo.location_city IS 'City of the site, as entered in the source.';
COMMENT ON COLUMN uls_lo.location_county IS 'County, borough or parish of the site.';
COMMENT ON COLUMN uls_lo.location_state IS 'State or territory (two letters).';
COMMENT ON COLUMN uls_lo.ground_elevation IS 'Ground elevation at the site. Unit not stated by the layout document.';
COMMENT ON COLUMN uls_lo.latitude IS 'Latitude in signed decimal degrees (north positive), derived from the source degrees/minutes/seconds. NULL on 2 rows with unusable source values (usi 12361 location 33: minutes 73; usi 13056 location 56: seconds empty). Open question d.';
COMMENT ON COLUMN uls_lo.longitude IS 'Longitude in signed decimal degrees (west negative), derived from degrees/minutes/seconds. Suspect, stored as in the source: usi 12510 location 55 (-3.94, a leading digit looks lost) and usi 11695 location 1 (+170.69, the direction letter looks wrong). Open question d.';
COMMENT ON COLUMN uls_lo.tower_registration_number IS 'FCC antenna structure registration number; NULL on about 40% of rows.';
COMMENT ON COLUMN uls_lo.height_of_support_structure IS 'Height of the support structure. Unit not stated by the layout document.';
COMMENT ON COLUMN uls_lo.overall_height_of_structure IS 'Overall height of the structure including appurtenances. Unit not stated by the layout document.';
COMMENT ON COLUMN uls_lo.structure_type IS 'Structure type text (for example TOWER, MAST), as in the source.';
COMMENT ON COLUMN uls_lo.location_name IS 'Site name, as entered in the source.';

COMMENT ON TABLE uls_file_counts IS 'Provenance of each ULS load: per service and record type, the source line count, zip SHA-256 and snapshot date.';
COMMENT ON COLUMN uls_file_counts.service IS 'ULS service (''cell'').';
COMMENT ON COLUMN uls_file_counts.record_type IS 'ULS record type loaded: HD, EN, FR or LO.';
COMMENT ON COLUMN uls_file_counts.line_count IS 'Number of lines in the source .dat file (equals the rows loaded).';
COMMENT ON COLUMN uls_file_counts.loaded_at IS 'When the file was loaded.';
COMMENT ON COLUMN uls_file_counts.sha256 IS 'SHA-256 of the source zip.';
COMMENT ON COLUMN uls_file_counts.snapshot_date IS 'Creation date of the FCC weekly file (2026-10-18).';

-- FCC equipment authorization open data (snapshots end 2021-03-22; open question f) ---------------
COMMENT ON TABLE fcc_grantees IS 'FCC equipment authorization grantees, 50,153 rows. grantee_code is the first part of every FCC ID. The snapshot ends 2021-03-22: grantees registered later are absent (open question f). Personal contact data and street addresses are not loaded.';
COMMENT ON COLUMN fcc_grantees.grantee_code IS 'Primary key: 3 characters (registered 1998-2013) or 5 characters (2013-2021), uppercase letters and digits. First part of every FCC ID.';
COMMENT ON COLUMN fcc_grantees.grantee_name IS 'Company name. NULL for one grantee (code OAF).';
COMMENT ON COLUMN fcc_grantees.city IS 'City as entered in the source. Placeholders N/A, NA and n/a are NULL; other text such as NIL is kept as written (open question i).';
COMMENT ON COLUMN fcc_grantees.state IS 'State or province as entered; NULL for 30,941 grantees (mostly outside the US, where the source says N/A).';
COMMENT ON COLUMN fcc_grantees.country IS 'Country name as entered; 111 distinct values. NULL for 158 grantees. Comparable with fcc_test_firms.country.';
COMMENT ON COLUMN fcc_grantees.date_received IS 'Registration date (the source has no time of day), 1998-04-14 to 2021-03-22.';

COMMENT ON TABLE fcc_test_firms IS 'FCC accredited test firms (labs), 591 rows. The snapshot ends 2021-03-22 and ALL expiration dates are in the past (2021-03-26 to 2023-03-18): it reflects accreditation around 2021, not current status (open question h). Key (designation_number, firm_name).';
COMMENT ON COLUMN fcc_test_firms.designation_number IS 'Designation number such as US1273 or JP0001. NOT unique (US5037 and US1273 each belong to two different firms): never join on it alone (open question g).';
COMMENT ON COLUMN fcc_test_firms.firm_name IS 'Name of the test firm (not unique on its own: some firms have several branches).';
COMMENT ON COLUMN fcc_test_firms.accreditation IS 'Name of the accrediting body.';
COMMENT ON COLUMN fcc_test_firms.mra IS 'Mutual recognition arrangement (for example APEC TEL, US-EU); NULL for 262 firms where the source says N/A.';
COMMENT ON COLUMN fcc_test_firms.expiration_date IS 'Accreditation expiration date, 2021-03-26 to 2023-03-18: every value is in the past.';
COMMENT ON COLUMN fcc_test_firms.city IS 'City of the firm.';
COMMENT ON COLUMN fcc_test_firms.state IS 'State or province of the firm; NULL for most (453).';
COMMENT ON COLUMN fcc_test_firms.country IS 'Country name; 29 distinct values. Comparable with fcc_grantees.country.';
COMMENT ON COLUMN fcc_test_firms.contract IS '''Y'' (574) or ''N'' (17): whether the firm said it is available on a contract basis, as in the source.';

-- views ------------------------------------------------------------------------------------------
-- View columns that mirror a base column inherit its comment (single source of truth).
DO $$
DECLARE
    m record;
BEGIN
    FOR m IN
        SELECT v AS view_name, b AS base_table, unnest(cols) AS col
        FROM (VALUES
            ('v_uls_licences', 'uls_hd', ARRAY['service', 'usi', 'call_sign', 'license_status',
                'radio_service_code', 'grant_date', 'expired_date', 'cancellation_date',
                'effective_date', 'last_action_date', 'auction_id']),
            ('v_uls_frequencies', 'uls_fr', ARRAY['service', 'usi', 'location_number',
                'antenna_number', 'freq_seq_id', 'class_station_code', 'frequency_assigned',
                'frequency_upper_band', 'power_output', 'power_erp', 'eirp']),
            ('v_secom_values', 'secom_measurements', ARRAY['run_id', 'feature_idx', 'value'])
        ) AS t(v, b, cols)
    LOOP
        EXECUTE format(
            'COMMENT ON COLUMN %I.%I IS %L', m.view_name, m.col,
            (SELECT col_description(m.base_table::regclass, a.attnum)
             FROM pg_attribute a WHERE a.attrelid = m.base_table::regclass AND a.attname = m.col));
    END LOOP;
END
$$;

COMMENT ON VIEW v_uls_licences IS 'One row per ULS licence with its licensee (the uls_en row of entity type ''L''). Same rows and keys as uls_hd. Use it for questions about licences, status, dates and licensee names.';
COMMENT ON COLUMN v_uls_licences.licensee_name IS 'entity_name of the licence''s ''L'' entity; the meaning of ''L'' is unconfirmed (open question b).';
COMMENT ON COLUMN v_uls_licences.licensee_id IS 'licensee_id of the ''L'' entity.';
COMMENT ON COLUMN v_uls_licences.licensee_frn IS 'FCC registration number (frn) of the ''L'' entity.';
COMMENT ON COLUMN v_uls_licences.licensee_city IS 'City of the ''L'' entity.';
COMMENT ON COLUMN v_uls_licences.licensee_state IS 'State of the ''L'' entity.';
COMMENT ON VIEW v_uls_frequencies IS 'ULS frequency rows (uls_fr, 236,958) with their licence''s call sign, status and licensee name. Units as in uls_fr: frequency ASSUMED MHz, power UNKNOWN.';
COMMENT ON COLUMN v_uls_frequencies.call_sign IS 'Call sign of the licence the frequency belongs to.';
COMMENT ON COLUMN v_uls_frequencies.license_status IS 'Status code of the licence (A, C, E, T; meanings unconfirmed).';
COMMENT ON COLUMN v_uls_frequencies.licensee_name IS 'Licensee name from v_uls_licences.';
COMMENT ON VIEW v_secom_values IS 'SECOM readings (924,530 rows) with their run''s label and timestamp: filter by label to compare a sensor between passing and failing runs.';
COMMENT ON COLUMN v_secom_values.label IS '''pass'' or ''fail'' of the run.';
COMMENT ON COLUMN v_secom_values.run_ts IS 'Timestamp of the run (no time zone).';
"""


def _set_vars(**values: str) -> None:
    bind = op.get_bind()
    for name, value in values.items():
        bind.execute(
            sa.text("SELECT set_config(:name, :value, false)"),
            {"name": f"rflens.{name}", "value": value},
        )


def upgrade() -> None:
    # exec_driver_sql: the script holds several statements and no bind parameters. psycopg reads
    # a bare % as a placeholder (format('%I'), '40% of rows'), so double them.
    op.get_bind().exec_driver_sql(VIEWS_AND_COMMENTS.replace("%", "%%"))

    op.execute(
        """
        CREATE TABLE sql_query_log (
            id               bigserial PRIMARY KEY,
            logged_at        timestamptz NOT NULL DEFAULT now(),
            tool             text NOT NULL
                CHECK (tool IN ('list_tables', 'describe_table', 'run_query')),
            question         text,
            sql_text         text NOT NULL,  -- the SQL; for the other tools, the call itself
            plan_cost        double precision,
            row_count        integer,
            duration_ms      integer,
            status           text NOT NULL CHECK (status IN ('ok', 'rejected', 'error')),
            rejection_reason text
        )
        """
    )
    op.execute(
        "COMMENT ON TABLE sql_query_log IS "
        "'sql-mcp audit log: one row per run_query call, accepted or rejected.'"
    )

    s = get_sql_mcp_settings()
    _set_vars(
        reader_role=s.sql_reader_user,
        reader_password=s.sql_reader_password,
        logger_role=s.sql_logger_user,
        logger_password=s.sql_logger_password,
        timeout=f"{s.sql_statement_timeout_ms}ms",
    )
    op.execute(
        f"""
        DO $$
        DECLARE
            rd text := current_setting('rflens.reader_role');
            lg text := current_setting('rflens.logger_role');
        BEGIN
            IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = rd) THEN
                EXECUTE format(
                    'CREATE ROLE %I LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE PASSWORD %L',
                    rd, current_setting('rflens.reader_password'));
            END IF;
            IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = lg) THEN
                EXECUTE format(
                    'CREATE ROLE %I LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE PASSWORD %L',
                    lg, current_setting('rflens.logger_password'));
            END IF;

            EXECUTE format('GRANT CONNECT ON DATABASE %I TO %I, %I', current_database(), rd, lg);
            EXECUTE format('GRANT USAGE ON SCHEMA public TO %I, %I', rd, lg);

            EXECUTE format('GRANT SELECT ON {READER_TABLES} TO %I', rd);
            EXECUTE format('ALTER ROLE %I SET default_transaction_read_only = on', rd);
            EXECUTE format('ALTER ROLE %I SET statement_timeout = %L', rd,
                           current_setting('rflens.timeout'));

            EXECUTE format('GRANT INSERT ON sql_query_log TO %I', lg);
            EXECUTE format('GRANT USAGE ON SEQUENCE sql_query_log_id_seq TO %I', lg);
        END
        $$
        """
    )
    _set_vars(reader_role="", reader_password="", logger_role="", logger_password="", timeout="")


def downgrade() -> None:
    s = get_sql_mcp_settings()
    _set_vars(reader_role=s.sql_reader_user, logger_role=s.sql_logger_user)
    op.execute(
        f"""
        DO $$
        DECLARE
            r text;
        BEGIN
            FOREACH r IN ARRAY ARRAY[current_setting('rflens.reader_role'),
                                     current_setting('rflens.logger_role')] LOOP
                IF EXISTS (SELECT FROM pg_roles WHERE rolname = r) THEN
                    EXECUTE format('REVOKE ALL ON {READER_TABLES}, sql_query_log FROM %I', r);
                    EXECUTE format('REVOKE ALL ON SEQUENCE sql_query_log_id_seq FROM %I', r);
                    EXECUTE format('REVOKE ALL ON SCHEMA public FROM %I', r);
                    EXECUTE format('REVOKE ALL ON DATABASE %I FROM %I', current_database(), r);
                    EXECUTE format('DROP ROLE %I', r);
                END IF;
            END LOOP;
        END
        $$
        """
    )
    _set_vars(reader_role="", logger_role="")
    op.execute("DROP TABLE sql_query_log")
    for view in VIEWS:
        op.execute(f"DROP VIEW {view}")
    # Remove the comments this migration added (only on relations that exist).
    for rel in COMMENTED_RELATIONS:
        op.execute(f"COMMENT ON TABLE {rel} IS NULL")
        op.execute(
            f"""
            DO $$
            DECLARE a record;
            BEGIN
                FOR a IN SELECT attname FROM pg_attribute
                         WHERE attrelid = '{rel}'::regclass AND attnum > 0 AND NOT attisdropped
                LOOP
                    EXECUTE format('COMMENT ON COLUMN {rel}.%I IS NULL', a.attname);
                END LOOP;
            END
            $$
            """
        )
