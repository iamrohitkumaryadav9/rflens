"""v_uls_licences.is_lease: separate spectrum leases from licences

Rows of uls_hd whose call_sign is "L" + 9 digits are spectrum LEASES, not licences: 3,174 of the
5,464 rows of the cell file. Evidence (checked 2026-10-24; CLAUDE.md, open data question j):
- FCC ULS Public Access Database Definitions v9.0 (record layouts): the Lease Dates record [LD]
  has the field "Call Sign (Lease Id)", the Lease Link record [LL] links a Call Sign to a Lease ID,
  and the Lease Classification record [LC] and the leased location / antenna / frequency records
  [L3, A3, F3, ...] are keyed by Lease ID.
- l_cell.zip (the loaded file) contains ll.dat, lc.dat, l3.dat, a3.dat and f3.dat: 3,169 of the
  3,174 matching HD rows have an LL and an LC record that link the lease ID to a licence call sign.
  The other 5 (L000001242-1245 and L000002482, status T, 2006-2007) match the pattern but have no
  LL / LC record in the file.
- None of the 3,174 has FR or LO rows: a lease's frequencies and locations are in the lease files.

The view gains one column at the end (CREATE OR REPLACE keeps v_uls_frequencies and every existing
comment). Comments that called every uls_hd row a licence are corrected.

Revision ID: 0010
Revises: 0009
Create Date: 2026-10-24
"""

# ruff: noqa: E501
# (the COMMENT ON strings are text for describe_table; they are not wrapped)

import importlib.util
import re
from pathlib import Path

import sqlalchemy as sa
from alembic import op

from rflens.config import get_sql_mcp_settings

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None

SOURCE_URL = "https://www.fcc.gov/sites/default/files/public_access_database_definitions_v9_0.pdf"

UPGRADE_SQL = (
    r"""
CREATE OR REPLACE VIEW v_uls_licences AS
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
       e.state       AS licensee_state,
       COALESCE(h.call_sign ~ '^L[0-9]{9}$', false) AS is_lease
FROM uls_hd h
LEFT JOIN uls_en e
       ON e.service = h.service AND e.usi = h.usi AND e.entity_type = 'L';

COMMENT ON COLUMN v_uls_licences.is_lease IS 'True when call_sign is L + 9 digits (for example L000070487): the row is a spectrum LEASE, not a licence (3,174 of 5,464 rows). Source: FCC ULS Public Access Database Definitions v9.0 ("""
    + SOURCE_URL
    + r"""): the Lease Dates record [LD] has the field "Call Sign (Lease Id)" and the Lease Link record [LL] links a call sign to a Lease ID; in l_cell.zip 3,169 of these rows have an LL and an LC (lease classification) record linking them to a licence call sign, 5 (status T, 2006-2007) match the pattern without such a record. A lease has no uls_fr or uls_lo rows (its frequencies and locations are in lease files that are not loaded), and licensee_name is the lessee. Use is_lease = false to count licences only. The lease classification codes (ML, TL) are not defined in the layout document.';

COMMENT ON VIEW v_uls_licences IS 'One row per ULS HD record with its licensee (the uls_en row of entity type ''L''): 2,290 licences plus 3,174 spectrum leases (is_lease = true; their call sign is a lease identifier). Same rows and keys as uls_hd. Filter is_lease = false for questions about licences. Use it for status, dates and licensee names.';

COMMENT ON COLUMN v_uls_licences.call_sign IS 'Licence call sign, unique within the cell service. For a spectrum lease (is_lease = true) it is the lease identifier, L + 9 digits.';

COMMENT ON COLUMN uls_hd.call_sign IS 'Licence call sign, unique within the cell service. For a spectrum lease it is the lease identifier, L + 9 digits (3,174 rows; see v_uls_licences.is_lease).';

COMMENT ON TABLE uls_hd IS 'FCC ULS header (HD record): one row per licence or spectrum lease. Loaded service: cell (Cellular), 5,464 rows = 2,290 licences + 3,174 leases (call_sign of the form L + 9 digits; see v_uls_licences.is_lease). Key (service, usi).';
"""
)


def _original_0008() -> str:
    """The text of 0008's SQL, to restore the pre-0010 definitions on downgrade."""
    path = Path(__file__).with_name("0008_sql_mcp_views_roles_log.py")
    spec = importlib.util.spec_from_file_location("migration_0008", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.VIEWS_AND_COMMENTS


def _restore_statements() -> list[str]:
    sql = _original_0008()
    create = [
        re.search(rf"CREATE VIEW {view} AS.*?;\n", sql, re.S).group(0)
        for view in ("v_uls_licences", "v_uls_frequencies")
    ]
    mirror = re.search(r"DO \$\$.*?\$\$;", sql, re.S).group(0)
    comments = re.findall(
        r"COMMENT ON (?:VIEW|COLUMN) v_uls_(?:licences|frequencies)(?:\.\w+)? "
        r"IS '(?:[^']|'')*';",
        sql,
    )
    uls_hd = re.findall(
        r"COMMENT ON (?:TABLE uls_hd|COLUMN uls_hd\.call_sign) IS '(?:[^']|'')*';", sql
    )
    # uls_hd first: the mirror block copies base-column comments onto the view columns.
    return [*create, *uls_hd, mirror, *comments]


def upgrade() -> None:
    # exec_driver_sql: several statements, no bind parameters; double % for psycopg.
    op.get_bind().exec_driver_sql(UPGRADE_SQL.replace("%", "%%"))


def downgrade() -> None:
    # Dropping a view drops its grants, so rflens_sql_reader must get SELECT back (0008 granted it).
    op.execute("DROP VIEW v_uls_frequencies")
    op.execute("DROP VIEW v_uls_licences")
    for statement in _restore_statements():
        op.get_bind().exec_driver_sql(statement.replace("%", "%%"))
    bind = op.get_bind()
    bind.execute(
        sa.text("SELECT set_config('rflens.reader_role', :r, false)"),
        {"r": get_sql_mcp_settings().sql_reader_user},
    )
    op.execute(
        """
        DO $$
        DECLARE r text := current_setting('rflens.reader_role');
        BEGIN
            IF EXISTS (SELECT FROM pg_roles WHERE rolname = r) THEN
                EXECUTE format('GRANT SELECT ON v_uls_licences, v_uls_frequencies TO %I', r);
            END IF;
        END
        $$
        """
    )
    bind.execute(sa.text("SELECT set_config('rflens.reader_role', '', false)"))
