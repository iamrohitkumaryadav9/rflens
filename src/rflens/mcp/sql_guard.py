"""Static guardrails for sql-mcp: what a query may contain before it is sent to Postgres.

check_query() parses with sqlglot (Postgres dialect) and rejects anything that is not one plain
SELECT over the allow-listed tables and views. The SQL that is executed is regenerated from the
parsed tree, not the caller's text, so what was inspected is exactly what runs.

This is the first of several layers. The database role (rflens_sql_reader: SELECT on the
allow-list only, read-only transactions, statement timeout) is the real boundary and the planner
cost gate and row cap are applied in sql_server.py; this module gives the caller a precise reason
instead of a bare permission error, and keeps obviously hostile statements away from the database.

Order of checks: parse, exactly one statement, SELECT only, no DML or DDL anywhere in the tree,
table allow-list (including CTEs and subqueries), function deny-list, LIMIT.
"""

from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError
from sqlglot.optimizer.scope import traverse_scope

# Tables and views the text-to-SQL agent may read. uls_en is deliberately absent: it holds contact
# persons' names (entity types CL and individuals) that no RF question needs (data minimisation);
# it is reachable only through v_uls_licences, which exposes the licensee entity only.
# chunks and documents belong to docs-mcp; alembic_version and sql_query_log are internal.
ALLOWED_TABLES = (
    "data_sources",
    "secom_runs",
    "secom_measurements",
    "uls_hd",
    "uls_fr",
    "uls_lo",
    "uls_file_counts",
    "fcc_grantees",
    "fcc_test_firms",
)
ALLOWED_VIEWS = ("v_uls_licences", "v_uls_frequencies", "v_secom_values")
ALLOWED_RELATIONS = ALLOWED_TABLES + ALLOWED_VIEWS

# Names that are refused wherever they appear in the tree, even as a CTE alias, so that no
# shadowing trick can hide a reference to them.
SENSITIVE_NAMES = frozenset({"uls_en", "chunks", "documents", "alembic_version", "sql_query_log"})

SYSTEM_SCHEMAS = frozenset({"pg_catalog", "information_schema", "pg_toast"})

# Function names refused by exact name or prefix (lower case): sleeping, file and large-object
# access, other databases, settings, sequences, catalog look-ups by name, and the *_to_xml family
# that runs a query given as a string.
DENIED_FUNCTION_PREFIXES = (
    "pg_",
    "lo_",
    "dblink",
    "query_to_xml",
    "table_to_xml",
    "cursor_to_xml",
    "schema_to_xml",
    "database_to_xml",
    "set_config",
    "current_setting",
    "nextval",
    "setval",
    "currval",
    "lastval",
    "to_reg",
    "has_",
    "txid_",
    "inet_",
)

_FORBIDDEN_NODE_NAMES = (
    "Insert", "Update", "Delete", "Merge", "Into", "Lock", "Create", "Drop", "Alter", "Command",
    "Copy", "Set", "TruncateTable", "Grant", "Revoke", "Transaction", "Commit", "Rollback",
)  # fmt: skip
FORBIDDEN_NODES = tuple(getattr(exp, n) for n in _FORBIDDEN_NODE_NAMES if hasattr(exp, n))


class Rejection(Exception):
    """A query refused by a guardrail. `code` is stable (logged); `message` is for the caller."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class CheckedQuery:
    sql: str  # regenerated from the parse tree, with the final LIMIT n (what the caller is shown)
    fetch_sql: str  # the same with LIMIT n + 1: what runs, so "more rows existed" can be seen
    limit: int
    notice: str | None  # set when a LIMIT was added or lowered


def _name(ident: exp.Expression | None) -> str:
    """Name of an identifier as Postgres resolves it: unquoted names fold to lower case."""
    if ident is None or not ident.name:
        return ""
    return ident.name if ident.args.get("quoted") else ident.name.lower()


def _function_name(fn: exp.Func) -> str:
    return (fn.name if isinstance(fn, exp.Anonymous) else fn.sql_name()).lower()


def _check_tables(tree: exp.Expression) -> None:
    for table in tree.find_all(exp.Table):
        if _name(table.this) in SENSITIVE_NAMES:
            raise Rejection(
                "table_not_allowed",
                f"table {table.this.name!r} is not available: "
                "only the tables and views from list_tables can be queried",
            )
    try:
        scopes = traverse_scope(tree)
    except SqlglotError as e:
        raise Rejection("parse_error", f"cannot analyse the tables of this query: {e}") from e
    for scope in scopes:
        for source in scope.sources.values():
            if not isinstance(source, exp.Table):
                continue  # a CTE or subquery: its own scope is checked in turn
            if not isinstance(source.this, exp.Identifier):
                raise Rejection(
                    "table_function",
                    "functions in FROM (generate_series, unnest, ...) are not allowed; "
                    "query the tables and views from list_tables",
                )
            name, schema = _name(source.this), _name(source.args.get("db"))
            if schema in SYSTEM_SCHEMAS or (not schema and name.startswith("pg_")):
                raise Rejection(
                    "system_catalog", f"system catalogs ({schema or 'pg_*'}) are not available"
                )
            if source.args.get("catalog") or schema not in ("", "public"):
                raise Rejection(
                    "table_not_allowed",
                    f"schema {schema!r} is not available; use plain table names",
                )
            if name not in ALLOWED_RELATIONS:
                raise Rejection(
                    "table_not_allowed",
                    f"table {source.this.name!r} is not on the allow-list; "
                    f"allowed: {', '.join(ALLOWED_RELATIONS)}",
                )


def _check_functions(tree: exp.Expression) -> None:
    for fn in tree.find_all(exp.Func):
        name = _function_name(fn)
        if name.startswith(DENIED_FUNCTION_PREFIXES):
            raise Rejection("function_denied", f"function {name}() is not allowed")


def _apply_limit(root: exp.Expression, default: int, maximum: int) -> tuple[int, str | None]:
    limit = root.args.get("limit")
    if isinstance(limit, exp.Fetch):
        raise Rejection("unsupported_limit", "use LIMIT n instead of FETCH FIRST n ROWS")
    if limit is None:
        root.limit(default, copy=False)
        return default, f"LIMIT {default} added (default; maximum {maximum})"
    wanted = limit.expression
    if isinstance(wanted, exp.Literal) and wanted.is_int:
        n = int(wanted.name)
        if n <= maximum:
            return n, None
    # Over the maximum, or LIMIT ALL / LIMIT NULL / an expression: use the maximum.
    limit.set("expression", exp.Literal.number(maximum))
    return maximum, f"LIMIT lowered to the maximum of {maximum}"


def check_query(sql: str, default_limit: int = 100, max_limit: int = 1000) -> CheckedQuery:
    """Return the query to run, or raise Rejection with a stable code."""
    try:
        trees = sqlglot.parse(sql, read="postgres")
    except SqlglotError as e:
        raise Rejection("parse_error", f"SQL does not parse (PostgreSQL dialect): {e}") from e
    statements = [t for t in trees if t is not None]  # `SELECT 1;` leaves an empty tail
    if not statements:
        raise Rejection("empty", "the query is empty")
    if len(statements) > 1:
        raise Rejection("multiple_statements", "send exactly one statement")
    root = statements[0].unnest() if isinstance(statements[0], exp.Subquery) else statements[0]
    if not isinstance(root, exp.Query):
        raise Rejection("not_select", f"only SELECT is allowed (got {type(root).__name__.upper()})")
    bad = root.find(*FORBIDDEN_NODES)
    if bad is not None:
        raise Rejection(
            "forbidden_construct",
            f"{type(bad).__name__.upper()} is not allowed anywhere in the query "
            "(no data changes, SELECT INTO, locking clauses or DDL)",
        )
    _check_tables(root)
    _check_functions(root)
    limit, notice = _apply_limit(root, default_limit, max_limit)
    # comments=False: sqlglot would otherwise carry comments into the SQL that runs.
    sql = root.sql(dialect="postgres", comments=False)
    root.args["limit"].set("expression", exp.Literal.number(limit + 1))
    fetch_sql = root.sql(dialect="postgres", comments=False)
    return CheckedQuery(sql=sql, fetch_sql=fetch_sql, limit=limit, notice=notice)
