"""Database-agnostic full-text search engine using the Strategy Pattern."""

import re
from typing import Any, Protocol, runtime_checkable

from sqlalchemy import ColumnElement, Select, String, Text, and_, or_, text
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import InstrumentedAttribute

from strapalchemy.logging.logger import logger

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _fts5_quote(token: str) -> str:
    """Wrap *token* in FTS5 phrase quotes so it is treated as a literal phrase.

    FTS5 treats ``"``, ``*``, ``OR``, ``AND``, ``NOT``, ``NEAR``, ``(``, ``)``,
    and column filter syntax as operators even inside a bound parameter, because
    the entire query string is parsed by the FTS5 engine.  Wrapping each token
    in double quotes disables operator interpretation; any literal ``"`` inside
    the token is escaped by doubling it (per FTS5 quoting rules).

    Args:
        token: A single search token, possibly containing FTS5 operator characters.

    Returns:
        The token enclosed in FTS5 phrase quotes with internal quotes doubled.
    """
    return '"' + token.replace('"', '""') + '"'


def _get_column(model: type, field_name: str) -> "InstrumentedAttribute[Any] | None":
    """Return the SQLAlchemy column attribute for *field_name* on *model*.

    Args:
        model: SQLAlchemy declarative model class.
        field_name: Attribute name to look up.

    Returns:
        Column attribute, or ``None`` if the attribute does not exist.
    """
    try:
        return getattr(model, field_name)
    except AttributeError:
        return None


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class SearchStrategy(Protocol):
    """Interface that every concrete search strategy must satisfy."""

    def build_conditions(
        self,
        model: type,
        fields: list[str],
        tokens: list[str],
    ) -> list[ColumnElement[bool]]:
        """Build a list of SQLAlchemy filter conditions for *tokens* over *fields*.

        Each element in the returned list represents one field's compound
        condition (all tokens must match that field).  The caller combines
        them with ``OR`` so that a row is a hit when *any* field satisfies
        *all* tokens.

        Args:
            model: SQLAlchemy declarative model class.
            fields: Column names declared in ``__searchable__``.
            tokens: Already-sanitized individual words from the search string.

        Returns:
            Flat list of SQLAlchemy ``ColumnElement[bool]`` objects.
        """
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


class PostgreSQLStrategy:
    """Search strategy for PostgreSQL using the native ``~*`` regex operator.

    Uses word-boundary anchors (``\\m`` / ``\\M``) for whole-word,
    case-insensitive matching — no extensions required.
    """

    def build_conditions(
        self,
        model: type,
        fields: list[str],
        tokens: list[str],
    ) -> list[ColumnElement[bool]]:
        """Build conditions using PostgreSQL case-insensitive regex matching.

        Args:
            model: SQLAlchemy declarative model class.
            fields: Column names to search.
            tokens: Individual search tokens.

        Returns:
            One condition per field where all tokens must match.
        """
        conditions: list[ColumnElement[bool]] = []
        for field_name in fields:
            col = _get_column(model, field_name)
            if col is None:
                continue
            token_conditions: list[ColumnElement[bool]] = [
                col.op("~*")(r"\m" + re.escape(t) + r"\M")
                for t in tokens
            ]
            if not token_conditions:
                continue
            conditions.append(
                and_(*token_conditions) if len(token_conditions) > 1 else token_conditions[0]
            )
        return conditions


class MySQLStrategy:
    """Search strategy for MySQL / MariaDB.

    Primary path: ``MATCH … AGAINST … IN BOOLEAN MODE`` (requires FULLTEXT
    index on the target columns).

    Automatic fallback: when the primary path raises (e.g. no FULLTEXT index),
    ``mark_fulltext_unavailable()`` is called internally and subsequent calls
    use ``SOUNDS LIKE`` (Soundex) phonetic matching instead.

    Degradation is scoped per field-set: only the specific combination of
    fields that triggered the failure degrades; other field combinations
    continue using FULLTEXT.
    """

    # MySQL FTS boolean mode operator characters that must be stripped from
    # user-supplied tokens before constructing the boolean query string.
    _MYSQL_FTS_OP_CHARS = re.compile(r'[+\-><()~*"@]')

    def __init__(self) -> None:
        self._failed_field_sets: set[frozenset[str]] = set()

    def mark_fulltext_unavailable(self, fields: list[str]) -> None:
        """Disable the FULLTEXT path for the given *fields* combination.

        Degradation is scoped: only this specific field-set falls back to
        Soundex; other field combinations are unaffected.

        Args:
            fields: Column names whose FULLTEXT path should be disabled.
        """
        self._failed_field_sets.add(frozenset(fields))

    def build_conditions(
        self,
        model: type,
        fields: list[str],
        tokens: list[str],
    ) -> list[ColumnElement[bool]]:
        """Build MySQL search conditions.

        Args:
            model: SQLAlchemy declarative model class.
            fields: Column names to search.
            tokens: Individual search tokens.

        Returns:
            FULLTEXT or Soundex conditions depending on availability.
        """
        if frozenset(fields) not in self._failed_field_sets:
            return self._build_fulltext_conditions(fields, tokens)
        return self._build_soundex_conditions(model, fields, tokens)

    def _sanitize_fts_token(self, token: str) -> str:
        """Strip MySQL FTS boolean mode operator characters from *token*.

        Args:
            token: A single search token, potentially containing operator chars.

        Returns:
            The token with all FTS operator characters removed and stripped.
        """
        return self._MYSQL_FTS_OP_CHARS.sub("", token).strip()

    def _build_fulltext_conditions(
        self,
        fields: list[str],
        tokens: list[str],
    ) -> list[ColumnElement[bool]]:
        # ``fields`` are developer-controlled column names from ``__searchable__``,
        # never user input.  User tokens go into the ``:q`` bound parameter only,
        # and are sanitized to strip MySQL FTS boolean operator characters.
        invalid = [f for f in fields if not _SAFE_IDENTIFIER.match(f)]
        if invalid:
            raise ValueError(
                f"Invalid field names for FULLTEXT search: {invalid!r}. "
                "Field names must match ^[A-Za-z_][A-Za-z0-9_]*$."
            )
        field_list = ", ".join(fields)
        sanitized_parts: list[str] = []
        for t in tokens:
            clean = self._sanitize_fts_token(t)
            if clean:
                sanitized_parts.append(f"+{clean}")
        if not sanitized_parts:
            return []
        boolean_query = " ".join(sanitized_parts)
        expr = text(f"MATCH({field_list}) AGAINST (:q IN BOOLEAN MODE)").bindparams(
            q=boolean_query
        )
        return [expr]  # type: ignore[list-item]

    def _build_soundex_conditions(
        self,
        model: type,
        fields: list[str],
        tokens: list[str],
    ) -> list[ColumnElement[bool]]:
        conditions: list[ColumnElement[bool]] = []
        for field_name in fields:
            col = _get_column(model, field_name)
            if col is None:
                continue
            token_conditions: list[ColumnElement[bool]] = [
                col.op("SOUNDS LIKE")(t) for t in tokens
            ]
            if not token_conditions:
                continue
            conditions.append(
                and_(*token_conditions) if len(token_conditions) > 1 else token_conditions[0]
            )
        return conditions


class SQLiteStrategy:
    """Search strategy for SQLite.

    Primary path (opt-in): if the model declares ``__fts5_table__: str``,
    a correlated FTS5 subquery is used.

    Automatic fallback: delegates to ``UniversalFallbackStrategy`` when no
    FTS5 virtual table is declared.
    """

    def build_conditions(
        self,
        model: type,
        fields: list[str],
        tokens: list[str],
    ) -> list[ColumnElement[bool]]:
        """Build SQLite search conditions.

        Args:
            model: SQLAlchemy declarative model class.
            fields: Column names to search.
            tokens: Individual search tokens.

        Returns:
            FTS5 subquery conditions or ILIKE fallback conditions.
        """
        fts_table: str | None = getattr(model, "__fts5_table__", None)
        if fts_table:
            return self._build_fts5_conditions(model, fts_table, tokens)
        return UniversalFallbackStrategy().build_conditions(model, fields, tokens)

    def _build_fts5_conditions(
        self,
        model: type,
        fts_table: str,
        tokens: list[str],
    ) -> list[ColumnElement[bool]]:
        # Validate ``fts_table`` against a safe identifier allowlist before
        # interpolating it into the SQL string.
        if not _SAFE_IDENTIFIER.match(fts_table):
            raise ValueError(
                f"__fts5_table__ value {fts_table!r} is not a valid SQL identifier. "
                "Use only letters, digits, and underscores."
            )

        insp = sa_inspect(model)
        # Use the ORM-mapped attribute for .in_() so SQLAlchemy generates
        # correct ORM-level SQL rather than raw Table column SQL.
        pk_attr_name = insp.mapper.primary_key[0].key
        pk_col = getattr(model, pk_attr_name)

        fts_query = " ".join(_fts5_quote(t) for t in tokens)  # FTS5 implicit AND, tokens quoted as literals
        subq = text(f"SELECT rowid FROM {fts_table} WHERE {fts_table} MATCH :q").bindparams(
            q=fts_query
        )
        return [pk_col.in_(subq)]  # type: ignore[list-item]


class UniversalFallbackStrategy:
    """Cross-database ILIKE/LIKE fallback using bound parameters.

    Each token produces a separate ``%token%`` condition; all tokens must
    match a field (AND) for that field to be considered a hit.

    When *fields* is empty, all ``String`` and ``Text`` columns of the model
    are searched automatically.
    """

    def build_conditions(
        self,
        model: type,
        fields: list[str],
        tokens: list[str],
    ) -> list[ColumnElement[bool]]:
        """Build parameterised ILIKE conditions across all relevant fields.

        Args:
            model: SQLAlchemy declarative model class.
            fields: Column names to search (auto-discovered when empty).
            tokens: Individual search tokens.

        Returns:
            One AND-compound condition per field.
        """
        effective_fields = fields
        if not effective_fields:
            effective_fields = [
                col.key
                for col in model.__table__.columns
                if isinstance(col.type, (String, Text))
            ]

        conditions: list[ColumnElement[bool]] = []
        for field_name in effective_fields:
            col = _get_column(model, field_name)
            if col is None:
                continue
            # .ilike() passes the full pattern as a bound parameter — injection-safe.
            token_conds: list[ColumnElement[bool]] = [col.ilike(f"%{t}%") for t in tokens]
            if not token_conds:
                continue
            conditions.append(
                and_(*token_conds) if len(token_conds) > 1 else token_conds[0]
            )
        return conditions


# ---------------------------------------------------------------------------
# SearchEngine
# ---------------------------------------------------------------------------


class SearchEngine:
    """Database-agnostic search engine that selects a strategy by dialect.

    Instantiate once per dialect (or once per request if the dialect varies).

    Example::

        engine = SearchEngine(dialect="postgresql")
        query = engine.apply_search(select(User), User, "alice developer")
    """

    def __init__(self, dialect: str = "sqlite") -> None:
        self._searchable_cache: dict[str, list[str]] = {}
        self._strategy: SearchStrategy = self._select_strategy(dialect)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_search(self, query: Select, model: type, search: str | None) -> Select:
        """Apply the appropriate search strategy to *query*.

        Args:
            query: SQLAlchemy ``Select`` statement to filter.
            model: SQLAlchemy declarative model class.
            search: Raw search string supplied by the caller.

        Returns:
            The same ``Select`` statement with a ``WHERE`` clause appended,
            or the original statement unchanged when *search* is blank.
        """
        if not search or not search.strip():
            return query

        tokens = self._tokenize_search_query(search)
        if not tokens:
            return query

        fields = self._get_searchable_fields(model)

        try:
            conditions = self._strategy.build_conditions(model, fields, tokens)
        except OperationalError as exc:
            # MySQL: FULLTEXT index may be absent — try the phonetic fallback.
            # Degradation is scoped to the specific field-set that failed.
            if hasattr(self._strategy, "mark_fulltext_unavailable"):
                logger.warning(
                    f"FULLTEXT search unavailable for fields {fields} on "
                    f"{model.__name__}: {exc}. Falling back to phonetic search."
                )
                self._strategy.mark_fulltext_unavailable(fields)  # type: ignore[union-attr]
                try:
                    conditions = self._strategy.build_conditions(model, fields, tokens)
                except OperationalError as exc2:
                    logger.error(f"Fallback search strategy also failed: {exc2}")
                    conditions = UniversalFallbackStrategy().build_conditions(
                        model, fields, tokens
                    )
            else:
                logger.error(f"Search strategy failed: {exc}")
                conditions = UniversalFallbackStrategy().build_conditions(
                    model, fields, tokens
                )

        if not conditions:
            return query

        return query.where(or_(*conditions))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _select_strategy(self, dialect: str) -> SearchStrategy:
        """Return the concrete strategy that matches *dialect*.

        Dialect strings are normalised before matching so that driver
        suffixes (e.g. ``postgresql+asyncpg``) and query strings are
        stripped automatically.

        Args:
            dialect: SQLAlchemy dialect name, optionally with driver suffix.

        Returns:
            An instance of the matching ``SearchStrategy`` implementation.
        """
        normalized = dialect.lower().split("+")[0].split("?")[0].strip()
        if normalized in ("postgresql", "postgres"):
            return PostgreSQLStrategy()
        if normalized in ("mysql", "mariadb"):
            return MySQLStrategy()
        if normalized == "sqlite":
            return SQLiteStrategy()
        return UniversalFallbackStrategy()

    def _get_searchable_fields(self, model: type) -> list[str]:
        """Return the list of searchable field names declared on *model*.

        Results are memoised by model name.

        Args:
            model: SQLAlchemy declarative model class.

        Returns:
            List of field name strings, possibly empty.
        """
        model_key = f"{model.__module__}.{model.__qualname__}"
        if model_key in self._searchable_cache:
            return self._searchable_cache[model_key]

        config = getattr(model, "__searchable__", None)
        if config is None:
            fields: list[str] = []
        elif isinstance(config, dict):
            fields = config.get("text_fields", [])
        elif isinstance(config, list):
            fields = list(config)
        else:
            fields = []

        self._searchable_cache[model_key] = fields
        return fields

    @staticmethod
    def _tokenize_search_query(search: str) -> list[str]:
        """Split *search* into a sanitised list of tokens.

        Processing steps (in order):

        1. Remove ASCII control characters (``\\x00``–``\\x1f``, ``\\x7f``).
        2. Normalise runs of whitespace to a single space and strip.
        3. Split on whitespace.
        4. Drop empty strings.
        5. Limit to 20 tokens to prevent abuse.

        Special characters such as ``'``, ``"``, ``-``, and ``;`` are
        **not** stripped — they are passed as bound parameters and are
        therefore injection-safe.

        Args:
            search: Raw search string supplied by the caller.

        Returns:
            List of at most 20 non-empty token strings.
        """
        search = re.sub(r"[\x00-\x1f\x7f]", "", search)
        search = re.sub(r"\s+", " ", search).strip()
        return [t for t in search.split() if t][:20]
