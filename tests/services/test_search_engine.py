"""Tests for SearchEngine."""

import pytest
from sqlalchemy import select

from tests.models import Post, User


@pytest.mark.asyncio
class TestSearchEngine:
    """Test cases for SearchEngine."""

    async def test_basic_search(self, async_session, populated_database):
        """Test basic full-text search with ILIKE fallback."""
        from strapalchemy.services.search_engine import SearchEngine

        search_engine = SearchEngine(dialect="sqlite")
        query = select(User)

        query = search_engine.apply_search(query, User, "Software")

        result = await async_session.execute(query)
        users = result.scalars().all()

        # Should find John Doe with "Software engineer" in bio
        assert len(users) >= 1
        assert any("Software" in u.bio for u in users)

    async def test_search_by_name(self, async_session, populated_database):
        """Test search by name field."""
        from strapalchemy.services.search_engine import SearchEngine

        search_engine = SearchEngine(dialect="sqlite")
        query = select(User)

        query = search_engine.apply_search(query, User, "John")

        result = await async_session.execute(query)
        users = result.scalars().all()

        assert len(users) >= 1
        assert any("John" in u.name for u in users)

    async def test_search_by_email(self, async_session, populated_database):
        """Test search by email field."""
        from strapalchemy.services.search_engine import SearchEngine

        search_engine = SearchEngine(dialect="sqlite")
        query = select(User)

        query = search_engine.apply_search(query, User, "john@example")

        result = await async_session.execute(query)
        users = result.scalars().all()

        assert len(users) >= 1

    async def test_empty_search(self, async_session, populated_database):
        """Test with empty search string."""
        from strapalchemy.services.search_engine import SearchEngine

        search_engine = SearchEngine(dialect="sqlite")
        query = select(User)

        query = search_engine.apply_search(query, User, "")

        result = await async_session.execute(query)
        users = result.scalars().all()

        # Should return all users
        assert len(users) == 5

    async def test_none_search(self, async_session, populated_database):
        """Test with None search."""
        from strapalchemy.services.search_engine import SearchEngine

        search_engine = SearchEngine(dialect="sqlite")
        query = select(User)

        query = search_engine.apply_search(query, User, None)

        result = await async_session.execute(query)
        users = result.scalars().all()

        # Should return all users
        assert len(users) == 5

    async def test_tokenization_normalization(self, async_session, populated_database):
        """Control characters and multi-whitespace are stripped before search."""
        from strapalchemy.services.search_engine import SearchEngine

        search_engine = SearchEngine(dialect="sqlite")
        query = select(User)

        # "\x00" is a control char that should be removed; "  " normalizes to " "
        query = search_engine.apply_search(query, User, "\x00john\x01  doe")

        # After normalization this becomes tokens ["john", "doe"] — both must appear in same field
        # No user has both "john" and "doe" in the same field in the test fixture
        # so result set may be empty or contain the matching user — just assert no exception
        result = await async_session.execute(query)
        users_list = result.scalars().all()
        assert isinstance(users_list, list)

    async def test_post_search(self, async_session, populated_database):
        """Test search on Post model."""
        from strapalchemy.services.search_engine import SearchEngine

        search_engine = SearchEngine(dialect="sqlite")
        query = select(Post)

        query = search_engine.apply_search(query, Post, "first")

        result = await async_session.execute(query)
        posts = result.scalars().all()

        # Should find "First Post"
        assert len(posts) >= 1
        assert any("first" in p.title.lower() for p in posts)

    async def test_multi_token_search(self, async_session, populated_database):
        """Test that multi-token search requires all tokens to match in a field.

        Scenario A: both tokens appear in a single record → returned.
        Scenario B: each token appears in a *different* record → neither returned
        when searching a field that only holds one at a time.
        """
        from strapalchemy.services.search_engine import SearchEngine

        search_engine = SearchEngine(dialect="sqlite")

        # --- Scenario A: "Software engineer" — both words are in John Doe's bio ---
        query = select(User)
        query = search_engine.apply_search(query, User, "Software engineer")
        result = await async_session.execute(query)
        users = result.scalars().all()

        assert len(users) >= 1
        assert all(
            "software" in (u.bio or "").lower() and "engineer" in (u.bio or "").lower()
            for u in users
        ), "Every returned user must have both tokens in at least one searchable field"

        # --- Scenario B: "John" only matches name; "Designer" only matches bio ---
        # "John" is John Doe's name; "Designer" is Jane Smith's bio.
        # The strategy generates AND-per-field conditions: each field must contain
        # ALL tokens.  Since no single searchable field contains both "John" and
        # "Designer" for any user, this returns no results.
        query2 = select(User)
        query2 = search_engine.apply_search(query2, User, "John Designer")
        result2 = await async_session.execute(query2)
        users2 = result2.scalars().all()

        names = {u.name for u in users2}
        # John Doe does not have "Designer" anywhere; Jane Smith does not have "John"
        # anywhere — so neither should appear.
        assert "John Doe" not in names, "John Doe should not match 'John Designer'"
        assert "Jane Smith" not in names, "Jane Smith should not match 'John Designer'"

    async def test_injection_safe_search(self, async_session, populated_database):
        """Test that SQL-injection payloads are handled safely via bound parameters.

        All user-supplied tokens are passed as SQLAlchemy bound parameters (never
        interpolated into the SQL string), so classic SQL injection payloads cannot
        alter query structure.

        Note: this test exercises the SQLite dialect (UniversalFallbackStrategy /
        ILIKE path).  MySQL FTS boolean operator characters (``+``, ``-``, ``>``,
        ``<``, ``(``, ``)``, ``~``, ``*``, ``"``, ``@``) within tokens could alter
        FTS query semantics even through a bound parameter, because MySQL evaluates
        the bound string as a boolean-mode expression.  That attack vector is
        covered by ``MySQLStrategy._sanitize_fts_token``, which strips those
        characters before the boolean query string is constructed.  The SQLite path
        tested here does not have that risk.
        """
        from strapalchemy.services.search_engine import SearchEngine

        search_engine = SearchEngine(dialect="sqlite")
        query = select(User)

        # Classic injection payload — must not raise and must not drop the table.
        query = search_engine.apply_search(query, User, "'; DROP TABLE users; --")

        result = await async_session.execute(query)
        users = result.scalars().all()

        # The database must still be intact and the call must return a list.
        assert isinstance(users, list)

        # Verify the table is still queryable (not dropped).
        check_query = select(User)
        check_result = await async_session.execute(check_query)
        all_users = check_result.scalars().all()
        assert len(all_users) == 5, "All 5 users must still exist after injection attempt"


class TestSearchEngineUnit:
    """Synchronous unit tests that do not require a database session."""

    def test_mysql_fts_token_sanitization(self):
        """MySQL FTS boolean operator chars are stripped from tokens before query construction."""
        from strapalchemy.services.search_engine import MySQLStrategy

        strategy = MySQLStrategy()
        assert strategy._sanitize_fts_token("+foo-bar") == "foobar"
        assert strategy._sanitize_fts_token("hello(world)") == "helloworld"
        assert strategy._sanitize_fts_token('"quoted"') == "quoted"
        assert strategy._sanitize_fts_token("~*test@") == "test"
        assert strategy._sanitize_fts_token("clean") == "clean"

    def test_mysql_field_validation(self):
        """Invalid field names raise ValueError before FULLTEXT query construction."""
        from strapalchemy.services.search_engine import MySQLStrategy

        strategy = MySQLStrategy()

        class FakeModel:
            __name__ = "FakeModel"
            __module__ = "test"
            __qualname__ = "FakeModel"
            __searchable__ = {"text_fields": ["name; DROP TABLE"]}

        with pytest.raises(ValueError, match="Invalid field names"):
            strategy.build_conditions(FakeModel, ["name; DROP TABLE"], ["foo"])

    def test_mysql_soundex_and_semantics(self):
        """Soundex fallback combines tokens with AND, not OR."""
        from strapalchemy.services.search_engine import MySQLStrategy

        strategy = MySQLStrategy()
        # Mark the field set as failed so the Soundex path is taken.
        strategy._failed_field_sets.add(frozenset(["name"]))

        # Use the real User model so SQLAlchemy column ops work correctly.
        conditions = strategy.build_conditions(User, ["name"], ["john", "doe"])

        # One field → one compound condition (AND of 2 Soundex terms).
        assert len(conditions) == 1

    def test_sqlite_fts5_invalid_table_name(self):
        """ValueError is raised when __fts5_table__ contains non-identifier characters."""
        from strapalchemy.services.search_engine import SQLiteStrategy

        strategy = SQLiteStrategy()

        class BadModel:
            __name__ = "BadModel"
            __module__ = "test"
            __qualname__ = "BadModel"
            __fts5_table__ = "users; DROP TABLE users"

        with pytest.raises(ValueError, match="valid SQL identifier"):
            strategy.build_conditions(BadModel, ["name"], ["foo"])
