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

        search_engine = SearchEngine()
        # Force ILIKE mode for SQLite compatibility
        search_engine._force_ilike = True
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

        search_engine = SearchEngine()
        search_engine._force_ilike = True
        query = select(User)

        query = search_engine.apply_search(query, User, "John")

        result = await async_session.execute(query)
        users = result.scalars().all()

        assert len(users) >= 1
        assert any("John" in u.name for u in users)

    async def test_search_by_email(self, async_session, populated_database):
        """Test search by email field."""
        from strapalchemy.services.search_engine import SearchEngine

        search_engine = SearchEngine()
        search_engine._force_ilike = True
        query = select(User)

        query = search_engine.apply_search(query, User, "john@example")

        result = await async_session.execute(query)
        users = result.scalars().all()

        assert len(users) >= 1

    async def test_empty_search(self, async_session, populated_database):
        """Test with empty search string."""
        from strapalchemy.services.search_engine import SearchEngine

        search_engine = SearchEngine()
        query = select(User)

        query = search_engine.apply_search(query, User, "")

        result = await async_session.execute(query)
        users = result.scalars().all()

        # Should return all users
        assert len(users) == 5

    async def test_none_search(self, async_session, populated_database):
        """Test with None search."""
        from strapalchemy.services.search_engine import SearchEngine

        search_engine = SearchEngine()
        query = select(User)

        query = search_engine.apply_search(query, User, None)

        result = await async_session.execute(query)
        users = result.scalars().all()

        # Should return all users
        assert len(users) == 5

    async def test_search_sanitization(self, async_session, populated_database):
        """Test that search query is sanitized."""
        from strapalchemy.services.search_engine import SearchEngine

        search_engine = SearchEngine()
        search_engine._force_ilike = True
        query = select(User)

        # Potentially dangerous input - gets sanitized
        query = search_engine.apply_search(query, User, "'; DROP TABLE users; --")

        # Should not raise an error
        result = await async_session.execute(query)
        users = result.scalars().all()

        # Query should execute safely
        assert isinstance(users, list)

    async def test_post_search(self, async_session, populated_database):
        """Test search on Post model."""
        from strapalchemy.services.search_engine import SearchEngine

        search_engine = SearchEngine()
        search_engine._force_ilike = True
        query = select(Post)

        query = search_engine.apply_search(query, Post, "first")

        result = await async_session.execute(query)
        posts = result.scalars().all()

        # Should find "First Post"
        assert len(posts) >= 1
        assert any("first" in p.title.lower() for p in posts)
