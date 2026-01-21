"""Tests for Paginator."""

import pytest
from sqlalchemy import select

from tests.models import User


@pytest.mark.asyncio
class TestPaginator:
    """Test cases for Paginator."""

    async def test_page_based_pagination(self, async_session, populated_database):
        """Test page-based pagination."""
        from strapalchemy.services.paginator import Paginator

        paginator = Paginator(async_session, User)
        query = select(User)

        query, meta = await paginator.apply_pagination(query, {"page": 1, "page_size": 2})

        result = await async_session.execute(query)
        users = result.scalars().all()

        assert len(users) == 2
        assert meta["page"] == 1
        assert meta["page_size"] == 2
        assert meta["total"] == 5
        assert meta["page_count"] == 3
        assert meta["has_next"] is True
        assert meta["has_previous"] is False

    async def test_second_page(self, async_session, populated_database):
        """Test second page of pagination."""
        from strapalchemy.services.paginator import Paginator

        paginator = Paginator(async_session, User)
        query = select(User)

        query, meta = await paginator.apply_pagination(query, {"page": 2, "page_size": 2})

        result = await async_session.execute(query)
        users = result.scalars().all()

        assert len(users) == 2
        assert meta["page"] == 2
        assert meta["has_next"] is True
        assert meta["has_previous"] is True

    async def test_last_page(self, async_session, populated_database):
        """Test last page of pagination."""
        from strapalchemy.services.paginator import Paginator

        paginator = Paginator(async_session, User)
        query = select(User)

        query, meta = await paginator.apply_pagination(query, {"page": 3, "page_size": 2})

        result = await async_session.execute(query)
        users = result.scalars().all()

        assert len(users) == 1
        assert meta["page"] == 3
        assert meta["has_next"] is False
        assert meta["has_previous"] is True

    async def test_offset_based_pagination(self, async_session, populated_database):
        """Test offset-based pagination."""
        from strapalchemy.services.paginator import Paginator

        paginator = Paginator(async_session, User)
        query = select(User)

        query, meta = await paginator.apply_pagination(query, {"start": 2, "limit": 2})

        result = await async_session.execute(query)
        users = result.scalars().all()

        assert len(users) == 2
        assert meta["start"] == 2
        assert meta["limit"] == 2
        assert meta["total"] == 5
        assert meta["has_next"] is True
        assert meta["has_previous"] is True

    async def test_no_pagination(self, async_session, populated_database):
        """Test query without pagination."""
        from strapalchemy.services.paginator import Paginator

        paginator = Paginator(async_session, User)
        query = select(User)

        query, meta = await paginator.apply_pagination(query, None)

        assert meta is None

        result = await async_session.execute(query)
        users = result.scalars().all()

        assert len(users) == 5

    async def test_page_size_limit(self, async_session, populated_database):
        """Test that page size is capped at 100."""
        from strapalchemy.services.paginator import Paginator

        paginator = Paginator(async_session, User)
        query = select(User)

        query, meta = await paginator.apply_pagination(query, {"page": 1, "page_size": 200})

        assert meta["page_size"] == 100

    async def test_invalid_page_defaults_to_one(self, async_session, populated_database):
        """Test that invalid page number defaults to 1."""
        from strapalchemy.services.paginator import Paginator

        paginator = Paginator(async_session, User)
        query = select(User)

        query, meta = await paginator.apply_pagination(query, {"page": "invalid", "page_size": 2})

        assert meta["page"] == 1

    async def test_with_count_flag(self, async_session, populated_database):
        """Test with_count flag in pagination."""
        from strapalchemy.services.paginator import Paginator

        paginator = Paginator(async_session, User)
        query = select(User)

        query, meta = await paginator.apply_pagination(
            query, {"page": 1, "page_size": 2, "with_count": True}
        )

        assert meta["with_count"] is True
