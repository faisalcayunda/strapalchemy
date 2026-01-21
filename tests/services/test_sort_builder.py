"""Tests for SortBuilder."""

import pytest
from sqlalchemy import select

from tests.models import User


@pytest.mark.asyncio
class TestSortBuilder:
    """Test cases for SortBuilder."""

    async def test_sort_ascending(self, async_session, populated_database):
        """Test ascending sort."""
        from strapalchemy.services.sort_builder import SortBuilder

        sort_builder = SortBuilder(User)
        query = select(User)

        query = await sort_builder.apply_sorting(query, ["name:asc"])

        result = await async_session.execute(query)
        users = result.scalars().all()

        assert len(users) == 5
        names = [u.name for u in users]
        assert names == sorted(names)

    async def test_sort_descending(self, async_session, populated_database):
        """Test descending sort."""
        from strapalchemy.services.sort_builder import SortBuilder

        sort_builder = SortBuilder(User)
        query = select(User)

        query = await sort_builder.apply_sorting(query, ["age:desc"])

        result = await async_session.execute(query)
        users = result.scalars().all()

        assert len(users) == 5
        ages = [u.age for u in users]
        assert ages == sorted(ages, reverse=True)

    async def test_sort_multiple_fields(self, async_session, populated_database):
        """Test sorting by multiple fields."""
        from strapalchemy.services.sort_builder import SortBuilder

        sort_builder = SortBuilder(User)
        query = select(User)

        # Sort by status descending, then by age ascending
        query = await sort_builder.apply_sorting(query, ["status:desc", "age:asc"])

        result = await async_session.execute(query)
        users = result.scalars().all()

        assert len(users) == 5
        # Verify sorting
        prev_status = None
        prev_age = None
        for user in users:
            if prev_status == user.status:
                # Within same status, age should be ascending
                if prev_age is not None:
                    assert user.age >= prev_age
                prev_age = user.age
            else:
                prev_status = user.status
                prev_age = user.age

    async def test_sort_default_direction(self, async_session, populated_database):
        """Test sort without explicit direction (defaults to asc)."""
        from strapalchemy.services.sort_builder import SortBuilder

        sort_builder = SortBuilder(User)
        query = select(User)

        query = await sort_builder.apply_sorting(query, ["name"])

        result = await async_session.execute(query)
        users = result.scalars().all()

        assert len(users) == 5
        names = [u.name for u in users]
        assert names == sorted(names)

    async def test_sort_by_relationship_field(self, async_session, populated_database):
        """Test sorting by relationship field."""
        from strapalchemy.services.sort_builder import SortBuilder

        sort_builder = SortBuilder(User)
        query = select(User)

        query = await sort_builder.apply_sorting(query, ["organization.name:asc"])

        result = await async_session.execute(query)
        users = result.scalars().all()

        assert len(users) == 5
        org_names = [u.organization.name for u in users]
        # Should be sorted by organization name
        assert org_names == sorted(org_names)

    async def test_no_sort_config(self, async_session, populated_database):
        """Test query without sort configuration."""
        from strapalchemy.services.sort_builder import SortBuilder

        sort_builder = SortBuilder(User)
        query = select(User)

        query = await sort_builder.apply_sorting(query, None)

        result = await async_session.execute(query)
        users = result.scalars().all()

        assert len(users) == 5

    async def test_invalid_field_skipped(self, async_session, populated_database):
        """Test with invalid field name (should be skipped)."""
        from strapalchemy.services.sort_builder import SortBuilder

        sort_builder = SortBuilder(User)
        query = select(User)

        query = await sort_builder.apply_sorting(query, ["invalid_field:asc"])

        result = await async_session.execute(query)
        users = result.scalars().all()

        assert len(users) == 5
