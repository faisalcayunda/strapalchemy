"""Tests for FieldSelector."""

import pytest
from sqlalchemy import select

from tests.models import User


@pytest.mark.asyncio
class TestFieldSelector:
    """Test cases for FieldSelector."""

    async def test_select_single_field(self, async_session, populated_database):
        """Test selecting a single field."""
        from strapalchemy.services.field_selector import FieldSelector

        field_selector = FieldSelector(User)
        query = select(User)

        query = await field_selector.apply_field_selection(query, ["name"])

        result = await async_session.execute(query)
        users = result.scalars().all()

        assert len(users) == 5
        # ID should be auto-included
        # Only id and name should be loaded
        for user in users:
            assert user.id is not None
            assert user.name is not None

    async def test_select_multiple_fields(self, async_session, populated_database):
        """Test selecting multiple fields."""
        from strapalchemy.services.field_selector import FieldSelector

        field_selector = FieldSelector(User)
        query = select(User)

        query = await field_selector.apply_field_selection(query, ["name", "email", "age"])

        result = await async_session.execute(query)
        users = result.scalars().all()

        assert len(users) == 5

    async def test_select_relationship_field(self, async_session, populated_database):
        """Test selecting relationship field."""
        from strapalchemy.services.field_selector import FieldSelector

        field_selector = FieldSelector(User)
        query = select(User)

        query = await field_selector.apply_field_selection(query, ["id", "name", "organization.slug"])

        result = await async_session.execute(query)
        users = result.scalars().all()

        assert len(users) == 5

    async def test_no_field_selection(self, async_session, populated_database):
        """Test without field selection."""
        from strapalchemy.services.field_selector import FieldSelector

        field_selector = FieldSelector(User)
        query = select(User)

        query = await field_selector.apply_field_selection(query, None)

        result = await async_session.execute(query)
        users = result.scalars().all()

        assert len(users) == 5

    async def test_empty_field_list(self, async_session, populated_database):
        """Test with empty field list."""
        from strapalchemy.services.field_selector import FieldSelector

        field_selector = FieldSelector(User)
        query = select(User)

        query = await field_selector.apply_field_selection(query, [])

        result = await async_session.execute(query)
        users = result.scalars().all()

        assert len(users) == 5

    async def test_invalid_field_skipped(self, async_session, populated_database):
        """Test with invalid field name (should be skipped)."""
        from strapalchemy.services.field_selector import FieldSelector

        field_selector = FieldSelector(User)
        query = select(User)

        query = await field_selector.apply_field_selection(query, ["name", "invalid_field"])

        result = await async_session.execute(query)
        users = result.scalars().all()

        assert len(users) == 5

    async def test_selected_fields_property(self, async_session, populated_database):
        """Test the selected_fields property."""
        from strapalchemy.services.field_selector import FieldSelector

        field_selector = FieldSelector(User)
        query = select(User)

        fields = ["name", "email", "age"]
        query = await field_selector.apply_field_selection(query, fields)

        assert field_selector.selected_fields == fields
