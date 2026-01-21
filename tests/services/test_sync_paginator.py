"""Tests for SyncPaginator."""

import pytest
from sqlalchemy import select

from strapalchemy.services.sync_paginator import SyncPaginator
from tests.models import User


class TestSyncPaginator:
    """Test cases for SyncPaginator."""

    def test_page_based_pagination(self, sync_session, populated_sync_database):
        """Test page-based pagination."""
        paginator = SyncPaginator(sync_session, User)
        query = select(User)

        query, meta = paginator.apply_pagination(query, {"page": 1, "page_size": 2})

        result = sync_session.execute(query)
        users = result.scalars().all()

        assert len(users) == 2
        assert meta["page"] == 1
        assert meta["page_size"] == 2
        assert meta["total"] == 5
        assert meta["page_count"] == 3
        assert meta["has_next"] is True
        assert meta["has_previous"] is False

    def test_offset_based_pagination(self, sync_session, populated_sync_database):
        """Test offset-based pagination."""
        paginator = SyncPaginator(sync_session, User)
        query = select(User)

        query, meta = paginator.apply_pagination(query, {"start": 2, "limit": 2})

        result = sync_session.execute(query)
        users = result.scalars().all()

        assert len(users) == 2
        assert meta["start"] == 2
        assert meta["limit"] == 2
        assert meta["total"] == 5
        assert meta["has_next"] is True
        assert meta["has_previous"] is True

    def test_default_pagination(self, sync_session, populated_sync_database):
        """Test default pagination when no page/start is specified."""
        paginator = SyncPaginator(sync_session, User)
        query = select(User)

        # Pass pagination config without 'page' or 'start' - should use default
        # The _apply_default_pagination is called when neither 'page' nor 'start' is in the config
        query, meta = paginator.apply_pagination(query, {"page_size": 10})

        result = sync_session.execute(query)
        users = result.scalars().all()

        # When using default pagination, it uses hardcoded limit of 25
        assert meta["page"] == 1
        assert meta["page_size"] == 25  # Default limit
        assert meta["total"] == 5

    def test_none_pagination_returns_query_without_meta(self, sync_session, populated_sync_database):
        """Test that None pagination returns original query without meta."""
        paginator = SyncPaginator(sync_session, User)
        query = select(User)

        query, meta = paginator.apply_pagination(query, None)

        assert meta is None

        result = sync_session.execute(query)
        users = result.scalars().all()

        # Should return all users since no pagination was applied
        assert len(users) == 5

    def test_invalid_page_defaults_to_1(self, sync_session, populated_sync_database):
        """Test that invalid page number defaults to 1."""
        paginator = SyncPaginator(sync_session, User)
        query = select(User)

        query, meta = paginator.apply_pagination(query, {"page": "invalid", "page_size": 2})

        assert meta["page"] == 1

    def test_page_size_capped_at_100(self, sync_session, populated_sync_database):
        """Test that page_size is capped at 100 for performance."""
        paginator = SyncPaginator(sync_session, User)
        query = select(User)

        query, meta = paginator.apply_pagination(query, {"page": 1, "page_size": 200})

        assert meta["page_size"] == 100

    def test_count_caching(self, sync_session, populated_sync_database):
        """Test that count queries are cached."""
        paginator = SyncPaginator(sync_session, User)
        query = select(User)

        # First call - should execute count query
        query1, meta1 = paginator.apply_pagination(query, {"page": 1, "page_size": 2})

        # Second call with same query - should use cached count
        query2, meta2 = paginator.apply_pagination(query, {"page": 2, "page_size": 2})

        # Both should have the same total count
        assert meta1["total"] == 5
        assert meta2["total"] == 5
        assert len(paginator._count_cache) == 1

    def test_last_page(self, sync_session, populated_sync_database):
        """Test pagination on last page."""
        paginator = SyncPaginator(sync_session, User)
        query = select(User)

        query, meta = paginator.apply_pagination(query, {"page": 3, "page_size": 2})

        result = sync_session.execute(query)
        users = result.scalars().all()

        assert len(users) == 1  # Only 1 user on last page
        assert meta["has_next"] is False
        assert meta["has_previous"] is True
