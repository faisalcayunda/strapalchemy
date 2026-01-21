"""Enhanced pagination builder for queries with performance optimizations and better error handling."""

from typing import Any

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from strapalchemy.logging.logger import logger
from strapalchemy.models.base import Base


class Paginator:
    """Enhanced pagination builder with performance optimizations and better error handling."""

    def __init__(self, session: AsyncSession, model: type[Base]):
        self.session = session
        self.model = model
        # Cache for count queries to avoid repeated execution
        self._count_cache: dict[str, int] = {}

    async def apply_pagination(
        self, query: Select, pagination: dict[str, Any] | None
    ) -> tuple[Select, dict[str, Any] | None]:
        """Apply enhanced pagination with performance optimizations and better error handling.

        Features:
        - Cached count queries for better performance
        - Optimized count query generation
        - Better error recovery
        - Enhanced metadata generation
        - Support for both page-based and offset-based pagination

        Args:
            query: SQLAlchemy Select query
            pagination: Pagination configuration

        Returns:
            Tuple of (modified query, metadata dict)
        """
        if not pagination:
            return query, None

        try:
            # Get total count with caching
            total_count = await self._get_total_count(query)

            # Page-based pagination
            if "page" in pagination:
                return self._apply_page_based_pagination(query, pagination, total_count)

            # Offset-based pagination
            elif "start" in pagination:
                return self._apply_offset_based_pagination(query, pagination, total_count)

            # Default pagination
            else:
                return self._apply_default_pagination(query, total_count)

        except Exception as e:
            logger.error(f"Error applying pagination: {e}")
            # Return original query without pagination if pagination fails
            return query, None

    def _apply_page_based_pagination(
        self, query: Select, pagination: dict[str, Any], total_count: int
    ) -> tuple[Select, dict[str, Any]]:
        """Apply page-based pagination with enhanced validation and error handling.

        Args:
            query: SQLAlchemy Select query
            pagination: Pagination configuration with 'page' and 'page_size'
            total_count: Total number of records

        Returns:
            Tuple of (modified query, metadata)
        """
        try:
            # Validate and sanitize page number
            try:
                page = int(pagination.get("page", 1))
            except (TypeError, ValueError):
                logger.warning("Invalid page number, defaulting to 1")
                page = 1
            page = max(1, page)

            # Validate and sanitize page size
            try:
                page_size = int(pagination.get("page_size", 25))
            except (TypeError, ValueError):
                logger.warning("Invalid page_size, defaulting to 25")
                page_size = 25
            page_size = max(1, min(100, page_size))  # Cap at 100 for performance

            offset = (page - 1) * page_size
            query = query.offset(offset).limit(page_size)

            # Calculate pagination metadata
            page_count = (total_count + page_size - 1) // page_size if page_size else 1

            meta = {
                "page": page,
                "page_size": page_size,
                "page_count": page_count,
                "total": total_count,
                "has_next": page < page_count,
                "has_previous": page > 1,
            }

            if pagination.get("with_count", False):
                meta["with_count"] = True

            return query, meta

        except Exception as e:
            logger.error(f"Error in page-based pagination: {e}")
            # Return original query with default pagination
            return query.limit(25), {
                "page": 1,
                "page_size": 25,
                "page_count": 1,
                "total": total_count,
                "has_next": False,
                "has_previous": False,
            }

    def _apply_offset_based_pagination(
        self, query: Select, pagination: dict[str, Any], total_count: int
    ) -> tuple[Select, dict[str, Any]]:
        """Apply offset-based pagination with enhanced validation and error handling.

        Args:
            query: SQLAlchemy Select query
            pagination: Pagination configuration with 'start' and 'limit'
            total_count: Total number of records

        Returns:
            Tuple of (modified query, metadata)
        """
        try:
            # Validate and sanitize start offset
            try:
                start = int(pagination.get("start", 0))
            except (TypeError, ValueError):
                logger.warning("Invalid start offset, defaulting to 0")
                start = 0
            start = max(0, start)

            # Validate and sanitize limit
            try:
                limit = int(pagination.get("limit", 25))
            except (TypeError, ValueError):
                logger.warning("Invalid limit, defaulting to 25")
                limit = 25
            limit = max(1, min(100, limit))  # Cap at 100 for performance

            query = query.offset(start).limit(limit)

            # Calculate page and page_count for more informative meta
            page = (start // limit) + 1 if limit else 1
            page_count = (total_count + limit - 1) // limit if limit else 1

            meta = {
                "start": start,
                "limit": limit,
                "page": page,
                "page_size": limit,
                "page_count": page_count,
                "total": total_count,
                "has_next": start + limit < total_count,
                "has_previous": start > 0,
            }

            if pagination.get("with_count", False):
                meta["with_count"] = True

            return query, meta

        except Exception as e:
            logger.error(f"Error in offset-based pagination: {e}")
            # Return original query with default pagination
            return query.limit(25), {
                "start": 0,
                "limit": 25,
                "page": 1,
                "page_size": 25,
                "page_count": 1,
                "total": total_count,
                "has_next": False,
                "has_previous": False,
            }

    def _apply_default_pagination(self, query: Select, total_count: int) -> tuple[Select, dict[str, Any]]:
        """Apply default pagination.

        Args:
            query: SQLAlchemy Select query
            total_count: Total number of records

        Returns:
            Tuple of (modified query, metadata)
        """
        query = query.limit(25)  # Default limit
        page_count = (total_count + 24) // 25
        meta = {
            "page": 1,
            "page_size": 25,
            "page_count": page_count,
            "total": total_count,
        }

        return query, meta

    async def _get_total_count(self, query: Select) -> int:
        """Get total count of records for the query with caching and performance optimizations.

        Args:
            query: SQLAlchemy Select query

        Returns:
            Total count of records
        """
        try:
            # Generate cache key based on query - use more stable key generation
            # This ensures cache consistency across requests
            cache_key = str(hash(str(query.compile(compile_kwargs={"literal_binds": True}))))
            if cache_key in self._count_cache:
                return self._count_cache[cache_key]

            # Remove order by for performance
            query_for_count = query.order_by(None)

            # Remove eager loading options
            query_for_count = query_for_count.options()

            # Use subquery to count distinct rows
            # This handles joins correctly by counting rows from the subquery
            count_query = select(func.count()).select_from(query_for_count.subquery())

            result = await self.session.scalar(count_query)
            count = result or 0

            # Cache the result with size limit to prevent memory issues
            self._count_cache[cache_key] = count

            # Limit cache size to prevent memory issues
            if len(self._count_cache) > 50:
                # Remove oldest entries (simple FIFO)
                oldest_keys = list(self._count_cache.keys())[:25]
                for key in oldest_keys:
                    del self._count_cache[key]

            return count

        except Exception as e:
            logger.error(f"Error counting rows: {e}")
            # Fallback to simple count if subquery fails
            return await self._get_simple_count()

    async def _get_simple_count(self) -> int:
        """Get simple count as fallback.

        Returns:
            Total count of records
        """
        try:
            # Try with id column if available
            if hasattr(self.model, "id"):
                simple_count = select(func.count(self.model.id)).select_from(self.model)
            else:
                # Use count(*) for tables without id
                simple_count = select(func.count()).select_from(self.model)

            if hasattr(self.model, "status"):
                simple_count = simple_count.where(self.model.status != "DELETED")

            result = await self.session.scalar(simple_count)
            return result or 0
        except Exception:
            return 0
