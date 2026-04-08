"""
Query optimization utilities to prevent N+1 queries and improve performance.
"""

from typing import Any

from sqlalchemy import Select
from sqlalchemy.ext.asyncio import AsyncSession

from strapalchemy.services._base_query_optimizer import _BaseQueryOptimizer


class QueryOptimizer(_BaseQueryOptimizer):
    """
    Advanced query optimization utilities to prevent N+1 queries and improve performance.
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def execute_optimized_query(
        self,
        query: Select,
        model: type,
        relationships: dict[str, str] | None = None,
        populate: Any | None = None,
        query_type: str = "list",
    ) -> Any:
        """
        Execute query with automatic optimization.

        Args:
            query: SQLAlchemy Select query
            model: SQLAlchemy model class
            relationships: Relationships configuration
            populate: Specific relationships to populate
            query_type: Type of query for optimization

        Returns:
            Query execution result
        """
        # Optimize relationships based on query type
        if relationships:
            optimized_relationships = self.optimize_relationships(relationships, query_type)
            query = self.apply_eager_loading(query, model, optimized_relationships, populate)

        # Execute query
        result = await self.session.execute(query)
        return result
