"""
Query optimization utilities to prevent N+1 queries and improve performance (Sync version).
"""

from typing import Any

from sqlalchemy import Select
from sqlalchemy.orm import Session

from strapalchemy.services._base_query_optimizer import _BaseQueryOptimizer


class SyncQueryOptimizer(_BaseQueryOptimizer):
    """
    Advanced query optimization utilities to prevent N+1 queries and improve performance (Sync version).

    Use this class with synchronous SQLAlchemy sessions.

    Example:
        ```python
        from sqlalchemy import create_engine, select
        from sqlalchemy.orm import Session
        from strapalchemy import SyncQueryOptimizer

        engine = create_engine("sqlite:///database.db")
        session = Session(engine)

        optimizer = SyncQueryOptimizer(session)
        query = select(User)

        result = optimizer.execute_optimized_query(query, User)
        users = result.scalars().all()
        ```
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    def execute_optimized_query(
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
        result = self.session.execute(query)
        return result
