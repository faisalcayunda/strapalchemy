"""
Query optimization utilities to prevent N+1 queries and improve performance (Sync version).
"""

from typing import Any

from sqlalchemy import Select
from sqlalchemy.orm import Session, contains_eager, joinedload, selectinload

from strapalchemy.logging.logger import logger


class SyncQueryOptimizer:
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

    def __init__(self, session: Session):
        self.session = session

    @staticmethod
    def apply_eager_loading(
        query: Select,
        model: type,
        relationships: dict[str, str] | None = None,
        populate: Any | None = None,
        strategy: str = "selectinload",
    ) -> Select:
        """
        Apply eager loading to prevent N+1 queries.

        Args:
            query: SQLAlchemy Select query
            model: SQLAlchemy model class
            relationships: Dictionary mapping relationship names to loading strategies
            populate: Specific relationships to populate
            strategy: Default loading strategy ('selectinload' or 'joinedload')

        Returns:
            Optimized query with eager loading
        """
        if not relationships:
            return query

        # Determine which relationships to load
        relationships_to_load = set()

        if populate:
            if isinstance(populate, dict):
                relationships_to_load.update(populate.keys())
            elif isinstance(populate, list):
                relationships_to_load.update(populate)
        else:
            # Load all relationships by default to prevent N+1
            relationships_to_load.update(relationships.keys())

        # Apply eager loading for each relationship
        for rel_name in relationships_to_load:
            if rel_name in relationships:
                load_strategy = relationships[rel_name]
                try:
                    if load_strategy == "selectinload":
                        query = query.options(selectinload(getattr(model, rel_name)))
                    elif load_strategy == "joinedload":
                        query = query.options(joinedload(getattr(model, rel_name)))
                    elif load_strategy == "contains_eager":
                        query = query.options(contains_eager(getattr(model, rel_name)))
                    else:
                        # Fallback to default strategy
                        if strategy == "selectinload":
                            query = query.options(selectinload(getattr(model, rel_name)))
                        else:
                            query = query.options(joinedload(getattr(model, rel_name)))
                except AttributeError as e:
                    logger.warning(f"Relationship '{rel_name}' not found on model {model.__name__}: {e}")
                    continue

        return query

    @staticmethod
    def optimize_relationships(relationships: dict[str, str], query_type: str = "list") -> dict[str, str]:
        """
        Optimize relationship loading strategies based on query type.

        Args:
            relationships: Original relationships configuration
            query_type: Type of query ('list', 'detail', 'count')

        Returns:
            Optimized relationships configuration
        """
        optimized = relationships.copy()

        if query_type == "list":
            # For list queries, prefer selectinload for better performance
            for rel_name, strategy in optimized.items():
                if strategy == "joinedload":
                    optimized[rel_name] = "selectinload"
        elif query_type == "detail":
            # For detail queries, prefer joinedload for single record
            for rel_name, strategy in optimized.items():
                if strategy == "selectinload":
                    optimized[rel_name] = "joinedload"
        elif query_type == "count":
            # For count queries, disable relationship loading
            return {}

        return optimized

    @staticmethod
    def detect_n_plus_one_risks(
        model: type, relationships: dict[str, str], query_plan: str | None = None
    ) -> list[str]:
        """
        Detect potential N+1 query risks in relationships.

        Args:
            model: SQLAlchemy model class
            relationships: Relationships configuration
            query_plan: Optional query execution plan for analysis

        Returns:
            List of potential N+1 risks
        """
        risks = []

        for rel_name, strategy in relationships.items():
            try:
                relationship = getattr(model, rel_name)
                # Check if relationship is configured for lazy loading
                if hasattr(relationship.property, "lazy") and relationship.property.lazy == "select":
                    risks.append(f"Relationship '{rel_name}' uses lazy loading - potential N+1 risk")
            except AttributeError:
                risks.append(f"Relationship '{rel_name}' not found on model {model.__name__}")

        return risks

    @staticmethod
    def create_batch_loader(model: type, relationships: dict[str, str], batch_size: int = 100) -> dict[str, Any]:
        """
        Create batch loading configuration for large datasets.

        Args:
            model: SQLAlchemy model class
            relationships: Relationships configuration
            batch_size: Size of batches for loading

        Returns:
            Batch loading configuration
        """
        batch_config = {"batch_size": batch_size, "relationships": relationships, "model": model}

        # Configure batch loading for each relationship
        for rel_name, strategy in relationships.items():
            if strategy == "selectinload":
                batch_config[f"{rel_name}_batch_size"] = batch_size

        return batch_config

    @staticmethod
    def analyze_query_complexity(query: Select, relationships: dict[str, str]) -> dict[str, Any]:
        """
        Analyze query complexity and provide optimization recommendations.

        Args:
            query: SQLAlchemy Select query
            relationships: Relationships configuration

        Returns:
            Analysis results with recommendations
        """
        analysis = {
            "total_relationships": len(relationships),
            "eager_loaded_relationships": 0,
            "lazy_loaded_relationships": 0,
            "recommendations": [],
        }

        # Count relationship loading strategies
        for strategy in relationships.values():
            if strategy in ["selectinload", "joinedload", "contains_eager"]:
                analysis["eager_loaded_relationships"] += 1
            else:
                analysis["lazy_loaded_relationships"] += 1

        # Generate recommendations
        if analysis["lazy_loaded_relationships"] > 0:
            analysis["recommendations"].append("Consider using eager loading to prevent N+1 queries")

        if analysis["total_relationships"] > 5:
            analysis["recommendations"].append(
                "Consider reducing the number of loaded relationships for better performance"
            )

        return analysis

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

    @staticmethod
    def get_loading_strategy_recommendations(relationship_type: str, expected_count: int) -> str:
        """
        Get recommended loading strategy based on relationship type and expected count.

        Args:
            relationship_type: Type of relationship ('one-to-many', 'many-to-one', 'many-to-many')
            expected_count: Expected number of related records

        Returns:
            Recommended loading strategy
        """
        if relationship_type == "one-to-many":
            if expected_count > 100:
                return "selectinload"
            else:
                return "joinedload"
        elif relationship_type == "many-to-one":
            return "joinedload"
        elif relationship_type == "many-to-many":
            return "selectinload"
        else:
            return "selectinload"
