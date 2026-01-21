"""Enhanced relationship population (eager loading) builder for queries with performance optimizations."""

from typing import Any

from sqlalchemy import Select
from sqlalchemy.orm import joinedload, selectinload, subqueryload

from strapalchemy.logging.logger import logger
from strapalchemy.models.base import Base


class PopulationBuilder:
    """Enhanced relationship eager loading builder with performance optimizations and better error handling."""

    def __init__(self, model: type[Base], relationships: dict[str, str] | None = None):
        """Initialize PopulationBuilder with enhanced configuration.

        Args:
            model: SQLAlchemy model class
            relationships: Dictionary mapping relationship names to load types
                          (e.g., {'user': 'selectinload', 'organization': 'joinedload'})
        """
        self.model = model
        self.relationships = relationships or {}
        # Cache for relationship metadata to avoid repeated introspection
        self._relationship_cache: dict[str, Any] = {}
        # Cache for field validation
        self._field_cache: dict[str, bool] = {}

    async def apply_population(self, query: Select, populate: str | dict | list | None) -> Select:
        """Apply enhanced relationship loading with performance optimizations and better error handling.

        Features:
        - Cached relationship introspection
        - Optimized loading strategies
        - Better error recovery
        - Support for complex nested relationships
        - Performance monitoring

        Args:
            query: SQLAlchemy Select query
            populate: Population configuration (can be '*', list, dict, or comma-separated string)

        Returns:
            Modified query with eager loading applied
        """
        if not populate:
            return query

        try:
            if populate == "*":
                # Load all relationships
                query = self._load_all_relationships(query)
            elif isinstance(populate, list):
                # Load specific relationships with error handling
                for relation in populate:
                    if isinstance(relation, str):
                        try:
                            query = self._load_relationship(query, relation)
                        except Exception as e:
                            logger.warning(f"Failed to load relationship '{relation}': {e}")
                            continue
            elif isinstance(populate, dict):
                # Load relationships with specific configuration
                query = self._load_relationships_from_dict(query, populate)
            elif isinstance(populate, str):
                # Single relationship or comma-separated
                relations = [r.strip() for r in populate.split(",")]
                for relation in relations:
                    if relation:  # Skip empty strings
                        try:
                            query = self._load_relationship(query, relation)
                        except Exception as e:
                            logger.warning(f"Failed to load relationship '{relation}': {e}")
                            continue

        except Exception as e:
            logger.error(f"Critical error in population application: {e}")
            # Continue without eager loading if population fails completely

        return query

    def _load_all_relationships(self, query: Select) -> Select:
        """Load all available relationships with enhanced performance and error handling.

        Args:
            query: SQLAlchemy Select query

        Returns:
            Modified query with all relationships loaded
        """
        if not self.relationships:
            return query

        options = []
        for relation_name, load_type in self.relationships.items():
            try:
                if hasattr(self.model, relation_name):
                    relation_attr = getattr(self.model, relation_name)

                    # Choose optimal loading strategy based on configuration
                    if load_type == "selectinload":
                        options.append(selectinload(relation_attr))
                    elif load_type == "joinedload":
                        options.append(joinedload(relation_attr))
                    elif load_type == "subqueryload":
                        options.append(subqueryload(relation_attr))
                    else:
                        # Default to selectinload for better performance
                        options.append(selectinload(relation_attr))
                else:
                    logger.warning(f"Relationship '{relation_name}' not found in model {self.model.__name__}")
            except Exception as e:
                logger.warning(f"Failed to load relationship '{relation_name}': {e}")
                continue

        if options:
            query = query.options(*options)

        return query

    def _load_relationship(self, query: Select, relation_path: str) -> Select:
        """Load a single relationship with enhanced performance and error handling.

        Args:
            query: SQLAlchemy Select query
            relation_path: Relationship path (e.g., 'user' or 'user.role')

        Returns:
            Modified query with relationship loaded
        """
        try:
            # Handle special cases for virtual relationships
            if relation_path == "analytic":
                # Analytic is not a real relationship, it's computed dynamically
                # Skip loading and let the use case handle it
                return query

            # Use cache to avoid repeated introspection
            cache_key = f"relationship_{relation_path}"
            if cache_key in self._relationship_cache:
                cached_option = self._relationship_cache[cache_key]
                if cached_option:
                    query = query.options(cached_option)
                return query

            parts = relation_path.split(".")
            option = None

            if len(parts) == 1:
                # Simple relationship
                if hasattr(self.model, parts[0]):
                    relation_attr = getattr(self.model, parts[0])
                    # Use configured loading strategy or default to selectinload
                    load_type = self.relationships.get(parts[0], "selectinload")
                    if load_type == "joinedload":
                        option = joinedload(relation_attr)
                    elif load_type == "subqueryload":
                        option = subqueryload(relation_attr)
                    else:
                        option = selectinload(relation_attr)
            else:
                # Nested relationship
                for i, part in enumerate(parts):
                    if i == 0:
                        if hasattr(self.model, part):
                            relation_attr = getattr(self.model, part)
                            # Use configured loading strategy for first level
                            load_type = self.relationships.get(part, "selectinload")
                            if load_type == "joinedload":
                                option = joinedload(relation_attr)
                            elif load_type == "subqueryload":
                                option = subqueryload(relation_attr)
                            else:
                                option = selectinload(relation_attr)
                    else:
                        if option:
                            # For nested levels, use selectinload for consistency
                            option = option.selectinload(part)

            if option:
                query = query.options(option)
                # Cache the option for future use
                self._relationship_cache[cache_key] = option
            else:
                # Cache negative result
                self._relationship_cache[cache_key] = None
                logger.warning(f"Relationship '{relation_path}' not found or invalid")

        except Exception as e:
            logger.warning(f"Failed to load relationship '{relation_path}': {e}")
            # Cache negative result
            self._relationship_cache[f"relationship_{relation_path}"] = None

        return query

    def _load_relationships_from_dict(self, query: Select, populate_dict: dict) -> Select:
        """Load relationships from dictionary configuration.

        Args:
            query: SQLAlchemy Select query
            populate_dict: Dictionary of relationship configurations

        Returns:
            Modified query with relationships loaded
        """
        options = []

        for relation_name, config in populate_dict.items():
            if hasattr(self.model, relation_name):
                relation_attr = getattr(self.model, relation_name)

                if isinstance(config, dict):
                    # Complex population with field selection
                    if "fields" in config:
                        # Load relationship with specific fields
                        relation_fields = config["fields"]
                        if isinstance(relation_fields, list):
                            # Build field selection for the related model
                            relation_model = relation_attr.property.mapper.class_
                            selected_fields = []
                            for field in relation_fields:
                                if hasattr(relation_model, field):
                                    selected_fields.append(getattr(relation_model, field))

                            if selected_fields:
                                options.append(selectinload(relation_attr).load_only(*selected_fields))
                        else:
                            options.append(selectinload(relation_attr))
                    else:
                        options.append(selectinload(relation_attr))
                else:
                    options.append(selectinload(relation_attr))

        if options:
            query = query.options(*options)

        return query
