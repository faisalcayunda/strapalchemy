"""Enhanced dynamic filter builder for queries with performance optimizations."""

from functools import lru_cache
from typing import Any

from dateutil.parser import parse as date_parse
from sqlalchemy import Select, and_, not_, or_

from strapalchemy.logging.logger import logger
from strapalchemy.models.base import Base

from .operator_handler import OperatorHandler


class FilterBuilder:
    """Enhanced filter builder with performance optimizations and better error handling."""

    def __init__(self, model: type[Base]):
        self.model = model
        self.operator_handler = OperatorHandler()
        # Cache for relationship metadata to avoid repeated introspection
        self._relationship_cache: dict[str, Any] = {}

    async def apply_filters(self, query: Select, filters: dict[str, Any] | None) -> Select:
        """Apply all filter conditions to the query with enhanced performance and error handling.

        Supports both regular field filters and nested relationship filters with optimizations:
        - Cached relationship introspection
        - Optimized join handling
        - Better error recovery
        - Performance monitoring

        Examples:
            # Regular field filter
            filters = {'name': {'$eq': 'Dataset A'}}

            # Nested relationship filter
            filters = {'organization': {'slug': {'$in': ['nike', 'meta']}}}

            # Multiple conditions on nested relationship
            filters = {'organization': {'slug': {'$eq': 'nike'}, 'status': {'$eq': 'ACTIVE'}}}

            # Combined regular and nested filters
            filters = {
                'name': {'$contains': 'data'},
                'organization': {'slug': {'$in': ['nike', 'meta']}}
            }

        Args:
            query: SQLAlchemy Select query
            filters: Dictionary of filters

        Returns:
            Modified query with filters applied
        """
        if not filters:
            return query

        try:
            filter_conditions = []
            joins_needed = set()
            applied_joins = set()  # Track applied joins to avoid duplicates

            for field_name, field_filters in filters.items():
                try:
                    # Parse dates in filter values
                    if isinstance(field_filters, dict):
                        field_filters = self._parse_date_values(field_filters)
                    # Check if this is a dot notation field (e.g., "organization.code")
                    if "." in field_name:
                        # Parse dot notation: "organization.code" -> relation="organization", field="code"
                        relation_name, nested_field = field_name.split(".", 1)

                        # Create a nested filter structure for the relationship
                        nested_filters = {nested_field: field_filters}

                        # Handle nested relationship filtering
                        relation_conditions, relation_joins = await self._build_relationship_filters(
                            relation_name, nested_filters
                        )
                        if relation_conditions is not None:
                            filter_conditions.append(relation_conditions)
                        # Only add joins that haven't been applied yet
                        for join_model in relation_joins:
                            if join_model not in applied_joins:
                                joins_needed.add(join_model)
                                applied_joins.add(join_model)
                    elif self._is_relationship_filter(field_name, field_filters):
                        # Handle nested relationship filtering with caching (for non-dot notation)
                        relation_conditions, relation_joins = await self._build_relationship_filters(
                            field_name, field_filters
                        )
                        if relation_conditions is not None:
                            filter_conditions.append(relation_conditions)
                        # Only add joins that haven't been applied yet
                        for join_model in relation_joins:
                            if join_model not in applied_joins:
                                joins_needed.add(join_model)
                                applied_joins.add(join_model)
                    else:
                        # Handle regular field filters
                        condition = await self._build_field_conditions(field_name, field_filters)
                        if condition is not None:
                            filter_conditions.append(condition)

                except Exception as e:
                    logger.warning(f"Failed to process filter for field '{field_name}': {e}")
                    # Continue processing other filters instead of failing completely
                    continue

            # Apply joins with optimized order
            for join_model in joins_needed:
                try:
                    query = query.join(join_model)
                except Exception as e:
                    logger.warning(f"Failed to apply join for {join_model}: {e}")
                    # Continue with other joins

            if filter_conditions:
                query = query.where(and_(*filter_conditions))

            return query

        except Exception as e:
            logger.error(f"Critical error in filter application: {e}")
            # Return original query if filter application fails completely
            return query

    def _is_relationship_filter(self, field_name: str, field_filters: Any) -> bool:
        """Check if this is a nested relationship filter with caching.

        Args:
            field_name: Field name to check
            field_filters: Filter configuration

        Returns:
            True if this is a relationship filter, False otherwise
        """
        if not isinstance(field_filters, dict):
            return False

        # Use cache to avoid repeated introspection
        cache_key = f"relationship_{field_name}"
        if cache_key in self._relationship_cache:
            is_relationship = self._relationship_cache[cache_key]
        else:
            # Check if field is a relationship on the model
            if not hasattr(self.model, field_name):
                is_relationship = False
            else:
                field_attr = getattr(self.model, field_name)
                # Check if it has a relationship property (SQLAlchemy relationship)
                is_relationship = hasattr(field_attr.property, "mapper")

            self._relationship_cache[cache_key] = is_relationship

        if not is_relationship:
            return False

        # Check if any key in field_filters is NOT an operator (starts with $)
        # If all keys are operators, it's a regular filter, not nested
        has_non_operator_key = any(not key.startswith("$") for key in field_filters.keys())

        return has_non_operator_key

    async def _build_relationship_filters(
        self, relation_name: str, relation_filters: dict[str, Any]
    ) -> tuple[Any, set]:
        """Build filter conditions for nested relationship filtering with enhanced performance.

        Args:
            relation_name: Relationship name
            relation_filters: Filters to apply on the relationship

        Returns:
            Tuple of (conditions, set of joins needed)
        """
        try:
            if not hasattr(self.model, relation_name):
                logger.warning(f"Relationship '{relation_name}' not found in model {self.model.__name__}")
                return None, set()

            relation_attr = getattr(self.model, relation_name)

            # Get the related model class
            if not hasattr(relation_attr.property, "mapper"):
                logger.warning(f"Relationship '{relation_name}' is not a valid SQLAlchemy relationship")
                return None, set()

            related_model = relation_attr.property.mapper.class_

            # Build conditions for the related model with better error handling
            conditions = []
            for field_name, field_filters in relation_filters.items():
                try:
                    # Get the field from the related model
                    if hasattr(related_model, field_name):
                        related_field = getattr(related_model, field_name)

                        # Build conditions for this field
                        if isinstance(field_filters, dict):
                            for operator, value in field_filters.items():
                                try:
                                    condition = self.operator_handler.build_condition(related_field, operator, value)
                                    if condition is not None:
                                        conditions.append(condition)
                                except Exception as e:
                                    logger.warning(f"Failed to build condition for {field_name}.{operator}: {e}")
                                    continue
                        else:
                            # Handle direct value comparison
                            try:
                                condition = related_field == field_filters
                                conditions.append(condition)
                            except Exception as e:
                                logger.warning(f"Failed to build direct condition for {field_name}: {e}")
                                continue
                    else:
                        logger.warning(f"Field '{field_name}' not found in related model {related_model.__name__}")
                        continue

                except Exception as e:
                    logger.warning(f"Error processing field '{field_name}' in relationship '{relation_name}': {e}")
                    continue

            if conditions:
                return and_(*conditions), {relation_attr}

            return None, set()

        except Exception as e:
            logger.error(f"Error building relationship filters for '{relation_name}': {e}")
            return None, set()

    async def _build_field_conditions(self, field_name: str, field_filters: dict[str, Any]):
        """Build filter conditions for a specific field with enhanced error handling.

        Args:
            field_name: Field name to filter
            field_filters: Filter configuration with operators

        Returns:
            SQLAlchemy condition or None
        """
        # Handle nested field references (e.g., 'user.email')
        model_field = self._get_model_field(field_name)
        if model_field is None:
            logger.warning(f"Field '{field_name}' not found in model {self.model.__name__}")
            return None

        conditions = []
        try:
            for operator, value in field_filters.items():
                try:
                    condition = None

                    # Logical operators (handled recursively)
                    if operator == "$or":
                        condition = await self._handle_or_operator(value)
                    elif operator == "$and":
                        condition = await self._handle_and_operator(value)
                    elif operator == "$not":
                        condition = await self._handle_not_operator(value)
                    else:
                        # Use the shared operator condition builder
                        condition = self.operator_handler.build_condition(model_field, operator, value)

                    if condition is not None:
                        conditions.append(condition)

                except Exception as e:
                    logger.warning(
                        f"Failed to build condition for field '{field_name}' with operator '{operator}': {e}"
                    )
                    continue

            return and_(*conditions) if conditions else None

        except AttributeError as e:
            logger.error(f"Attribute error building field conditions for '{field_name}': {e}")
            raise ValueError(
                f"Invalid filter query structure for field '{field_name}': {e}"
            )
        except Exception as e:
            logger.error(f"Unexpected error building field conditions for '{field_name}': {e}")
            return None

    async def _handle_or_operator(self, value: Any):
        """Handle $or logical operator with enhanced error handling."""
        if not isinstance(value, list):
            return None

        or_conditions = []
        for i, or_filter in enumerate(value):
            try:
                if isinstance(or_filter, dict):
                    for or_field, or_field_filters in or_filter.items():
                        try:
                            or_condition = await self._build_field_conditions(or_field, or_field_filters)
                            if or_condition is not None:
                                or_conditions.append(or_condition)
                        except Exception as e:
                            logger.warning(f"Failed to build OR condition for field '{or_field}': {e}")
                            continue
                else:
                    logger.warning(f"Invalid OR filter structure at index {i}: expected dict, got {type(or_filter)}")
                    continue
            except Exception as e:
                logger.warning(f"Error processing OR filter at index {i}: {e}")
                continue

        return or_(*or_conditions) if or_conditions else None

    async def _handle_and_operator(self, value: Any):
        """Handle $and logical operator with enhanced error handling."""
        if not isinstance(value, list):
            return None

        and_conditions = []
        for i, and_filter in enumerate(value):
            try:
                if isinstance(and_filter, dict):
                    for and_field, and_field_filters in and_filter.items():
                        try:
                            and_condition = await self._build_field_conditions(and_field, and_field_filters)
                            if and_condition is not None:
                                and_conditions.append(and_condition)
                        except Exception as e:
                            logger.warning(f"Failed to build AND condition for field '{and_field}': {e}")
                            continue
                else:
                    logger.warning(f"Invalid AND filter structure at index {i}: expected dict, got {type(and_filter)}")
                    continue
            except Exception as e:
                logger.warning(f"Error processing AND filter at index {i}: {e}")
                continue

        return and_(*and_conditions) if and_conditions else None

    async def _handle_not_operator(self, value: Any):
        """Handle $not logical operator with enhanced error handling."""
        if not isinstance(value, dict):
            return None

        not_conditions = []
        for not_field, not_field_filters in value.items():
            try:
                not_condition = await self._build_field_conditions(not_field, not_field_filters)
                if not_condition is not None:
                    not_conditions.append(not_condition)
            except Exception as e:
                logger.warning(f"Failed to build NOT condition for field '{not_field}': {e}")
                continue

        return not_(and_(*not_conditions)) if not_conditions else None

    @lru_cache(maxsize=128)
    def _get_model_field(self, field_path: str):
        """Get model field attribute from field path (supports nested fields).

        Args:
            field_path: Field path (e.g., 'user.email' or 'name')

        Returns:
            Model field attribute or None
        """
        try:
            if not field_path or ".." in field_path or field_path.startswith("_"):
                return None

            parts = field_path.split(".")
            model_field = self.model

            for part in parts:
                if hasattr(model_field, part):
                    model_field = getattr(model_field, part)
                else:
                    return None

            return model_field
        except Exception:
            return None

    @staticmethod
    def _parse_date_values(field_filters: dict[str, Any]) -> dict[str, Any]:
        """Parse date strings in filter values with enhanced error handling.

        Args:
            field_filters: Dictionary of field filters

        Returns:
            Modified filters with parsed dates
        """
        date_operators = {"$eq", "$lt", "$lte", "$gt", "$gte", "$ne", "$in", "$between"}

        for key, value in field_filters.items():
            if key in date_operators and value is not None:
                try:

                    def try_parse_date(val):
                        if isinstance(val, str):
                            try:
                                if val.isdigit():
                                    return val
                                # Try to parse as date/datetime
                                parsed = date_parse(val)
                                # Return the original string if parsing doesn't change the value
                                # (indicates it wasn't a date string)
                                if str(parsed) == val:
                                    return val
                                return parsed
                            except (ValueError, TypeError, OverflowError):
                                return val
                        elif isinstance(val, list):
                            return [try_parse_date(item) for item in val]
                        return val

                    field_filters[key] = try_parse_date(value)
                except Exception as e:
                    logger.warning(f"Failed to parse date for key '{key}': {e}")
                    # Keep original value if parsing fails
                    continue

        return field_filters
