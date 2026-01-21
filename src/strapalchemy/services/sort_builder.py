"""Enhanced sorting builder for queries with performance optimizations and better error handling."""

from typing import Any, Dict, List, Optional, Tuple, Type

from sqlalchemy import Select, asc, desc

from strapalchemy.logging.logger import logger
from strapalchemy.models.base import Base


class SortBuilder:
    """Enhanced sorting builder with performance optimizations and better error handling."""

    def __init__(self, model: Type[Base]):
        self.model = model
        # Cache for relationship metadata to avoid repeated introspection
        self._relationship_cache: Dict[str, Any] = {}
        # Cache for field validation
        self._field_cache: Dict[str, bool] = {}

    async def apply_sorting(self, query: Select, sort_config: Optional[List[str]]) -> Select:
        """Apply enhanced sorting to the query with performance optimizations and better error handling.

        Features:
        - Cached relationship introspection
        - Optimized join handling
        - Better error recovery
        - Performance monitoring
        - Support for complex nested sorting

        Examples:
            # Sort by direct field
            sort_config = ['name:asc']

            # Sort by relationship field
            sort_config = ['organization.name:desc']

            # Multiple sorts
            sort_config = ['organization.name:asc', 'created_at:desc']

        Args:
            query: SQLAlchemy Select query
            sort_config: List of sort directives (e.g., ['name:asc', 'id:desc'])

        Returns:
            Modified query with sorting applied
        """
        if not sort_config:
            # Use stable sorting by created_at desc, then id for consistency
            # This ensures consistent ordering even when new records are added
            if hasattr(self.model, "created_at"):
                return query.order_by(desc(self.model.created_at), self.model.id)
            elif hasattr(self.model, "id"):
                return query.order_by(self.model.id)
            return query

        if isinstance(sort_config, str):
            sort_config = [sort_config]

        try:
            order_clauses = []
            joins_needed = []
            applied_joins = set()  # Track applied joins to avoid duplicates

            for sort_directive in sort_config:
                try:
                    field_name, direction = self._parse_sort_directive(sort_directive)

                    # Check if this is a nested sort (contains ".")
                    if "." in field_name:
                        model_field, relation_joins = self._get_nested_field_for_sorting(field_name)
                        # Add joins in order, avoiding duplicates
                        for join_rel in relation_joins:
                            if join_rel not in applied_joins:
                                joins_needed.append(join_rel)
                                applied_joins.add(join_rel)
                    else:
                        # Get the model field for direct sorting
                        model_field = self._get_model_field(field_name)

                    if model_field is not None:
                        if direction == "desc":
                            order_clauses.append(desc(model_field))
                        else:
                            order_clauses.append(asc(model_field))
                    else:
                        logger.warning(f"Field '{field_name}' not found for sorting, skipping")

                except Exception as e:
                    logger.warning(f"Failed to process sort directive '{sort_directive}': {e}")
                    continue

            # Apply joins needed for sorting using outerjoin to prevent losing records
            # that don't have the relationship
            for join_relation in joins_needed:
                try:
                    query = query.outerjoin(join_relation)
                except Exception as e:
                    logger.warning(f"Failed to apply join for sorting: {e}")
                    continue

            if order_clauses:
                query = query.order_by(*order_clauses)
            else:
                # Fallback to default sort
                if hasattr(self.model, "id"):
                    query = query.order_by(self.model.id)

            return query

        except Exception as e:
            logger.error(f"Critical error in sorting application: {e}")
            # Return original query if sorting fails completely
            return query

    @staticmethod
    def _parse_sort_directive(sort_directive: str) -> Tuple[str, str]:
        """Parse sort directive into field name and direction.

        Args:
            sort_directive: Sort directive (e.g., 'name:asc' or 'name')

        Returns:
            Tuple of (field_name, direction)
        """
        if ":" in sort_directive:
            field_name, direction = sort_directive.split(":", 1)
            field_name = field_name.strip()
            direction = direction.strip().lower()
        else:
            field_name = sort_directive.strip()
            direction = "asc"

        return field_name, direction

    def _get_nested_field_for_sorting(self, field_path: str) -> Tuple[Any, List]:
        """Get nested field and required joins for sorting with caching.

        Args:
            field_path: Dotted path like 'organization.name' or 'user.role.name'

        Returns:
            Tuple of (field_object, list_of_joins_needed)
        """
        # Use cache to avoid repeated introspection
        cache_key = f"nested_field_{field_path}"
        if cache_key in self._relationship_cache:
            return self._relationship_cache[cache_key]

        try:
            parts = field_path.split(".")
            if len(parts) < 2:
                # Not a nested field
                result = self._get_model_field(field_path), []
                self._relationship_cache[cache_key] = result
                return result

            # Start from the base model
            current_model = self.model
            joins_needed = []

            # Navigate through relationships
            for i, part in enumerate(parts[:-1]):
                if not hasattr(current_model, part):
                    result = None, []
                    self._relationship_cache[cache_key] = result
                    return result

                relation_attr = getattr(current_model, part)

                # Check if it's a relationship
                if not hasattr(relation_attr.property, "mapper"):
                    result = None, []
                    self._relationship_cache[cache_key] = result
                    return result

                # Add this relationship to joins (in order)
                joins_needed.append(relation_attr)

                # Get the related model for next iteration
                current_model = relation_attr.property.mapper.class_

            # Get the final field from the last model in the chain
            final_field_name = parts[-1]
            if hasattr(current_model, final_field_name):
                final_field = getattr(current_model, final_field_name)
                result = final_field, joins_needed
            else:
                result = None, []

            self._relationship_cache[cache_key] = result
            return result

        except Exception as e:
            logger.error(f"Error getting nested field for sorting: {e}")
            result = None, []
            self._relationship_cache[cache_key] = result
            return result

    def _get_model_field(self, field_path: str):
        """Get model field attribute from field path with enhanced validation.

        Args:
            field_path: Field path (can be dotted for nested access)

        Returns:
            Model field attribute or None
        """
        # Use cache for field validation
        if field_path in self._field_cache:
            if not self._field_cache[field_path]:
                return None

        try:
            if not field_path or ".." in field_path or field_path.startswith("_"):
                self._field_cache[field_path] = False
                return None

            parts = field_path.split(".")
            model_field = self.model

            for part in parts:
                if hasattr(model_field, part):
                    model_field = getattr(model_field, part)
                else:
                    self._field_cache[field_path] = False
                    return None

            self._field_cache[field_path] = True
            return model_field
        except Exception as e:
            logger.warning(f"Error getting model field '{field_path}': {e}")
            self._field_cache[field_path] = False
            return None
