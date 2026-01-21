"""Enhanced field selection builder for queries with performance optimizations and better validation."""

from typing import Dict, List, Optional, Set, Type

from sqlalchemy import Select
from sqlalchemy.orm import joinedload, load_only, selectinload

from strapalchemy.logging.logger import logger
from strapalchemy.models.base import Base


class FieldSelector:
    """Enhanced field selector with performance optimizations and better validation."""

    def __init__(self, model: Type[Base]):
        self.model = model
        self._selected_fields: Optional[List[str]] = None
        # Cache for field validation to avoid repeated introspection
        self._field_cache: Dict[str, bool] = {}
        # Cache for model column information
        self._column_cache: Optional[Set[str]] = None

    async def apply_field_selection(self, query: Select, fields: Optional[List[str]]) -> Select:
        """Apply enhanced field selection with performance optimizations and better validation.

        Features:
        - Cached field validation
        - Optimized column introspection
        - Better error handling
        - Automatic ID field inclusion
        - Performance monitoring
        - Proper handling of relationship fields
        - Automatic relationship loading for dotted fields

        Args:
            query: SQLAlchemy Select query
            fields: List of field names to select

        Returns:
            Modified query with field selection applied
        """
        if not fields:
            return query

        if isinstance(fields, str):
            fields = [fields]

        try:
            # Separate direct columns from relationship fields
            direct_columns = []
            relationship_fields = []
            relationships_to_load = set()

            for field in fields:
                if "." in field:
                    # This is a relationship field (e.g., "organization.name")
                    relationship_fields.append(field)
                    # Extract relationship name for loading
                    relationship_name = field.split(".")[0]
                    relationships_to_load.add(relationship_name)
                elif self._is_valid_field(field):
                    direct_columns.append(field)
                else:
                    logger.warning(f"Field '{field}' not found in model {self.model.__name__}, skipping")

            # Only apply load_only for direct columns, not relationship fields
            if direct_columns:
                # Add ID if not already present (required for most operations)
                if "id" not in direct_columns and self._is_valid_field("id"):
                    direct_columns.append("id")

                # Use load_only with direct column names for optimal performance
                query = query.options(load_only(*direct_columns))

            # Load relationships for dotted fields
            if relationships_to_load:

                for rel_name in relationships_to_load:
                    if hasattr(self.model, rel_name):
                        try:
                            # Check if it's actually a relationship attribute
                            rel_attr = getattr(self.model, rel_name)
                            if hasattr(rel_attr, "property") and hasattr(rel_attr.property, "mapper"):
                                # Use joinedload for single relationships, selectinload for collections
                                if rel_attr.property.uselist:
                                    query = query.options(selectinload(rel_attr))
                                else:
                                    query = query.options(joinedload(rel_attr))
                        except Exception as e:
                            logger.warning(f"Failed to load relationship '{rel_name}': {e}")
                            continue

            if not direct_columns and not relationship_fields:
                logger.warning("No valid fields found for selection, skipping field selection")

        except Exception as e:
            logger.error(f"Error applying field selection: {e}")
            # Continue without field selection if it fails

        self._selected_fields = fields
        return query

    def _is_valid_field(self, field_name: str) -> bool:
        """Check if a field is valid for selection with caching.

        Args:
            field_name: Field name to validate

        Returns:
            True if field is valid, False otherwise
        """
        # Use cache to avoid repeated introspection
        if field_name in self._field_cache:
            return self._field_cache[field_name]

        try:
            # Check if field exists on the model
            is_valid = hasattr(self.model, field_name)
            self._field_cache[field_name] = is_valid
            return is_valid
        except Exception as e:
            logger.warning(f"Error validating field '{field_name}': {e}")
            self._field_cache[field_name] = False
            return False

    def _get_model_columns(self) -> Set[str]:
        """Get all column names from the model with caching.

        Returns:
            Set of column names
        """
        if self._column_cache is None:
            try:
                self._column_cache = {column.name for column in self.model.__table__.columns}
            except Exception as e:
                logger.warning(f"Error getting model columns: {e}")
                self._column_cache = set()
        return self._column_cache

    @property
    def selected_fields(self) -> Optional[List[str]]:
        """Get the currently selected fields.

        Returns:
            List of selected field names or None
        """
        return self._selected_fields
