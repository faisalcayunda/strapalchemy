"""Enhanced high-performance model serialization utility with memory optimizations."""

from collections.abc import Mapping
from datetime import date, datetime
from decimal import Decimal
from ipaddress import IPv4Address, IPv6Address
from typing import Any
from uuid import UUID

from sqlalchemy.inspection import inspect

from strapalchemy.logging.logger import logger


class ModelSerializer:
    """Enhanced high-performance SQLAlchemy model serializer with memory optimizations and better error handling."""

    # Enhanced type converter dispatch table for better performance
    _TYPE_CONVERTERS = {
        datetime: lambda v: v.isoformat(),
        date: lambda v: v.isoformat(),
        Decimal: lambda v: float(v),
        UUID: lambda v: str(v),
        IPv4Address: lambda v: str(v),
        IPv6Address: lambda v: str(v),
    }

    # Cache for model introspection to avoid repeated calls
    _model_cache: dict[str, Any] = {}

    @classmethod
    def serialize(
        cls,
        models: Any,
        fields: list[str] = None,
        populate: str | dict[str, Any] | list[str] = None,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """Serialize SQLAlchemy models to dictionaries with enhanced performance and memory optimizations.

        Features:
        - Memory-efficient serialization
        - Cached model introspection
        - Better error handling
        - Optimized field processing
        - Reduced memory allocation

        Args:
            models: Single model or list/tuple of models
            fields: List of field names to include (supports dot notation for nested fields)
            populate: Relationships to include (can be '*', list, dict, or comma-separated string)

        Returns:
            Dictionary or list of dictionaries
        """
        # Early return for None
        if models is None:
            return {}

        try:
            # Normalize fields input
            if isinstance(fields, str):
                fields = [fields]

            # Parse populate configuration with caching
            populate_dict = cls._parse_populate(populate)

            # Pre-parse fields to separate dotted and regular fields
            dotted_fields, regular_fields = cls._parse_fields(fields)

            # Handle list/tuple vs single model with optimized processing
            if isinstance(models, (list, tuple)):
                # Use list comprehension for better performance
                return [
                    cls._serialize_single_model(model, regular_fields, dotted_fields, populate_dict)
                    for model in models
                ]

            return cls._serialize_single_model(models, regular_fields, dotted_fields, populate_dict)

        except Exception as e:
            logger.error(f"Error in model serialization: {e}")
            # Return empty result if serialization fails
            return [] if isinstance(models, (list, tuple)) else {}

    @classmethod
    def _serialize_single_model(cls, model, regular_fields, dotted_fields, populate_dict) -> dict[str, Any]:
        """Serialize a single model instance with enhanced performance and memory optimization.

        Args:
            model: SQLAlchemy model instance
            regular_fields: List of regular field names
            dotted_fields: List of tuples (field_name, field_parts) for nested fields
            populate_dict: Dictionary of relationships to populate

        Returns:
            Dictionary representation of model
        """
        try:
            item = {}
            model_dict = model.__dict__  # Cache model dict access

            # Get model state with caching
            model_key = f"{model.__class__.__name__}_{id(model)}"
            if model_key not in cls._model_cache:
                state = inspect(model)
                cls._model_cache[model_key] = state
            else:
                state = cls._model_cache[model_key]

            unloaded = state.unloaded

            # Serialize fields with optimized processing
            if regular_fields or dotted_fields:
                # Process regular fields (faster path)
                for field in regular_fields:
                    # Always try to serialize explicitly requested fields, even if unloaded
                    # This ensures consistent response structure
                    try:
                        value = model_dict.get(field) or getattr(model, field, None)
                        item[field] = cls._convert_to_serializable(value)
                    except (AttributeError, Exception) as e:
                        logger.warning(f"Failed to serialize field '{field}': {e}")
                        # Set to None for missing fields to maintain consistent structure
                        item[field] = None
                        continue

                # Process dotted fields - create nested structure
                for field_name, field_parts in dotted_fields:
                    try:
                        value = cls._get_nested_attr(model, field_parts)
                        cls._set_nested_value(item, field_parts, cls._convert_to_serializable(value))
                    except Exception as e:
                        logger.warning(f"Failed to serialize dotted field '{field_name}': {e}")
                        # Set to None for missing relationship fields to maintain consistent structure
                        cls._set_nested_value(item, field_parts, None)
                        continue
            else:
                # Serialize all columns with optimized processing
                if hasattr(model, "to_dict"):
                    try:
                        item = model.to_dict()
                    except Exception as e:
                        logger.warning(f"Failed to use model.to_dict(): {e}")
                        item = {}
                else:
                    # Serialize all columns consistently - always include all columns for consistent response
                    for column in model.__table__.columns:
                        col_name = column.name
                        try:
                            # Try to get the value from model_dict first, then from model attributes
                            value = model_dict.get(col_name)
                            if value is None and hasattr(model, col_name):
                                value = getattr(model, col_name, None)

                            # Only set to None if the column is actually unloaded AND we can't get the value
                            if col_name in unloaded and value is None:
                                # For truly unloaded columns, set to None to maintain consistent structure
                                item[col_name] = None
                            else:
                                # Use the actual value if available
                                item[col_name] = cls._convert_to_serializable(value)
                        except Exception as e:
                            logger.warning(f"Failed to serialize column '{col_name}': {e}")
                            # Always include the field with None value for consistency
                            item[col_name] = None
                            continue

            # Serialize relationships according to populate param
            if populate_dict == "*":
                cls._serialize_all_relationships(model, model_dict, item)
            elif populate_dict:
                cls._serialize_selected_relationships(model, model_dict, item, populate_dict)

            return item

        except Exception as e:
            logger.error(f"Critical error serializing model: {e}")
            return {}

    @classmethod
    def _serialize_all_relationships(cls, model, model_dict, item):
        """Serialize all relationships with proper unloaded state checking."""
        state = inspect(model)
        unloaded = state.unloaded

        for relationship in model.__mapper__.relationships:
            rel_key = relationship.key

            # Always include relationship field for consistency
            if rel_key in unloaded:
                # Set to None for unloaded relationships to maintain consistent structure
                item[rel_key] = None
                continue

            if rel_key in model_dict:
                rel_value = model_dict[rel_key]
                if rel_value is not None:
                    if isinstance(rel_value, list):
                        item[rel_key] = [cls.serialize(r) for r in rel_value]
                    else:
                        item[rel_key] = cls.serialize(rel_value)
                else:
                    # Include field even if None for consistency
                    item[rel_key] = None
            else:
                # Include field even if not in model_dict for consistency
                item[rel_key] = None

    @classmethod
    def _serialize_selected_relationships(cls, model, model_dict, item, populate_dict):
        """Serialize selected relationships based on populate configuration.

        Args:
            model: SQLAlchemy model instance
            model_dict: Model's __dict__ cache
            item: Dictionary to update with serialized relationships
            populate_dict: Dictionary of relationships to populate
        """
        state = inspect(model)
        unloaded = state.unloaded

        for rel_key, rel_populate in populate_dict.items():
            # Always include relationship field for consistency
            if rel_key in unloaded:
                # Set to None for unloaded relationships to maintain consistent structure
                item[rel_key] = None
                continue

            # Use dict get for faster lookup
            rel_value = model_dict.get(rel_key)
            if rel_value is None and hasattr(model, rel_key):
                rel_value = getattr(model, rel_key)

            if rel_value is not None:
                if isinstance(rel_value, list):
                    item[rel_key] = [cls.serialize(r, populate=rel_populate) for r in rel_value]
                else:
                    item[rel_key] = cls.serialize(rel_value, populate=rel_populate)
            else:
                # Include field even if None for consistency
                item[rel_key] = None

    @classmethod
    def _convert_to_serializable(cls, value: Any) -> Any:
        """Convert value to JSON-serializable type.

        Args:
            value: Value to convert

        Returns:
            JSON-serializable value
        """
        if value is None:
            return value

        value_type = type(value)

        # Fast path for common types
        converter = cls._TYPE_CONVERTERS.get(value_type)
        if converter:
            return converter(value)

        # Fallback for subclasses
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        elif isinstance(value, (IPv4Address, IPv6Address)):
            return str(value)
        elif hasattr(value, "__table__"):
            # Nested SQLAlchemy model (check __table__ only, faster)
            return cls.serialize(value)

        return value

    @staticmethod
    def _get_nested_attr(obj, attr_parts):
        """Get nested attribute from object using dotted path.

        Args:
            obj: Object to traverse
            attr_parts: List of attribute names

        Returns:
            Nested attribute value or None
        """
        value = obj
        for part in attr_parts:
            value = getattr(value, part, None)
            if value is None:
                break
        return value

    @staticmethod
    def _set_nested_value(target_dict, path_parts, value):
        """Set a value in nested dict structure.

        Example: set_nested_value(dict, ["dataset", "name"], "value")
        Results in: {"dataset": {"name": "value"}}

        Args:
            target_dict: Dictionary to update
            path_parts: List of keys representing the nested path
            value: Value to set
        """
        current = target_dict
        for part in path_parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[path_parts[-1]] = value

    @staticmethod
    def _parse_fields(fields):
        """Parse fields list into dotted and regular fields.

        Args:
            fields: List of field names (may include dot notation)

        Returns:
            Tuple of (dotted_fields, regular_fields)
        """
        dotted_fields = []
        regular_fields = []

        if fields:
            for field in fields:
                if "." in field:
                    # Pre-split and cache the parts
                    dotted_fields.append((field, field.split(".")))
                else:
                    regular_fields.append(field)

        return dotted_fields, regular_fields

    @staticmethod
    def _parse_populate(populate_param):
        """Parse populate parameter into normalized dictionary.

        Args:
            populate_param: Population configuration (can be '*', list, dict, or string)

        Returns:
            Dictionary or '*' representing relationships to populate
        """
        if not populate_param:
            return {}
        if populate_param == "*":
            return "*"

        # Helper function to convert dot notation to nested dict
        def build_nested_dict(path: str) -> dict[str, Any]:
            """Convert dot notation path to nested dict.

            Example: "user.role" -> {"user": {"role": {}}}
            """
            parts = path.strip().split(".")
            if len(parts) == 1:
                return {}

            result = {}
            current = result
            for part in parts[:-1]:
                current[part] = {}
                current = current[part]
            # Last part gets empty dict
            current[parts[-1]] = {}

            return result[parts[0]] if parts else {}

        # Helper to merge nested dicts
        def merge_nested(target: dict, source: dict) -> None:
            """Merge source dict into target dict recursively."""
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    merge_nested(target[key], value)
                else:
                    target[key] = value

        if isinstance(populate_param, str):
            # Handle comma-separated string: "user,organization" or "user.role,organization"
            result = {}
            for s in populate_param.split(","):
                s = s.strip()
                if not s:
                    continue
                if "." in s:
                    # Nested relationship: "user.role"
                    parts = s.split(".")
                    first_key = parts[0]
                    nested = build_nested_dict(s)
                    if first_key in result:
                        if nested:
                            merge_nested(result[first_key], nested)
                    else:
                        result[first_key] = nested
                else:
                    # Simple relationship: "user"
                    if s not in result:
                        result[s] = {}
            return result

        if isinstance(populate_param, list):
            # Handle list: ["user.role", "organization"] or ["user", "organization"]
            result = {}
            for item in populate_param:
                if not isinstance(item, str):
                    continue
                item = item.strip()
                if not item:
                    continue
                if "." in item:
                    # Nested relationship: "user.role"
                    parts = item.split(".")
                    first_key = parts[0]
                    nested = build_nested_dict(item)
                    if first_key in result:
                        if nested:
                            merge_nested(result[first_key], nested)
                    else:
                        result[first_key] = nested
                else:
                    # Simple relationship: "user"
                    if item not in result:
                        result[item] = {}
            return result

        if isinstance(populate_param, Mapping):
            return dict(populate_param)

        return {}
