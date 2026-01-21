"""Type conversion utilities for query building."""

from typing import Any

from sqlalchemy import Boolean, Float, Integer, String


class TypeConverter:
    """Handles type conversion between Python and SQLAlchemy types."""

    @staticmethod
    def convert_value_type(model_field, value: Any) -> Any:
        """Convert filter value to match the model field type.

        This handles common type mismatches like string '1' vs integer 1.

        Args:
            model_field: SQLAlchemy model field
            value: Value to convert

        Returns:
            Converted value matching field type
        """
        if value is None:
            return value

        try:
            field_type = TypeConverter._get_python_type(model_field)

            if field_type and not isinstance(value, field_type):
                return TypeConverter._cast_to_type(value, field_type)

        except (ValueError, TypeError, AttributeError):
            # If conversion fails, return original value and let SQLAlchemy handle it
            pass

        return value

    @staticmethod
    def _get_python_type(model_field):
        """Get Python type from SQLAlchemy field type."""
        if not hasattr(model_field, "type"):
            return None

        type_mapping = {
            Integer: int,
            String: str,
            Boolean: bool,
            Float: float,
        }

        for sqlalchemy_type, python_type in type_mapping.items():
            if isinstance(model_field.type, sqlalchemy_type):
                return python_type

        return None

    @staticmethod
    def _cast_to_type(value: Any, target_type: type) -> Any:
        """Cast value to target type with special handling for booleans."""
        if target_type == int and isinstance(value, str):
            return int(value)
        elif target_type == str and not isinstance(value, str):
            return str(value)
        elif target_type == bool and isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        elif target_type == float and isinstance(value, (str, int)):
            return float(value)

        return value

    @staticmethod
    def convert_list_values(model_field, values: list) -> list:
        """Convert list of values to match field type."""
        return [TypeConverter.convert_value_type(model_field, v) for v in values]
