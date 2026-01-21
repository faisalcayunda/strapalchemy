"""Operator condition builder for query filters."""

import ast
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import func

from .type_converter import TypeConverter

# Supported Strapi-style operators
STRAPI_OPERATORS: set[str] = {
    "$eq",
    "$eqi",
    "$ne",
    "$nei",
    "$lt",
    "$lte",
    "$gt",
    "$gte",
    "$in",
    "$notIn",
    "$contains",
    "$notContains",
    "$containsi",
    "$notContainsi",
    "$null",
    "$notNull",
    "$between",
    "$startsWith",
    "$startsWithi",
    "$endsWith",
    "$endsWithi",
    "$or",
    "$and",
    "$not",
}


class OperatorHandler:
    """Builds SQLAlchemy conditions from Strapi-style operators."""

    def __init__(self):
        self.type_converter = TypeConverter()
        # Dispatch table for better performance
        self._operator_handlers = {
            "$eq": self._handle_eq,
            "$eqi": self._handle_eqi,
            "$ne": self._handle_ne,
            "$nei": self._handle_nei,
            "$lt": self._handle_lt,
            "$lte": self._handle_lte,
            "$gt": self._handle_gt,
            "$gte": self._handle_gte,
            "$in": self._handle_in,
            "$notIn": self._handle_not_in,
            "$contains": self._handle_contains,
            "$notContains": self._handle_not_contains,
            "$containsi": self._handle_containsi,
            "$notContainsi": self._handle_not_containsi,
            "$startsWith": self._handle_starts_with,
            "$startsWithi": self._handle_starts_withi,
            "$endsWith": self._handle_ends_with,
            "$endsWithi": self._handle_ends_withi,
            "$null": self._handle_null,
            "$notNull": self._handle_not_null,
            "$between": self._handle_between,
        }

    def build_condition(self, model_field, operator: str, value: Any) -> Any | None:
        """Build a single operator condition for a field.

        Args:
            model_field: SQLAlchemy model field
            operator: Strapi-style operator (e.g., '$eq', '$in')
            value: Filter value

        Returns:
            SQLAlchemy condition or None if operator not found
        """
        handler = self._operator_handlers.get(operator)
        if handler:
            return handler(model_field, value)
        return None

    # Equality operators
    def _handle_eq(self, field, value):
        value = self.type_converter.convert_value_type(field, value)
        return field == value

    def _handle_eqi(self, field, value):
        return func.lower(field) == func.lower(str(value))

    def _handle_ne(self, field, value):
        value = self.type_converter.convert_value_type(field, value)
        return field != value

    def _handle_nei(self, field, value):
        return func.lower(field) != func.lower(str(value))

    # Comparison operators
    def _handle_lt(self, field, value):
        value = self.type_converter.convert_value_type(field, value)
        return field < value

    def _handle_lte(self, field, value):
        value = self.type_converter.convert_value_type(field, value)
        # Handle date/datetime special case
        if hasattr(field.type, "python_type") and field.type.python_type in (date, datetime):
            if isinstance(value, (date, datetime)):
                value = value + timedelta(days=1)
        return field < value

    def _handle_gt(self, field, value):
        value = self.type_converter.convert_value_type(field, value)
        return field > value

    def _handle_gte(self, field, value):
        value = self.type_converter.convert_value_type(field, value)
        return field >= value

    # Array operators
    def _handle_in(self, field, value):
        parsed_values = self._parse_array_value(value)
        if isinstance(parsed_values, (list, tuple)):
            converted = self.type_converter.convert_list_values(field, parsed_values)
            return field.in_(converted)
        return None

    def _handle_not_in(self, field, value):
        parsed_values = self._parse_array_value(value)
        if isinstance(parsed_values, (list, tuple)):
            converted = self.type_converter.convert_list_values(field, parsed_values)
            return ~field.in_(converted)
        return None

    # String operators
    def _handle_contains(self, field, value):
        return field.contains(str(value))

    def _handle_not_contains(self, field, value):
        return ~field.contains(str(value))

    def _handle_containsi(self, field, value):
        return func.lower(field).contains(func.lower(str(value)))

    def _handle_not_containsi(self, field, value):
        return ~func.lower(field).contains(func.lower(str(value)))

    def _handle_starts_with(self, field, value):
        return field.startswith(str(value))

    def _handle_starts_withi(self, field, value):
        return func.lower(field).startswith(func.lower(str(value)))

    def _handle_ends_with(self, field, value):
        return field.endswith(str(value))

    def _handle_ends_withi(self, field, value):
        return func.lower(field).endswith(func.lower(str(value)))

    # Null operators
    def _handle_null(self, field, value):
        return field.is_(None) if value else field.isnot(None)

    def _handle_not_null(self, field, value):
        return field.isnot(None) if value else field.is_(None)

    # Range operator
    def _handle_between(self, field, value):
        if isinstance(value, str):
            value = [value]
        if isinstance(value, list) and len(value) == 2:
            return field.between(value[0], value[1])
        return None

    @staticmethod
    def _parse_array_value(value: Any) -> Any:
        """Parse string array representation into actual list."""
        if isinstance(value, str):
            stripped = value.strip()
            # Check if it's array-like string
            if (stripped.startswith("[") and stripped.endswith("]")) or (
                stripped.startswith("(") and stripped.endswith(")")
            ):
                try:
                    parsed = ast.literal_eval(stripped)
                    if isinstance(parsed, (list, tuple)):
                        return list(parsed)
                except Exception:
                    pass
            # Handle comma-separated values
            if "," in value:
                return [v.strip() for v in value.split(",")]
            return [value]
        return value

    @staticmethod
    def is_operator(key: str) -> bool:
        """Check if key is a valid operator."""
        return key in STRAPI_OPERATORS
