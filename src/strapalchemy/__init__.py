"""
StrapAlchemy - Enhanced SQLAlchemy Query Builder Library.

A powerful query builder library for SQLAlchemy that provides advanced
database query capabilities including search, filtering, sorting, pagination,
and more.

Example:
    ```python
    from strapalchemy import FilterBuilder, SortBuilder, Paginator
    from strapalchemy.models import Base

    # Define your model
    class User(Base):
        __tablename__ = "users"
        id = Column(Integer, primary_key=True)
        name = Column(String)
        email = Column(String)

    # Use the builders
    filter_builder = FilterBuilder(User)
    query = select(User)
    query = await filter_builder.apply_filters(query, {"name": {"$contains": "John"}})
    ```
"""

# Services
from strapalchemy.services.field_selector import FieldSelector
from strapalchemy.services.filter_builder import FilterBuilder
from strapalchemy.services.operator_handler import OperatorHandler, STRAPI_OPERATORS
from strapalchemy.services.paginator import Paginator
from strapalchemy.services.population_builder import PopulationBuilder
from strapalchemy.services.query_optimizer import QueryOptimizer
from strapalchemy.services.search_engine import SearchEngine
from strapalchemy.services.serializer import ModelSerializer
from strapalchemy.services.sort_builder import SortBuilder
from strapalchemy.services.type_converter import TypeConverter

# Models
from strapalchemy.models.base import Base

# Logging
from strapalchemy.logging import logger, get_logger, setup_logging_from_ini

__version__ = "0.1.0"

__all__ = [
    # Services
    "FieldSelector",
    "FilterBuilder",
    "OperatorHandler",
    "Paginator",
    "PopulationBuilder",
    "QueryOptimizer",
    "SearchEngine",
    "ModelSerializer",
    "SortBuilder",
    "TypeConverter",
    "STRAPI_OPERATORS",
    # Models
    "Base",
    # Logging
    "logger",
    "get_logger",
    "setup_logging_from_ini",
    "__version__",
]
