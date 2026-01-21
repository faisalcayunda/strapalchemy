"""
StrapAlchemy - Enhanced SQLAlchemy Query Builder Library.

A powerful query builder library for SQLAlchemy that provides advanced
database query capabilities including search, filtering, sorting, pagination,
and more.

Example:
    ```python
    from strapalchemy import FilterBuilder, SortBuilder, SearchEngine, Base
    from sqlalchemy import select, Column, Integer, String

    # Define your model
    class User(Base):
        __tablename__ = "users"
        id = Column(Integer, primary_key=True)
        name = Column(String)
        email = Column(String)

    # Use the builders
    query = select(User)

    # Filtering (sync - no await needed)
    filter_builder = FilterBuilder(User)
    query = filter_builder.apply_filters(query, {"name": {"$contains": "John"}})

    # Sorting (sync)
    sort_builder = SortBuilder(User)
    query = sort_builder.apply_sorting(query, ["name:asc"])

    # Search (sync)
    search_engine = SearchEngine()
    query = search_engine.apply_search(query, User, "search term")
    ```
"""

# Services
# Logging
from strapalchemy.logging import get_logger, logger, setup_logging_from_ini

# Models
from strapalchemy.models.base import Base
from strapalchemy.services.field_selector import FieldSelector
from strapalchemy.services.filter_builder import FilterBuilder
from strapalchemy.services.operator_handler import STRAPI_OPERATORS, OperatorHandler
from strapalchemy.services.paginator import Paginator
from strapalchemy.services.population_builder import PopulationBuilder
from strapalchemy.services.query_optimizer import QueryOptimizer
from strapalchemy.services.search_engine import SearchEngine
from strapalchemy.services.serializer import ModelSerializer
from strapalchemy.services.sort_builder import SortBuilder
from strapalchemy.services.sync_paginator import SyncPaginator
from strapalchemy.services.sync_query_optimizer import SyncQueryOptimizer
from strapalchemy.services.type_converter import TypeConverter

__version__ = "0.2.5"

__all__ = [
    # Services
    "FieldSelector",
    "FilterBuilder",
    "OperatorHandler",
    "Paginator",
    "SyncPaginator",
    "PopulationBuilder",
    "QueryOptimizer",
    "SyncQueryOptimizer",
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
