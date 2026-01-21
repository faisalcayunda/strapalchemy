# StrapAlchemy

[![PyPI version](https://badge.fury.io/py/strapalchemy.svg)](https://badge.fury.io/py/strapalchemy)

> Enhanced SQLAlchemy query builder with advanced filtering, sorting, pagination, and search capabilities.

StrapAlchemy is a powerful query builder library for SQLAlchemy that provides Strapi-style query syntax for building complex database queries with ease.

## Features

- **Sync & Async Support**: Choose between sync and async based on your session type
- **Advanced Filtering**: Strapi-style operators (`$eq`, `$in`, `$contains`, `$between`, etc.)
- **Nested Relationship Filtering**: Filter through related models with dot notation
- **Flexible Sorting**: Sort by direct fields or relationship fields
- **Pagination**: Support for both page-based and offset-based pagination
- **Full-Text Search**: BM25 search with ParadeDB integration and ILIKE fallback
- **Field Selection**: Select specific fields to optimize query performance
- **Relationship Population**: Eager load relationships to prevent N+1 queries
- **Query Optimization**: Built-in caching and optimization for better performance
- **Model Serialization**: Convert SQLAlchemy models to dictionaries easily

## Installation

```bash
pip install strapalchemy
```

## Quick Start

### For Async Sessions (AsyncEngine + AsyncSession)

```python
from sqlalchemy import select
from strapalchemy import FilterBuilder, SortBuilder, Paginator, SearchEngine, Base
from sqlalchemy import Column, Integer, String

# Define your model
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String)
    status = Column(String)

# Build your query
query = select(User)

# Apply filters (sync - no await needed)
filter_builder = FilterBuilder(User)
query = filter_builder.apply_filters(query, {
    "name": {"$contains": "John"},
    "status": {"$eq": "active"}
})

# Apply sorting (sync - no await needed)
sort_builder = SortBuilder(User)
query = sort_builder.apply_sorting(query, ["name:asc", "created_at:desc"])

# Apply search (sync - no await needed)
search_engine = SearchEngine()
query = search_engine.apply_search(query, User, "search term")

# Apply pagination (async - use with AsyncSession)
paginator = Paginator(session, User)
query, meta = await paginator.apply_pagination(query, {"page": 1, "page_size": 20})

# Execute
result = await session.execute(query)
users = result.scalars().all()
```

### For Sync Sessions (Engine + Session)

```python
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from strapalchemy import FilterBuilder, SortBuilder, SyncPaginator, SearchEngine, Base
from sqlalchemy import Column, Integer, String

# Define your model
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String)
    status = Column(String)

# Create sync session
engine = create_engine("sqlite:///database.db")
session = Session(engine)

# Build your query
query = select(User)

# Apply filters (sync - same as async)
filter_builder = FilterBuilder(User)
query = filter_builder.apply_filters(query, {
    "name": {"$contains": "John"},
    "status": {"$eq": "active"}
})

# Apply sorting (sync - same as async)
sort_builder = SortBuilder(User)
query = sort_builder.apply_sorting(query, ["name:asc", "created_at:desc"])

# Apply search (sync - same as async)
search_engine = SearchEngine()
query = search_engine.apply_search(query, User, "search term")

# Apply pagination (sync - use SyncPaginator with regular Session)
paginator = SyncPaginator(session, User)
query, meta = paginator.apply_pagination(query, {"page": 1, "page_size": 20})

# Execute
result = session.execute(query)
users = result.scalars().all()
```

## Filtering

StrapAlchemy supports Strapi-style filtering operators:

| Operator | Description | Example |
|----------|-------------|---------|
| `$eq` | Equal | `{"status": {"$eq": "active"}}` |
| `$ne` | Not equal | `{"status": {"$ne": "deleted"}}` |
| `$lt` | Less than | `{"age": {"$lt": 18}}` |
| `$lte` | Less than or equal | `{"age": {"$lte": 18}}` |
| `$gt` | Greater than | `{"age": {"$gt": 18}}` |
| `$gte` | Greater than or equal | `{"age": {"$gte": 18}}` |
| `$in` | In list | `{"status": {"$in": ["active", "pending"]}}` |
| `$notIn` | Not in list | `{"status": {"$notIn": ["deleted"]}}` |
| `$contains` | Contains | `{"name": {"$contains": "John"}}` |
| `$containsi` | Contains (case insensitive) | `{"name": {"$containsi": "john"}}` |
| `$startsWith` | Starts with | `{"email": {"$startsWith": "admin"}}` |
| `$endsWith` | Ends with | `{"email": {"$endsWith": "@example.com"}}` |
| `$null` | Is null | `{"deleted_at": {"$null": true}}` |
| `$notNull` | Is not null | `{"email": {"$notNull": true}}` |
| `$between` | Between | `{"created_at": {"$between": ["2024-01-01", "2024-12-31"]}}` |
| `$or` | Logical OR | `{"$or": [{"status": {"$eq": "active"}}, {"status": {"$eq": "pending"}}]}` |
| `$and` | Logical AND | `{"$and": [{"status": {"$eq": "active"}}, {"verified": {"$eq": true}}]}` |

### Nested Relationship Filtering

```python
# Filter by relationship fields (sync - no await needed)
query = filter_builder.apply_filters(query, {
    "organization": {"slug": {"$eq": "acme"}}
})

# Or use dot notation (sync - no await needed)
query = filter_builder.apply_filters(query, {
    "organization.slug": {"$eq": "acme"}
})
```

## Sorting

```python
# Sort by single field (sync - no await needed)
query = sort_builder.apply_sorting(query, "name:asc")

# Sort by multiple fields (sync - no await needed)
query = sort_builder.apply_sorting(query, ["name:asc", "created_at:desc"])

# Sort by relationship field (sync - no await needed)
query = sort_builder.apply_sorting(query, ["organization.name:asc"])
```

## Pagination

> **Note**: Choose `Paginator` for async sessions and `SyncPaginator` for sync sessions. Both have the same API.

### For Async Sessions

```python
from strapalchemy import Paginator

paginator = Paginator(async_session, User)
query, meta = await paginator.apply_pagination(query, {
    "page": 1,
    "page_size": 20
})
```

### For Sync Sessions

```python
from strapalchemy import SyncPaginator

paginator = SyncPaginator(session, User)
query, meta = paginator.apply_pagination(query, {
    "page": 1,
    "page_size": 20
})
```

### Response Metadata

Both return the same metadata structure:
```python
# {
#     "page": 1,
#     "page_size": 20,
#     "page_count": 5,
#     "total": 100,
#     "has_next": True,
#     "has_previous": False
# }
```

### Page-based Pagination

```python
# Async
query, meta = await paginator.apply_pagination(query, {"page": 1, "page_size": 20})

# Sync
query, meta = paginator.apply_pagination(query, {"page": 1, "page_size": 20})
```

### Offset-based Pagination

```python
# Async
query, meta = await paginator.apply_pagination(query, {"start": 0, "limit": 20})

# Sync
query, meta = paginator.apply_pagination(query, {"start": 0, "limit": 20})
```

## Field Selection

```python
from strapalchemy import FieldSelector

field_selector = FieldSelector(User)
query = field_selector.apply_field_selection(query, ["id", "name", "email"])

# Select relationship fields (sync - no await needed)
query = field_selector.apply_field_selection(query, ["id", "name", "organization.slug"])
```

## Model Serialization

```python
from strapalchemy import ModelSerializer

# Serialize a single model
data = ModelSerializer.serialize(user, fields=["id", "name", "email"])

# Serialize a list
data = ModelSerializer.serialize(users, fields=["id", "name"])

# Serialize with relationships
data = ModelSerializer.serialize(user, populate="organization")

# Serialize with nested relationships
data = ModelSerializer.serialize(user, populate=["organization", "user.role"])
```

## Search

```python
from strapalchemy import SearchEngine

search_engine = SearchEngine()

# Add searchable fields to your model
class User(Base):
    __tablename__ = "users"
    __searchable__ = {
        "text_fields": ["name", "email", "bio"]
    }
    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String)
    bio = Column(String)

# Apply search (sync - no await needed)
query = search_engine.apply_search(query, User, "John Doe")
```

## Advanced Usage

### Choosing Between Sync and Async

| Class | Sync Version | Async Version | Session Type |
|-------|--------------|---------------|--------------|
| Paginator | `SyncPaginator` | `Paginator` | Session vs AsyncSession |
| QueryOptimizer | `SyncQueryOptimizer` | `QueryOptimizer` | Session vs AsyncSession |

### API Reference

All builders (FilterBuilder, SortBuilder, SearchEngine, etc.) are **sync** and work with both session types:

| Method | Type | Session Type |
|--------|------|--------------|
| `FilterBuilder.apply_filters()` | **Sync** | Works with both |
| `SortBuilder.apply_sorting()` | **Sync** | Works with both |
| `SearchEngine.apply_search()` | **Sync** | Works with both |
| `FieldSelector.apply_field_selection()` | **Sync** | Works with both |
| `PopulationBuilder.apply_population()` | **Sync** | Works with both |
| `SyncPaginator.apply_pagination()` | **Sync** | For sync Session only |
| `Paginator.apply_pagination()` | **Async** | For async AsyncSession only |
| `SyncQueryOptimizer.execute_optimized_query()` | **Sync** | For sync Session only |
| `QueryOptimizer.execute_optimized_query()` | **Async** | For async AsyncSession only |

### Combining Multiple Builders

#### Async Example

```python
from strapalchemy import FilterBuilder, SortBuilder, Paginator, SearchEngine

async def get_users(async_session, filters=None, sort=None, search=None, page=None):
    query = select(User)

    if filters:
        query = filter_builder.apply_filters(query, filters)  # sync

    if sort:
        query = sort_builder.apply_sorting(query, sort)  # sync

    if search:
        query = search_engine.apply_search(query, User, search)  # sync

    if page:
        paginator = Paginator(async_session, User)
        query, meta = await paginator.apply_pagination(query, page)  # async

    result = await async_session.execute(query)
    return result.scalars().all(), meta
```

#### Sync Example

```python
from strapalchemy import FilterBuilder, SortBuilder, SyncPaginator, SearchEngine

def get_users(session, filters=None, sort=None, search=None, page=None):
    query = select(User)

    if filters:
        query = filter_builder.apply_filters(query, filters)  # sync

    if sort:
        query = sort_builder.apply_sorting(query, sort)  # sync

    if search:
        query = search_engine.apply_search(query, User, search)  # sync

    if page:
        paginator = SyncPaginator(session, User)
        query, meta = paginator.apply_pagination(query, page)  # sync

    result = session.execute(query)
    return result.scalars().all(), meta
```

## Requirements

- Python >= 3.12
- SQLAlchemy >= 2.0.45
- python-dateutil >= 2.9.0
- rich >= 13.0.0

## Changelog

### 0.2.5

- Added `SyncPaginator` for synchronous SQLAlchemy sessions
- Added `SyncQueryOptimizer` for synchronous query execution
- Updated README with comprehensive sync/async examples
- Added sync session fixtures for testing
- All builders now work with both sync and async sessions

### 0.2.4

- Converted `FilterBuilder.apply_filters` to sync (no async overhead needed)
- Updated all documentation and examples to reflect sync API

### 0.2.3

- Fixed import path for `Base` - now import from `strapalchemy` directly
- Updated documentation with correct import examples

### 0.2.2

- Converted `_handle_or_operator` to sync (no async overhead needed)
- Converted `_apply_default_pagination` to sync (no async overhead needed)
- Performance improvements for sync operations

### 0.2.1

- Converted `SortBuilder.apply_sorting` to sync
- Converted `SearchEngine.apply_search` to sync
- Converted `FieldSelector.apply_field_selection` to sync
- Converted `PopulationBuilder.apply_population` to sync
- Converted FilterBuilder helper methods to sync
- Updated README with correct async/sync API usage

### 0.2.0

- Initial release with core features
- Advanced filtering with Strapi-style operators
- Nested relationship filtering
- Flexible sorting
- Pagination (page-based and offset-based)
- Full-text search with BM25/ILIKE fallback
- Field selection
- Relationship population
- Query optimization
- Model serialization

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please use the GitHub issue tracker.
