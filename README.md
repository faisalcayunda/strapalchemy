# StrapAlchemy

> Enhanced SQLAlchemy query builder with advanced filtering, sorting, pagination, and search capabilities.

StrapAlchemy is a powerful query builder library for SQLAlchemy that provides Strapi-style query syntax for building complex database queries with ease.

## Features

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

```python
from sqlalchemy import select
from strapalchemy import FilterBuilder, SortBuilder, Paginator, SearchEngine
from strapalchemy.models import Base
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

# Apply filters
filter_builder = FilterBuilder(User)
query = await filter_builder.apply_filters(query, {
    "name": {"$contains": "John"},
    "status": {"$eq": "active"}
})

# Apply sorting
sort_builder = SortBuilder(User)
query = await sort_builder.apply_sorting(query, ["name:asc", "created_at:desc"])

# Apply search
search_engine = SearchEngine()
query = await search_engine.apply_search(query, User, "search term")

# Apply pagination
paginator = Paginator(session, User)
query, meta = await paginator.apply_pagination(query, {"page": 1, "page_size": 20})

# Execute
result = await session.execute(query)
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
# Filter by relationship fields
query = await filter_builder.apply_filters(query, {
    "organization": {"slug": {"$eq": "acme"}}
})

# Or use dot notation
query = await filter_builder.apply_filters(query, {
    "organization.slug": {"$eq": "acme"}
})
```

## Sorting

```python
# Sort by single field
query = await sort_builder.apply_sorting(query, "name:asc")

# Sort by multiple fields
query = await sort_builder.apply_sorting(query, ["name:asc", "created_at:desc"])

# Sort by relationship field
query = await sort_builder.apply_sorting(query, ["organization.name:asc"])
```

## Pagination

### Page-based Pagination

```python
query, meta = await paginator.apply_pagination(query, {
    "page": 1,
    "page_size": 20
})

# meta contains:
# {
#     "page": 1,
#     "page_size": 20,
#     "page_count": 5,
#     "total": 100,
#     "has_next": True,
#     "has_previous": False
# }
```

### Offset-based Pagination

```python
query, meta = await paginator.apply_pagination(query, {
    "start": 0,
    "limit": 20
})
```

## Field Selection

```python
from strapalchemy import FieldSelector

field_selector = FieldSelector(User)
query = await field_selector.apply_field_selection(query, ["id", "name", "email"])

# Select relationship fields
query = await field_selector.apply_field_selection(query, ["id", "name", "organization.slug"])
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

# Apply search
query = await search_engine.apply_search(query, User, "John Doe")
```

## Advanced Usage

### Combining Multiple Builders

```python
async def get_users(filters=None, sort=None, search=None, page=None):
    query = select(User)

    if filters:
        query = await filter_builder.apply_filters(query, filters)

    if sort:
        query = await sort_builder.apply_sorting(query, sort)

    if search:
        query = await search_engine.apply_search(query, User, search)

    if page:
        query, meta = await paginator.apply_pagination(query, page)

    result = await session.execute(query)
    return result.scalars().all(), meta
```

## Requirements

- Python >= 3.12
- SQLAlchemy >= 2.0.45
- python-dateutil >= 2.9.0
- rich >= 13.0.0

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please use the GitHub issue tracker.
