"""Enhanced full-text and fuzzy search engine for queries with BM25 and performance optimizations."""

import re
from typing import Any, Dict, List, Optional, Type

from sqlalchemy import Select, String, Text, or_, text

from strapalchemy.logging.logger import logger
from strapalchemy.models.base import Base


class SearchEngine:
    """Enhanced search engine with BM25, fuzzy search, and performance optimizations."""

    def __init__(self):
        # Cache for searchable field configurations
        self._searchable_cache: Dict[str, Any] = {}
        # Flag to track if ParadeDB is available (will be set on first use)
        self._paradedb_available: Optional[bool] = None
        # Flag to force ILIKE search (for testing or when ParadeDB fails)
        self._force_ilike: bool = False

    async def apply_search(self, query: Select, model: Type[Base], search: Optional[str]) -> Select:
        """Apply enhanced full-text search with BM25, fuzzy search, and performance optimizations.

        Features:
        - BM25 search for exact matches (high priority)
        - Fuzzy search with configurable tolerance
        - Fallback to ILIKE search
        - Performance optimizations with caching
        - Better error handling

        Args:
            query: SQLAlchemy Select query
            model: SQLAlchemy model class
            search: Search string to find in searchable fields

        Returns:
            Modified query with search filter applied
        """
        if not search or not search.strip():
            return query

        try:
            # Get searchable configuration with caching
            searchable_config = self._get_searchable_config(model)
            if not searchable_config:
                return self._apply_fallback_search(query, model, search)

            text_fields = searchable_config.get("text_fields", [])

            if not text_fields:
                return self._apply_fallback_search(query, model, search)

            # Sanitize search query
            sanitized_search = self._sanitize_search_query(search)

            # If ParadeDB is known to be unavailable or force_ilike is set, use ILIKE directly
            if self._force_ilike or self._paradedb_available is False:
                logger.info("Using ILIKE search (ParadeDB unavailable or disabled)")
                return self._apply_ilike_search(query, model, text_fields, sanitized_search)

            # Try to apply ParadeDB search strategy
            return self._apply_hybrid_search(query, model, text_fields, sanitized_search)

        except Exception as e:
            logger.error(f"Error applying search: {e}")
            # Fallback to simple ILIKE search
            return self._apply_fallback_search(query, model, search)

    def mark_paradedb_unavailable(self):
        """Mark ParadeDB as unavailable and force fallback to ILIKE search."""
        self._paradedb_available = False
        self._force_ilike = True
        logger.warning("ParadeDB marked as unavailable. All searches will use ILIKE fallback.")

    def _get_searchable_config(self, model: Type[Base]) -> Optional[Dict[str, Any]]:
        """Get searchable configuration with caching.

        Args:
            model: SQLAlchemy model class

        Returns:
            Searchable configuration or None
        """
        model_name = model.__name__
        if model_name in self._searchable_cache:
            return self._searchable_cache[model_name]

        if not hasattr(model, "__searchable__"):
            self._searchable_cache[model_name] = None
            return None

        config = getattr(model, "__searchable__", None)
        self._searchable_cache[model_name] = config
        return config

    def _apply_fallback_search(self, query: Select, model: Type[Base], search: str) -> Select:
        """Apply fallback ILIKE search when no configuration is available.

        Args:
            query: SQLAlchemy Select query
            model: SQLAlchemy model class
            search: Search string

        Returns:
            Modified query with fallback search
        """
        sanitized_search = self._sanitize_search_query(search)
        return self._apply_ilike_search(query, model, [], sanitized_search)

    def _apply_hybrid_search(
        self, query: Select, model: Type[Base], text_fields: List[str], search: str, fuzzy_tolerance: int = 2
    ) -> Select:
        """Apply hybrid BM25 + fuzzy search for best results.

        Falls back to ILIKE search if ParadeDB is not available.

        Args:
            query: SQLAlchemy Select query
            model: SQLAlchemy model class
            text_fields: List of field names to search
            search: Sanitized search string
            fuzzy_tolerance: Fuzzy search tolerance level

        Returns:
            Modified query with hybrid search applied
        """
        try:
            table_name = model.__tablename__

            # BM25 search for exact matches (high priority)
            bm25_condition = self._build_bm25_condition(table_name, text_fields, search)

            # Fuzzy search using ParadeDB's fuzzy term query syntax
            fuzzy_conditions = []
            for field in text_fields:
                # ParadeDB fuzzy syntax: field:term~distance
                fuzzy_conditions.append(f"{field}:{search}~{fuzzy_tolerance}")

            # Build combined fuzzy query
            if fuzzy_conditions:
                fuzzy_query = " OR ".join(fuzzy_conditions)
                combined_condition = f"(({bm25_condition}) OR ({table_name} @@@ '{fuzzy_query}'))"
            else:
                combined_condition = bm25_condition

            return query.where(text(f"({combined_condition})"))
        except Exception as e:
            # If ParadeDB is not available or query fails, fallback to ILIKE search
            logger.warning(f"ParadeDB search failed, falling back to ILIKE search: {e}")
            return self._apply_ilike_search(query, model, text_fields, search)

    def _apply_bm25_search(self, query: Select, model: Type[Base], text_fields: List[str], search: str) -> Select:
        """Apply BM25 full-text search.

        Falls back to ILIKE search if ParadeDB is not available.

        Args:
            query: SQLAlchemy Select query
            model: SQLAlchemy model class
            text_fields: List of field names to search
            search: Sanitized search string

        Returns:
            Modified query with BM25 search applied
        """
        try:
            table_name = model.__tablename__
            bm25_condition = self._build_bm25_condition(table_name, text_fields, search)
            return query.where(text(f"({bm25_condition})"))
        except Exception as e:
            # If ParadeDB is not available, fallback to ILIKE search
            logger.warning(f"BM25 search failed, falling back to ILIKE search: {e}")
            return self._apply_ilike_search(query, model, text_fields, search)

    def _build_bm25_condition(self, table_name: str, text_fields: List[str], search: str) -> str:
        """Build BM25 search condition.

        Args:
            table_name: Database table name
            text_fields: List of field names to search
            search: Sanitized search string

        Returns:
            BM25 search condition string
        """
        field_queries = [f"{field}:{search}" for field in text_fields]
        pgsearch_query = " OR ".join(field_queries)
        return f"{table_name} @@@ '{pgsearch_query}'"

    @staticmethod
    def _sanitize_search_query(search: str) -> str:
        """Sanitize search query to prevent injection and clean up input.

        Args:
            search: Raw search string

        Returns:
            Sanitized search string
        """
        search_query = search.strip()
        # Remove control characters
        search_query = re.sub(r"[\x00-\x1f\x7f]", "", search_query)
        # Remove dangerous characters that could cause SQL injection
        search_query = search_query.replace('"', "").replace("'", "").replace(";", "").replace("--", "")
        # Remove SQL keywords that could be dangerous
        dangerous_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "CREATE", "ALTER"]
        for keyword in dangerous_keywords:
            search_query = re.sub(re.escape(keyword), "", search_query, flags=re.IGNORECASE)
        # Normalize whitespace
        search_query = re.sub(r"\s+", " ", search_query)
        return search_query

    def _apply_fuzzy_search(
        self, query: Select, model: Type[Base], text_fields: List[str], search: str, fuzzy_tolerance: int = 2
    ) -> Select:
        """Apply ParadeDB fuzzy search with configurable tolerance.

        Falls back to ILIKE search if ParadeDB is not available.

        Args:
            query: SQLAlchemy Select query
            model: SQLAlchemy model class
            text_fields: List of field names to search
            search: Sanitized search string
            fuzzy_tolerance: Fuzzy search tolerance level (1-3)

        Returns:
            Modified query with fuzzy search applied
        """
        try:
            table_name = model.__tablename__
            # Clamp tolerance between 1 and 3
            tolerance = max(1, min(3, fuzzy_tolerance))

            # ParadeDB fuzzy syntax: field:term~distance
            field_conditions = [f"{field}:{search}~{tolerance}" for field in text_fields]
            fuzzy_query = " OR ".join(field_conditions)
            return query.where(text(f"({table_name} @@@ '{fuzzy_query}')"))
        except Exception as e:
            # If ParadeDB is not available, fallback to ILIKE search
            logger.warning(f"Fuzzy search failed, falling back to ILIKE search: {e}")
            return self._apply_ilike_search(query, model, text_fields, search)

    def _apply_ilike_search(self, query: Select, model: Type[Base], text_fields: List[str], search: str) -> Select:
        """Apply ILIKE-based search with enhanced field selection.

        Args:
            query: SQLAlchemy Select query
            model: SQLAlchemy model class
            text_fields: List of field names to search (if empty, scans all String/Text fields)
            search: Sanitized search string

        Returns:
            Modified query with ILIKE search applied
        """
        conditions = []
        if text_fields:
            # Search only specified fields
            for field_name in text_fields:
                if hasattr(model, field_name):
                    field_obj = getattr(model, field_name)
                    conditions.append(field_obj.ilike(f"%{search}%"))
        else:
            # Search all String/Text columns
            for column in model.__table__.columns:
                if isinstance(column.type, (String, Text)):
                    conditions.append(column.ilike(f"%{search}%"))

        if conditions:
            return query.where(or_(*conditions))

        return query
