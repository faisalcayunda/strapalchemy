from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all strapalchemy ORM models.

    Import and subclass this to define mapped models:

        from strapalchemy.models.base import Base

        class User(Base):
            __tablename__ = "users"
            ...
    """
