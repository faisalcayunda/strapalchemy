"""Test models for StrapAlchemy tests."""

from sqlalchemy import Column, DateTime, Integer, ForeignKey, String, Text, func
from sqlalchemy.orm import DeclarativeBase, relationship


class TestBase(DeclarativeBase):
    """Base class for test models."""
    pass


class Organization(TestBase):
    __tablename__ = "organizations"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    slug = Column(String(50), nullable=False, unique=True)
    status = Column(String(20), default="active")
    created_at = Column(DateTime, server_default=func.now())

    users = relationship("User", back_populates="organization", lazy="selectin")


class User(TestBase):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    email = Column(String(255), nullable=False, unique=True)
    status = Column(String(20), default="active")
    age = Column(Integer)
    bio = Column(Text)
    organization_id = Column(Integer, ForeignKey("organizations.id"))
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now())

    organization = relationship("Organization", back_populates="users", lazy="selectin")
    posts = relationship("Post", back_populates="author", lazy="selectin")

    __searchable__ = {
        "text_fields": ["name", "email", "bio"]
    }


class Post(TestBase):
    __tablename__ = "posts"

    id = Column(Integer, primary_key=True)
    title = Column(String(200), nullable=False)
    content = Column(Text)
    status = Column(String(20), default="draft")
    author_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, server_default=func.now())

    author = relationship("User", back_populates="posts", lazy="selectin")

    __searchable__ = {
        "text_fields": ["title", "content"]
    }
