"""Test configuration and fixtures for StrapAlchemy tests."""

import asyncio
from collections.abc import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from tests.models import Organization, Post, TestBase, User

# Database URLs for testing
ASYNC_TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"
SYNC_TEST_DATABASE_URL = "sqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def async_engine():
    """Create async engine for testing."""
    engine = create_async_engine(
        ASYNC_TEST_DATABASE_URL,
        echo=False,
        future=True,
    )

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(TestBase.metadata.create_all)

    yield engine

    # Drop tables
    async with engine.begin() as conn:
        await conn.run_sync(TestBase.metadata.drop_all)

    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def async_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create async session for testing."""
    async_session_maker = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session_maker() as session:
        yield session


@pytest.fixture
def sample_organizations():
    """Sample organization data."""
    return [
        Organization(name="Acme Corp", slug="acme", status="active"),
        Organization(name="Beta Inc", slug="beta", status="active"),
        Organization(name="Gamma LLC", slug="gamma", status="inactive"),
    ]


@pytest.fixture
def sample_users(sample_organizations):
    """Sample user data."""
    return [
        User(
            name="John Doe",
            email="john@example.com",
            status="active",
            age=30,
            bio="Software engineer",
            organization_id=1,
        ),
        User(
            name="Jane Smith",
            email="jane@example.com",
            status="active",
            age=25,
            bio="Designer",
            organization_id=1,
        ),
        User(
            name="Bob Johnson",
            email="bob@example.com",
            status="inactive",
            age=35,
            bio="Manager",
            organization_id=2,
        ),
        User(
            name="Alice Brown",
            email="alice@example.com",
            status="active",
            age=28,
            bio="Developer",
            organization_id=2,
        ),
        User(
            name="Charlie Wilson",
            email="charlie@example.com",
            status="pending",
            age=22,
            bio="Intern",
            organization_id=3,
        ),
    ]


@pytest.fixture
def sample_posts(sample_users):
    """Sample post data."""
    return [
        Post(
            title="First Post",
            content="This is my first post",
            status="published",
            author_id=1,
        ),
        Post(
            title="Second Post",
            content="Another post",
            status="draft",
            author_id=1,
        ),
        Post(
            title="Third Post",
            content="More content",
            status="published",
            author_id=2,
        ),
    ]


@pytest_asyncio.fixture(scope="function")
async def populated_database(
    async_session, sample_organizations, sample_users, sample_posts
):
    """Populate database with sample data."""
    async with async_session.begin():
        async_session.add_all(sample_organizations)
        async_session.add_all(sample_users)
        async_session.add_all(sample_posts)

    await async_session.commit()

    # Refresh all instances
    for org in sample_organizations:
        await async_session.refresh(org)
    for user in sample_users:
        await async_session.refresh(user)
    for post in sample_posts:
        await async_session.refresh(post)

    return {
        "organizations": sample_organizations,
        "users": sample_users,
        "posts": sample_posts,
    }


# === Sync fixtures for testing SyncPaginator and SyncQueryOptimizer ===


@pytest.fixture(scope="function")
def sync_engine():
    """Create sync engine for testing."""
    engine = create_engine(
        SYNC_TEST_DATABASE_URL,
        echo=False,
    )

    # Create tables
    TestBase.metadata.create_all(engine)

    yield engine

    # Drop tables
    TestBase.metadata.drop_all(engine)

    engine.dispose()


@pytest.fixture(scope="function")
def sync_session(sync_engine) -> Generator[Session, None, None]:
    """Create sync session for testing."""
    session_maker = sessionmaker(
        sync_engine,
        class_=Session,
        expire_on_commit=False,
    )

    with session_maker() as session:
        yield session


@pytest.fixture(scope="function")
def populated_sync_database(
    sync_session, sample_organizations, sample_users, sample_posts
):
    """Populate sync database with sample data."""
    with sync_session.begin():
        sync_session.add_all(sample_organizations)
        sync_session.add_all(sample_users)
        sync_session.add_all(sample_posts)

    sync_session.commit()

    # Refresh all instances
    for org in sample_organizations:
        sync_session.refresh(org)
    for user in sample_users:
        sync_session.refresh(user)
    for post in sample_posts:
        sync_session.refresh(post)

    return {
        "organizations": sample_organizations,
        "users": sample_users,
        "posts": sample_posts,
    }
