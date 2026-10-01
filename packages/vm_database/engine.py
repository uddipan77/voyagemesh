"""Async SQLAlchemy engine and session management.

One engine per process, a session factory, and a context-managed session. Business logic
never touches a session directly — it receives a repository (see :mod:`vm_database.repositories`),
so the persistence layer can change without the domain layer noticing (brief §18).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from vm_config.settings import DatabaseSettings

__all__ = ["Database"]

logger = logging.getLogger(__name__)


class Database:
    """Owns the async engine and hands out sessions."""

    def __init__(self, settings: DatabaseSettings) -> None:
        self._settings = settings
        self._engine: AsyncEngine = create_async_engine(
            settings.url,
            echo=settings.echo,
            pool_size=settings.pool_size,
            max_overflow=settings.max_overflow,
            pool_timeout=settings.pool_timeout_seconds,
            pool_pre_ping=True,  # a stale connection is transparently replaced
        )
        self._sessionmaker = async_sessionmaker(
            self._engine, expire_on_commit=False, class_=AsyncSession
        )

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """A transactional session. Commits on success, rolls back on any exception."""
        async with self._sessionmaker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def health_check(self) -> bool:
        """True when the database answers a trivial query. Never raises."""
        from sqlalchemy import text

        try:
            async with self._engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
        except Exception:
            return False
        return True

    async def aclose(self) -> None:
        await self._engine.dispose()
