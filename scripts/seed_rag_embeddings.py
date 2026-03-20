#!/usr/bin/env python3
"""BEAN AI v1 — Seed RAG technique embeddings.

Generate and store vector embeddings for rows in rag_techniques where
embedding IS NULL.

Usage:
    python scripts/seed_rag_embeddings.py
    python scripts/seed_rag_embeddings.py --db-batch-size 100 --concurrency 3
    python scripts/seed_rag_embeddings.py --limit 500 --log-level DEBUG
    python scripts/seed_rag_embeddings.py --failure-report seed_failures.jsonl

Requirements:
    - Migrations already applied
    - SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, OPENAI_API_KEY set

Behavior:
    - Safe to re-run
    - Processes only rows with embedding IS NULL
    - Uses keyset pagination to avoid poison-row refetch amplification
    - Uses bounded worker pool
    - Retries transient failures with exponential backoff + full jitter
    - Reuses one Supabase client for the full run
    - Supports optional embedding batch API if available
    - Stops promptly on shutdown signals
    - Uses bounded concurrent database writes for batch update mode
    - Streams failures to JSONL immediately
    - Exits non-zero if any rows failed
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
import os
import random
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, TextIO, TypeVar

# Allow running from project root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LOGGER = logging.getLogger("seed_rag_embeddings")

DEFAULT_DB_BATCH_SIZE = 100
DEFAULT_CONCURRENCY = 3
DEFAULT_DB_WRITE_CONCURRENCY = 10
DEFAULT_RETRIES = 4
DEFAULT_EMBED_TIMEOUT_SECONDS = 30
DEFAULT_UPDATE_TIMEOUT_SECONDS = 30
DEFAULT_BASE_RETRY_DELAY_SECONDS = 0.5
DEFAULT_MAX_RETRY_DELAY_SECONDS = 8.0
DEFAULT_QUEUE_MULTIPLIER = 2

T = TypeVar("T")


@dataclass(slots=True)
class TechniqueRow:
    id: str
    name: str
    description: str
    example: str | None
    category: str | None


@dataclass(slots=True)
class ProcessResult:
    technique_id: str
    name: str
    success: bool
    error: str | None = None


@dataclass(slots=True)
class RunStats:
    started_monotonic: float
    finished_monotonic: float | None = None
    total_processed: int = 0
    total_succeeded: int = 0
    total_failed: int = 0
    embedded_total: int = 0
    remaining_total: int = 0

    @property
    def duration_seconds(self) -> float:
        end = (
            self.finished_monotonic
            if self.finished_monotonic is not None
            else time.monotonic()
        )
        return max(0.0, end - self.started_monotonic)


class GracefulShutdown:
    """Track termination signals so workers can stop cleanly."""

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            LOGGER.debug(
                "No running event loop available for signal handler installation"
            )
            return

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._request_shutdown, sig)
            except (NotImplementedError, RuntimeError):
                LOGGER.debug("Signal handler not supported for signal %s", sig)

        self._installed = True

    def _request_shutdown(self, signum: int) -> None:
        LOGGER.warning("Received signal %s, shutting down gracefully", signum)
        self._event.set()

    @property
    def requested(self) -> bool:
        return self._event.is_set()

    async def wait(self) -> None:
        await self._event.wait()


class FailureReporter:
    """Append failures to a JSONL file immediately, with minimal memory use."""

    def __init__(self, path_str: str) -> None:
        self.enabled = bool(path_str)
        self.path: Path | None = Path(path_str) if path_str else None
        self._fh: TextIO | None = None

    def open(self) -> None:
        if not self.enabled or self.path is None:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def write(self, result: ProcessResult) -> None:
        if self._fh is None:
            return

        payload = {
            "technique_id": result.technique_id,
            "name": result.name,
            "success": result.success,
            "error": result.error,
        }
        self._fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed missing embeddings for rag_techniques.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of rows to process in this run (0 = no limit).",
    )
    parser.add_argument(
        "--db-batch-size",
        type=int,
        default=DEFAULT_DB_BATCH_SIZE,
        help=f"Rows fetched from DB per loop (default: {DEFAULT_DB_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Concurrent worker count for single-row embedding mode (default: {DEFAULT_CONCURRENCY}).",
    )
    parser.add_argument(
        "--db-write-concurrency",
        type=int,
        default=DEFAULT_DB_WRITE_CONCURRENCY,
        help=f"Maximum concurrent database writes in batch embedding mode (default: {DEFAULT_DB_WRITE_CONCURRENCY}).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help=f"Retry attempts per operation (default: {DEFAULT_RETRIES}).",
    )
    parser.add_argument(
        "--embed-timeout",
        type=int,
        default=DEFAULT_EMBED_TIMEOUT_SECONDS,
        help=f"Timeout in seconds for one embedding request (default: {DEFAULT_EMBED_TIMEOUT_SECONDS}).",
    )
    parser.add_argument(
        "--update-timeout",
        type=int,
        default=DEFAULT_UPDATE_TIMEOUT_SECONDS,
        help=f"Timeout in seconds for one database update (default: {DEFAULT_UPDATE_TIMEOUT_SECONDS}).",
    )
    parser.add_argument(
        "--failure-report",
        type=str,
        default="",
        help="Optional path to append failures as JSONL.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging level.",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        LOGGER.debug("python-dotenv not installed; relying on shell environment")


def validate_env() -> None:
    required = [
        "SUPABASE_URL",
        "SUPABASE_SERVICE_ROLE_KEY",
        "OPENAI_API_KEY",
    ]
    missing = [key for key in required if not os.environ.get(key)]
    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
        )


def validate_args(args: argparse.Namespace) -> None:
    if args.limit < 0:
        raise ValueError("--limit must be 0 or greater")
    if args.db_batch_size <= 0:
        raise ValueError("--db-batch-size must be greater than 0")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be greater than 0")
    if args.db_write_concurrency <= 0:
        raise ValueError("--db-write-concurrency must be greater than 0")
    if args.retries <= 0:
        raise ValueError("--retries must be greater than 0")
    if args.embed_timeout <= 0:
        raise ValueError("--embed-timeout must be greater than 0")
    if args.update_timeout <= 0:
        raise ValueError("--update-timeout must be greater than 0")


def sanitize_text(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip()


def build_embedding_text(row: TechniqueRow) -> str:
    title = sanitize_text(row.name)
    description = sanitize_text(row.description)
    example = sanitize_text(row.example)

    if not title:
        raise ValueError(f"Technique {row.id} is missing a valid name")
    if not description:
        raise ValueError(f"Technique {row.id} is missing a valid description")

    parts = [f"{title}: {description}"]
    if example:
        parts.append(f"Example: {example}")
    return "\n".join(parts)


def full_jitter_delay(attempt: int) -> float:
    capped = min(
        DEFAULT_BASE_RETRY_DELAY_SECONDS * (2 ** (attempt - 1)),
        DEFAULT_MAX_RETRY_DELAY_SECONDS,
    )
    return random.uniform(0, capped)


async def retry_async(
    operation_name: str,
    coro_factory: Callable[[], Awaitable[T]],
    retries: int,
) -> T:
    last_exc: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            return await coro_factory()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            last_exc = exc

            if attempt >= retries:
                break

            delay = full_jitter_delay(attempt)
            LOGGER.warning(
                "%s failed on attempt %d/%d: %s. Retrying in %.2fs",
                operation_name,
                attempt,
                retries,
                exc,
                delay,
            )
            await asyncio.sleep(delay)

    assert last_exc is not None
    raise last_exc


def normalize_embedding(vector: Any) -> list[float]:
    if not isinstance(vector, list) or not vector:
        raise ValueError("Embedding service returned an empty or invalid vector")

    if not all(isinstance(value, (float, int)) for value in vector):
        raise ValueError("Embedding service returned a non-numeric vector")

    normalized = [float(value) for value in vector]

    for value in normalized:
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("Embedding service returned a non-finite vector")

    return normalized


async def fetch_unembedded_batch(
    client: Any,
    batch_size: int,
    last_seen_id: str | None,
) -> list[TechniqueRow]:
    query = (
        client.table("rag_techniques")
        .select("id, name, description, example, category")
        .is_("embedding", "null")
        .order("id")
        .limit(batch_size)
    )

    if last_seen_id is not None:
        query = query.gt("id", last_seen_id)

    result = await query.execute()

    rows = result.data or []
    techniques: list[TechniqueRow] = []

    for row in rows:
        techniques.append(
            TechniqueRow(
                id=str(row["id"]),
                name=str(row["name"]),
                description=str(row["description"]),
                example=row.get("example"),
                category=row.get("category"),
            )
        )

    return techniques


async def count_embedded_rows(client: Any) -> int:
    # FIX: use .neq() instead of .not_("embedding", "is", "null") for
    # broader supabase-py v2 compatibility across PostgREST versions.
    result = (
        await client.table("rag_techniques")
        .select("id", count="planned", head=True)
        .neq("embedding", None)
        .execute()
    )
    return int(result.count or 0)


async def count_remaining_rows(client: Any) -> int:
    result = (
        await client.table("rag_techniques")
        .select("id", count="planned", head=True)
        .is_("embedding", "null")
        .execute()
    )
    return int(result.count or 0)


async def resolve_embedding_functions() -> tuple[Any, Any]:
    """
    Return (single_embedding_fn, batch_embedding_fn_or_none).

    Expected signatures:
        async def get_embedding(text: str) -> list[float]
        async def get_embeddings(texts: list[str]) -> list[list[float]]  # optional

    Note: services/embedding_service.py currently only exposes get_embedding
    (singular). batch_fn will be None and the worker pool path will always be
    used. If get_embeddings is added later, batch mode activates automatically.
    """
    from services import embedding_service  # type: ignore[attr-defined]

    single_fn = getattr(embedding_service, "get_embedding", None)
    batch_fn = getattr(embedding_service, "get_embeddings", None)

    if single_fn is None or not callable(single_fn):
        raise RuntimeError("services.embedding_service.get_embedding is required")

    if batch_fn is not None and not callable(batch_fn):
        batch_fn = None

    return single_fn, batch_fn


async def generate_single_embedding(
    single_embedding_fn: Callable[[str], Awaitable[Any]],
    text: str,
    retries: int,
    timeout_seconds: int,
) -> list[float]:
    async def _call() -> list[float]:
        raw = await asyncio.wait_for(
            single_embedding_fn(text),
            timeout=timeout_seconds,
        )
        return normalize_embedding(raw)

    return await retry_async(
        operation_name="Embedding request",
        coro_factory=_call,
        retries=retries,
    )


async def generate_batch_embeddings(
    batch_embedding_fn: Callable[[list[str]], Awaitable[Any]],
    texts: list[str],
    retries: int,
    timeout_seconds: int,
) -> list[list[float]]:
    async def _call() -> list[list[float]]:
        raw = await asyncio.wait_for(
            batch_embedding_fn(texts),
            timeout=timeout_seconds,
        )

        if not isinstance(raw, list):
            raise ValueError("Batch embedding service returned a non-list response")

        if len(raw) != len(texts):
            raise ValueError(
                f"Batch embedding service returned {len(raw)} vectors for {len(texts)} texts"
            )

        return [normalize_embedding(item) for item in raw]

    return await retry_async(
        operation_name=f"Batch embedding request ({len(texts)} items)",
        coro_factory=_call,
        retries=retries,
    )


async def update_embedding(
    client: Any,
    technique_id: str,
    embedding: list[float],
    retries: int,
    timeout_seconds: int,
) -> None:
    async def _call() -> None:
        result = await asyncio.wait_for(
            (
                client.table("rag_techniques")
                .update({"embedding": embedding})
                .eq("id", technique_id)
                .is_("embedding", "null")
                .execute()
            ),
            timeout=timeout_seconds,
        )

        if result.data == []:
            LOGGER.debug(
                "Technique %s update returned no rows; it may already be embedded",
                technique_id,
            )

    await retry_async(
        operation_name=f"Database update for technique {technique_id}",
        coro_factory=_call,
        retries=retries,
    )


async def process_single_technique(
    client: Any,
    single_embedding_fn: Callable[[str], Awaitable[Any]],
    row: TechniqueRow,
    retries: int,
    embed_timeout: int,
    update_timeout: int,
) -> ProcessResult:
    try:
        embedding_text = build_embedding_text(row)

        embedding = await generate_single_embedding(
            single_embedding_fn=single_embedding_fn,
            text=embedding_text,
            retries=retries,
            timeout_seconds=embed_timeout,
        )

        await update_embedding(
            client=client,
            technique_id=row.id,
            embedding=embedding,
            retries=retries,
            timeout_seconds=update_timeout,
        )

        LOGGER.info(
            "Embedded technique id=%s name=%s category=%s",
            row.id,
            row.name,
            row.category or "unknown",
        )
        return ProcessResult(
            technique_id=row.id,
            name=row.name,
            success=True,
        )

    except asyncio.CancelledError:
        raise
    except Exception as exc:
        LOGGER.exception(
            "Failed to process technique id=%s name=%s: %s",
            row.id,
            row.name,
            exc,
        )
        return ProcessResult(
            technique_id=row.id,
            name=row.name,
            success=False,
            error=str(exc),
        )


async def update_rows_after_batch_embeddings(
    client: Any,
    rows: list[TechniqueRow],
    embeddings: list[list[float]],
    retries: int,
    update_timeout: int,
    write_concurrency: int,
    shutdown: GracefulShutdown,
) -> list[ProcessResult]:
    semaphore = asyncio.Semaphore(write_concurrency)

    async def _update_one(row: TechniqueRow, embedding: list[float]) -> ProcessResult:
        if shutdown.requested:
            return ProcessResult(
                technique_id=row.id,
                name=row.name,
                success=False,
                error="Skipped due to shutdown request",
            )

        async with semaphore:
            if shutdown.requested:
                return ProcessResult(
                    technique_id=row.id,
                    name=row.name,
                    success=False,
                    error="Skipped due to shutdown request",
                )

            try:
                await update_embedding(
                    client=client,
                    technique_id=row.id,
                    embedding=embedding,
                    retries=retries,
                    timeout_seconds=update_timeout,
                )
                LOGGER.info(
                    "Embedded technique id=%s name=%s category=%s",
                    row.id,
                    row.name,
                    row.category or "unknown",
                )
                return ProcessResult(
                    technique_id=row.id,
                    name=row.name,
                    success=True,
                )
            except Exception as exc:
                LOGGER.exception(
                    "Failed to update technique after batch embedding id=%s name=%s: %s",
                    row.id,
                    row.name,
                    exc,
                )
                return ProcessResult(
                    technique_id=row.id,
                    name=row.name,
                    success=False,
                    error=str(exc),
                )

    tasks = [
        asyncio.create_task(_update_one(row, embedding))
        for row, embedding in zip(rows, embeddings, strict=True)
    ]
    return await asyncio.gather(*tasks)


async def process_batch_with_batch_embeddings(
    client: Any,
    batch_embedding_fn: Callable[[list[str]], Awaitable[Any]],
    rows: list[TechniqueRow],
    retries: int,
    embed_timeout: int,
    update_timeout: int,
    write_concurrency: int,
    shutdown: GracefulShutdown,
) -> list[ProcessResult]:
    if not rows:
        return []

    if shutdown.requested:
        return [
            ProcessResult(
                technique_id=row.id,
                name=row.name,
                success=False,
                error="Skipped due to shutdown request",
            )
            for row in rows
        ]

    try:
        texts = [build_embedding_text(row) for row in rows]
        embeddings = await generate_batch_embeddings(
            batch_embedding_fn=batch_embedding_fn,
            texts=texts,
            retries=retries,
            timeout_seconds=embed_timeout,
        )
    except Exception as exc:
        LOGGER.warning(
            "Batch embedding failed for %d rows: %s. Falling back to per-row processing.",
            len(rows),
            exc,
        )
        return []

    return await update_rows_after_batch_embeddings(
        client=client,
        rows=rows,
        embeddings=embeddings,
        retries=retries,
        update_timeout=update_timeout,
        write_concurrency=write_concurrency,
        shutdown=shutdown,
    )


async def drain_queue_without_processing(
    queue: asyncio.Queue[TechniqueRow | None],
) -> None:
    while True:
        try:
            item = queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        else:
            queue.task_done()
            if item is None:
                continue


async def worker(
    name: str,
    client: Any,
    single_embedding_fn: Callable[[str], Awaitable[Any]],
    queue: asyncio.Queue[TechniqueRow | None],
    results: list[ProcessResult],
    retries: int,
    embed_timeout: int,
    update_timeout: int,
    shutdown: GracefulShutdown,
) -> None:
    while True:
        if shutdown.requested:
            LOGGER.debug("Worker %s stopping immediately due to shutdown", name)
            await drain_queue_without_processing(queue)
            return

        row = await queue.get()
        try:
            if row is None:
                LOGGER.debug("Worker %s received stop sentinel", name)
                return

            if shutdown.requested:
                LOGGER.debug(
                    "Worker %s skipping technique %s due to shutdown",
                    name,
                    row.id,
                )
                results.append(
                    ProcessResult(
                        technique_id=row.id,
                        name=row.name,
                        success=False,
                        error="Skipped due to shutdown request",
                    )
                )
                await drain_queue_without_processing(queue)
                return

            LOGGER.debug("Worker %s processing technique %s", name, row.id)

            result = await process_single_technique(
                client=client,
                single_embedding_fn=single_embedding_fn,
                row=row,
                retries=retries,
                embed_timeout=embed_timeout,
                update_timeout=update_timeout,
            )
            results.append(result)
        finally:
            queue.task_done()


async def process_batch_with_worker_pool(
    client: Any,
    single_embedding_fn: Callable[[str], Awaitable[Any]],
    batch: list[TechniqueRow],
    args: argparse.Namespace,
    shutdown: GracefulShutdown,
) -> list[ProcessResult]:
    queue_maxsize = max(args.concurrency * DEFAULT_QUEUE_MULTIPLIER, args.concurrency)
    queue: asyncio.Queue[TechniqueRow | None] = asyncio.Queue(maxsize=queue_maxsize)
    results: list[ProcessResult] = []

    workers = [
        asyncio.create_task(
            worker(
                name=f"worker-{index + 1}",
                client=client,
                single_embedding_fn=single_embedding_fn,
                queue=queue,
                results=results,
                retries=args.retries,
                embed_timeout=args.embed_timeout,
                update_timeout=args.update_timeout,
                shutdown=shutdown,
            )
        )
        for index in range(args.concurrency)
    ]

    try:
        for row in batch:
            if shutdown.requested:
                break
            await queue.put(row)

        for _ in workers:
            await queue.put(None)

        await queue.join()
    finally:
        if shutdown.requested:
            await drain_queue_without_processing(queue)

        for task in workers:
            if not task.done():
                task.cancel()

        await asyncio.gather(*workers, return_exceptions=True)

    return results


def supports_batch_embedding(batch_fn: Any) -> bool:
    if batch_fn is None:
        return False

    try:
        signature = inspect.signature(batch_fn)
        return len(signature.parameters) >= 1
    except (TypeError, ValueError):
        return True


def log_final_summary(stats: RunStats) -> None:
    LOGGER.info(
        "Run complete. duration_seconds=%.2f processed=%d succeeded=%d failed=%d embedded_total=%d remaining_without_embedding=%d",
        stats.duration_seconds,
        stats.total_processed,
        stats.total_succeeded,
        stats.total_failed,
        stats.embedded_total,
        stats.remaining_total,
    )


async def close_supabase_client(client: Any) -> None:
    """Best-effort cleanup for underlying async HTTP clients/sessions."""
    seen: set[int] = set()

    async def _try_aclose(obj: Any) -> None:
        if obj is None:
            return
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)

        aclose = getattr(obj, "aclose", None)
        if callable(aclose):
            try:
                await aclose()
            except Exception as exc:
                LOGGER.debug("Ignoring client close error: %s", exc)

        close = getattr(obj, "close", None)
        if callable(close):
            try:
                result = close()
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:
                LOGGER.debug("Ignoring client close error: %s", exc)

    candidates = [
        client,
        getattr(client, "session", None),
        getattr(client, "_session", None),
        getattr(client, "postgrest", None),
        getattr(client, "_postgrest", None),
        getattr(client, "auth", None),
        getattr(client, "_auth", None),
        getattr(getattr(client, "postgrest", None), "session", None),
        getattr(getattr(client, "_postgrest", None), "session", None),
        getattr(getattr(client, "auth", None), "session", None),
        getattr(getattr(client, "_auth", None), "session", None),
        getattr(getattr(client, "rest", None), "session", None),
        getattr(getattr(client, "_rest", None), "session", None),
    ]

    for candidate in candidates:
        await _try_aclose(candidate)


async def seed_embeddings(args: argparse.Namespace) -> int:
    from services.supabase_client import get_service_client

    validate_env()
    validate_args(args)

    shutdown = GracefulShutdown()
    shutdown.install()

    stats = RunStats(started_monotonic=time.monotonic())
    failure_reporter = FailureReporter(args.failure_report)
    failure_reporter.open()

    client: Any | None = None

    try:
        LOGGER.info("Connecting to Supabase")
        client = await get_service_client()

        single_embedding_fn, batch_embedding_fn = await resolve_embedding_functions()
        use_batch_embedding_api = supports_batch_embedding(batch_embedding_fn)

        LOGGER.info(
            "Embedding batch API available: %s",
            "yes" if use_batch_embedding_api else "no",
        )

        remaining_limit = args.limit if args.limit > 0 else None
        last_seen_id: str | None = None

        while not shutdown.requested:
            current_batch_size = args.db_batch_size
            if remaining_limit is not None:
                if remaining_limit <= 0:
                    break
                current_batch_size = min(current_batch_size, remaining_limit)

            batch = await fetch_unembedded_batch(
                client=client,
                batch_size=current_batch_size,
                last_seen_id=last_seen_id,
            )

            if not batch:
                LOGGER.info(
                    "No more eligible rows with missing embeddings in this pass"
                )
                break

            last_seen_id = batch[-1].id

            LOGGER.info(
                "Fetched batch with %d technique(s); cursor_last_seen_id=%s",
                len(batch),
                last_seen_id,
            )

            results: list[ProcessResult] = []

            if use_batch_embedding_api and batch_embedding_fn is not None:
                results = await process_batch_with_batch_embeddings(
                    client=client,
                    batch_embedding_fn=batch_embedding_fn,
                    rows=batch,
                    retries=args.retries,
                    embed_timeout=args.embed_timeout,
                    update_timeout=args.update_timeout,
                    write_concurrency=args.db_write_concurrency,
                    shutdown=shutdown,
                )

            if not results and not shutdown.requested:
                results = await process_batch_with_worker_pool(
                    client=client,
                    single_embedding_fn=single_embedding_fn,
                    batch=batch,
                    args=args,
                    shutdown=shutdown,
                )

            batch_processed = len(results)
            batch_succeeded = sum(1 for result in results if result.success)
            batch_failed = [result for result in results if not result.success]

            for result in batch_failed:
                failure_reporter.write(result)

            stats.total_processed += batch_processed
            stats.total_succeeded += batch_succeeded
            stats.total_failed += len(batch_failed)

            LOGGER.info(
                "Batch complete. processed=%d succeeded=%d failed=%d cursor_last_seen_id=%s",
                batch_processed,
                batch_succeeded,
                len(batch_failed),
                last_seen_id,
            )

            if batch_failed:
                LOGGER.error("Failed techniques in this batch:")
                for result in batch_failed:
                    LOGGER.error(
                        " - %s (%s): %s",
                        result.name,
                        result.technique_id,
                        result.error,
                    )

            if remaining_limit is not None:
                remaining_limit -= len(batch)

            if not shutdown.requested and batch_succeeded == 0 and batch_failed:
                LOGGER.error(
                    "Stopping early because an entire batch failed with zero successes. "
                    "This usually indicates persistent bad data, credential issues, "
                    "or an upstream API or database problem."
                )
                break

        if client is not None:
            stats.embedded_total = await count_embedded_rows(client)
            stats.remaining_total = await count_remaining_rows(client)
        stats.finished_monotonic = time.monotonic()

        log_final_summary(stats)

        if shutdown.requested:
            LOGGER.warning("Run ended due to shutdown request")
            return 130

        if stats.total_failed:
            LOGGER.error("One or more techniques failed during this run")
            return 1

        return 0

    finally:
        failure_reporter.close()
        if client is not None:
            await close_supabase_client(client)


async def async_main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    load_env()

    try:
        return await seed_embeddings(args)
    except KeyboardInterrupt:
        LOGGER.error("Interrupted by user")
        return 130
    except Exception as exc:
        LOGGER.exception("Fatal error: %s", exc)
        return 1


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
