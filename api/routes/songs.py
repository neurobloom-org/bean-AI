"""BEAN AI v1 — Songs / Music Library routes.

Endpoints:
    GET    /api/v1/songs              — list songs available to the user
    POST   /api/v1/songs              — upload a new song
    DELETE /api/v1/songs/{song_id}    — delete the user's song

Songs are stored in Supabase Storage (bean-music bucket) and metadata is
tracked in the user_songs table.

Upload flow:
    1. Client POSTs multipart form-data with the audio file + metadata.
    2. Server validates MIME type and file size.
    3. File is streamed directly to Supabase Storage (never written to disk).
    4. Metadata row is inserted into user_songs.
    5. Song is immediately available for playback.

Supported formats: MP3, OGG, WAV, M4A/AAC
Max file size: 20 MB
"""

from __future__ import annotations

import logging
import mimetypes
import uuid
from typing import Annotated, Literal

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from pydantic import BaseModel

from api.routes.auth import get_current_bearer_token, get_current_user_id
from services.music_service import (
    ALLOWED_MIME_TYPES,
    MAX_UPLOAD_BYTES,
    STORAGE_BUCKET,
    VALID_MOODS,
    delete_user_song,
    get_songs_for_mood,
    register_uploaded_song,
)
from services.supabase_client import get_service_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/songs", tags=["songs"])

MoodLiteral = Literal["calm", "happy", "sad", "lofi", "nature", "classical"]

# ── Response models ───────────────────────────────────────────────────────────


class SongResponse(BaseModel):
    id: str
    title: str
    mood: str
    duration_seconds: float | None = None
    file_size_bytes: int | None = None
    is_default: bool
    created_at: str | None = None


class SongListResponse(BaseModel):
    songs: list[SongResponse]
    total: int


class UploadResponse(BaseModel):
    id: str
    title: str
    mood: str
    message: str


# ── List songs ────────────────────────────────────────────────────────────────


@router.get("/", response_model=SongListResponse)
async def list_songs(
    current_user_id: Annotated[str, Depends(get_current_user_id)],
    mood: MoodLiteral | None = Query(
        default=None,
        description="Filter by mood. If omitted, returns all moods.",
    ),
) -> SongListResponse:
    """List songs available to this user (their uploads + BEAN defaults).

    Returns metadata only — never storage paths or signed URLs.
    """
    moods_to_fetch = [mood] if mood else list(VALID_MOODS)
    all_songs: list[dict] = []

    for m in moods_to_fetch:
        songs = await get_songs_for_mood(current_user_id, m)
        all_songs.extend(songs)

    # Sort: user's own songs first, then defaults; alphabetical within each group
    all_songs.sort(key=lambda s: (s.get("is_default", True), s.get("title", "")))

    items = [
        SongResponse(
            id=str(s["id"]),
            title=s["title"],
            mood=s["mood"],
            duration_seconds=s.get("duration_seconds"),
            file_size_bytes=s.get("file_size_bytes"),
            is_default=bool(s.get("is_default", False)),
            created_at=str(s.get("created_at", "")),
        )
        for s in all_songs
    ]

    return SongListResponse(songs=items, total=len(items))


# ── Upload a song ─────────────────────────────────────────────────────────────


@router.post("/", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_song(
    current_user_id: Annotated[str, Depends(get_current_user_id)],
    _token: Annotated[str, Depends(get_current_bearer_token)],
    file: UploadFile = File(..., description="Audio file (MP3, OGG, WAV, M4A, AAC)"),
    title: str = Form(..., min_length=1, max_length=200, description="Song title"),
    mood: MoodLiteral = Form(..., description="Mood category"),
) -> UploadResponse:
    """Upload a personal song to your BEAN music library.

    The file is streamed directly to Supabase Storage — it is never written
    to the server's disk. After upload, the song is immediately available
    for playback via 'play {mood} music'.

    Constraints:
        - Max file size: 20 MB
        - Formats: MP3, OGG, WAV, M4A/AAC
    """
    # ── Validate MIME type ────────────────────────────────────────────────────
    # Trust Content-Type header + filename extension (belt-and-suspenders)
    content_type: str = file.content_type or ""

    # Try to infer from filename if content_type is generic
    if content_type in ("application/octet-stream", ""):
        ext = (file.filename or "").rsplit(".", 1)[-1].lower()
        guessed, _ = mimetypes.guess_type(f"file.{ext}")
        content_type = guessed or ""

    # Normalise common aliases
    _mime_aliases = {
        "audio/mp3": "audio/mpeg",
        "audio/x-mpeg": "audio/mpeg",
        "audio/x-wav": "audio/wav",
        "audio/x-ogg": "audio/ogg",
        "audio/m4a": "audio/mp4",
        "audio/x-m4a": "audio/mp4",
    }
    content_type = _mime_aliases.get(content_type, content_type)

    if content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type. Allowed: MP3, OGG, WAV, M4A/AAC. "
                f"Detected: {content_type or 'unknown'}"
            ),
        )

    # ── Read file into memory (max 20 MB) ─────────────────────────────────────
    # Supabase Python SDK's storage.upload() accepts bytes.
    # We enforce the size limit before touching storage.
    contents = await file.read(MAX_UPLOAD_BYTES + 1)

    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {MAX_UPLOAD_BYTES // 1024 // 1024} MB.",
        )

    if len(contents) < 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is too small to be a valid audio file.",
        )

    # ── Build storage path ────────────────────────────────────────────────────
    _mime_exts = {
        "audio/mpeg": "mp3",
        "audio/ogg": "ogg",
        "audio/wav": "wav",
        "audio/mp4": "m4a",
        "audio/aac": "aac",
    }
    ext = _mime_exts.get(content_type, "mp3")
    song_id = str(uuid.uuid4())
    storage_path = f"user_uploads/{current_user_id}/{song_id}.{ext}"

    # ── Upload to Supabase Storage ────────────────────────────────────────────
    try:
        client = await get_service_client()
        await client.storage.from_(STORAGE_BUCKET).upload(
            path=storage_path,
            file=contents,
            file_options={"content-type": content_type, "upsert": "false"},
        )
        logger.info(
            "Song uploaded to storage: path=%s user=%s size=%d",
            storage_path,
            current_user_id[:8],
            len(contents),
        )
    except Exception as exc:
        logger.error("Storage upload failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload file to storage. Please try again.",
        ) from exc

    # ── Register metadata in DB ───────────────────────────────────────────────
    try:
        row = await register_uploaded_song(
            user_id=current_user_id,
            storage_path=storage_path,
            title=title.strip(),
            mood=mood,
            mime_type=content_type,
            file_size_bytes=len(contents),
        )
    except Exception as exc:
        # Attempt to clean up orphaned storage file
        try:
            cleanup_client = await get_service_client()
            await cleanup_client.storage.from_(STORAGE_BUCKET).remove([storage_path])
            logger.info("Cleaned up orphaned storage file: %s", storage_path)
        except Exception as cleanup_exc:
            logger.warning(
                "Failed to clean up orphaned storage file %s: %s",
                storage_path,
                cleanup_exc,
            )
        logger.error("Song metadata registration failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="File was uploaded but metadata could not be saved. Please try again.",
        ) from exc

    return UploadResponse(
        id=str(row["id"]),
        title=row["title"],
        mood=row["mood"],
        message=f"'{title}' added to your {mood} playlist!",
    )


# ── Delete a song ─────────────────────────────────────────────────────────────


@router.delete("/{song_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_song(
    song_id: str,
    current_user_id: Annotated[str, Depends(get_current_user_id)],
) -> None:
    """Delete one of your uploaded songs.

    Cannot delete BEAN's default songs.
    Both the storage file and the metadata row are removed.
    """
    # Basic UUID format guard (avoids a DB round-trip for obvious bad input)
    try:
        uuid.UUID(song_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Song not found.",
        )

    deleted = await delete_user_song(song_id=song_id, user_id=current_user_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Song not found or you do not have permission to delete it.",
        )