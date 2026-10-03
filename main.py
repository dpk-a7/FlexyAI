import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
from contextlib import asynccontextmanager, closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlencode

import jwt
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException

from flexy_runtime import PlatformError, create_runtime

BASE_DIRECTORY = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIRECTORY / "data" / "users.sqlite3"
SCRYPT_N = 2**17
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_MAXMEM = 256 * 1024 * 1024


class Credentials(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    username: str = Field(min_length=1, max_length=128, pattern=r"^\S+$")
    password: SecretStr = Field(min_length=8, max_length=1024)


class GPURequest(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    device: int = Field(default=0, ge=0)


class LoadFlexyAppRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    app: str = Field(min_length=1, max_length=128)
    logging: bool = False
    config: dict[str, Any] = Field(default_factory=dict)
    gpu: GPURequest = Field(default_factory=GPURequest)


class RunnerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    runner_uuid: str = Field(min_length=1, max_length=128)


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=SCRYPT_N, r=SCRYPT_R,
        p=SCRYPT_P, maxmem=SCRYPT_MAXMEM, dklen=32,
    )
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    algorithm, n, r, p, salt, expected = encoded.split("$")
    if algorithm != "scrypt":
        raise ValueError("Unsupported stored password hash")
    actual = hashlib.scrypt(
        password.encode("utf-8"), salt=bytes.fromhex(salt), n=int(n), r=int(r),
        p=int(p), maxmem=SCRYPT_MAXMEM, dklen=32,
    )
    return hmac.compare_digest(actual, bytes.fromhex(expected))


def parse_token_expiration(value: str) -> timedelta:
    match = re.fullmatch(r"([1-9][0-9]*)([smhd])", value)
    if match is None:
        raise ValueError("TOKEN_EXPIRATION must be a positive duration such as 30m, 2h, or 1d")
    seconds = int(match[1]) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[match[2]]
    return timedelta(seconds=seconds)


def initialize_database() -> None:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(DATABASE_PATH)) as connection, connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                is_superuser INTEGER NOT NULL DEFAULT 0 CHECK (is_superuser IN (0, 1))
            )
        """)
        if connection.execute("SELECT 1 FROM users WHERE is_superuser = 1").fetchone():
            return
        username = os.getenv("SUPERUSER_USERNAME", "")
        password = os.getenv("SUPERUSER_PASSWORD", "")
        if not username or not password:
            raise ValueError("Set SUPERUSER_USERNAME and SUPERUSER_PASSWORD before first startup")
        credentials = Credentials(username=username, password=password)
        connection.execute(
            "INSERT INTO users (username, password_hash, is_superuser) VALUES (?, ?, 1)",
            (credentials.username, hash_password(credentials.password.get_secret_value())),
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv(BASE_DIRECTORY / ".env")
    secret = os.getenv("APP_SECRET_KEY", "")
    if len(secret.encode("utf-8")) < 32:
        raise ValueError("APP_SECRET_KEY must contain at least 32 bytes of random secret material")
    app.state.secret = secret
    app.state.token_lifetime = parse_token_expiration(os.getenv("TOKEN_EXPIRATION", ""))
    initialize_database()
    # Unknown usernames still perform a password hash verification.
    app.state.dummy_hash = hash_password(secrets.token_urlsafe(32))
    runtime = create_runtime(secret)
    await runtime.start()
    app.state.runtime = runtime
    try:
        yield
    finally:
        await runtime.stop()


app = FastAPI(title="Flexy AI", lifespan=lifespan)
bearer = HTTPBearer(auto_error=False)


@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    error = exc.detail if isinstance(exc.detail, dict) else {
        "code": "HTTP_ERROR", "message": exc.detail,
    }
    return JSONResponse(
        status_code=exc.status_code, content={"success": False, "error": error},
        headers=exc.headers,
    )


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    # Pydantic's default error response can include the submitted password.
    return JSONResponse(status_code=422, content={
        "success": False,
        "error": {"code": "INVALID_REQUEST", "message": "Request validation failed."},
    })


@app.exception_handler(PlatformError)
async def platform_error(request: Request, exc: PlatformError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={
        "success": False,
        "error": {"code": exc.code, "message": exc.message},
    })


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={
        "success": False,
        "error": {"code": "INTERNAL_ERROR", "message": "An unexpected server error occurred."},
    })


def get_current_user(
    request: Request,
    authorization: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> sqlite3.Row:
    unauthorized = HTTPException(
        status_code=401,
        detail={"code": "INVALID_TOKEN", "message": "A valid bearer token is required."},
        headers={"WWW-Authenticate": "Bearer"},
    )
    if authorization is None:
        raise unauthorized
    try:
        claims = jwt.decode(
            authorization.credentials, request.app.state.secret, algorithms=["HS256"],
            options={"require": ["sub", "exp", "iat"]},
        )
    except jwt.InvalidTokenError:
        raise unauthorized from None
    with closing(sqlite3.connect(DATABASE_PATH)) as connection:
        connection.row_factory = sqlite3.Row
        user = connection.execute(
            "SELECT id, username, is_superuser FROM users WHERE id = ?", (claims["sub"],),
        ).fetchone()
    if user is None:
        raise unauthorized
    return user


def require_superuser(user: Annotated[sqlite3.Row, Depends(get_current_user)]) -> None:
    if not user["is_superuser"]:
        raise HTTPException(status_code=403, detail={
            "code": "SUPERUSER_REQUIRED", "message": "Only the superuser can register users.",
        })


def get_runtime(request: Request):
    return request.app.state.runtime


def _owner_id(user: sqlite3.Row) -> str:
    return str(user["id"])


async def parse_inference_form(request: Request) -> tuple[str, dict[str, Any], dict[str, list[UploadFile]]]:
    content_length = request.headers.get("content-length")
    if content_length is None:
        raise PlatformError("CONTENT_LENGTH_REQUIRED", "Content-Length is required for media uploads.", 411)
    try:
        request_size = int(content_length)
    except ValueError:
        raise PlatformError("INVALID_REQUEST", "Invalid Content-Length header.", 422) from None
    if request_size < 0:
        raise PlatformError("INVALID_REQUEST", "Invalid Content-Length header.", 422)
    if request_size > request.app.state.runtime.settings.max_upload_bytes:
        raise PlatformError("UPLOAD_TOO_LARGE", "Upload exceeds the configured size limit.", 413)
    try:
        form = await request.form()
    except Exception:
        raise PlatformError("INVALID_REQUEST", "Request must be valid multipart form data.", 422) from None
    scalar_fields: dict[str, str] = {}
    uploads: dict[str, list[UploadFile]] = {}
    try:
        for name, value in form.multi_items():
            if isinstance(value, UploadFile):
                uploads.setdefault(name, []).append(value)
            elif name in scalar_fields:
                raise PlatformError("INVALID_REQUEST", "Duplicate request field.", 422)
            else:
                scalar_fields[name] = str(value)
        if set(scalar_fields) - {"runner_uuid", "payload", "logging"}:
            raise PlatformError("INVALID_REQUEST", "Unsupported request field.", 422)
        runner_uuid = scalar_fields.get("runner_uuid", "")
        if not runner_uuid:
            raise PlatformError("INVALID_REQUEST", "runner_uuid is required.", 422)
        try:
            payload = json.loads(scalar_fields.get("payload", "{}"))
        except json.JSONDecodeError:
            raise PlatformError("INVALID_REQUEST", "payload must be a JSON object.", 422) from None
        if not isinstance(payload, dict):
            raise PlatformError("INVALID_REQUEST", "payload must be a JSON object.", 422)
        logging = scalar_fields.get("logging", "false").lower()
        if logging not in {"true", "false"}:
            raise PlatformError("INVALID_REQUEST", "logging must be true or false.", 422)
        return runner_uuid, payload, uploads
    except Exception:
        for files in uploads.values():
            for upload in files:
                await upload.close()
        raise


@app.post("/v1/api/get_token")
def get_token(credentials: Credentials, request: Request) -> dict[str, str]:
    with closing(sqlite3.connect(DATABASE_PATH)) as connection:
        connection.row_factory = sqlite3.Row
        user = connection.execute(
            "SELECT id, password_hash FROM users WHERE username = ?", (credentials.username,),
        ).fetchone()
    stored_hash = user["password_hash"] if user else request.app.state.dummy_hash
    valid = verify_password(credentials.password.get_secret_value(), stored_hash)
    if user is None or not valid:
        raise HTTPException(status_code=401, detail={
            "code": "INVALID_CREDENTIALS", "message": "Invalid username or password.",
        })
    now = datetime.now(timezone.utc)
    token = jwt.encode(
        {"sub": str(user["id"]), "iat": now, "exp": now + request.app.state.token_lifetime},
        request.app.state.secret, algorithm="HS256",
    )
    return {"token": token}


@app.post("/v1/api/register", status_code=201, dependencies=[Depends(require_superuser)])
def register_user(credentials: Credentials) -> dict[str, int | str]:
    password_hash = hash_password(credentials.password.get_secret_value())
    try:
        with closing(sqlite3.connect(DATABASE_PATH)) as connection, connection:
            cursor = connection.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (credentials.username, password_hash),
            )
            user_id = cursor.lastrowid
    except sqlite3.IntegrityError as exc:
        if exc.sqlite_errorname != "SQLITE_CONSTRAINT_UNIQUE":
            raise
        raise HTTPException(status_code=409, detail={
            "code": "USERNAME_EXISTS", "message": "Username already exists.",
        }) from None
    return {"id": user_id, "username": credentials.username}


@app.post("/v1/api/load_flexy_app")
async def load_flexy_app(
    payload: LoadFlexyAppRequest,
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    runtime=Depends(get_runtime),
) -> dict[str, str]:
    return await runtime.load_runner(_owner_id(user), payload.app, payload.config, payload.gpu.device)


@app.get("/v1/api/get_flexy_status")
async def get_flexy_status(
    runner_uuid: str,
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    runtime=Depends(get_runtime),
) -> dict[str, Any]:
    return await runtime.runner_status(_owner_id(user), runner_uuid)


@app.post("/v1/api/inference")
async def inference(
    request: Request,
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    runtime=Depends(get_runtime),
) -> dict[str, Any]:
    runner_uuid, payload, uploads = await parse_inference_form(request)
    result = await runtime.submit_inference(_owner_id(user), runner_uuid, payload, uploads)
    return {"success": True, "job_uuid": result["job_uuid"], "status": result["status"]}


@app.get("/v1/api/inference_status")
async def inference_status(
    request: Request,
    runner_uuid: str,
    job_uuid: str,
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    runtime=Depends(get_runtime),
) -> dict[str, Any]:
    result = await runtime.job_status(_owner_id(user), runner_uuid, job_uuid)
    signature = result.pop("download_signature", None)
    expires_at = result.pop("download_expires_at", None)
    if signature is not None and expires_at is not None:
        path = request.app.url_path_for("download_artifact", job_uuid=job_uuid)
        result["download_url"] = f"{path}?{urlencode({'expires': expires_at, 'signature': signature})}"
    return result


@app.post("/v1/api/unload_flexy_app")
async def unload_flexy_app(
    payload: RunnerRequest,
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    runtime=Depends(get_runtime),
) -> dict[str, str]:
    return await runtime.unload_runner(_owner_id(user), payload.runner_uuid)


@app.get("/v1/api/artifacts/{job_uuid}/output.mp4", name="download_artifact")
async def download_artifact(
    job_uuid: str,
    expires: int,
    signature: str,
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    runtime=Depends(get_runtime),
) -> FileResponse:
    path = await runtime.download_path(_owner_id(user), job_uuid, expires, signature)
    return FileResponse(path, media_type="video/mp4", filename="output.mp4")


@app.get("/v1/api/admin/resources", dependencies=[Depends(require_superuser)])
async def resource_status(runtime=Depends(get_runtime)) -> dict[str, Any]:
    return await runtime.resource_status()
