
# Flexy AI

Flexy AI is a private, single-runner video/audio inference server. SQLite stores
users and password hashes; password-protected Redis stores only runner/job state,
queue IDs, and artifact expiry metadata. The current server deliberately allows
one isolated native worker at a time, with one selected CUDA GPU and CPU/RAM
offload where the video app needs it.

Standalone MiniMax H3 and WAN SCAIL-2 app packages are documented in
[docs/inference.md](docs/inference.md), including the HTTP contract, exact
checkpoint filenames, and native inference dependencies. Control-plane, queue,
artifact, authentication, and resource-policy tests have run; real GPU inference,
output parity, and memory use remain unverified.

Install Python 3.12+ and the dependencies:

```sh
python -m pip install -r requirements.txt
```

That installs the API control plane. A server that loads one of the native
video/audio apps must run from the same CUDA-enabled environment as the worker;
install the Torch build and `requirements-inference.txt` described in
[docs/inference.md](docs/inference.md) before loading an app.

Set these values in `.env` (existing process environment variables take precedence):

```dotenv
APP_SECRET_KEY=<random secret of at least 32 bytes>
TOKEN_EXPIRATION="30m"
SUPERUSER_USERNAME=<your administrator username>
SUPERUSER_PASSWORD=<your administrator password>
REDIS_PASSWORD=<random Redis password>
MODEL_PATH=D:/ComfyModels/models
ARTIFACT_ROOT=D:/FlexyAI/artifacts
AUTO_DELETE_MEDIA=30m
DOWNLOAD_URL_TTL=10m
```

`ARTIFACT_ROOT` is required and must be a dedicated temporary-media directory,
not the source tree or model directory.
Redis authentication is required: set `REDIS_PASSWORD`, or use an authenticated
`REDIS_URL` for a separately managed instance.

Generate a signing secret with `python -c "import secrets; print(secrets.token_urlsafe(32))"`.
Usernames are case-sensitive, contain no whitespace, and are at most 128 characters.
Passwords must contain 8–1024 characters. Token durations accept `s`, `m`, `h`, or `d`.

```sh
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --no-access-log
```

First startup creates `data/users.sqlite3` and seeds the superuser from `.env`.
SQLite stores scrypt hashes with a separate random salt per password, including the
superuser password. Existing accounts survive restarts; changing the bootstrap
credentials in `.env` does not reset them. After the first successful startup,
`SUPERUSER_PASSWORD` can be removed from `.env`. Keep `APP_SECRET_KEY` stable so
existing tokens remain valid. Preserve the `data/` directory across deployments.

1. Log in as the superuser using `POST /v1/api/get_token`:

   ```json
   {"username": "<superuser username>", "password": "<superuser password>"}
   ```

   The response is `{"token": "<JWT>"}`; its lifetime comes from `TOKEN_EXPIRATION`.

2. Register a regular user using `POST /v1/api/register` with the header
   `Authorization: Bearer <superuser JWT>`:

   ```json
   {"username": "alice", "password": "<user password>"}
   ```

   Returns HTTP 201 with the user's `id` and `username`. Only the stored superuser
   role authorizes registration; clients cannot assign roles. Duplicate usernames
   return 409, invalid/missing tokens return 401, and ordinary users receive 403.

3. The new user logs in through `/v1/api/get_token` with their own credentials.

## Video/audio inference API

The current safe allowlist contains `mmh3_fi2v`, `mmh3_fr2v`, and `wan_p_iv2v`.
Load one with `POST /v1/api/load_flexy_app`, then poll
`GET /v1/api/get_flexy_status?runner_uuid=...`. A load returns immediately while
the isolated child process initializes. Runner states are `loading`, `ready`,
`busy`, `stopping`, `stopped`, and `failed`; job states are `queued`, `running`,
`completed`, and `failed`. Terminal job metadata expires with its temporary
artifact window.

`POST /v1/api/inference` is an authenticated multipart request containing
`runner_uuid`, a JSON-string `payload`, and the app's named media fields. The
server generates owner/job-scoped filenames; clients cannot supply server input or
output paths. Clients must send `Content-Length`; chunked media uploads are
rejected before multipart parsing. Jobs are FIFO and only one job is dispatched to
the loaded runner.
`GET /v1/api/inference_status` returns a short-lived, JWT-protected signed URL
when the job completes. Failed-job inputs/outputs are deleted immediately;
successful outputs expire after `AUTO_DELETE_MEDIA`.

The scheduler samples available RAM, swap pressure, offload headroom, VRAM, and GPU temperature.
VRAM fullness by itself does not terminate these CPU-offloading apps. A sole worker
is stopped only for host/offload RAM hard stops, a severe swap hard stop, or a
sustained thermal hard stop.
Owners see a safe resource snapshot with their runner; the superuser can use
`GET /v1/api/admin/resources` for host-wide resource telemetry. Multi-GPU model
splitting and vLLM are intentionally not enabled for the current video/audio apps.

Interactive API documentation is available at `/docs`. Use HTTPS at the proxy
when accessing these credential-bearing endpoints over a network.
Authentication isolates Flexy users from each other; it cannot protect media from
Windows host administrators or infrastructure operators, who control the machine
and its storage.

## Windows-native deployment infrastructure

The Windows development profile keeps FastAPI and native video workers on the
host, where they can accurately monitor host RAM and use the documented Windows
CUDA environment. Docker runs only Redis, Nginx, and an optional Ngrok tunnel.
This avoids treating Docker Desktop's WSL memory view as the host's memory view.

Add `REDIS_PASSWORD`, `REDIS_PORT`, and `FLEXY_HTTP_PORT` from
`.env.example` to your untracked `.env`. Do not replace existing credentials or
commit `.env`.

Start the local infrastructure:

```powershell
docker compose --env-file .env -f docker-services/docker-compose.yml up -d redis nginx
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --no-access-log
```

When `REDIS_URL` is omitted, the native server derives its authenticated loopback
connection from `REDIS_PASSWORD` and `REDIS_PORT`. Set `REDIS_URL` only for a
separately managed Redis instance.

Nginx is then available only at `http://127.0.0.1:8080` by default. Redis is
also bound only to loopback so the native server can use it; it has no public
Docker port. Nginx has no artifact mount and returns 404 for `/artifacts/`.

To open a tunnel, set `NGROK_AUTHTOKEN` in `.env` and start the optional
profile explicitly:

```powershell
docker compose --env-file .env -f docker-services/docker-compose.yml --profile tunnel up -d
```

Validate the Compose configuration without starting services:

```powershell
docker compose --env-file .env -f docker-services/docker-compose.yml config --quiet
```

Run the focused tests with a temporary SQLite database:

```sh
python -m pip install httpx
python -m unittest -v test_main test_platform test_resource_manager test_inference_helpers
```
