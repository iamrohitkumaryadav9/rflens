"""make api: serve the API on 127.0.0.1 only (the host is not configurable)."""

import sys

import uvicorn

from rflens.config import get_api_settings

HOST = "127.0.0.1"


def main() -> int:
    settings = get_api_settings()
    if settings.api_key is None or not settings.api_key.get_secret_value():
        print("API_KEY is not set in .env: set a long random value first.", file=sys.stderr)
        return 2
    print(f"RFLens API on http://{HOST}:{settings.api_port}  (X-API-Key required)", flush=True)
    uvicorn.run("rflens.api.main:app", host=HOST, port=settings.api_port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
