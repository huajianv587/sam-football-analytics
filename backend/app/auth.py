from functools import lru_cache

import jwt
from fastapi import Header, HTTPException, status

from .config import get_settings


@lru_cache
def _jwks_client(url: str) -> jwt.PyJWKClient:
    return jwt.PyJWKClient(f"{url.rstrip('/')}/auth/v1/.well-known/jwks.json")


async def current_user_id(authorization: str | None = Header(default=None)) -> str:
    settings = get_settings()
    if settings.auth_disabled:
        if not settings.local_admin_user_id:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LOCAL_ADMIN_USER_ID is not configured",
            )
        return settings.local_admin_user_id
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing bearer token")
    if not settings.supabase_url:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Supabase is not configured")

    token = authorization.removeprefix("Bearer ").strip()
    try:
        signing_key = _jwks_client(settings.supabase_url).get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256", "ES256"],
            audience="authenticated",
        )
        return str(claims["sub"])
    except (jwt.PyJWTError, KeyError) as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid token") from exc
