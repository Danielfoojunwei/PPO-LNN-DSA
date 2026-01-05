"""
Libiao RCS Authentication Module

Flexible authentication supporting multiple methods:
- None (no auth)
- Bearer token
- API key (header or query param)
- Cookie-based session
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
import time


class AuthProvider(ABC):
    """Base class for authentication providers."""

    @abstractmethod
    def get_headers(self) -> Dict[str, str]:
        """Get authentication headers to add to requests."""
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, str]:
        """Get authentication query parameters."""
        pass

    @abstractmethod
    def get_cookies(self) -> Dict[str, str]:
        """Get authentication cookies."""
        pass

    def is_expired(self) -> bool:
        """Check if credentials are expired."""
        return False

    def refresh(self) -> bool:
        """Refresh credentials if supported. Returns True on success."""
        return True


@dataclass
class NoAuth(AuthProvider):
    """No authentication required."""

    def get_headers(self) -> Dict[str, str]:
        return {}

    def get_params(self) -> Dict[str, str]:
        return {}

    def get_cookies(self) -> Dict[str, str]:
        return {}


@dataclass
class BearerTokenAuth(AuthProvider):
    """Bearer token authentication (Authorization: Bearer <token>)."""
    token: str
    expires_at: Optional[float] = None

    def get_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    def get_params(self) -> Dict[str, str]:
        return {}

    def get_cookies(self) -> Dict[str, str]:
        return {}

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


@dataclass
class ApiKeyAuth(AuthProvider):
    """
    API key authentication.

    Supports both header and query parameter modes.
    """
    api_key: str
    header_name: str = "X-API-Key"
    use_header: bool = True
    param_name: str = "api_key"

    def get_headers(self) -> Dict[str, str]:
        if self.use_header:
            return {self.header_name: self.api_key}
        return {}

    def get_params(self) -> Dict[str, str]:
        if not self.use_header:
            return {self.param_name: self.api_key}
        return {}

    def get_cookies(self) -> Dict[str, str]:
        return {}


@dataclass
class CookieAuth(AuthProvider):
    """Cookie-based session authentication."""
    session_cookie: str
    cookie_name: str = "session"
    expires_at: Optional[float] = None

    def get_headers(self) -> Dict[str, str]:
        return {}

    def get_params(self) -> Dict[str, str]:
        return {}

    def get_cookies(self) -> Dict[str, str]:
        return {self.cookie_name: self.session_cookie}

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


@dataclass
class BasicAuth(AuthProvider):
    """HTTP Basic authentication."""
    username: str
    password: str

    def get_headers(self) -> Dict[str, str]:
        import base64
        credentials = f"{self.username}:{self.password}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return {"Authorization": f"Basic {encoded}"}

    def get_params(self) -> Dict[str, str]:
        return {}

    def get_cookies(self) -> Dict[str, str]:
        return {}


@dataclass
class CompositeAuth(AuthProvider):
    """Combine multiple auth providers."""
    providers: list

    def get_headers(self) -> Dict[str, str]:
        headers = {}
        for provider in self.providers:
            headers.update(provider.get_headers())
        return headers

    def get_params(self) -> Dict[str, str]:
        params = {}
        for provider in self.providers:
            params.update(provider.get_params())
        return params

    def get_cookies(self) -> Dict[str, str]:
        cookies = {}
        for provider in self.providers:
            cookies.update(provider.get_cookies())
        return cookies

    def is_expired(self) -> bool:
        return any(p.is_expired() for p in self.providers)


class AuthFactory:
    """Factory for creating auth providers from config."""

    @staticmethod
    def from_config(config: Dict[str, Any]) -> AuthProvider:
        """
        Create auth provider from configuration dict.

        Config format:
        {
            "type": "none" | "bearer" | "api_key" | "cookie" | "basic",
            ... type-specific fields
        }
        """
        auth_type = config.get("type", "none").lower()

        if auth_type == "none":
            return NoAuth()

        elif auth_type == "bearer":
            return BearerTokenAuth(
                token=config["token"],
                expires_at=config.get("expires_at")
            )

        elif auth_type == "api_key":
            return ApiKeyAuth(
                api_key=config["api_key"],
                header_name=config.get("header_name", "X-API-Key"),
                use_header=config.get("use_header", True),
                param_name=config.get("param_name", "api_key")
            )

        elif auth_type == "cookie":
            return CookieAuth(
                session_cookie=config["session_cookie"],
                cookie_name=config.get("cookie_name", "session"),
                expires_at=config.get("expires_at")
            )

        elif auth_type == "basic":
            return BasicAuth(
                username=config["username"],
                password=config["password"]
            )

        else:
            raise ValueError(f"Unknown auth type: {auth_type}")

    @staticmethod
    def from_env() -> AuthProvider:
        """
        Create auth provider from environment variables.

        Checks for:
        - RCS_AUTH_TYPE: none, bearer, api_key, cookie, basic
        - RCS_BEARER_TOKEN: for bearer auth
        - RCS_API_KEY: for api_key auth
        - RCS_API_KEY_HEADER: header name (default X-API-Key)
        - RCS_SESSION_COOKIE: for cookie auth
        - RCS_COOKIE_NAME: cookie name (default session)
        - RCS_USERNAME, RCS_PASSWORD: for basic auth
        """
        import os

        auth_type = os.environ.get("RCS_AUTH_TYPE", "none").lower()

        if auth_type == "none":
            return NoAuth()

        elif auth_type == "bearer":
            token = os.environ.get("RCS_BEARER_TOKEN")
            if not token:
                raise ValueError("RCS_BEARER_TOKEN required for bearer auth")
            return BearerTokenAuth(token=token)

        elif auth_type == "api_key":
            api_key = os.environ.get("RCS_API_KEY")
            if not api_key:
                raise ValueError("RCS_API_KEY required for api_key auth")
            return ApiKeyAuth(
                api_key=api_key,
                header_name=os.environ.get("RCS_API_KEY_HEADER", "X-API-Key"),
                use_header=os.environ.get("RCS_API_KEY_USE_HEADER", "true").lower() == "true"
            )

        elif auth_type == "cookie":
            cookie = os.environ.get("RCS_SESSION_COOKIE")
            if not cookie:
                raise ValueError("RCS_SESSION_COOKIE required for cookie auth")
            return CookieAuth(
                session_cookie=cookie,
                cookie_name=os.environ.get("RCS_COOKIE_NAME", "session")
            )

        elif auth_type == "basic":
            username = os.environ.get("RCS_USERNAME")
            password = os.environ.get("RCS_PASSWORD")
            if not username or not password:
                raise ValueError("RCS_USERNAME and RCS_PASSWORD required for basic auth")
            return BasicAuth(username=username, password=password)

        else:
            raise ValueError(f"Unknown RCS_AUTH_TYPE: {auth_type}")
