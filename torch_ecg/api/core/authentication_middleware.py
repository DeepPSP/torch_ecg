import base64
import logging
from typing import Tuple, Union

from fastapi import HTTPException
from starlette import status
from starlette.authentication import AuthCredentials, AuthenticationBackend, AuthenticationError, SimpleUser
from starlette.requests import HTTPConnection
from starlette.responses import PlainTextResponse, Response

logger = logging.getLogger(__name__)


def on_authentication_error(_conn: HTTPConnection, exc: Exception) -> Response:
    return PlainTextResponse(str(exc), status_code=status.HTTP_401_UNAUTHORIZED)


class BasicAuthBackend(AuthenticationBackend):
    """Backend for starlette.middleware.authentication.AuthenticationMiddleware.

    Usage:
        from starlette.middleware.authentication import AuthenticationMiddleware
        from shared_dependencies.auth_middleware import TokenAuthBackend, on_authentication_error

        app = FastAPI()
        app.add_middleware(
            AuthenticationMiddleware,
            backend=BasicAuthBackend(
                basic_creds=("username", "password"),
            ),
            on_error=on_authentication_error,
        )
    """

    def __init__(
        self,
        basic_creds: Union[Tuple[str, str], None] = None,
    ):
        """Backend for starlette.middleware.authentication.AuthenticationMiddleware.

        Args:
            basic_creds (Tuple[str, str] | None): Username and password for basic authentication.
        """
        self.basic_creds = basic_creds

    async def authenticate(self, conn: HTTPConnection) -> Tuple[AuthCredentials, SimpleUser]:
        if "Authorization" not in conn.headers:
            raise AuthenticationError("NO_AUTHORIZATION_HEADER")

        auth = conn.headers["Authorization"]

        if self.basic_creds is None:
            logger.error(f"basic_creds is unset. Aborting access to {conn.url.path}.")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"message": "Server did not configure credentials for this route."},
            )

        if auth.split()[0].lower() != "basic":
            raise AuthenticationError("NO_BASIC_AUTH_HEADER")

        scheme, token = auth.split()
        username, password = base64.b64decode(token).decode("utf-8").split(":")
        if (username, password) != self.basic_creds:
            raise AuthenticationError("INCORRECT_CREDENTIALS")
        return AuthCredentials(["authenticated"]), SimpleUser(username)
