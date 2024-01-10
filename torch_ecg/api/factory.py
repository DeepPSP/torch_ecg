from fastapi import FastAPI
from starlette.middleware.authentication import AuthenticationMiddleware

from torch_ecg_volta.api.config import settings
from torch_ecg_volta.api.core.authentication_middleware import BasicAuthBackend, on_authentication_error
from torch_ecg_volta.api.v1.router import router as router_v1


def setup_routers(app: FastAPI):
    app.include_router(router_v1)


def setup_middlewares(app: FastAPI):
    if settings.ENABLE_AUTHENTICATION:
        app.add_middleware(
            AuthenticationMiddleware,
            backend=BasicAuthBackend(
                basic_creds=("username", "password"),
            ),
            on_error=on_authentication_error,
        )


def create_app() -> FastAPI:
    app = FastAPI()
    setup_routers(app)
    setup_middlewares(app)
    return app
