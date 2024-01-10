import uvicorn

from torch_ecg_volta.api.config import settings


def main():
    uvicorn.run(
        app="torch_ecg_volta.api:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        workers=settings.APP_WORKERS,
    )


if __name__ == "__main__":
    main()
