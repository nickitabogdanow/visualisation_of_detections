from __future__ import annotations

import os
import sys
from multiprocessing import freeze_support

import uvicorn


def main() -> None:
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8002"))
    is_frozen = getattr(sys, "frozen", False)
    reload_enabled = (
        not is_frozen and os.getenv("RELOAD", "true").lower() in {"1", "true", "yes"}
    )

    print(f"Запускаю сервер: http://{host}:{port}")
    print("Для остановки нажмите Ctrl+C")
    uvicorn.run("main:app", host=host, port=port, reload=reload_enabled)


if __name__ == "__main__":
    freeze_support()
    main()
