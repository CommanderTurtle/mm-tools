import argparse
import os
import threading

import uvicorn

from core import stop_decoder
from server import app, manager


class Server(uvicorn.Server):
    shutdown_timer = None

    def handle_exit(self, sig, frame):
        # Stop between chunks before Uvicorn starts waiting for active requests.
        manager.stopping = True
        manager.cancel.set()
        stop_decoder()
        if self.shutdown_timer is None:
            # A native device call cannot be interrupted by asyncio cancellation.
            # This deadline is armed only on shutdown; process exit releases it.
            self.shutdown_timer = threading.Timer(30, os._exit, args=(128 + sig,))
            self.shutdown_timer.daemon = True
            self.shutdown_timer.start()
        super().handle_exit(sig, frame)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8293)
    parser.add_argument("--ssl-certfile")
    parser.add_argument("--ssl-keyfile")
    args = parser.parse_args()
    server = Server(uvicorn.Config(app, **vars(args), timeout_graceful_shutdown=20))
    try:
        server.run()
    finally:
        if manager.lock.acquire(blocking=False):
            try:
                manager.close()
                if server.shutdown_timer is not None:
                    server.shutdown_timer.cancel()
            finally:
                manager.lock.release()


if __name__ == "__main__":
    main()
