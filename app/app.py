"""
Thin entrypoint script.

Why:
- Delegate app creation to a factory for better testability and structure.
"""

import os
import threading
import webbrowser

from app.config import Config
from app.factory import create_app


def _open_browser():
    """Open the browser once after the server starts (avoid double-open with reloader)."""
    webbrowser.open(f"http://localhost:{Config.PORT}/")


app = create_app()


if __name__ == '__main__':
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        threading.Timer(1.0, _open_browser).start()
    app.run(debug=Config.DEBUG, port=Config.PORT)
