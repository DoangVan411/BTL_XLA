"""
Application factory.

Why:
- Initialize the Flask app in a reusable/testable function.
- Register blueprints and extensions here.
"""

from flask import Flask
from flask_cors import CORS

from app.api.routes import bp
from app.config import Config


def create_app() -> Flask:
    """Create and configure Flask application instance."""
    # Khởi tạo Flask app, nạp cấu hình và đăng ký các route/CORS
    app = Flask(__name__, template_folder=Config.TEMPLATES_DIR, static_folder=Config.STATIC_DIR)
    app.config.from_object(Config)

    # Register HTTP routes
    app.register_blueprint(bp)

    # Enable CORS (limit origins for security)
    CORS(app, resources={r"/*": {"origins": Config.CORS_ORIGINS}})

    return app


