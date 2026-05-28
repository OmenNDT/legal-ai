"""Simple API Key authentication for RAG Extract (Flask-compatible).

No user database required. Uses a single API key set via environment variable.
Admin operations require the admin key. Read operations require any valid key.
"""
import os
from functools import wraps
from flask import request, jsonify, g

# API keys from environment
API_KEY = os.getenv("RAG_API_KEY")
ADMIN_API_KEY = os.getenv("RAG_ADMIN_API_KEY")

if not API_KEY:
    # Generate a random key if not set (for development)
    import secrets
    API_KEY = secrets.token_urlsafe(32)
    print(f"[rag_extract.auth] Generated API_KEY (set RAG_API_KEY env to override): {API_KEY}")

if not ADMIN_API_KEY:
    ADMIN_API_KEY = API_KEY  # Default: admin key same as read key


class SimpleUser:
    """Minimal user stand-in for compatibility with existing code."""
    def __init__(self, role: str = "user"):
        self.id = 1
        self.username = "api_user"
        self.role = role
        self.is_active = True


def require_api_key(admin_only: bool = False):
    """Flask decorator to validate API key via X-API-Key header.
    
    Localhost requests (127.0.0.1, ::1) are allowed without API key for development.
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            # Allow localhost without API key (safe for dev)
            remote_addr = request.remote_addr or ''
            if remote_addr in ('127.0.0.1', '::1', 'localhost'):
                g.rag_user = SimpleUser(role="admin")
                return fn(*args, **kwargs)
            
            api_key = request.headers.get("X-API-Key", "")
            if not api_key:
                return jsonify({"error": "Missing X-API-Key header"}), 401
            if api_key not in (API_KEY, ADMIN_API_KEY):
                return jsonify({"error": "Invalid API key"}), 401
            if admin_only and api_key != ADMIN_API_KEY:
                return jsonify({"error": "Admin key required"}), 403
            
            role = "admin" if api_key == ADMIN_API_KEY else "user"
            g.rag_user = SimpleUser(role=role)
            return fn(*args, **kwargs)
        return wrapper
    return decorator


def get_current_user() -> SimpleUser:
    """Get the current user from Flask g object."""
    return getattr(g, "rag_user", SimpleUser(role="user"))


def require_admin():
    """Decorator for admin-only endpoints."""
    return require_api_key(admin_only=True)
