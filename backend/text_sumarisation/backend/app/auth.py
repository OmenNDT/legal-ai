import os
import time
import bcrypt
import jwt
import psycopg2
import psycopg2.extras
from contextlib import contextmanager
from psycopg2.pool import ThreadedConnectionPool
from flask import Blueprint, jsonify, request

# DSN trỏ về database legal_ai (table users, password_reset_tokens, users_history)
DSN = os.environ.get(
    "BDP_LEGAL_DSN",
    "postgresql://postgres:%24TTn120897%24@100.81.215.111:5432/legal_ai"
)
JWT_SECRET = os.environ.get(
    "BDP_LEGAL_JWT_SECRET",
    "text-summarization-dev-secret-change-me-9f3e7a1c2d4b6e8"
)
JWT_ALGO = "HS256"
TOKEN_TTL_HOURS = 24

# Pool kết nối, init một lần khi blueprint được register
_pool: ThreadedConnectionPool | None = None

def _get_pool() -> ThreadedConnectionPool:
    global _pool
    if _pool is None:
        _pool = ThreadedConnectionPool(1, 8, DSN)
    return _pool

@contextmanager
def _cur():
    pool = _get_pool()
    c = pool.getconn()
    try:
        cu = c.cursor(cursor_factory = psycopg2.extras.RealDictCursor)
        yield cu
        c.commit()
        cu.close()
    except Exception:
        c.rollback()
        raise
    finally:
        pool.putconn(c)

def _hash(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt(rounds = 12)).decode()

def _check(pw: str, h: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode(), h.encode())
    except ValueError:
        return False

def _user_public(row: dict) -> dict:
    return {
        "id": str(row["user_id"]),
        "email": row["email"],
        "name": row["full_name"],
        "is_active": row["is_active"],
        "avatar_url": row.get("avatar_url")
    }

def _issue_token(user_id) -> str:
    payload = {
        "sub": str(user_id),
        "iat": int(time.time()),
        "exp": int(time.time()) + TOKEN_TTL_HOURS * 3600
    }
    return jwt.encode(payload, JWT_SECRET, algorithm = JWT_ALGO)

def _user_from_request() -> dict | None:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    token = auth[7:]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms = [JWT_ALGO])
    except jwt.PyJWTError:
        return None
    uid = payload.get("sub")
    with _cur() as cu:
        cu.execute(
            "SELECT user_id, email, password_hash, full_name, avatar_url, is_active "
            "FROM users WHERE user_id = %s AND is_deleted = false",
            (uid,)
        )
        row = cu.fetchone()
    if not row or not row["is_active"]:
        return None
    return row

def _error(msg: str, code: int = 400):
    return jsonify({"error": msg}), code

bp = Blueprint("auth", __name__, url_prefix = "/api/auth")

# Đăng ký user mới vào bảng legal_ai.users
@bp.post("/register")
def register():
    body = request.get_json(silent = True) or {}
    name = (body.get("name") or "").strip()
    email = (body.get("email") or "").strip().lower()
    password = body.get("password") or ""
    if not name or not email or len(password) < 6:
        return _error("Vui lòng điền đầy đủ thông tin (mật khẩu ≥ 6 ký tự).")
    with _cur() as cu:
        cu.execute("SELECT 1 FROM users WHERE LOWER(email) = %s", (email,))
        if cu.fetchone():
            return _error("Email đã tồn tại.", 409)
        cu.execute(
            "INSERT INTO users (email, password_hash, full_name) VALUES (%s, %s, %s) "
            "RETURNING user_id, email, password_hash, full_name, avatar_url, is_active",
            (email, _hash(password), name)
        )
        row = cu.fetchone()
    token = _issue_token(row["user_id"])
    return jsonify({"user": _user_public(row), "token": token})

# Đăng nhập bằng email + password
@bp.post("/login")
def login():
    body = request.get_json(silent = True) or {}
    email = (body.get("email") or "").strip().lower()
    password = body.get("password") or ""
    if not email or not password:
        return _error("Vui lòng nhập email và mật khẩu.")
    with _cur() as cu:
        cu.execute(
            "SELECT user_id, email, password_hash, full_name, avatar_url, is_active, is_deleted "
            "FROM users WHERE LOWER(email) = %s",
            (email,),
        )
        row = cu.fetchone()
    if not row or row["is_deleted"] or not _check(password, row["password_hash"]):
        return _error("Sai email hoặc mật khẩu.", 401)
    if not row["is_active"]:
        return _error("Tài khoản đã bị vô hiệu hoá.", 403)
    token = _issue_token(row["user_id"])
    return jsonify({"user": _user_public(row), "token": token})

# Đặt lại mật khẩu trực tiếp bằng email - không OTP
@bp.post("/reset-password")
def reset_password():
    body = request.get_json(silent = True) or {}
    email = (body.get("email") or "").strip().lower()
    password = body.get("password") or ""
    if not email or len(password) < 6:
        return _error("Mật khẩu phải có ít nhất 6 ký tự.")
    with _cur() as cu:
        cu.execute(
            "UPDATE users SET password_hash = %s, updated_at = NOW() "
            "WHERE LOWER(email) = %s AND is_deleted = false RETURNING user_id",
            (_hash(password), email)
        )
        if cu.fetchone() is None:
            return _error("Email không tồn tại trong hệ thống.", 404)
    return jsonify({"ok": True})

# Endpoint cũ - chỉ verify email tồn tại (giữ tương thích nếu frontend còn gọi)
@bp.post("/forgot-password")
def forgot_password():
    body = request.get_json(silent = True) or {}
    email = (body.get("email") or "").strip().lower()
    if not email:
        return _error("Vui lòng nhập email.")
    with _cur() as cu:
        cu.execute(
            "SELECT 1 FROM users WHERE LOWER(email) = %s AND is_deleted = false",
            (email,)
        )
        exists = cu.fetchone() is not None
    if not exists:
        return _error("Email không tồn tại trong hệ thống.", 404)
    return jsonify({"sent": True})

# Thông tin user hiện tại
@bp.get("/me")
def me():
    row = _user_from_request()
    if not row:
        return _error("Chưa đăng nhập.", 401)
    return jsonify({"user": _user_public(row)})

# Soft-delete tài khoản
@bp.delete("/me")
def delete_me():
    row = _user_from_request()
    if not row:
        return _error("Chưa đăng nhập.", 401)
    with _cur() as cu:
        cu.execute(
            "UPDATE users SET is_deleted = true, deleted_at = NOW() WHERE user_id = %s",
            (row["user_id"],)
        )
    return jsonify({"ok": True})
