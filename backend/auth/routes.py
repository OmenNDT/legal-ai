from functools import wraps
from flask import Blueprint, g, jsonify, request
from . import service
from .security import decode_jwt
import jwt as pyjwt

auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")

def _err(message: str, status: int = 400):
    return jsonify({"error": message}), status

def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        header = request.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            return _err("Thiếu token xác thực.", 401)
        token = header[7:]
        try:
            payload = decode_jwt(token)
        except pyjwt.ExpiredSignatureError:
            return _err("Phiên đăng nhập đã hết hạn.", 401)
        except pyjwt.PyJWTError:
            return _err("Token không hợp lệ.", 401)
        g.user_id = payload["sub"]
        g.user_email = payload.get("email")
        return fn(*args, **kwargs)
    return wrapper

@auth_bp.post("/register")
def register():
    body = request.get_json(silent=True) or {}
    try:
        result = service.register(
            email=body.get("email", ""),
            password=body.get("password", ""),
            full_name=body.get("name", ""),
        )
        return jsonify(result)
    except service.AuthError as e:
        return _err(str(e), e.status)

@auth_bp.post("/login")
def login():
    body = request.get_json(silent=True) or {}
    try:
        result = service.login(
            email=body.get("email", ""),
            password=body.get("password", ""),
        )
        return jsonify(result)
    except service.AuthError as e:
        return _err(str(e), e.status)

@auth_bp.post("/forgot-password")
def forgot_password():
    body = request.get_json(silent=True) or {}
    try:
        result = service.request_password_reset(email=body.get("email", ""))
        return jsonify(result)
    except service.AuthError as e:
        return _err(str(e), e.status)

@auth_bp.post("/reset-password")
def reset_password():
    body = request.get_json(silent=True) or {}
    try:
        result = service.reset_password(
            email=body.get("email", ""),
            otp=body.get("otp", ""),
            new_password=body.get("password", ""),
        )
        return jsonify(result)
    except service.AuthError as e:
        return _err(str(e), e.status)

@auth_bp.get("/me")
@login_required
def me():
    user = service.get_user_by_id(g.user_id)
    if not user:
        return _err("Người dùng không tồn tại.", 404)
    return jsonify({"user": user})

@auth_bp.delete("/me")
@login_required
def delete_me():
    service.soft_delete_user(g.user_id)
    return jsonify({"deleted": True})
