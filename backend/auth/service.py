import random
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from .db import get_cursor
from .security import create_jwt, hash_password, verify_password

EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
MIN_PASSWORD_LEN = 6
OTP_TTL_MINUTES = 10

class AuthError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status

def _validate_email(email: str) -> str:
    email = (email or "").strip().lower()
    if not EMAIL_RE.match(email):
        raise AuthError("Email không hợp lệ.")
    return email

def _validate_password(password: str) -> None:
    if not password or len(password) < MIN_PASSWORD_LEN:
        raise AuthError(f"Mật khẩu phải có ít nhất {MIN_PASSWORD_LEN} ký tự.")

def _row_to_public(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "user_id": str(row["user_id"]),
        "email": row["email"],
        "name": row["full_name"],
        "avatar": row.get("avatar_url"),
        "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
    }

def register(email: str, password: str, full_name: str) -> Dict[str, Any]:
    email = _validate_email(email)
    _validate_password(password)
    full_name = (full_name or "").strip()
    if not full_name:
        raise AuthError("Họ và tên không được để trống.")

    with get_cursor() as cur:
        cur.execute(
            "SELECT user_id, is_deleted FROM users WHERE LOWER(email) = %s",
            (email,),
        )
        existing = cur.fetchone()
        if existing and not existing["is_deleted"]:
            raise AuthError("Email đã được sử dụng.", status=409)

        pwd_hash = hash_password(password)

        if existing and existing["is_deleted"]:
            cur.execute(
                """
                UPDATE users
                    SET password_hash = %s,
                        full_name = %s,
                        is_deleted = FALSE,
                        deleted_at = NULL,
                        updated_at = NOW()
                    WHERE user_id = %s
                RETURNING user_id, email, full_name, avatar_url, created_at
                """,
                (pwd_hash, full_name, existing["user_id"]),
            )
        else:
            cur.execute(
                """
                INSERT INTO users (email, password_hash, full_name)
                VALUES (%s, %s, %s)
                RETURNING user_id, email, full_name, avatar_url, created_at
                """,
                (email, pwd_hash, full_name),
            )

        row = cur.fetchone()

    user = _row_to_public(row)
    token = create_jwt(user["user_id"], user["email"])
    return {"user": user, "token": token}

def login(email: str, password: str) -> Dict[str, Any]:
    email = _validate_email(email)
    if not password:
        raise AuthError("Vui lòng nhập mật khẩu.")

    with get_cursor() as cur:
        cur.execute(
            """
            SELECT user_id, email, password_hash, full_name, avatar_url,
                   is_active, is_deleted, created_at
            FROM users
            WHERE LOWER(email) = %s
            """,
            (email,),
        )
        row = cur.fetchone()

    if not row or row["is_deleted"]:
        raise AuthError("Email hoặc mật khẩu không đúng.", status=401)
    if not row["is_active"]:
        raise AuthError("Tài khoản đã bị vô hiệu hoá.", status=403)
    if not verify_password(password, row["password_hash"]):
        raise AuthError("Email hoặc mật khẩu không đúng.", status=401)

    user = _row_to_public(row)
    token = create_jwt(user["user_id"], user["email"])
    return {"user": user, "token": token}

def request_password_reset(email: str) -> Dict[str, Any]:
    email = _validate_email(email)
    otp = f"{random.randint(0, 999999):06d}"
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=OTP_TTL_MINUTES)

    with get_cursor() as cur:
        cur.execute(
            "SELECT user_id FROM users WHERE LOWER(email) = %s AND is_deleted = FALSE",
            (email,),
        )
        row = cur.fetchone()
        if not row:
            return {"sent": True, "dev_otp": None}

        cur.execute(
            "UPDATE password_reset_tokens SET used_at = NOW() "
            "WHERE user_id = %s AND used_at IS NULL",
            (row["user_id"],),
        )
        cur.execute(
            """
            INSERT INTO password_reset_tokens (user_id, otp_code, expires_at)
            VALUES (%s, %s, %s)
            """,
            (row["user_id"], otp, expires_at),
        )

    return {"sent": True, "dev_otp": otp, "expires_in_minutes": OTP_TTL_MINUTES}

def reset_password(email: str, otp: str, new_password: str) -> Dict[str, Any]:
    email = _validate_email(email)
    _validate_password(new_password)
    otp = (otp or "").strip()
    if not otp:
        raise AuthError("Vui lòng nhập mã OTP.")

    with get_cursor() as cur:
        cur.execute(
            "SELECT user_id FROM users WHERE LOWER(email) = %s AND is_deleted = FALSE",
            (email,),
        )
        user_row = cur.fetchone()
        if not user_row:
            raise AuthError("Mã OTP không hợp lệ hoặc đã hết hạn.", status=400)

        cur.execute(
            """
            SELECT token_id, expires_at, used_at
            FROM password_reset_tokens
            WHERE user_id = %s AND otp_code = %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (user_row["user_id"], otp),
        )
        token_row = cur.fetchone()

        if (
            not token_row
            or token_row["used_at"] is not None
            or token_row["expires_at"] < datetime.now(timezone.utc)
        ):
            raise AuthError("Mã OTP không hợp lệ hoặc đã hết hạn.", status=400)

        cur.execute(
            "UPDATE password_reset_tokens SET used_at = NOW() WHERE token_id = %s",
            (token_row["token_id"],),
        )
        cur.execute(
            """
            UPDATE users
            SET password_hash = %s, updated_at = NOW()
            WHERE user_id = %s
            """,
            (hash_password(new_password), user_row["user_id"]),
        )

    return {"reset": True}

def soft_delete_user(user_id: str, reason: Optional[str] = None) -> None:
    del reason
    with get_cursor() as cur:
        cur.execute(
            """
            UPDATE users
                SET is_deleted = TRUE,
                    deleted_at = NOW(),
                    updated_at = NOW()
                WHERE user_id = %s AND is_deleted = FALSE
            """,
            (user_id,),
        )

def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT user_id, email, full_name, avatar_url, created_at
            FROM users
            WHERE user_id = %s AND is_deleted = FALSE
            """,
            (user_id,),
        )
        row = cur.fetchone()
    return _row_to_public(row) if row else None