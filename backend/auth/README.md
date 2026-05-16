# Auth module

Authentication backend for Legal AI: register, login, forgot/reset password,
soft-delete, with SCD-2 history on the `users` table.

## Architecture

```
backend/auth/
├── db_config.py    # Reads config/database/config.yaml (parses JDBC URL, ${ENV} vars)
├── db.py           # psycopg2 ThreadedConnectionPool + context managers
├── security.py     # bcrypt password hashing + JWT (HS256) helpers
├── service.py      # Business logic + validation (AuthError exceptions)
├── routes.py       # Flask Blueprint /api/auth/*
└── init_db.py      # Applies config/database/schema_auth.sql
```

## Database design

### `users` — current state, 1 row per account
Used for fast email lookups on login. Fields: `email`, `password_hash` (bcrypt),
`full_name`, `avatar_url`, `is_active`, `is_deleted`, timestamps.

### `users_history` — SCD Type 2 audit log
Every INSERT/UPDATE on `users` fires `fn_users_scd2` which:
1. Closes the previous "current" row (`valid_to = updated_at`, `is_current = false`)
2. Inserts a new row with `valid_from = updated_at`, `is_current = true`,
   `change_type ∈ {CREATE, UPDATE, DELETE, RESTORE}`

This gives a full timeline of every field for any user.

### Soft delete
`DELETE /api/auth/me` does NOT remove the row — it sets
`is_deleted = TRUE, deleted_at = NOW()`. The trigger logs change_type=`DELETE`.
Re-registering the same email later restores the row (change_type=`RESTORE`)
with the new password.

### `password_reset_tokens`
OTP codes for forgot-password flow. Each OTP lives 10 minutes; previous unused
OTPs for the same user are invalidated when a new one is issued.

## Setup

### 1. Environment
```bash
export POSTGRES_PASSWORD='your-postgres-password'
export JWT_SECRET='a-long-random-secret-for-prod'   # optional in dev
```

### 2. Install deps
```bash
pip install -r requirements.txt
```

### 3. Create tables on Postgres
```bash
python -m backend.auth.init_db
```

### 4. Start backend
```bash
python -m backend.app   # port 9010
```

### 5. Start frontend
```bash
cd frontend && npm run dev   # vite proxies /api → :9010
```

## API

| Method | Path                       | Auth | Body |
|--------|----------------------------|------|------|
| POST   | `/api/auth/register`       | —    | `{email, password, name}` |
| POST   | `/api/auth/login`          | —    | `{email, password}` |
| POST   | `/api/auth/forgot-password`| —    | `{email}` → returns `dev_otp` |
| POST   | `/api/auth/reset-password` | —    | `{email, otp, password}` |
| GET    | `/api/auth/me`             | JWT  | — |
| DELETE | `/api/auth/me`             | JWT  | — (soft delete) |

All successful login/register responses return `{user, token}`. The token is a
JWT (HS256) with payload `{sub, email, iat, exp}` and a 24h TTL.

## Querying the history

Latest snapshot per user:
```sql
SELECT * FROM users_history WHERE is_current = TRUE;
```

Full timeline for one user:
```sql
SELECT change_type, valid_from, valid_to, full_name, email, is_deleted
FROM users_history
WHERE user_id = '...' ORDER BY valid_from;
```

What did the system look like on 2025-01-01?
```sql
SELECT * FROM users_history
WHERE valid_from <= '2025-01-01'
  AND (valid_to > '2025-01-01' OR valid_to IS NULL);
```
