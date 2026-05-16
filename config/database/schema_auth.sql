-- =====================================================================
-- Auth schema for Legal AI
-- =====================================================================
-- Design notes:
--   * `users` is the CURRENT-state view (1 row per user_id) used for
--     fast lookup on login. The `password_hash` is bcrypt.
--   * `users_history` is the SCD Type 2 audit table. Every INSERT /
--     UPDATE / soft-DELETE on users produces one row here with
--     [valid_from, valid_to) windows and a change_type label.
--   * Soft delete: `users.is_deleted = TRUE` + `deleted_at` timestamp.
--     The row is never physically removed; it is just flagged.
--   * Triggers keep the two tables in sync automatically.
-- =====================================================================

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ---------- CURRENT STATE ---------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    user_id        UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    email          VARCHAR(255) NOT NULL UNIQUE,
    password_hash  VARCHAR(255) NOT NULL,
    full_name      VARCHAR(255) NOT NULL,
    avatar_url     TEXT,
    is_active      BOOLEAN      NOT NULL DEFAULT TRUE,
    is_deleted     BOOLEAN      NOT NULL DEFAULT FALSE,
    created_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    deleted_at     TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_users_email_active
    ON users (LOWER(email))
    WHERE is_deleted = FALSE;

-- ---------- SCD TYPE 2 AUDIT ------------------------------------------
CREATE TABLE IF NOT EXISTS users_history (
    history_id     BIGSERIAL    PRIMARY KEY,
    user_id        UUID         NOT NULL,
    email          VARCHAR(255) NOT NULL,
    password_hash  VARCHAR(255) NOT NULL,
    full_name      VARCHAR(255) NOT NULL,
    avatar_url     TEXT,
    is_active      BOOLEAN      NOT NULL,
    is_deleted     BOOLEAN      NOT NULL,
    valid_from     TIMESTAMPTZ  NOT NULL,
    valid_to       TIMESTAMPTZ,                       -- NULL = current
    is_current     BOOLEAN      NOT NULL,
    change_type    VARCHAR(16)  NOT NULL,             -- CREATE | UPDATE | DELETE | RESTORE
    changed_by     UUID,                              -- who triggered the change (NULL = self)
    change_reason  TEXT
);

CREATE INDEX IF NOT EXISTS idx_users_history_user
    ON users_history (user_id, valid_from DESC);

CREATE INDEX IF NOT EXISTS idx_users_history_current
    ON users_history (user_id)
    WHERE is_current = TRUE;

-- ---------- TRIGGERS to maintain SCD2 ---------------------------------
CREATE OR REPLACE FUNCTION fn_users_scd2()
RETURNS TRIGGER AS $$
DECLARE
    v_change_type VARCHAR(16);
BEGIN
    IF TG_OP = 'INSERT' THEN
        v_change_type := 'CREATE';

        INSERT INTO users_history (
            user_id, email, password_hash, full_name, avatar_url,
            is_active, is_deleted, valid_from, valid_to, is_current,
            change_type
        ) VALUES (
            NEW.user_id, NEW.email, NEW.password_hash, NEW.full_name, NEW.avatar_url,
            NEW.is_active, NEW.is_deleted, NEW.created_at, NULL, TRUE,
            v_change_type
        );

    ELSIF TG_OP = 'UPDATE' THEN
        -- Decide change type
        IF NEW.is_deleted = TRUE AND OLD.is_deleted = FALSE THEN
            v_change_type := 'DELETE';
        ELSIF NEW.is_deleted = FALSE AND OLD.is_deleted = TRUE THEN
            v_change_type := 'RESTORE';
        ELSE
            v_change_type := 'UPDATE';
        END IF;

        -- Close out the previous current row
        UPDATE users_history
           SET valid_to   = NEW.updated_at,
               is_current = FALSE
         WHERE user_id    = NEW.user_id
           AND is_current = TRUE;

        -- Insert a new current row
        INSERT INTO users_history (
            user_id, email, password_hash, full_name, avatar_url,
            is_active, is_deleted, valid_from, valid_to, is_current,
            change_type
        ) VALUES (
            NEW.user_id, NEW.email, NEW.password_hash, NEW.full_name, NEW.avatar_url,
            NEW.is_active, NEW.is_deleted, NEW.updated_at, NULL, TRUE,
            v_change_type
        );
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_users_scd2 ON users;
CREATE TRIGGER trg_users_scd2
AFTER INSERT OR UPDATE ON users
FOR EACH ROW EXECUTE FUNCTION fn_users_scd2();

-- ---------- PASSWORD RESET TOKENS (OTP) -------------------------------
CREATE TABLE IF NOT EXISTS password_reset_tokens (
    token_id     UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      UUID         NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    otp_code     VARCHAR(10)  NOT NULL,
    expires_at   TIMESTAMPTZ  NOT NULL,
    used_at      TIMESTAMPTZ,
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_prt_user_active
    ON password_reset_tokens (user_id)
    WHERE used_at IS NULL;
