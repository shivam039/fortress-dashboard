import re

with open("engine/utils/db.py", "r") as f:
    content = f.read()

target = """def _ensure_user_broker_connections_neon():"""

replacement = """def _ensure_user_broker_connections_neon():
    _exec(
        \"\"\"
        CREATE TABLE IF NOT EXISTS user_broker_connections (
            connection_id BIGSERIAL PRIMARY KEY,
            user_id BIGINT NOT NULL REFERENCES app_users(user_id) ON DELETE CASCADE,
            broker_name TEXT NOT NULL,
            broker_client_id TEXT,
            access_token_encrypted TEXT,
            refresh_token_encrypted TEXT,
            expires_at TIMESTAMPTZ,
            connected_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            is_active BOOLEAN DEFAULT TRUE,
            metadata_json JSONB,
            UNIQUE (user_id, broker_name)
        )
        \"\"\"
    )
    _exec("CREATE INDEX IF NOT EXISTS idx_broker_connections_user ON user_broker_connections (user_id)")
    _exec("CREATE INDEX IF NOT EXISTS idx_broker_connections_active ON user_broker_connections (is_active)")

def _ensure_user_broker_connections_neon_fallback():"""

# Just ensure they actually exist, but they are already defined in neon block based on earlier exploration
# Wait, let me check first.
