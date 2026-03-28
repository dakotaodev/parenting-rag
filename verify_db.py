"""Verify Supabase PostgreSQL connection."""

import os
import sys
from urllib.parse import quote

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # Fall back to environment variables already set

try:
    import psycopg2
except ImportError:
    print("psycopg2 not installed. Run: uv add psycopg2-binary")
    sys.exit(1)

conn_string = os.getenv("SUPABASE_URL")
if not conn_string:
    print("SUPABASE_URL not set. Add it to your .env file.")
    print("Find it in Supabase: Settings > Database > Connection string (URI mode)")
    sys.exit(1)


def encode_connection_string(url: str) -> str:
    """URL-encode special characters in the password to prevent parse errors."""
    scheme_end = url.index("://") + 3
    at_pos = url.rfind("@")
    userinfo = url[scheme_end:at_pos]
    rest = url[at_pos + 1 :]

    colon_pos = userinfo.index(":")
    user = userinfo[:colon_pos]
    password = userinfo[colon_pos + 1 :]

    # Strip surrounding brackets if copied from a template like [password]
    if password.startswith("[") and password.endswith("]"):
        password = password[1:-1]

    encoded_password = quote(password, safe="")
    return f"{url[:scheme_end]}{user}:{encoded_password}@{rest}"


conn_string = encode_connection_string(conn_string)
print(f"Connecting to: {conn_string[:40]}...")

try:
    conn = psycopg2.connect(conn_string, connect_timeout=10)
    cur = conn.cursor()
    cur.execute("SELECT version();")
    version = cur.fetchone()[0]
    cur.close()
    conn.close()
    print(f"Connected successfully!")
    print(f"PostgreSQL version: {version}")
except psycopg2.OperationalError as e:
    print(f"Connection failed: {e}")
    sys.exit(1)
