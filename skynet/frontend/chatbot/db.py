import logging
import random
import string
import time
from datetime import datetime

import docker
import psycopg2
import asyncpg

from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from contextlib import contextmanager as cm

from skynet.constants import (
    DEFAULT_ROLE, DEFAULT_MODEL, DEFAULT_STEP,
    DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_GUIDANCE,
    DEFAULT_STRENGTH, DEFAULT_UPSCALER
)
from skynet.frontend.chatbot.types import BaseFileInput

DB_INIT_SQL = """
CREATE SCHEMA IF NOT EXISTS skynet;

CREATE TABLE IF NOT EXISTS skynet.user(
    id BIGSERIAL PRIMARY KEY NOT NULL,
    generated INT NOT NULL,
    joined TIMESTAMP NOT NULL,
    last_method TEXT,
    last_prompt TEXT,
    last_inputs TEXT,
    role VARCHAR(128) NOT NULL
);

CREATE TABLE IF NOT EXISTS skynet.user_config(
    id BIGSERIAL NOT NULL,
    model VARCHAR(512) NOT NULL,
    step INT NOT NULL,
    width INT NOT NULL,
    height INT NOT NULL,
    seed NUMERIC,
    guidance DECIMAL NOT NULL,
    strength DECIMAL NOT NULL,
    upscaler VARCHAR(128),
    autoconf BOOLEAN DEFAULT TRUE,
    CONSTRAINT fk_config
      FOREIGN KEY(id)
        REFERENCES skynet.user(id)
);

CREATE TABLE IF NOT EXISTS skynet.user_requests(
    id BIGSERIAL NOT NULL,
    user_id BIGSERIAL NOT NULL,
    sent TIMESTAMP NOT NULL,
    status TEXT NOT NULL,
    status_msg BIGSERIAL PRIMARY KEY NOT NULL,
    CONSTRAINT fk_user_req
      FOREIGN KEY(user_id)
        REFERENCES skynet.user(id)
);
"""


@cm
def open_new_database(cleanup: bool = True):
    """
    Context manager that spins up a temporary Postgres Docker container,
    creates a 'skynet' user and database, and yields (container, password, host).
    Stops the container on exit if 'cleanup' is True.
    """
    root_password = "".join(random.choice(string.ascii_lowercase) for _ in range(12))
    skynet_password = "".join(random.choice(string.ascii_lowercase) for _ in range(12))

    dclient = docker.from_env()
    container = dclient.containers.run(
        "postgres",
        name="skynet-test-postgres",
        ports={"5432/tcp": None},
        environment={"POSTGRES_PASSWORD": root_password},
        detach=True,
    )

    try:
        # Wait for Postgres to be ready
        for log_line in container.logs(stream=True):
            line = log_line.decode().rstrip()
            logging.info(line)
            if (
                "database system is ready to accept connections" in line
                or "database system is shut down" in line
            ):
                break

        container.reload()
        port_info = container.ports["5432/tcp"][0]
        port = port_info["HostPort"]
        db_host = f"localhost:{port}"

        # Let PostgreSQL settle
        time.sleep(1)
        logging.info("Creating 'skynet' database...")

        conn = psycopg2.connect(
            user="postgres", password=root_password, host="localhost", port=port
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        conn.autocommit = True
        cursor = conn.cursor()
        cursor.execute(f"CREATE USER skynet WITH PASSWORD '{skynet_password}'")
        cursor.execute("CREATE DATABASE skynet")
        cursor.execute("GRANT ALL PRIVILEGES ON DATABASE skynet TO skynet")
        cursor.close()
        conn.close()

        logging.info("Database setup complete.")
        yield container, skynet_password, db_host

    finally:
        if container and cleanup:
            container.stop()


class FrontendUserDB:
    """
    A class that manages the connection pool for the 'skynet' database,
    initializes the schema if needed, and provides high-level methods
    for interacting with the 'skynet' tables.
    """

    def __init__(
        self,
        db_user: str,
        db_pass: str,
        db_host: str,
        db_name: str
    ):
        self.db_user = db_user
        self.db_pass = db_pass
        self.db_host = db_host
        self.db_name = db_name
        self.pool: asyncpg.Pool | None = None

    async def __aenter__(self) -> "FrontendUserDB":
        dsn = f"postgres://{self.db_user}:{self.db_pass}@{self.db_host}/{self.db_name}"
        self.pool = await asyncpg.create_pool(dsn=dsn)
        await self._init_db()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.pool:
            await self.pool.close()

    async def _init_db(self):
        """
        Ensures the 'skynet' schema and tables exist. Also checks for
        missing columns and adds them if necessary.
        """
        async with self.pool.acquire() as conn:
            # Check if schema is already initialized
            result = await conn.fetch("""
                SELECT DISTINCT table_schema
                FROM information_schema.tables
                WHERE table_schema = 'skynet'
            """)
            if not result:
                await conn.execute(DB_INIT_SQL)

            # Check if 'autoconf' column exists in user_config
            col_check = await conn.fetch("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'user_config' AND column_name = 'autoconf'
            """)
            if not col_check:
                await conn.execute(
                    "ALTER TABLE skynet.user_config ADD COLUMN autoconf BOOLEAN DEFAULT TRUE;"
                )

    # -------------
    # USER METHODS
    # -------------

    async def get_user_config(self, user_id: int):
        """
        Fetches the user_config for the given user ID.
        Returns the record if found, otherwise None.
        """
        async with self.pool.acquire() as conn:
            records = await conn.fetch(
                "SELECT * FROM skynet.user_config WHERE id = $1", user_id
            )
        return dict(records[0]) if len(records) == 1 else None

    async def get_user(self, user_id: int):
        """Alias for get_user_config (same data returned)."""
        return await self.get_user_config(user_id)

    async def new_user(self, user_id: int):
        """
        Inserts a new user in skynet.user and its corresponding user_config record.
        Raises ValueError if the user already exists.
        """
        existing = await self.get_user(user_id)
        if existing:
            raise ValueError("User already present in DB")

        logging.info(f"New user! {user_id}")
        now = datetime.utcnow()

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO skynet.user(
                        id, generated, joined,
                        last_method, last_prompt, last_inputs, role
                    )
                    VALUES($1, 0, $2, 'txt2img', NULL, NULL, $3)
                    """,
                    user_id,
                    now,
                    DEFAULT_ROLE,
                )
                await conn.execute(
                    """
                    INSERT INTO skynet.user_config(
                        id, model, step, width,
                        height, guidance, strength, upscaler
                    )
                    VALUES($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    user_id,
                    DEFAULT_MODEL,
                    DEFAULT_STEP,
                    DEFAULT_WIDTH,
                    DEFAULT_HEIGHT,
                    DEFAULT_GUIDANCE,
                    DEFAULT_STRENGTH,
                    DEFAULT_UPSCALER,
                )

    async def get_or_create_user(self, user_id: int):
        """
        Retrieves a user_config record for the given user_id.
        If none exists, creates the user and returns the new record.
        """
        user_cfg = await self.get_user(user_id)
        if not user_cfg:
            await self.new_user(user_id)
            user_cfg = await self.get_user(user_id)
        return user_cfg

    async def update_user(self, user_id: int, attr: str, val):
        """
        Generic function to update a single field in skynet.user for a given user_id.
        """
        async with self.pool.acquire() as conn:
            await conn.execute(
                f"UPDATE skynet.user SET {attr} = $2 WHERE id = $1", user_id, val
            )

    async def update_user_config(self, user_id: int, attr: str, val):
        """
        Generic function to update a single field in skynet.user_config for a given user_id.
        """
        async with self.pool.acquire() as conn:
            await conn.execute(
                f"UPDATE skynet.user_config SET {attr} = $2 WHERE id = $1", user_id, val
            )

    async def get_user_stats(self, user_id: int):
        """
        Returns (generated, joined, role) for the given user_id.
        """
        async with self.pool.acquire() as conn:
            records = await conn.fetch(
                """
                SELECT generated, joined, role
                FROM skynet.user
                WHERE id = $1
                """,
                user_id,
            )
        return records[0] if records else None

    async def increment_generated(self, user_id: int):
        """
        Increments the 'generated' count for a given user by 1.
        """
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE skynet.user
                SET generated = generated + 1
                WHERE id = $1
                """,
                user_id,
            )

    async def update_user_stats(
        self,
        user_id: int,
        method: str,
        last_prompt: str | None = None,
        last_inputs: list | None = None
    ):
        """
        Updates various 'last_*' fields in skynet.user.
        """
        await self.update_user(user_id, "last_method", method)
        if last_prompt is not None:
            await self.update_user(user_id, "last_prompt", last_prompt)

        last_inputs_str = None
        if isinstance(last_inputs, list):
            last_inputs_str = ','.join((f'{f.id}:{f.cid}' for f in last_inputs))
            await self.update_user(user_id, "last_inputs", last_inputs_str)

        logging.info("Updated user stats: %s", (method, last_prompt, last_inputs_str))

    # ----------------------
    # USER REQUESTS METHODS
    # ----------------------

    async def get_user_request(self, request_id: int):
        """
        Fetches all matching rows for a given request_id.
        """
        async with self.pool.acquire() as conn:
            return await conn.fetch(
                "SELECT * FROM skynet.user_requests WHERE id = $1", request_id
            )

    async def get_user_request_by_sid(self, status_msg_id: int):
        """
        Fetches exactly one row (first row) by status_msg primary key.
        """
        async with self.pool.acquire() as conn:
            records = await conn.fetch(
                "SELECT * FROM skynet.user_requests WHERE status_msg = $1", status_msg_id
            )
        return records[0] if records else None

    async def new_user_request(
        self,
        user_id: int,
        request_id: int,
        status_msg_id: int,
        status: str = "started processing request..."
    ):
        """
        Inserts a new row in skynet.user_requests.
        """
        now = datetime.utcnow()
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO skynet.user_requests(
                        id, user_id, sent, status, status_msg
                    )
                    VALUES($1, $2, $3, $4, $5)
                    """,
                    request_id, user_id, now, status, status_msg_id
                )

    async def update_user_request(self, request_id: int, status: str):
        """
        Updates the 'status' for a user request identified by 'request_id'.
        """
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE skynet.user_requests
                SET status = $2
                WHERE id = $1
                """,
                request_id, status
            )

    async def update_user_request_by_sid(self, sid: int, status: str):
        """
        Updates the 'status' for a user request identified by 'status_msg'.
        """
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE skynet.user_requests
                SET status = $2
                WHERE status_msg = $1
                """,
                sid, status
            )

    # ----------------------------
    # Convenience "Get Last" Helpers
    # ----------------------------

    async def get_last_method_of(self, user_id: int) -> str | None:
        async with self.pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT last_method FROM skynet.user WHERE id = $1", user_id
            )

    async def get_last_prompt_of(self, user_id: int) -> str | None:
        async with self.pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT last_prompt FROM skynet.user WHERE id = $1", user_id
            )

    async def get_last_inputs_of(self, user_id: int) -> list[BaseFileInput] | None:
        async with self.pool.acquire() as conn:
            last_inputs_str = await conn.fetchval(
                "SELECT last_inputs FROM skynet.user WHERE id = $1", user_id
            )

            if not last_inputs_str:
                return []

            last_inputs = []
            for i in last_inputs_str.split(','):
                id, cid = i.split(':')
                last_inputs.from_values(id, cid)

            return last_inputs
