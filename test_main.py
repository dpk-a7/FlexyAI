import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import jwt
from fastapi.testclient import TestClient

import main


class NoopRuntime:
    async def start(self):
        pass

    async def stop(self):
        pass


class AuthenticationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.database = Path(temporary.name) / "users.sqlite3"
        self.enterContext(patch.object(main, "DATABASE_PATH", self.database))
        self.enterContext(patch.dict(os.environ, {
            "APP_SECRET_KEY": "test-only-signing-secret-at-least-32-bytes",
            "TOKEN_EXPIRATION": "30m",
            "SUPERUSER_USERNAME": "admin",
            "SUPERUSER_PASSWORD": "test-admin-password",
        }))
        self.enterContext(patch.object(main, "create_runtime", return_value=NoopRuntime()))
        self.client = self.enterContext(TestClient(main.app))

    def login(self, username="admin", password="test-admin-password"):
        response = self.client.post("/v1/api/get_token", json={
            "username": username, "password": password,
        })
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["token"]

    def test_superuser_registers_salted_users_and_users_cannot_register(self):
        headers = {"Authorization": f"Bearer {self.login()}"}
        password = "same-test-password"
        for username in ("alice", "bob"):
            response = self.client.post("/v1/api/register", headers=headers, json={
                "username": username, "password": password,
            })
            self.assertEqual(response.status_code, 201, response.text)
            self.assertEqual(set(response.json()), {"id", "username"})
        with closing(sqlite3.connect(self.database)) as connection:
            rows = connection.execute(
                "SELECT password_hash, is_superuser FROM users WHERE username IN ('alice', 'bob')"
            ).fetchall()
        self.assertNotEqual(rows[0][0], rows[1][0])
        for encoded, role in rows:
            self.assertNotIn(password, encoded)
            self.assertTrue(main.verify_password(password, encoded))
            self.assertEqual(role, 0)
        token = self.login("alice", password)
        response = self.client.post("/v1/api/register", headers={
            "Authorization": f"Bearer {token}",
        }, json={"username": "mallory", "password": password})
        self.assertEqual(response.status_code, 403)
        duplicate = self.client.post("/v1/api/register", headers=headers, json={
            "username": "alice", "password": "different-password",
        })
        self.assertEqual(duplicate.status_code, 409)
        self.login("alice", password)
        with TestClient(main.app) as restarted:
            response = restarted.post("/v1/api/get_token", json={
                "username": "alice", "password": password,
            })
            self.assertEqual(response.status_code, 200)

    def test_invalid_credentials_and_invalid_tokens_are_rejected(self):
        for username in ("admin", "missing", "'OR'1'='1"):
            response = self.client.post("/v1/api/get_token", json={
                "username": username, "password": "wrong-password",
            })
            self.assertEqual(response.status_code, 401)
            self.assertNotIn("wrong-password", response.text)
        claims = jwt.decode(self.login(), main.app.state.secret, algorithms=["HS256"])
        self.assertEqual(claims["exp"] - claims["iat"], 1800)
        expired = jwt.encode({**claims, "iat": 1, "exp": 2}, main.app.state.secret, algorithm="HS256")
        forged = jwt.encode(claims, "different-signing-secret-at-least-32-bytes", algorithm="HS256")
        missing_exp = jwt.encode({"sub": claims["sub"], "iat": claims["iat"]}, main.app.state.secret, algorithm="HS256")
        for token in (None, "invalid-token", expired, forged, missing_exp):
            headers = {} if token is None else {"Authorization": f"Bearer {token}"}
            response = self.client.post("/v1/api/register", headers=headers, json={
                "username": "alice", "password": "test-user-password",
            })
            self.assertEqual(response.status_code, 401, response.text)

    def test_registration_cannot_assign_superuser_and_errors_hide_passwords(self):
        headers = {"Authorization": f"Bearer {self.login()}"}
        for payload in (
            {"username": "alice", "password": "test-user-password", "is_superuser": True},
            {"username": "alice", "password": "short"},
        ):
            response = self.client.post("/v1/api/register", headers=headers, json=payload)
            self.assertEqual(response.status_code, 422)
            self.assertNotIn(payload["password"], response.text)

    def test_restart_preserves_superuser_password_and_checks_current_role(self):
        token = self.login()
        with patch.dict(os.environ, {"SUPERUSER_PASSWORD": "changed-password"}):
            main.initialize_database()
        self.login()
        with closing(sqlite3.connect(self.database)) as connection, connection:
            connection.execute("UPDATE users SET is_superuser = 0 WHERE username = 'admin'")
        response = self.client.post("/v1/api/register", headers={
            "Authorization": f"Bearer {token}",
        }, json={"username": "alice", "password": "test-user-password"})
        self.assertEqual(response.status_code, 403)

    def test_configured_lifetime_and_missing_configuration(self):
        with patch.dict(os.environ, {"TOKEN_EXPIRATION": "2h"}):
            with TestClient(main.app) as client:
                response = client.post("/v1/api/get_token", json={
                    "username": "admin", "password": "test-admin-password",
                })
                self.assertEqual(response.status_code, 200)
                claims = jwt.decode(response.json()["token"], main.app.state.secret, algorithms=["HS256"])
                self.assertEqual(claims["exp"] - claims["iat"], 7200)
        for value in ("", "0m", "30", "-1h", "forever"):
            with self.assertRaises(ValueError):
                main.parse_token_expiration(value)
        with patch.dict(os.environ, {"APP_SECRET_KEY": ""}):
            with self.assertRaisesRegex(ValueError, "APP_SECRET_KEY"):
                with TestClient(main.app):
                    pass
        with patch.object(main, "DATABASE_PATH", self.database.parent / "fresh.sqlite3"):
            with patch.dict(os.environ, {"SUPERUSER_PASSWORD": ""}):
                with self.assertRaisesRegex(ValueError, "SUPERUSER_PASSWORD"):
                    main.initialize_database()


if __name__ == "__main__":
    unittest.main()
