import aiohttp
import jwt
import datetime
from typing import List, Optional


class ZypryxApi:
    def __init__(self, base_url: str, jwt_token: str):
        self.base_url = base_url.rstrip("/")
        self.jwt_token = jwt_token
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self.jwt_token}",
                "Content-Type": "application/json"
            }
        )
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.close()

    # ---------------------------------------------------------
    # JWT GENERATION (matches C# JwtFactory)
    # ---------------------------------------------------------
    @staticmethod
    def generate_jwt(
        key: str,
        issuer: str,
        audience: str,
        subject: str = "InternalService",
        role: str = "InternalService",
        minutes: int = 60
    ):
        now = datetime.datetime.utcnow()
        exp = now + datetime.timedelta(minutes=minutes)

        payload = {
            "sub": subject,
            "role": role,
            "iss": issuer,
            "aud": audience,
            "iat": now,
            "exp": exp
        }

        token = jwt.encode(payload, key, algorithm="HS256")
        return token

    # Allow updating token mid-session
    def set_jwt_token(self, token: str):
        self.jwt_token = token
        if self.session:
            self.session.headers.update({"Authorization": f"Bearer {token}"})

    # ---------------------------------------------------------
    # Internal request handler
    # ---------------------------------------------------------
    async def _send(self, method: str, route: str, **kwargs):
        url = f"{self.base_url}/{route.lstrip('/')}"
        try:
            async with self.session.request(method, url, ssl=False, **kwargs) as res:
                text = await res.text()

                if res.status >= 400:
                    print(f"[API ERROR] {res.status}: {text}")
                    return None

                if not text.strip():
                    return None

                return await res.json()

        except Exception as ex:
            print(f"[REQUEST ERROR] {ex}")
            return None

    # ---------------------------------------------------------
    # COIN ENDPOINTS
    # ---------------------------------------------------------
    async def get_all_coins(self) -> Optional[List[dict]]:
        return await self._send("GET", "api/coin")

    async def get_active_coins(self) -> Optional[List[dict]]:
        return await self._send("GET", "api/coin/active")

    async def get_coin(self, coin_id: int) -> Optional[dict]:
        return await self._send("GET", f"api/coin/{coin_id}")

    # ---------------------------------------------------------
    # KLINE ENDPOINTS
    # ---------------------------------------------------------
    async def get_klines(self, coin_id: int, interval: str):
        route = f"api/kline?coinId={coin_id}&interval={interval}"
        return await self._send("GET", route)

    async def insert_klines(self, klines: list):
        return await self._send("POST", "api/kline/insert", json=klines)

    async def get_latest_kline(self, coin_id: int, interval: str):
        return await self._send(
            "GET",
            f"api/kline/latest?coinId={coin_id}&interval={interval}"
        )

    async def get_earliest_kline(self, coin_id: int, interval: str):
        return await self._send(
            "GET",
            f"api/kline/earliest?coinId={coin_id}&interval={interval}"
        )

    async def delete_klines(self, start_ms: int, end_ms: int):
        return await self._send(
            "DELETE",
            f"api/kline?startDate={start_ms}&endDate={end_ms}"
        )

    # ---------------------------------------------------------
    # CONFIGURATION
    # ---------------------------------------------------------
    async def insert_configuration(self, config_dict: dict):
        return await self._send("POST", "api/configuration", json=config_dict)

    # ---------------------------------------------------------
    # READING ENDPOINT
    # ---------------------------------------------------------
    async def insert_reading(self, reading_dict: dict):
        return await self._send("POST", "api/reading", json=reading_dict)
