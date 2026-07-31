"""
ecw_client.py

Read-only client for the eClinicalWorks FHIR R4 API, scoped to the
Patient resource (US Core Patient 6.1.0 profile).

Design notes
------------
- This client is intentionally READ-ONLY. No POST/PUT/PATCH/DELETE
  methods exist here. If write capability is ever needed later
  (e.g. for the MAR system), it should live in a separate client
  class so the QR/scan display path can never accidentally write
  to ECW.
- ECW rate-limits FHIR resource calls to 250/minute per practice
  code (per base URL). A simple token-bucket limiter is included
  so this client is safe to call from a live Streamlit session
  without needing to think about it at the call site.
- Auth: SMART Backend Services (asymmetric client authentication,
  RFC 7523 JWT-bearer client assertion) — this is what ECW's dev
  portal expects once you register a JWKS URL instead of a client
  secret. Each token request is authenticated with a short-lived
  JWT signed by your private key; ECW verifies it against the
  public key it fetched from your hosted jwks.json.

Usage
-----
    client = ECWClient(
        base_url="https://staging-fhir.ecwcloud.com/fhir/r4/FFBJCD",
        client_id=os.environ["ECW_CLIENT_ID"],
        token_url="https://staging-oauthserver.ecwcloud.com/oauth/oauth2/token",
        private_key_path=os.environ["ECW_PRIVATE_KEY_PATH"],
        kid=os.environ["ECW_KID"],
    )
    patient = client.get_patient_by_id("U4AqQsgbW6hNV8HfIb2RTFL9i1K9X8ASlFdXtZ1Zcm4")
"""

from __future__ import annotations

import time
import uuid
import threading
from dataclasses import dataclass, field
from typing import Optional, Any

import jwt as pyjwt  # PyJWT — pip install pyjwt cryptography

import requests


ECW_RATE_LIMIT_PER_MINUTE = 250


class ECWAuthError(Exception):
    """Raised when token acquisition fails."""


class ECWAPIError(Exception):
    """Raised for non-2xx responses from the ECW FHIR API."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"ECW API error {status_code}: {message}")


class _RateLimiter:
    """Simple sliding-window limiter so this client self-enforces
    ECW's 250 calls/minute/practice-code cap instead of relying on
    the caller to remember."""

    def __init__(self, max_calls: int, period_seconds: float = 60.0):
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self._calls: list[float] = []
        self._lock = threading.Lock()

    def wait_if_needed(self) -> None:
        with self._lock:
            now = time.monotonic()
            self._calls = [t for t in self._calls if now - t < self.period_seconds]
            if len(self._calls) >= self.max_calls:
                sleep_for = self.period_seconds - (now - self._calls[0])
                if sleep_for > 0:
                    time.sleep(sleep_for)
                now = time.monotonic()
                self._calls = [t for t in self._calls if now - t < self.period_seconds]
            self._calls.append(time.monotonic())


@dataclass
class PatientRecord:
    """Flattened, display-ready view of a US Core Patient resource.
    Only fields relevant to the QR/scan display card are pulled out;
    `raw` retains the full parsed FHIR JSON in case more is needed
    later (e.g. surfacing generalPractitioner or pharmacy extension)."""

    fhir_id: str
    mrn: Optional[str]
    full_name: str
    given_name: Optional[str]
    family_name: Optional[str]
    gender: Optional[str]
    birth_date: Optional[str]
    phone_home: Optional[str]
    phone_mobile: Optional[str]
    email: Optional[str]
    address_line: Optional[str]
    address_city: Optional[str]
    address_state: Optional[str]
    address_postal_code: Optional[str]
    active: Optional[bool]
    last_updated: Optional[str]
    raw: dict = field(default_factory=dict, repr=False)


class ECWClient:
    def __init__(
        self,
        base_url: str,
        client_id: str,
        token_url: str,
        private_key_path: str,
        kid: str,
        scope: str = "system/Patient.read",
        timeout_seconds: float = 15.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id
        self.token_url = token_url
        self.kid = kid
        self.scope = scope
        self.timeout_seconds = timeout_seconds

        with open(private_key_path, "rb") as f:
            self._private_key_pem = f.read()

        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0.0
        self._rate_limiter = _RateLimiter(ECW_RATE_LIMIT_PER_MINUTE)

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def _build_client_assertion(self) -> str:
        """Builds and signs the short-lived JWT that authenticates
        this client to ECW's token endpoint, per SMART Backend
        Services (RFC 7523). ECW verifies the signature using the
        public key it fetched from your hosted jwks.json — matched
        by the `kid` in this JWT's header."""
        now = int(time.time())
        claims = {
            "iss": self.client_id,
            "sub": self.client_id,
            "aud": self.token_url,
            "jti": str(uuid.uuid4()),
            "iat": now,
            "exp": now + 300,  # SMART spec: assertion must be short-lived (<=5 min)
        }
        return pyjwt.encode(
            claims,
            self._private_key_pem,
            algorithm="RS384",
            headers={"kid": self.kid, "typ": "JWT"},
        )

    def _fetch_access_token(self) -> None:
        client_assertion = self._build_client_assertion()
        resp = requests.post(
            self.token_url,
            data={
                "grant_type": "client_credentials",
                "client_assertion_type": "urn:ietf:params:oauth:client-assertion-type:jwt-bearer",
                "client_assertion": client_assertion,
                "scope": self.scope,
            },
            timeout=self.timeout_seconds,
        )
        if resp.status_code != 200:
            raise ECWAuthError(
                f"Token request failed ({resp.status_code}): {resp.text}"
            )
        payload = resp.json()
        self._access_token = payload["access_token"]
        expires_in = payload.get("expires_in", 3300)
        # refresh a little early to avoid edge-of-expiry failures
        self._token_expires_at = time.monotonic() + expires_in - 60

    def _get_valid_token(self) -> str:
        if self._access_token is None or time.monotonic() >= self._token_expires_at:
            self._fetch_access_token()
        assert self._access_token is not None
        return self._access_token

    # ------------------------------------------------------------------
    # Low-level GET
    # ------------------------------------------------------------------

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        self._rate_limiter.wait_if_needed()
        token = self._get_valid_token()
        url = f"{self.base_url}/{path.lstrip('/')}"
        resp = requests.get(
            url,
            headers={
                "Accept": "application/json+fhir",
                "Authorization": f"Bearer {token}",
            },
            params=params or {},
            timeout=self.timeout_seconds,
        )
        if resp.status_code == 401:
            # token may have been invalidated server-side; retry once
            self._access_token = None
            token = self._get_valid_token()
            resp = requests.get(
                url,
                headers={
                    "Accept": "application/json+fhir",
                    "Authorization": f"Bearer {token}",
                },
                params=params or {},
                timeout=self.timeout_seconds,
            )
        if resp.status_code >= 400:
            raise ECWAPIError(resp.status_code, resp.text)
        return resp.json()

    # ------------------------------------------------------------------
    # Patient resource — read only
    # ------------------------------------------------------------------

    def get_patient_by_id(self, patient_id: str) -> PatientRecord:
        """GET [base]/Patient/[id]"""
        raw = self._get(f"Patient/{patient_id}")
        return self._parse_patient(raw)

    def search_patient(
        self,
        *,
        family: Optional[str] = None,
        given: Optional[str] = None,
        birthdate: Optional[str] = None,  # yyyy-mm-dd
        gender: Optional[str] = None,
        phone: Optional[str] = None,
        identifier: Optional[str] = None,
    ) -> list[PatientRecord]:
        """Search using any combination of ECW's supported query
        parameter pairs (see docs: family+given+birthdate, etc.)."""
        params: dict[str, Any] = {}
        if family:
            params["family"] = family
        if given:
            params["given"] = given
        if birthdate:
            params["birthdate"] = birthdate
        if gender:
            params["gender"] = gender
        if phone:
            params["phone"] = phone
        if identifier:
            params["identifier"] = identifier

        raw = self._get("Patient", params=params)
        entries = raw.get("entry", [])
        return [self._parse_patient(e["resource"]) for e in entries]

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_patient(raw: dict) -> PatientRecord:
        def first(items, default=None):
            return items[0] if items else default

        names = raw.get("name", [])
        usual_name = next((n for n in names if n.get("use") == "usual"), first(names, {}))
        full_name = usual_name.get("text") or " ".join(
            usual_name.get("given", []) + [usual_name.get("family", "")]
        ).strip()

        identifiers = raw.get("identifier", [])
        mrn = next(
            (i["value"] for i in identifiers if i.get("use") == "secondary"),
            None,
        )

        telecoms = raw.get("telecom", [])
        phone_home = next(
            (t["value"] for t in telecoms if t.get("system") == "phone" and t.get("use") == "home"),
            None,
        )
        phone_mobile = next(
            (t["value"] for t in telecoms if t.get("system") == "phone" and t.get("use") == "mobile"),
            None,
        )
        email = next((t["value"] for t in telecoms if t.get("system") == "email"), None)

        addresses = raw.get("address", [])
        current_address = next(
            (a for a in addresses if a.get("use") == "home"), first(addresses, {})
        )

        return PatientRecord(
            fhir_id=raw.get("id", ""),
            mrn=mrn,
            full_name=full_name,
            given_name=usual_name.get("given", [None])[0] if usual_name.get("given") else None,
            family_name=usual_name.get("family"),
            gender=raw.get("gender"),
            birth_date=raw.get("birthDate"),
            phone_home=phone_home,
            phone_mobile=phone_mobile,
            email=email,
            address_line=", ".join(current_address.get("line", [])) or None,
            address_city=current_address.get("city"),
            address_state=current_address.get("state"),
            address_postal_code=current_address.get("postalCode"),
            active=raw.get("active"),
            last_updated=raw.get("meta", {}).get("lastUpdated"),
            raw=raw,
        )