"""
Phase 1 — FastAPI Auth: APIM Header Identity
=============================================
APIM validates the Entra ID JWT and injects user identity into HTTP headers
before the request reaches FastAPI.  FastAPI reads those headers here —
zero JWT crypto, zero network calls, zero third-party auth libraries.

Header contract (set by apim/policies/global.xml):
  X-User-ID    → Entra OID (stable, unique across tenants)
  X-User-Email → preferred_username (UPN, e.g. doctor@hospital.com)
  X-User-Name  → display name (e.g. "Dr. Ahmed Hassan")
  X-User-Roles → comma-separated app roles (e.g. "Doctor,ClinicalAdmin")

Dev mode (APIM_ENABLED=false):
  All requests are treated as the ANONYMOUS user, which has both Doctor
  and ClinicalAdmin roles so every route is accessible locally without
  running APIM.  Set APIM_ENABLED=true only when traffic flows through APIM.

Security model:
  • In production FastAPI is VNet-internal — no public ingress.
  • APIM enforces JWT + roles before forwarding; X-User-* headers are
    overwritten (exists-action="override") so callers cannot spoof them.
  • FastAPI trusts these headers as authoritative — defence-in-depth only;
    the real security boundary is the VNet + APIM policy.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, HTTPException, Request

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Runtime flag — read at module import (after bootstrap() has run)
# ---------------------------------------------------------------------------
_APIM_ENABLED: bool = os.getenv("APIM_ENABLED", "false").lower() == "true"


# ---------------------------------------------------------------------------
# Identity model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AuthenticatedUser:
    """
    Immutable, hashable user identity extracted from APIM-injected headers.

    Frozen + slots = safe to cache, fast attribute access, no accidental mutation.
    """
    user_id:      str          # Entra OID — stable unique identifier
    email:        str          # preferred_username / UPN
    display_name: str          # human-readable name for logs and UI
    roles:        tuple[str, ...]  # app roles from the JWT "roles" claim

    # ── Role helpers ──────────────────────────────────────────────────────

    @property
    def is_admin(self) -> bool:
        """True if the user holds the ClinicalAdmin role."""
        return "ClinicalAdmin" in self.roles

    @property
    def is_stakeholder(self) -> bool:
        """True if the user holds ExecutiveStakeholder or FinancialAuditor role."""
        return any(r in self.roles for r in ("ExecutiveStakeholder", "FinancialAuditor"))

    @property
    def can_view_dashboard(self) -> bool:
        """True for ClinicalAdmin or ExecutiveStakeholder roles."""
        return self.is_admin or self.is_stakeholder

    @property
    def is_doctor(self) -> bool:
        """True for any clinical user (Doctor or ClinicalAdmin)."""
        return "Doctor" in self.roles or self.is_admin

    @property
    def identity_key(self) -> str:
        """Best available stable key for logging / rate-limit counters."""
        return self.user_id or self.email or "anonymous"

    @property
    def display(self) -> str:
        """Short human-readable string for log lines."""
        return self.display_name or self.email or self.user_id or "unknown"


# ---------------------------------------------------------------------------
# Singleton for dev / no-APIM mode
# ---------------------------------------------------------------------------

ANONYMOUS = AuthenticatedUser(
    user_id="00000000-0000-0000-0000-000000000000",
    email="dev@localhost",
    display_name="Dev Mode",
    roles=("Doctor", "ClinicalAdmin", "ExecutiveStakeholder"),
)


# ---------------------------------------------------------------------------
# Core extractor
# ---------------------------------------------------------------------------

def get_current_user(request: Request) -> AuthenticatedUser:
    """
    Extract and return the authenticated user from APIM-injected headers.

    Performance: pure header dict lookups — O(1), no I/O, no crypto.
    Called once per request via FastAPI dependency injection.

    In dev mode (APIM_ENABLED=false): returns ANONYMOUS by default, but allows
    developer header emulation (e.g. X-User-Roles, X-User-Email) or query param ?role=
    for testing role gates locally without standing up APIM.
    In prod mode (APIM_ENABLED=true): returns the user from headers,
    or ANONYMOUS with a warning if X-User-ID is missing (APIM misconfigured).
    """
    if not _APIM_ENABLED:
        # Dev / Staging role emulation support
        dev_roles = request.headers.get("x-user-roles") or request.query_params.get("role")
        if dev_roles:
            roles = tuple(r.strip() for r in dev_roles.split(",") if r.strip())
            return AuthenticatedUser(
                user_id=request.headers.get("x-user-id", "dev-emulator-id"),
                email=request.headers.get("x-user-email", "dev@localhost"),
                display_name=request.headers.get("x-user-name", "Dev User"),
                roles=roles,
            )
        return ANONYMOUS

    user_id   = request.headers.get("x-user-id",    "").strip()
    email     = request.headers.get("x-user-email", "").strip()
    name      = request.headers.get("x-user-name",  "").strip()
    roles_raw = request.headers.get("x-user-roles", "").strip()

    roles = tuple(r.strip() for r in roles_raw.split(",") if r.strip())

    if not user_id:
        # APIM global.xml always sets X-User-ID — if it's missing, APIM is
        # either misconfigured or bypassed.  Log loudly; return anonymous so
        # the app degrades gracefully rather than crashing.
        logger.warning(
            "[Auth] Request reached FastAPI without X-User-ID "
            "(path=%s) — APIM misconfigured or request bypassed gateway",
            request.url.path,
        )
        return ANONYMOUS

    user = AuthenticatedUser(
        user_id=user_id,
        email=email,
        display_name=name,
        roles=roles,
    )
    logger.debug("[Auth] %s roles=%s path=%s", user.display, roles_raw, request.url.path)
    return user


# ---------------------------------------------------------------------------
# FastAPI dependency factories
# ---------------------------------------------------------------------------

def require_authenticated(request: Request) -> AuthenticatedUser:
    """
    Dependency: any authenticated user (Doctor, ClinicalAdmin, ExecutiveStakeholder).

    In prod: APIM already enforced this via validate-jwt + required-claims.
    Here we double-check as defence-in-depth.
    """
    user = get_current_user(request)
    if _APIM_ENABLED and not (user.is_doctor or user.is_admin or user.is_stakeholder):
        raise HTTPException(
            status_code=403,
            detail="Access denied: valid app role required",
        )
    return user


def require_admin(request: Request) -> AuthenticatedUser:
    """
    Dependency: ClinicalAdmin role required.

    In prod: APIM admin.xml already blocked non-admins at the gateway;
    this is the Python-side duplicate check.
    """
    user = get_current_user(request)
    if not user.is_admin:
        raise HTTPException(
            status_code=403,
            detail="ClinicalAdmin role required",
        )
    return user


def require_dashboard_access(request: Request) -> AuthenticatedUser:
    """
    Dependency: ClinicalAdmin OR ExecutiveStakeholder role required.

    Used by /dashboard and dashboard analytic endpoints.
    Allows leadership and financial auditors to access high-level tracking,
    while restricting standard clinicians to clinical encounter entry.
    """
    user = get_current_user(request)
    if not user.can_view_dashboard:
        raise HTTPException(
            status_code=403,
            detail="Access denied: ClinicalAdmin or ExecutiveStakeholder role required",
        )
    return user


# ---------------------------------------------------------------------------
# Convenience: extract SSO user string for usage-log entries
# ---------------------------------------------------------------------------

def get_sso_user(request: Request) -> Optional[str]:
    """
    Return the authenticated user's display string for usage/audit logs.

    Replaces the old _get_client_user() in app.py which scanned several
    reverse-proxy headers.  Now we have a single authoritative source: APIM.
    Returns None in dev mode (ANONYMOUS has email dev@localhost, not useful).
    """
    if not _APIM_ENABLED:
        # Check legacy headers too so the transition is smooth in staging
        for hdr in ("x-forwarded-user", "x-remote-user", "x-auth-user", "remote-user"):
            val = request.headers.get(hdr, "").strip()
            if val:
                return val
        return None

    user = get_current_user(request)
    return user.display_name or user.email or user.user_id or None
