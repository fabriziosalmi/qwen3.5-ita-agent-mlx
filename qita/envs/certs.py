"""TLS certificate manager server (test-only env), English tool names."""

from __future__ import annotations

import datetime as dt

from .base import Env, ToolError

TODAY = dt.date(2026, 11, 4)
RENEW_DAYS = 90
SEED = [
    {"domain": "ilfaro.example", "expires": "2026-11-12", "auto_renew": True, "dns_provider": "cloudflare",
     "sans": ["ilfaro.example", "www.ilfaro.example"], "revoked": False},
    {"domain": "shop.ilfaro.example", "expires": "2027-01-20", "auto_renew": True, "dns_provider": "cloudflare",
     "sans": ["shop.ilfaro.example"], "revoked": False},
    {"domain": "mail.ilfaro.example", "expires": "2026-10-30", "auto_renew": False, "dns_provider": "route53",
     "sans": ["mail.ilfaro.example"], "revoked": False},
    {"domain": "api.studio-levante.example", "expires": "2026-11-25", "auto_renew": False, "dns_provider": "hetzner",
     "sans": ["api.studio-levante.example"], "revoked": False},
    {"domain": "blog.ilfaro.example", "expires": "2027-03-01", "auto_renew": True, "dns_provider": "cloudflare",
     "sans": ["blog.ilfaro.example"], "revoked": False},
]


def days_left(c: dict) -> int:
    return (dt.date.fromisoformat(c["expires"]) - TODAY).days


def status(c: dict) -> str:
    if c["revoked"]:
        return "revoked"
    d = days_left(c)
    return "expired" if d < 0 else "expiring" if d <= 30 else "valid"


def line(c: dict) -> str:
    return (f"{c['domain']} | {status(c)} | expires {c['expires']} ({days_left(c)} days) | "
            f"auto_renew={'on' if c['auto_renew'] else 'off'}")


class CertsEnv(Env):
    name = "certs"
    title = "certificates"

    def build(self) -> None:
        self.certs = {c["domain"]: dict(c, sans=list(c["sans"])) for c in SEED}
        s = self.server

        def get(domain: str) -> dict:
            d = domain.strip().lower()
            if d not in self.certs:
                raise ToolError(f"Certificate not found: {domain}")
            return self.certs[d]

        @s.tool()
        def list_certificates(status_filter: str | None = None) -> str:
            """List managed TLS certificates. Optional status_filter: valid, expiring (30 days or
            less), expired, revoked."""
            if status_filter and status_filter not in ("valid", "expiring", "expired", "revoked"):
                raise ToolError("status_filter must be one of: valid, expiring, expired, revoked")
            rows = [line(c) for c in sorted(self.certs.values(), key=lambda c: c["expires"])
                    if status_filter is None or status(c) == status_filter]
            return "\n".join(rows) or "No certificates match"

        @s.tool()
        def get_certificate(domain: str) -> str:
            """Show details of one certificate: SANs, expiry, status, DNS provider, auto-renewal."""
            c = get(domain)
            return (f"domain: {c['domain']}\nsans: {', '.join(c['sans'])}\nstatus: {status(c)}\n"
                    f"expires: {c['expires']} ({days_left(c)} days)\ndns_provider: {c['dns_provider']}\n"
                    f"auto_renew: {'on' if c['auto_renew'] else 'off'}\nissuer: Let's Encrypt R12")

        @s.tool()
        def renew_certificate(domain: str) -> str:
            """Renew a certificate now via ACME DNS-01. The new certificate is valid for 90 days."""
            c = get(domain)
            if c["revoked"]:
                raise ToolError(f"{domain} is revoked; issue a new certificate instead")
            c["expires"] = (TODAY + dt.timedelta(days=RENEW_DAYS)).isoformat()
            return f"Renewed {c['domain']}: new expiry {c['expires']}"

        @s.tool()
        def set_auto_renew(domain: str, enabled: bool) -> str:
            """Turn automatic renewal on or off for a certificate."""
            c = get(domain)
            c["auto_renew"] = bool(enabled)
            return f"auto_renew for {c['domain']} is now {'on' if enabled else 'off'}"

        @s.tool()
        def issue_certificate(domain: str, dns_provider: str = "cloudflare") -> str:
            """Issue a new certificate for a domain that is not managed yet."""
            d = domain.strip().lower()
            if d in self.certs:
                raise ToolError(f"{d} already has a certificate; renew it instead")
            if "." not in d or " " in d:
                raise ToolError(f"Invalid domain: {domain}")
            self.certs[d] = {"domain": d, "expires": (TODAY + dt.timedelta(days=RENEW_DAYS)).isoformat(),
                             "auto_renew": True, "dns_provider": dns_provider, "sans": [d], "revoked": False}
            return f"Issued {d}, expires {self.certs[d]['expires']}"

        @s.tool()
        def revoke_certificate(domain: str, reason: str = "unspecified") -> str:
            """Revoke a certificate. Browsers stop trusting it immediately. This cannot be undone."""
            c = get(domain)
            c["revoked"] = True
            return f"Revoked {c['domain']} (reason: {reason})"

    def check(self, kind: str, spec: dict) -> bool | None:
        c = self.certs.get(spec.get("domain", ""))
        if kind == "cert_expires":
            return c is not None and c["expires"] == spec["date"]
        if kind == "cert_revoked":
            return c is not None and c["revoked"] is spec.get("value", True)
        if kind == "cert_auto_renew":
            return c is not None and c["auto_renew"] is spec["value"]
        if kind == "cert_exists":
            return c is not None
        if kind == "cert_count":
            return len(self.certs) == spec["n"]
        return None
