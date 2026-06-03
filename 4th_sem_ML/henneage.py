"""
HENNGE Challenge — Mission 3 Submission Helper
================================================
Fill in YOUR_EMAIL, YOUR_GIST_URL below, then run:
    python mission3_submit.py

What this script does:
  1. Generates a 10-digit TOTP using HMAC-SHA-512 (RFC 6238)
     Secret = your_email + "HENNGECHALLENGE004"
  2. Sends an HTTP POST with Basic Auth to the HENNGE API
"""

import hmac
import hashlib
import struct
import time
import base64
import json
import urllib.request
import urllib.error

# ─────────────────────────────────────────────
# ✏️  FILL THESE IN BEFORE RUNNING
# ─────────────────────────────────────────────
YOUR_EMAIL    = "laleshkumarraj.work@gmail.com"          # ← your real email
YOUR_GIST_URL = "https://gist.github.com/Lalesh2005/02feddabd6ec2b1b9b54dad0972a4792"  # ← your secret gist URL
SOLUTION_LANG = "python"                          # "python" or "golang"
# ─────────────────────────────────────────────


API_URL   = "https://api.challenge.hennge.com/challenges/backend-recursion/004"
TOTP_SALT = "HENNGECHALLENGE004"


def generate_totp(userid: str, digits: int = 10, time_step: int = 30, t0: int = 0) -> str:
    """
    RFC 6238 TOTP with HMAC-SHA-512.

    Steps (from RFC 6238 + errata):
    1. T  = floor((current_unix_time - T0) / X)    [X=30, T0=0]
    2. msg = T packed as big-endian unsigned 64-bit int
    3. h   = HMAC-SHA-512(secret_bytes, msg)        [64-byte digest]
    4. offset = last_byte(h) & 0x0F                 [0..15]
    5. P   = 4 bytes starting at h[offset], MSB first, mask top bit
    6. OTP = P % 10^digits, zero-padded to `digits` chars
    """
    secret = (userid + TOTP_SALT).encode("utf-8")
    T = (int(time.time()) - t0) // time_step
    msg = struct.pack(">Q", T)                          # 8-byte big-endian

    h = hmac.new(secret, msg, hashlib.sha512).digest()  # 64 bytes (SHA-512)

    offset = h[-1] & 0x0F                               # last nibble
    p = struct.unpack(">I", h[offset : offset + 4])[0]  # 4-byte big-endian int
    p &= 0x7FFFFFFF                                      # strip top bit (RFC 4226 §5.3)

    otp = p % (10 ** digits)
    return str(otp).zfill(digits)


def build_basic_auth(userid: str, password: str) -> str:
    """Base64-encode 'userid:password' for HTTP Basic Authentication."""
    credentials = f"{userid}:{password}"
    encoded = base64.b64encode(credentials.encode("utf-8")).decode("utf-8")
    return f"Basic {encoded}"


def submit(email: str, gist_url: str, lang: str) -> None:
    totp = generate_totp(email)
    print(f"[TOTP]  Generated OTP : {totp}")
    print(f"[AUTH]  userid         : {email}")

    payload = json.dumps({
        "github_url": gist_url,
        "contact_email": email,
        "solution_language": lang,
    }).encode("utf-8")

    auth_header = build_basic_auth(email, totp)

    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": auth_header,
        },
        method="POST",
    )

    print(f"\n[POST]  {API_URL}")
    print(f"[BODY]  {payload.decode()}\n")

    try:
        with urllib.request.urlopen(req) as resp:
            body = resp.read().decode("utf-8")
            print(f"[{resp.status}] ✅  {body}")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        print(f"[{e.code}] ❌  {body}")
        if e.code == 401:
            print(
                "\nHint: 401 Unauthorized usually means the TOTP was generated\n"
                "      with the wrong timestamp or secret. Double-check:\n"
                "      - secret = your_email + 'HENNGECHALLENGE004'\n"
                "      - HMAC-SHA-512 (not SHA-1)\n"
                "      - T = floor(unix_time / 30)  [T0=0, step=30]\n"
                "      - Digits = 10\n"
                "      Your system clock should be accurate (NTP-synced)."
            )
        elif e.code == 400:
            print("\nHint: 400 Bad Request — check your JSON payload fields.")
    except urllib.error.URLError as e:
        print(f"[ERROR] Network error: {e.reason}")


def main():
    if "your_email" in YOUR_EMAIL:
        print("⚠️  Please set YOUR_EMAIL and YOUR_GIST_URL at the top of this file first.")
        return
    submit(YOUR_EMAIL, YOUR_GIST_URL, SOLUTION_LANG)


if __name__ == "__main__":
    main()