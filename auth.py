"""Registration / login logic (no Streamlit code here so it is easy to test).

Passwords are hashed with scrypt (memory-hard KDF from the standard library)
using a random per-user salt. Plain-text passwords are never stored.
"""
import hashlib
import hmac
import os
import re

import database

_N, _R, _P = 2 ** 14, 8, 1
EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P, dklen=32)
    return f"scrypt${_N}${_R}${_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt_hex, hash_hex = stored.split("$")
        if scheme != "scrypt":
            return False
        digest = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex),
                                n=int(n), r=int(r), p=int(p), dklen=len(hash_hex) // 2)
        return hmac.compare_digest(digest.hex(), hash_hex)
    except (ValueError, TypeError):
        return False


def validate_registration(name, email, password, confirm) -> list:
    errors = []
    if not name or len(name.strip()) < 2:
        errors.append("Please enter your full name (at least 2 characters).")
    if not email or not EMAIL_RE.match(email.strip()):
        errors.append("Please enter a valid email address.")
    if not password or len(password) < 8:
        errors.append("Password must be at least 8 characters long.")
    elif not (re.search(r"[A-Za-z]", password) and re.search(r"\d", password)):
        errors.append("Password must contain at least one letter and one number.")
    if password != confirm:
        errors.append("Passwords do not match.")
    return errors


def register(name, email, password, confirm):
    """Returns (success: bool, message: str)."""
    errors = validate_registration(name, email, password, confirm)
    if errors:
        return False, " ".join(errors)
    ok = database.create_user(name.strip(), email.strip().lower(), hash_password(password))
    if not ok:
        return False, "An account with this email already exists."
    return True, "Account created successfully. Please log in."


def authenticate(email, password):
    """Returns {'id','name','email'} on success, otherwise None."""
    if not email or not password:
        return None
    user = database.get_user_by_email(email.strip().lower())
    if user and verify_password(password, user["password_hash"]):
        return {"id": user["id"], "name": user["name"], "email": user["email"]}
    # Spend similar time when the user doesn't exist (avoids user enumeration by timing)
    if not user:
        verify_password(password, hash_password("dummy-password-1"))
    return None
