from app.core.security import create_access_token, decode_token, hash_password, verify_password


def test_password_round_trip():
    hashed = hash_password("secret123")
    assert hashed != "secret123"
    assert verify_password("secret123", hashed) is True
    assert verify_password("wrong", hashed) is False


def test_jwt_round_trip():
    token = create_access_token({"sub": "user@test.com"})
    payload = decode_token(token)
    assert payload["sub"] == "user@test.com"


def test_jwt_invalid_token_returns_none():
    result = decode_token("not.a.valid.token")
    assert result is None
