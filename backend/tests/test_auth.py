"""Unit tests for the authentication service."""

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from app.core.exceptions import (
    ConflictException,
    UnauthorizedException,
    ValidationException,
)
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    validate_password_complexity,
    verify_password,
)
from app.services.auth_service import AuthService


# --- Password Complexity Tests ---


class TestPasswordComplexity:
    """Tests for password complexity validation."""

    def test_valid_password_three_categories(self):
        """Password with uppercase, lowercase, and digits passes."""
        is_valid, msg = validate_password_complexity("Abcdef12")
        assert is_valid is True
        assert msg == ""

    def test_valid_password_four_categories(self):
        """Password with all four categories passes."""
        is_valid, msg = validate_password_complexity("Abc123!@")
        assert is_valid is True

    def test_valid_password_upper_lower_special(self):
        """Password with uppercase, lowercase, and special chars passes."""
        is_valid, msg = validate_password_complexity("Abcdef!@")
        assert is_valid is True

    def test_too_short(self):
        """Password shorter than 8 chars fails."""
        is_valid, msg = validate_password_complexity("Ab1!xyz")
        assert is_valid is False
        assert "8" in msg

    def test_too_long(self):
        """Password longer than 64 chars fails."""
        is_valid, msg = validate_password_complexity("A" * 65)
        assert is_valid is False
        assert "64" in msg

    def test_only_two_categories(self):
        """Password with only two categories fails."""
        is_valid, msg = validate_password_complexity("abcdefgh")
        assert is_valid is False
        assert "三类" in msg

    def test_only_lowercase_digits(self):
        """Password with only lowercase and digits fails."""
        is_valid, msg = validate_password_complexity("abcdef12")
        assert is_valid is False

    def test_exact_min_length(self):
        """Password with exactly 8 chars and 3 categories passes."""
        is_valid, msg = validate_password_complexity("Abcdef1!")
        assert is_valid is True

    def test_exact_max_length(self):
        """Password with exactly 64 chars and 3 categories passes."""
        password = "A" * 31 + "a" * 31 + "1!"
        is_valid, msg = validate_password_complexity(password)
        assert is_valid is True


# --- Password Hashing Tests ---


class TestPasswordHashing:
    """Tests for bcrypt password hashing."""

    def test_hash_and_verify(self):
        """Hashed password can be verified."""
        password = "SecurePass123!"
        hashed = hash_password(password)
        assert verify_password(password, hashed) is True

    def test_wrong_password_fails(self):
        """Wrong password does not verify."""
        hashed = hash_password("CorrectPass1!")
        assert verify_password("WrongPass1!", hashed) is False

    def test_hash_is_different_each_time(self):
        """bcrypt produces different hashes for the same password (salt)."""
        password = "SamePass123!"
        hash1 = hash_password(password)
        hash2 = hash_password(password)
        assert hash1 != hash2
        # But both verify
        assert verify_password(password, hash1) is True
        assert verify_password(password, hash2) is True


# --- JWT Token Tests ---


class TestJWTTokens:
    """Tests for JWT token creation and decoding."""

    def test_create_and_decode_access_token(self):
        """Access token can be created and decoded."""
        user_id = str(uuid.uuid4())
        token = create_access_token(subject=user_id)
        payload = decode_token(token)
        assert payload is not None
        assert payload["sub"] == user_id
        assert payload["type"] == "access"

    def test_create_and_decode_refresh_token(self):
        """Refresh token can be created and decoded."""
        user_id = str(uuid.uuid4())
        token = create_refresh_token(subject=user_id)
        payload = decode_token(token)
        assert payload is not None
        assert payload["sub"] == user_id
        assert payload["type"] == "refresh"

    def test_access_token_has_correct_expiry(self):
        """Access token expires in ~30 minutes."""
        user_id = str(uuid.uuid4())
        token = create_access_token(subject=user_id)
        payload = decode_token(token)
        exp = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
        now = datetime.now(timezone.utc)
        diff = exp - now
        # Should be close to 30 minutes (allow 5 seconds tolerance)
        assert timedelta(minutes=29, seconds=55) < diff < timedelta(minutes=30, seconds=5)

    def test_refresh_token_has_correct_expiry(self):
        """Refresh token expires in ~7 days."""
        user_id = str(uuid.uuid4())
        token = create_refresh_token(subject=user_id)
        payload = decode_token(token)
        exp = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
        now = datetime.now(timezone.utc)
        diff = exp - now
        assert timedelta(days=6, hours=23, minutes=59) < diff < timedelta(days=7, seconds=5)

    def test_invalid_token_returns_none(self):
        """Invalid token returns None."""
        payload = decode_token("invalid.token.here")
        assert payload is None

    def test_extra_claims_in_access_token(self):
        """Extra claims are included in access token."""
        user_id = str(uuid.uuid4())
        token = create_access_token(subject=user_id, extra_claims={"email": "test@example.com"})
        payload = decode_token(token)
        assert payload["email"] == "test@example.com"


# --- Auth Service Tests ---


@pytest.fixture
def mock_db():
    """Create a mock async database session."""
    db = AsyncMock()
    db.add = MagicMock()  # add() is synchronous on SQLAlchemy session
    return db


@pytest.fixture
def mock_redis():
    """Create a mock async Redis client."""
    redis = AsyncMock()
    redis.hgetall = AsyncMock(return_value={})
    redis.hset = AsyncMock()
    redis.expire = AsyncMock()
    redis.delete = AsyncMock()
    return redis


@pytest.fixture
def auth_service(mock_db, mock_redis):
    """Create an AuthService instance with mocked dependencies."""
    return AuthService(db=mock_db, redis=mock_redis)


class TestAuthServiceRegister:
    """Tests for user registration."""

    @pytest.mark.asyncio
    async def test_register_success(self, auth_service, mock_db):
        """Successful registration creates a user."""
        # Mock: no existing user
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)
        mock_db.flush = AsyncMock()

        user = await auth_service.register(
            email="new@example.com",
            password="ValidPass1!",
            display_name="New User",
        )

        assert user.email == "new@example.com"
        assert user.display_name == "New User"
        assert user.password_hash is not None
        mock_db.add.assert_called_once()
        mock_db.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_duplicate_email(self, auth_service, mock_db):
        """Registration with existing email raises ConflictException."""
        existing_user = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing_user
        mock_db.execute = AsyncMock(return_value=mock_result)

        with pytest.raises(ConflictException, match="已被注册"):
            await auth_service.register(
                email="existing@example.com",
                password="ValidPass1!",
                display_name="User",
            )

    @pytest.mark.asyncio
    async def test_register_weak_password(self, auth_service):
        """Registration with weak password raises ValidationException."""
        with pytest.raises(ValidationException):
            await auth_service.register(
                email="user@example.com",
                password="weak",
                display_name="User",
            )


class TestAuthServiceLogin:
    """Tests for user login."""

    @pytest.mark.asyncio
    async def test_login_success(self, auth_service, mock_db, mock_redis):
        """Successful login returns token pair."""
        user_id = uuid.uuid4()
        user = MagicMock()
        user.id = user_id
        user.email = "user@example.com"
        user.password_hash = hash_password("ValidPass1!")

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = user
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await auth_service.login("user@example.com", "ValidPass1!")

        assert "access_token" in result
        assert "refresh_token" in result
        assert result["token_type"] == "bearer"

    @pytest.mark.asyncio
    async def test_login_wrong_password(self, auth_service, mock_db, mock_redis):
        """Login with wrong password raises UnauthorizedException."""
        user = MagicMock()
        user.id = uuid.uuid4()
        user.password_hash = hash_password("CorrectPass1!")

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = user
        mock_db.execute = AsyncMock(return_value=mock_result)

        with pytest.raises(UnauthorizedException, match="邮箱或密码错误"):
            await auth_service.login("user@example.com", "WrongPass1!")

    @pytest.mark.asyncio
    async def test_login_nonexistent_user(self, auth_service, mock_db, mock_redis):
        """Login with non-existent email raises UnauthorizedException."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        with pytest.raises(UnauthorizedException, match="邮箱或密码错误"):
            await auth_service.login("nobody@example.com", "SomePass1!")


class TestAuthServiceRefresh:
    """Tests for token refresh."""

    @pytest.mark.asyncio
    async def test_refresh_success(self, auth_service, mock_db):
        """Valid refresh token returns new token pair."""
        user_id = uuid.uuid4()
        refresh = create_refresh_token(subject=str(user_id))

        user = MagicMock()
        user.id = user_id

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = user
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await auth_service.refresh_token(refresh)
        assert "access_token" in result
        assert "refresh_token" in result

    @pytest.mark.asyncio
    async def test_refresh_invalid_token(self, auth_service):
        """Invalid refresh token raises UnauthorizedException."""
        with pytest.raises(UnauthorizedException):
            await auth_service.refresh_token("invalid.token")

    @pytest.mark.asyncio
    async def test_refresh_with_access_token_fails(self, auth_service):
        """Using an access token for refresh raises UnauthorizedException."""
        user_id = str(uuid.uuid4())
        access = create_access_token(subject=user_id)

        with pytest.raises(UnauthorizedException, match="Token 类型"):
            await auth_service.refresh_token(access)


class TestAuthServiceLockout:
    """Tests for login lockout mechanism."""

    @pytest.mark.asyncio
    async def test_lockout_after_5_failures(self, auth_service, mock_db, mock_redis):
        """Account is locked after 5 failed attempts."""
        # Simulate 4 previous failures
        mock_redis.hgetall = AsyncMock(return_value={"attempts": "4"})

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        with pytest.raises(UnauthorizedException):
            await auth_service.login("user@example.com", "WrongPass1!")

        # Verify lockout was set (hset called with locked_until)
        mock_redis.hset.assert_called()
        call_kwargs = mock_redis.hset.call_args
        mapping = call_kwargs.kwargs.get("mapping") or call_kwargs[1].get("mapping")
        assert "locked_until" in mapping

    @pytest.mark.asyncio
    async def test_locked_account_rejects_login(self, auth_service, mock_redis):
        """Locked account rejects login attempts."""
        locked_until = (
            datetime.now(timezone.utc) + timedelta(minutes=10)
        ).isoformat()
        mock_redis.hgetall = AsyncMock(
            return_value={"attempts": "5", "locked_until": locked_until}
        )

        with pytest.raises(UnauthorizedException, match="锁定"):
            await auth_service.login("user@example.com", "AnyPass1!")

    @pytest.mark.asyncio
    async def test_expired_lock_allows_login(self, auth_service, mock_db, mock_redis):
        """Expired lock allows login attempt."""
        locked_until = (
            datetime.now(timezone.utc) - timedelta(minutes=1)
        ).isoformat()
        mock_redis.hgetall = AsyncMock(
            return_value={"attempts": "5", "locked_until": locked_until}
        )

        user_id = uuid.uuid4()
        user = MagicMock()
        user.id = user_id
        user.password_hash = hash_password("ValidPass1!")

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = user
        mock_db.execute = AsyncMock(return_value=mock_result)

        # After expired lock, hgetall returns expired data, then delete is called
        # On second call (after delete), return empty
        call_count = [0]
        original_hgetall = mock_redis.hgetall

        async def side_effect(key):
            call_count[0] += 1
            if call_count[0] == 1:
                return {"attempts": "5", "locked_until": locked_until}
            return {}

        mock_redis.hgetall = AsyncMock(side_effect=side_effect)

        result = await auth_service.login("user@example.com", "ValidPass1!")
        assert "access_token" in result


class TestAuthServiceOIDC:
    """Tests for OIDC user creation/binding."""

    @pytest.mark.asyncio
    async def test_create_new_oidc_user(self, auth_service, mock_db):
        """New OIDC user is created when no match exists."""
        # No existing user by OIDC identity
        mock_result_oidc = MagicMock()
        mock_result_oidc.scalar_one_or_none.return_value = None

        # No existing user by email
        mock_result_email = MagicMock()
        mock_result_email.scalar_one_or_none.return_value = None

        mock_db.execute = AsyncMock(
            side_effect=[mock_result_oidc, mock_result_email]
        )
        mock_db.flush = AsyncMock()

        user = await auth_service.get_or_create_oidc_user(
            provider="keycloak.example.com",
            subject="oidc-sub-123",
            email="oidc@example.com",
            display_name="OIDC User",
        )

        assert user.email == "oidc@example.com"
        assert user.oidc_provider == "keycloak.example.com"
        assert user.oidc_subject == "oidc-sub-123"
        mock_db.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_bind_oidc_to_existing_email(self, auth_service, mock_db):
        """OIDC identity is bound to existing user with same email."""
        # No existing user by OIDC identity
        mock_result_oidc = MagicMock()
        mock_result_oidc.scalar_one_or_none.return_value = None

        # Existing user by email
        existing_user = MagicMock()
        existing_user.email = "user@example.com"
        existing_user.oidc_provider = None
        existing_user.oidc_subject = None

        mock_result_email = MagicMock()
        mock_result_email.scalar_one_or_none.return_value = existing_user

        mock_db.execute = AsyncMock(
            side_effect=[mock_result_oidc, mock_result_email]
        )
        mock_db.flush = AsyncMock()

        user = await auth_service.get_or_create_oidc_user(
            provider="okta.example.com",
            subject="okta-sub-456",
            email="user@example.com",
            display_name="Existing User",
        )

        assert user.oidc_provider == "okta.example.com"
        assert user.oidc_subject == "okta-sub-456"
        mock_db.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_return_existing_oidc_user(self, auth_service, mock_db):
        """Existing OIDC user is returned directly."""
        existing_user = MagicMock()
        existing_user.oidc_provider = "keycloak.example.com"
        existing_user.oidc_subject = "oidc-sub-123"

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing_user
        mock_db.execute = AsyncMock(return_value=mock_result)

        user = await auth_service.get_or_create_oidc_user(
            provider="keycloak.example.com",
            subject="oidc-sub-123",
            email="oidc@example.com",
            display_name="OIDC User",
        )

        assert user is existing_user
