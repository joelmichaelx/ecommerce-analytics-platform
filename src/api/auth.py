"""
Authentication and Authorization
JWT-based authentication for E-commerce Analytics API
"""

import jwt
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
from passlib.context import CryptContext
from fastapi import HTTPException, status
import logging

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT configuration
SECRET_KEY = "your-secret-key-here"  # In production, use environment variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# User database (in production, use a real database)
USERS_DB = {
    "admin": {
        "user_id": "1",
        "username": "admin",
        "email": "admin@company.com",
        "hashed_password": pwd_context.hash("admin123"),
        "role": "admin",
        "permissions": ["read", "write", "admin"]
    },
    "analyst": {
        "user_id": "2",
        "username": "analyst",
        "email": "analyst@company.com",
        "hashed_password": pwd_context.hash("analyst123"),
        "role": "analyst",
        "permissions": ["read", "write"]
    },
    "viewer": {
        "user_id": "3",
        "username": "viewer",
        "email": "viewer@company.com",
        "hashed_password": pwd_context.hash("viewer123"),
        "role": "viewer",
        "permissions": ["read"]
    }
}

class AuthenticationService:
    """Authentication service for user management and JWT tokens"""
    
    def __init__(self, secret_key: str = SECRET_KEY, algorithm: str = ALGORITHM):
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash a password"""
        return pwd_context.hash(password)
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate a user with username and password"""
        try:
            user = USERS_DB.get(username)
            if not user:
                return None
            
            if not self.verify_password(password, user["hashed_password"]):
                return None
            
            return {
                "user_id": user["user_id"],
                "username": user["username"],
                "email": user["email"],
                "role": user["role"],
                "permissions": user["permissions"]
            }
            
        except Exception as e:
            logger.error(f"Authentication failed for user {username}: {str(e)}")
            return None
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token"""
        try:
            to_encode = data.copy()
            if expires_delta:
                expire = datetime.utcnow() + expires_delta
            else:
                expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            
            to_encode.update({"exp": expire})
            encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
            return encoded_jwt
            
        except Exception as e:
            logger.error(f"Failed to create access token: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Token creation failed"
            )
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify a JWT token and return user data"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            username: str = payload.get("sub")
            if username is None:
                return None
            
            user = USERS_DB.get(username)
            if user is None:
                return None
            
            return {
                "user_id": user["user_id"],
                "username": user["username"],
                "email": user["email"],
                "role": user["role"],
                "permissions": user["permissions"]
            }
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.JWTError as e:
            logger.warning(f"Invalid token: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Token verification failed: {str(e)}")
            return None
    
    def refresh_token(self, token: str) -> Optional[str]:
        """Refresh an access token"""
        try:
            user_data = self.verify_token(token)
            if not user_data:
                return None
            
            return self.create_access_token(
                data={"sub": user_data["username"]},
                expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            )
            
        except Exception as e:
            logger.error(f"Token refresh failed: {str(e)}")
            return None
    
    def revoke_token(self, token: str) -> bool:
        """Revoke a token (in production, maintain a blacklist)"""
        try:
            # In production, add token to blacklist
            # For now, just return True
            return True
            
        except Exception as e:
            logger.error(f"Token revocation failed: {str(e)}")
            return False

class AuthorizationService:
    """Authorization service for permission checking"""
    
    def __init__(self):
        self.permission_hierarchy = {
            "admin": ["read", "write", "delete", "admin"],
            "analyst": ["read", "write"],
            "viewer": ["read"]
        }
    
    def check_permission(self, user_permissions: list, required_permission: str) -> bool:
        """Check if user has required permission"""
        return required_permission in user_permissions
    
    def check_role_permission(self, user_role: str, required_permission: str) -> bool:
        """Check if user role has required permission"""
        role_permissions = self.permission_hierarchy.get(user_role, [])
        return required_permission in role_permissions
    
    def get_user_permissions(self, user_role: str) -> list:
        """Get permissions for a user role"""
        return self.permission_hierarchy.get(user_role, [])
    
    def is_admin(self, user_role: str) -> bool:
        """Check if user is admin"""
        return user_role == "admin"
    
    def can_access_resource(self, user_permissions: list, resource: str, action: str) -> bool:
        """Check if user can access a specific resource with specific action"""
        # Define resource-based permissions
        resource_permissions = {
            "analytics": ["read"],
            "realtime": ["read"],
            "ml_models": ["read", "write"],
            "pipeline": ["read", "write", "admin"],
            "monitoring": ["read", "write"],
            "export": ["read", "write"],
            "admin": ["admin"]
        }
        
        required_permissions = resource_permissions.get(resource, ["read"])
        
        if action == "read":
            return "read" in user_permissions
        elif action == "write":
            return "write" in user_permissions
        elif action == "admin":
            return "admin" in user_permissions
        
        return False

# Global instances
auth_service = AuthenticationService()
authz_service = AuthorizationService()

# FastAPI dependency functions
async def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """FastAPI dependency to verify JWT token"""
    return auth_service.verify_token(token)

async def get_current_user(token: str) -> Optional[Dict[str, Any]]:
    """FastAPI dependency to get current user from token"""
    return await verify_token(token)

async def require_permission(required_permission: str):
    """FastAPI dependency to require specific permission"""
    def permission_checker(current_user: Dict[str, Any] = Depends(get_current_user)):
        if not current_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )
        
        user_permissions = current_user.get("permissions", [])
        if not authz_service.check_permission(user_permissions, required_permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{required_permission}' required"
            )
        
        return current_user
    
    return permission_checker

async def require_admin(current_user: Dict[str, Any] = Depends(get_current_user)):
    """FastAPI dependency to require admin role"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    if not authz_service.is_admin(current_user.get("role", "")):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required"
        )
    
    return current_user

# Utility functions
def generate_api_key() -> str:
    """Generate a secure API key"""
    return secrets.token_urlsafe(32)

def hash_api_key(api_key: str) -> str:
    """Hash an API key for storage"""
    return hashlib.sha256(api_key.encode()).hexdigest()

def verify_api_key(api_key: str, hashed_key: str) -> bool:
    """Verify an API key against its hash"""
    return hash_api_key(api_key) == hashed_key

# Session management
class SessionManager:
    """Manage user sessions"""
    
    def __init__(self):
        self.active_sessions = {}
    
    def create_session(self, user_id: str, token: str) -> str:
        """Create a new session"""
        session_id = secrets.token_urlsafe(16)
        self.active_sessions[session_id] = {
            "user_id": user_id,
            "token": token,
            "created_at": datetime.utcnow(),
            "last_activity": datetime.utcnow()
        }
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data"""
        return self.active_sessions.get(session_id)
    
    def update_session_activity(self, session_id: str):
        """Update session last activity"""
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["last_activity"] = datetime.utcnow()
    
    def destroy_session(self, session_id: str) -> bool:
        """Destroy a session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            return True
        return False
    
    def cleanup_expired_sessions(self, max_age_hours: int = 24):
        """Clean up expired sessions"""
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        expired_sessions = [
            session_id for session_id, session_data in self.active_sessions.items()
            if session_data["last_activity"] < cutoff_time
        ]
        
        for session_id in expired_sessions:
            del self.active_sessions[session_id]
        
        return len(expired_sessions)

# Global session manager
session_manager = SessionManager()

# Rate limiting
class RateLimiter:
    """Simple rate limiter for API endpoints"""
    
    def __init__(self):
        self.requests = {}
    
    def is_allowed(self, user_id: str, endpoint: str, max_requests: int = 100, window_minutes: int = 60) -> bool:
        """Check if request is allowed based on rate limits"""
        key = f"{user_id}:{endpoint}"
        now = datetime.utcnow()
        window_start = now - timedelta(minutes=window_minutes)
        
        if key not in self.requests:
            self.requests[key] = []
        
        # Clean old requests
        self.requests[key] = [
            req_time for req_time in self.requests[key]
            if req_time > window_start
        ]
        
        # Check if under limit
        if len(self.requests[key]) < max_requests:
            self.requests[key].append(now)
            return True
        
        return False
    
    def get_remaining_requests(self, user_id: str, endpoint: str, max_requests: int = 100, window_minutes: int = 60) -> int:
        """Get remaining requests for user/endpoint"""
        key = f"{user_id}:{endpoint}"
        now = datetime.utcnow()
        window_start = now - timedelta(minutes=window_minutes)
        
        if key not in self.requests:
            return max_requests
        
        # Clean old requests
        self.requests[key] = [
            req_time for req_time in self.requests[key]
            if req_time > window_start
        ]
        
        return max(0, max_requests - len(self.requests[key]))

# Global rate limiter
rate_limiter = RateLimiter()

# Example usage
if __name__ == "__main__":
    # Test authentication
    auth = AuthenticationService()
    
    # Test user authentication
    user = auth.authenticate_user("admin", "admin123")
    if user:
        print("Authentication successful:", user)
        
        # Create token
        token = auth.create_access_token(data={"sub": user["username"]})
        print("Access token created:", token)
        
        # Verify token
        verified_user = auth.verify_token(token)
        print("Token verification:", verified_user)
    else:
        print("Authentication failed")
