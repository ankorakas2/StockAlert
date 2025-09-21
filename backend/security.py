# security.py - Complete Security Configuration for Production
import os
import secrets
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from cryptography.fernet import Fernet
import redis
import logging
from functools import wraps
import ipaddress
from sqlalchemy import create_engine, Column, String, DateTime, Integer, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import re

logger = logging.getLogger(__name__)

# Security Configuration
class SecurityConfig:
    # JWT Configuration
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
    JWT_ALGORITHM = "HS256"
    JWT_EXPIRATION_HOURS = 24
    JWT_REFRESH_EXPIRATION_DAYS = 30
    
    # Encryption
    ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", Fernet.generate_key())
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS = 100
    RATE_LIMIT_WINDOW = 60  # seconds
    
    # IP Whitelist/Blacklist
    ALLOWED_IPS = os.getenv("ALLOWED_IPS", "").split(",") if os.getenv("ALLOWED_IPS") else []
    BLOCKED_IPS = os.getenv("BLOCKED_IPS", "").split(",") if os.getenv("BLOCKED_IPS") else []
    
    # Security Headers
    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
    }
    
    # Password Policy
    PASSWORD_MIN_LENGTH = 12
    PASSWORD_REQUIRE_UPPERCASE = True
    PASSWORD_REQUIRE_LOWERCASE = True
    PASSWORD_REQUIRE_DIGITS = True
    PASSWORD_REQUIRE_SPECIAL = True
    PASSWORD_SPECIAL_CHARS = "!@#$%^&*()_+-=[]{}|;:,.<>?"
    
    # Session Configuration
    SESSION_TIMEOUT_MINUTES = 30
    MAX_SESSIONS_PER_USER = 5
    
    # API Key Configuration
    API_KEY_LENGTH = 32
    API_KEY_PREFIX = "sk_"
    
    # Database Encryption
    ENCRYPT_PII = True
    ENCRYPT_FINANCIAL_DATA = True

# Database Models for Security
Base = declarative_base()

class UserSession(Base):
    __tablename__ = "user_sessions"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, index=True)
    token_hash = Column(String)
    ip_address = Column(String)
    user_agent = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    is_active = Column(Boolean, default=True)

class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String)
    action = Column(String)
    resource = Column(String)
    ip_address = Column(String)
    user_agent = Column(String)
    request_data = Column(String)
    response_code = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)

class SecurityIncident(Base):
    __tablename__ = "security_incidents"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    incident_type = Column(String)
    severity = Column(String)
    user_id = Column(String, nullable=True)
    ip_address = Column(String)
    description = Column(String)
    metadata = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    resolved = Column(Boolean, default=False)

# Security Services
class PasswordValidator:
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
    def validate_password_strength(self, password: str) -> tuple[bool, str]:
        """Validate password against security policy"""
        if len(password) < SecurityConfig.PASSWORD_MIN_LENGTH:
            return False, f"Password must be at least {SecurityConfig.PASSWORD_MIN_LENGTH} characters long"
        
        if SecurityConfig.PASSWORD_REQUIRE_UPPERCASE and not re.search(r"[A-Z]", password):
            return False, "Password must contain at least one uppercase letter"
        
        if SecurityConfig.PASSWORD_REQUIRE_LOWERCASE and not re.search(r"[a-z]", password):
            return False, "Password must contain at least one lowercase letter"
        
        if SecurityConfig.PASSWORD_REQUIRE_DIGITS and not re.search(r"\d", password):
            return False, "Password must contain at least one digit"
        
        if SecurityConfig.PASSWORD_REQUIRE_SPECIAL:
            if not any(char in SecurityConfig.PASSWORD_SPECIAL_CHARS for char in password):
                return False, "Password must contain at least one special character"
        
        # Check for common passwords
        common_passwords = ["password", "123456", "password123", "admin", "letmein"]
        if password.lower() in common_passwords:
            return False, "Password is too common"
        
        return True, "Password is strong"
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return self.pwd_context.verify(plain_password, hashed_password)

class TokenManager:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=0,
            decode_responses=True
        ) if os.getenv("REDIS_ENABLED", "false").lower() == "true" else None
        
    def create_access_token(self, user_id: str, additional_claims: Dict = None) -> str:
        """Create JWT access token"""
        expire = datetime.utcnow() + timedelta(hours=SecurityConfig.JWT_EXPIRATION_HOURS)
        
        payload = {
            "sub": user_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access",
            "jti": secrets.token_urlsafe(16)
        }
        
        if additional_claims:
            payload.update(additional_claims)
        
        token = jwt.encode(payload, SecurityConfig.JWT_SECRET_KEY, algorithm=SecurityConfig.JWT_ALGORITHM)
        
        # Store in Redis for blacklisting capability
        if self.redis_client:
            self.redis_client.setex(
                f"token:{payload['jti']}",
                int(SecurityConfig.JWT_EXPIRATION_HOURS * 3600),
                user_id
            )
        
        return token
    
    def create_refresh_token(self, user_id: str) -> str:
        """Create JWT refresh token"""
        expire = datetime.utcnow() + timedelta(days=SecurityConfig.JWT_REFRESH_EXPIRATION_DAYS)
        
        payload = {
            "sub": user_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh",
            "jti": secrets.token_urlsafe(16)
        }
        
        token = jwt.encode(payload, SecurityConfig.JWT_SECRET_KEY, algorithm=SecurityConfig.JWT_ALGORITHM)
        
        if self.redis_client:
            self.redis_client.setex(
                f"refresh:{payload['jti']}",
                int(SecurityConfig.JWT_REFRESH_EXPIRATION_DAYS * 86400),
                user_id
            )
        
        return token
    
    def verify_token(self, token: str) -> Dict:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(
                token,
                SecurityConfig.JWT_SECRET_KEY,
                algorithms=[SecurityConfig.JWT_ALGORITHM]
            )
            
            # Check if token is blacklisted
            if self.redis_client:
                jti = payload.get("jti")
                if jti and self.redis_client.get(f"blacklist:{jti}"):
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Token has been revoked"
                    )
            
            return payload
            
        except JWTError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {str(e)}"
            )
    
    def revoke_token(self, token: str):
        """Revoke a token by adding to blacklist"""
        try:
            payload = jwt.decode(
                token,
                SecurityConfig.JWT_SECRET_KEY,
                algorithms=[SecurityConfig.JWT_ALGORITHM],
                options={"verify_exp": False}
            )
            
            jti = payload.get("jti")
            if jti and self.redis_client:
                # Calculate remaining TTL
                exp = payload.get("exp")
                if exp:
                    ttl = exp - datetime.utcnow().timestamp()
                    if ttl > 0:
                        self.redis_client.setex(f"blacklist:{jti}", int(ttl), "1")
                        
        except JWTError:
            pass  # Token is already invalid

class DataEncryption:
    def __init__(self):
        self.cipher = Fernet(SecurityConfig.ENCRYPTION_KEY)
        
    def encrypt(self, data: str) -> str:
        """Encrypt sensitive data"""
        if not data:
            return data
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        if not encrypted_data:
            return encrypted_data
        return self.cipher.decrypt(encrypted_data.encode()).decode()
    
    def encrypt_dict(self, data: Dict, fields_to_encrypt: List[str]) -> Dict:
        """Encrypt specific fields in a dictionary"""
        encrypted_data = data.copy()
        for field in fields_to_encrypt:
            if field in encrypted_data and encrypted_data[field]:
                encrypted_data[field] = self.encrypt(str(encrypted_data[field]))
        return encrypted_data
    
    def decrypt_dict(self, data: Dict, fields_to_decrypt: List[str]) -> Dict:
        """Decrypt specific fields in a dictionary"""
        decrypted_data = data.copy()
        for field in fields_to_decrypt:
            if field in decrypted_data and decrypted_data[field]:
                decrypted_data[field] = self.decrypt(decrypted_data[field])
        return decrypted_data

class IPFilter:
    def __init__(self):
        self.whitelist = [ipaddress.ip_network(ip) for ip in SecurityConfig.ALLOWED_IPS if ip]
        self.blacklist = [ipaddress.ip_network(ip) for ip in SecurityConfig.BLOCKED_IPS if ip]
        
    def is_ip_allowed(self, ip: str) -> bool:
        """Check if IP is allowed"""
        try:
            ip_addr = ipaddress.ip_address(ip)
            
            # Check blacklist first
            for network in self.blacklist:
                if ip_addr in network:
                    return False
            
            # If whitelist is configured, IP must be in whitelist
            if self.whitelist:
                for network in self.whitelist:
                    if ip_addr in network:
                        return True
                return False
            
            # If no whitelist, allow all non-blacklisted IPs
            return True
            
        except ValueError:
            return False  # Invalid IP

class SecurityAuditor:
    def __init__(self, db_session):
        self.db_session = db_session
        
    def log_access(self, user_id: str, action: str, resource: str, 
                   ip_address: str, user_agent: str, request_data: Dict,
                   response_code: int):
        """Log API access for audit trail"""
        audit_log = AuditLog(
            user_id=user_id,
            action=action,
            resource=resource,
            ip_address=ip_address,
            user_agent=user_agent,
            request_data=json.dumps(request_data) if request_data else None,
            response_code=response_code,
            timestamp=datetime.utcnow()
        )
        self.db_session.add(audit_log)
        self.db_session.commit()
    
    def log_security_incident(self, incident_type: str, severity: str,
                            ip_address: str, description: str,
                            user_id: Optional[str] = None,
                            metadata: Optional[Dict] = None):
        """Log security incident"""
        incident = SecurityIncident(
            incident_type=incident_type,
            severity=severity,
            user_id=user_id,
            ip_address=ip_address,
            description=description,
            metadata=json.dumps(metadata) if metadata else None,
            timestamp=datetime.utcnow()
        )
        self.db_session.add(incident)
        self.db_session.commit()
        
        # Send alert for high severity incidents
        if severity in ["HIGH", "CRITICAL"]:
            self._send_security_alert(incident)
    
    def _send_security_alert(self, incident: SecurityIncident):
        """Send security alert to administrators"""
        # Implement email/SMS/Slack notification
        logger.critical(f"Security Incident: {incident.incident_type} - {incident.description}")

class APIKeyManager:
    def __init__(self, db_session):
        self.db_session = db_session
        
    def generate_api_key(self, user_id: str, name: str = "default") -> str:
        """Generate new API key for user"""
        key = SecurityConfig.API_KEY_PREFIX + secrets.token_urlsafe(SecurityConfig.API_KEY_LENGTH)
        
        # Hash the key for storage
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        # Store in database (implement APIKey model)
        # api_key = APIKey(
        #     user_id=user_id,
        #     name=name,
        #     key_hash=key_hash,
        #     created_at=datetime.utcnow()
        # )
        # self.db_session.add(api_key)
        # self.db_session.commit()
        
        return key
    
    def verify_api_key(self, api_key: str) -> Optional[str]:
        """Verify API key and return user_id"""
        if not api_key.startswith(SecurityConfig.API_KEY_PREFIX):
            return None
        
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Query database for key_hash
        # api_key_record = self.db_session.query(APIKey).filter_by(
        #     key_hash=key_hash,
        #     is_active=True
        # ).first()
        
        # if api_key_record:
        #     return api_key_record.user_id
        
        return None

# Middleware
async def security_headers_middleware(request: Request, call_next):
    """Add security headers to response"""
    response = await call_next(request)
    for header, value in SecurityConfig.SECURITY_HEADERS.items():
        response.headers[header] = value
    return response

async def ip_filter_middleware(request: Request, call_next):
    """Filter requests by IP address"""
    client_ip = request.client.host
    ip_filter = IPFilter()
    
    if not ip_filter.is_ip_allowed(client_ip):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied from this IP address"
        )
    
    response = await call_next(request)
    return response

# Decorators
def require_permissions(*permissions):
    """Decorator to check user permissions"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Check permissions logic
            user = kwargs.get('current_user')
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            user_permissions = user.get('permissions', [])
            if not all(perm in user_permissions for perm in permissions):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Environment Configuration
# .env.production
"""
# Security
ENVIRONMENT=production
JWT_SECRET_KEY=your-secret-key-here
ENCRYPTION_KEY=your-encryption-key-here
API_KEY=your-api-key-here

# SSL/TLS
SSL_CERT_PATH=/path/to/cert.pem
SSL_KEY_PATH=/path/to/key.pem
USE_HTTPS=true

# Database
DATABASE_URL=postgresql://user:password@localhost/stockalert
DATABASE_ENCRYPT_PII=true

# Redis
REDIS_ENABLED=true
REDIS_HOST=localhost
REDIS_PORT=6379

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# CORS
CORS_ORIGINS=https://app.stockalertpro.com,https://stockalertpro.com
ALLOWED_HOSTS=app.stockalertpro.com,api.stockalertpro.com

# Firebase
FIREBASE_CONFIG={"type":"service_account",...}

# News APIs
NEWSAPI_KEY=your-newsapi-key
ALPHAVANTAGE_KEY=your-alphavantage-key

# Monitoring
SENTRY_DSN=your-sentry-dsn
DATADOG_API_KEY=your-datadog-key

# Email
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=alerts@stockalertpro.com
SMTP_PASSWORD=your-smtp-password

# IP Filtering (optional)
ALLOWED_IPS=
BLOCKED_IPS=

# Session
SESSION_SECRET=your-session-secret
SESSION_TIMEOUT_MINUTES=30

# Feature Flags
ENABLE_PORTFOLIO=true
ENABLE_TECHNICAL_ANALYSIS=true
ENABLE_BACKTESTING=true
ENABLE_PAPER_TRADING=false
"""

# Docker Configuration for Production
"""
# Dockerfile.production
FROM python:3.11-slim

# Security: Run as non-root user
RUN useradd -m -u 1000 stockalert && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        postgresql-client \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=stockalert:stockalert . .

# Security: Don't run as root
USER stockalert

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Use gunicorn for production
CMD ["gunicorn", "main:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "4", \
     "--bind", "0.0.0.0:8000", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--log-level", "info"]
"""

# Nginx Configuration for HTTPS
"""
# nginx.conf
server {
    listen 80;
    server_name api.stockalertpro.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.stockalertpro.com;
    
    # SSL Configuration
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    
    # Security Headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Content-Security-Policy "default-src 'self'" always;
    
    # Rate Limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;
    
    # Proxy to backend
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # Buffering
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
        proxy_busy_buffers_size 8k;
    }
    
    # Block sensitive paths
    location ~ /\. {
        deny all;
    }
    
    location ~ /\.git {
        deny all;
    }
}
"""

# Monitoring and Logging Configuration
class MonitoringConfig:
    """
    # prometheus_config.yml
    global:
      scrape_interval: 15s
      evaluation_interval: 15s
    
    scrape_configs:
      - job_name: 'stockalert-api'
        static_configs:
          - targets: ['localhost:8000']
        metrics_path: '/metrics'
    
    # grafana_dashboard.json
    {
      "dashboard": {
        "title": "StockAlert Pro Monitoring",
        "panels": [
          {
            "title": "API Request Rate",
            "targets": [
              {
                "expr": "rate(http_requests_total[5m])"
              }
            ]
          },
          {
            "title": "Response Time",
            "targets": [
              {
                "expr": "histogram_quantile(0.95, http_request_duration_seconds_bucket)"
              }
            ]
          },
          {
            "title": "Error Rate",
            "targets": [
              {
                "expr": "rate(http_requests_total{status=~'5..'}[5m])"
              }
            ]
          },
          {
            "title": "Active Users",
            "targets": [
              {
                "expr": "active_users_total"
              }
            ]
          }
        ]
      }
    }
    """
    pass

# Database Migration Script
"""
-- migrations/001_security_tables.sql
CREATE TABLE IF NOT EXISTS user_sessions (
    id VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    token_hash VARCHAR(255) NOT NULL,
    ip_address VARCHAR(45),
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    INDEX idx_user_id (user_id),
    INDEX idx_expires_at (expires_at)
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(255),
    action VARCHAR(100),
    resource VARCHAR(255),
    ip_address VARCHAR(45),
    user_agent TEXT,
    request_data JSON,
    response_code INT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user_id (user_id),
    INDEX idx_timestamp (timestamp)
);

CREATE TABLE IF NOT EXISTS security_incidents (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    incident_type VARCHAR(100),
    severity VARCHAR(20),
    user_id VARCHAR(255),
    ip_address VARCHAR(45),
    description TEXT,
    metadata JSON,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved BOOLEAN DEFAULT FALSE,
    INDEX idx_severity (severity),
    INDEX idx_timestamp (timestamp)
);

CREATE TABLE IF NOT EXISTS api_keys (
    id VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    name VARCHAR(100),
    key_hash VARCHAR(255) UNIQUE NOT NULL,
    permissions JSON,
    rate_limit INT DEFAULT 1000,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMP,
    expires_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    INDEX idx_user_id (user_id),
    INDEX idx_key_hash (key_hash)
);
"""