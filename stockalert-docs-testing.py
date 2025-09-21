# api_documentation.py - OpenAPI Documentation & Swagger UI
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.staticfiles import StaticFiles

def custom_openapi(app: FastAPI):
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="StockAlert Pro API",
        version="2.0.0",
        description="""
        ## 🚀 StockAlert Pro API Documentation
        
        Professional stock monitoring and analysis platform with real-time alerts, 
        portfolio tracking, and AI-powered trading signals.
        
        ### Features
        - 📊 **Portfolio Management**: Track your investments with real-time P&L
        - 📈 **Technical Analysis**: RSI, MACD, Bollinger Bands, and more
        - 🤖 **AI Trading Signals**: ML-powered buy/sell recommendations
        - 📰 **News Sentiment**: Analyze market sentiment from trusted sources
        - ⚡ **Real-time Alerts**: Push notifications for price movements
        - 🔄 **Backtesting**: Test strategies on historical data
        
        ### Authentication
        All protected endpoints require a valid JWT token in the Authorization header:
        ```
        Authorization: Bearer <your-jwt-token>
        ```
        
        ### Rate Limiting
        - Public endpoints: 120 requests/minute
        - Authenticated endpoints: 300 requests/minute
        - Technical analysis: 60 requests/minute
        - Backtesting: 10 requests/minute
        
        ### Response Codes
        - `200`: Success
        - `201`: Created
        - `400`: Bad Request
        - `401`: Unauthorized
        - `403`: Forbidden
        - `404`: Not Found
        - `429`: Too Many Requests
        - `500`: Internal Server Error
        
        ### Support
        - Email: api-support@stockalertpro.com
        - Documentation: https://docs.stockalertpro.com
        - Status Page: https://status.stockalertpro.com
        """,
        routes=app.routes,
        tags=[
            {
                "name": "Authentication",
                "description": "User authentication and authorization endpoints"
            },
            {
                "name": "Portfolio",
                "description": "Portfolio management and tracking"
            },
            {
                "name": "Stocks",
                "description": "Stock prices and information"
            },
            {
                "name": "Technical Analysis",
                "description": "Technical indicators and signals"
            },
            {
                "name": "News",
                "description": "Market news and sentiment analysis"
            },
            {
                "name": "Backtesting",
                "description": "Strategy backtesting and performance"
            },
            {
                "name": "Alerts",
                "description": "Price and signal alerts management"
            },
            {
                "name": "Watchlist",
                "description": "Personal watchlist management"
            }
        ],
        servers=[
            {"url": "https://api.stockalertpro.com", "description": "Production"},
            {"url": "https://staging-api.stockalertpro.com", "description": "Staging"},
            {"url": "http://localhost:8000", "description": "Development"}
        ],
        components={
            "securitySchemes": {
                "Bearer": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "JWT",
                    "description": "Enter your JWT token"
                },
                "ApiKey": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key",
                    "description": "Enter your API key"
                }
            }
        }
    )
    
    # Add examples for each endpoint
    for path in openapi_schema["paths"]:
        for method in openapi_schema["paths"][path]:
            if method in ["get", "post", "put", "delete", "patch"]:
                operation = openapi_schema["paths"][path][method]
                
                # Add request examples
                if "requestBody" in operation:
                    operation["requestBody"]["content"]["application/json"]["examples"] = {
                        "example1": {
                            "summary": "Example request",
                            "value": get_request_example(path, method)
                        }
                    }
                
                # Add response examples
                if "responses" in operation:
                    for status_code in operation["responses"]:
                        if status_code.startswith("2"):
                            operation["responses"][status_code]["content"] = {
                                "application/json": {
                                    "schema": operation["responses"][status_code].get("content", {}).get("application/json", {}).get("schema", {}),
                                    "examples": {
                                        "example1": {
                                            "summary": "Successful response",
                                            "value": get_response_example(path, method, status_code)
                                        }
                                    }
                                }
                            }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

def get_request_example(path: str, method: str) -> dict:
    """Get request example for endpoint"""
    examples = {
        ("/api/portfolio/transaction", "post"): {
            "symbol": "AAPL",
            "type": "BUY",
            "shares": 100,
            "price": 150.50,
            "commission": 0
        },
        ("/api/auth/login", "post"): {
            "email": "user@example.com",
            "password": "SecurePassword123!"
        },
        ("/api/watchlist/add", "post"): {
            "symbol": "TSLA",
            "target_price": 250.00,
            "alert_enabled": True
        }
    }
    return examples.get((path, method), {})

def get_response_example(path: str, method: str, status_code: str) -> dict:
    """Get response example for endpoint"""
    examples = {
        ("/api/portfolio", "get", "200"): {
            "positions": [
                {
                    "symbol": "AAPL",
                    "shares": 100,
                    "average_price": 145.50,
                    "current_price": 150.25,
                    "total_value": 15025.00,
                    "gain_loss": 475.00,
                    "gain_loss_percent": 3.26
                }
            ],
            "total_value": 15025.00,
            "total_gain_loss": 475.00,
            "total_gain_loss_percent": 3.26
        },
        ("/api/technical/{symbol}", "get", "200"): {
            "symbol": "AAPL",
            "rsi": 65.5,
            "macd": {
                "macd": 2.15,
                "signal": 1.98,
                "histogram": 0.17
            },
            "bollinger_bands": {
                "upper": 155.20,
                "middle": 150.00,
                "lower": 144.80
            },
            "sma_20": 149.50,
            "sma_50": 147.30,
            "sma_200": 140.20
        }
    }
    return examples.get((path, method, status_code), {})

# tests/test_backend.py - Complete Backend Testing Suite
import pytest
import asyncio
from httpx import AsyncClient
from datetime import datetime, timedelta
import json
from unittest.mock import Mock, patch, AsyncMock
from main import app
from security import TokenManager, PasswordValidator, DataEncryption

# Test Configuration
@pytest.fixture
def test_config():
    return {
        "test_user": {
            "email": "test@example.com",
            "password": "Test123!@#",
            "user_id": "test-user-123"
        },
        "test_stock": {
            "symbol": "AAPL",
            "price": 150.00
        }
    }

@pytest.fixture
async def async_client():
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

@pytest.fixture
def auth_headers(test_config):
    token_manager = TokenManager()
    token = token_manager.create_access_token(test_config["test_user"]["user_id"])
    return {"Authorization": f"Bearer {token}"}

# Authentication Tests
class TestAuthentication:
    @pytest.mark.asyncio
    async def test_user_registration(self, async_client):
        response = await async_client.post(
            "/api/auth/register",
            json={
                "email": "newuser@example.com",
                "password": "SecurePass123!",
                "confirm_password": "SecurePass123!"
            }
        )
        assert response.status_code == 201
        assert "user_id" in response.json()
        assert "access_token" in response.json()
    
    @pytest.mark.asyncio
    async def test_user_login(self, async_client, test_config):
        response = await async_client.post(
            "/api/auth/login",
            json={
                "email": test_config["test_user"]["email"],
                "password": test_config["test_user"]["password"]
            }
        )
        assert response.status_code == 200
        assert "access_token" in response.json()
        assert "refresh_token" in response.json()
    
    @pytest.mark.asyncio
    async def test_invalid_credentials(self, async_client):
        response = await async_client.post(
            "/api/auth/login",
            json={
                "email": "wrong@example.com",
                "password": "WrongPassword"
            }
        )
        assert response.status_code == 401
    
    @pytest.mark.asyncio
    async def test_token_refresh(self, async_client, test_config):
        # First login
        login_response = await async_client.post(
            "/api/auth/login",
            json={
                "email": test_config["test_user"]["email"],
                "password": test_config["test_user"]["password"]
            }
        )
        refresh_token = login_response.json()["refresh_token"]
        
        # Refresh token
        response = await async_client.post(
            "/api/auth/refresh",
            json={"refresh_token": refresh_token}
        )
        assert response.status_code == 200
        assert "access_token" in response.json()
    
    @pytest.mark.asyncio
    async def test_password_reset(self, async_client, test_config):
        response = await async_client.post(
            "/api/auth/forgot-password",
            json={"email": test_config["test_user"]["email"]}
        )
        assert response.status_code == 200
        assert response.json()["message"] == "Password reset email sent"

# Portfolio Tests
class TestPortfolio:
    @pytest.mark.asyncio
    async def test_get_portfolio(self, async_client, auth_headers):
        response = await async_client.get(
            "/api/portfolio",
            headers=auth_headers
        )
        assert response.status_code == 200
        assert "positions" in response.json()
        assert "total_value" in response.json()
    
    @pytest.mark.asyncio
    async def test_add_transaction_buy(self, async_client, auth_headers):
        response = await async_client.post(
            "/api/portfolio/transaction",
            headers=auth_headers,
            json={
                "symbol": "AAPL",
                "type": "BUY",
                "shares": 10,
                "price": 150.00,
                "commission": 0
            }
        )
        assert response.status_code == 200
        assert "new_position" in response.json()
    
    @pytest.mark.asyncio
    async def test_add_transaction_sell(self, async_client, auth_headers):
        # First buy
        await async_client.post(
            "/api/portfolio/transaction",
            headers=auth_headers,
            json={
                "symbol": "AAPL",
                "type": "BUY",
                "shares": 10,
                "price": 150.00,
                "commission": 0
            }
        )
        
        # Then sell
        response = await async_client.post(
            "/api/portfolio/transaction",
            headers=auth_headers,
            json={
                "symbol": "AAPL",
                "type": "SELL",
                "shares": 5,
                "price": 155.00,
                "commission": 0
            }
        )
        assert response.status_code == 200
        assert response.json()["new_position"]["shares"] == 5
    
    @pytest.mark.asyncio
    async def test_insufficient_shares_error(self, async_client, auth_headers):
        response = await async_client.post(
            "/api/portfolio/transaction",
            headers=auth_headers,
            json={
                "symbol": "AAPL",
                "type": "SELL",
                "shares": 100,
                "price": 155.00,
                "commission": 0
            }
        )
        assert response.status_code == 400
        assert "Insufficient shares" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_portfolio_performance(self, async_client, auth_headers):
        response = await async_client.get(
            "/api/portfolio/performance?period=1M",
            headers=auth_headers
        )
        assert response.status_code == 200
        assert "performance" in response.json()

# Technical Analysis Tests
class TestTechnicalAnalysis:
    @pytest.mark.asyncio
    @patch('yfinance.Ticker')
    async def test_get_technical_indicators(self, mock_ticker, async_client):
        # Mock yfinance data
        mock_ticker.return_value.history.return_value = Mock(
            Close=Mock(values=[150, 151, 152, 153, 154]),
            High=Mock(values=[151, 152, 153, 154, 155]),
            Low=Mock(values=[149, 150, 151, 152, 153]),
            Volume=Mock(values=[1000000, 1100000, 1200000, 1300000, 1400000])
        )
        
        response = await async_client.get("/api/technical/AAPL")
        assert response.status_code == 200
        assert "rsi" in response.json()
        assert "macd" in response.json()
        assert "bollinger_bands" in response.json()
    
    @pytest.mark.asyncio
    async def test_get_trading_signals(self, async_client):
        response = await async_client.get("/api/technical/AAPL/signals")
        assert response.status_code == 200
        assert "overall_signal" in response.json()
        assert "confidence" in response.json()
        assert response.json()["overall_signal"] in ["BUY", "SELL", "HOLD"]
    
    @pytest.mark.asyncio
    async def test_invalid_symbol(self, async_client):
        response = await async_client.get("/api/technical/INVALID123")
        assert response.status_code == 404

# Backtesting Tests
class TestBacktesting:
    @pytest.mark.asyncio
    @patch('yfinance.Ticker')
    async def test_backtest_rsi_strategy(self, mock_ticker, async_client):
        # Mock historical data
        mock_data = Mock()
        mock_data.empty = False
        mock_data['Close'] = Mock(values=[150 + i for i in range(252)])
        mock_ticker.return_value.history.return_value = mock_data
        
        response = await async_client.get(
            "/api/backtest/AAPL?strategy=RSI&period=1y"
        )
        assert response.status_code == 200
        assert "total_trades" in response.json()
        assert "win_rate" in response.json()
        assert "sharpe_ratio" in response.json()
    
    @pytest.mark.asyncio
    async def test_compare_strategies(self, async_client):
        response = await async_client.get("/api/backtest/AAPL/compare")
        assert response.status_code == 200
        assert "results" in response.json()
        assert len(response.json()["results"]) == 3  # RSI, MACD, BOLLINGER
        assert "best_strategy" in response.json()

# Security Tests
class TestSecurity:
    def test_password_validation(self):
        validator = PasswordValidator()
        
        # Test weak password
        is_valid, message = validator.validate_password_strength("weak")
        assert not is_valid
        assert "at least 12 characters" in message
        
        # Test strong password
        is_valid, message = validator.validate_password_strength("StrongP@ssw0rd123!")
        assert is_valid
        assert message == "Password is strong"
    
    def test_password_hashing(self):
        validator = PasswordValidator()
        password = "TestPassword123!"
        
        hashed = validator.hash_password(password)
        assert hashed != password
        assert validator.verify_password(password, hashed)
        assert not validator.verify_password("WrongPassword", hashed)
    
    def test_data_encryption(self):
        encryption = DataEncryption()
        
        # Test string encryption
        original = "sensitive data"
        encrypted = encryption.encrypt(original)
        assert encrypted != original
        assert encryption.decrypt(encrypted) == original
        
        # Test dict encryption
        data = {"ssn": "123-45-6789", "name": "John Doe"}
        encrypted_data = encryption.encrypt_dict(data, ["ssn"])
        assert encrypted_data["ssn"] != data["ssn"]
        assert encrypted_data["name"] == data["name"]
        
        decrypted_data = encryption.decrypt_dict(encrypted_data, ["ssn"])
        assert decrypted_data["ssn"] == data["ssn"]
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, async_client):
        # Make multiple requests quickly
        responses = []
        for _ in range(15):
            response = await async_client.get("/api/stock/AAPL/price")
            responses.append(response.status_code)
        
        # Should hit rate limit
        assert 429 in responses
    
    @pytest.mark.asyncio
    async def test_sql_injection_protection(self, async_client):
        # Try SQL injection
        response = await async_client.get(
            "/api/search/'; DROP TABLE users; --"
        )
        assert response.status_code in [200, 404]  # Should handle safely
    
    @pytest.mark.asyncio
    async def test_xss_protection(self, async_client, auth_headers):
        # Try XSS in transaction
        response = await async_client.post(
            "/api/portfolio/transaction",
            headers=auth_headers,
            json={
                "symbol": "<script>alert('XSS')</script>",
                "type": "BUY",
                "shares": 10,
                "price": 100
            }
        )
        assert response.status_code == 400  # Should reject malicious input

# Integration Tests
class TestIntegration:
    @pytest.mark.asyncio
    async def test_complete_user_flow(self, async_client):
        """Test complete user journey from registration to trading"""
        
        # 1. Register new user
        register_response = await async_client.post(
            "/api/auth/register",
            json={
                "email": "integration@test.com",
                "password": "IntegrationTest123!"
            }
        )
        assert register_response.status_code == 201
        token = register_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # 2. Search for stock
        search_response = await async_client.get("/api/search/AAPL")
        assert search_response.status_code == 200
        
        # 3. Add to watchlist
        watchlist_response = await async_client.post(
            "/api/watchlist/add",
            headers=headers,
            json={"symbol": "AAPL"}
        )
        assert watchlist_response.status_code == 200
        
        # 4. Get technical analysis
        tech_response = await async_client.get("/api/technical/AAPL")
        assert tech_response.status_code == 200
        
        # 5. Add to portfolio
        portfolio_response = await async_client.post(
            "/api/portfolio/transaction",
            headers=headers,
            json={
                "symbol": "AAPL",
                "type": "BUY",
                "shares": 10,
                "price": 150.00
            }
        )
        assert portfolio_response.status_code == 200
        
        # 6. Check portfolio
        portfolio_check = await async_client.get(
            "/api/portfolio",
            headers=headers
        )
        assert portfolio_check.status_code == 200
        assert len(portfolio_check.json()["positions"]) > 0

# Performance Tests
class TestPerformance:
    @pytest.mark.asyncio
    async def test_api_response_time(self, async_client):
        """Test that API responds within acceptable time"""
        import time
        
        endpoints = [
            "/api/stock/AAPL/price",
            "/api/search/AAPL",
            "/health"
        ]
        
        for endpoint in endpoints:
            start = time.time()
            response = await async_client.get(endpoint)
            elapsed = time.time() - start
            
            assert elapsed < 2.0, f"{endpoint} took {elapsed}s"
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, async_client):
        """Test handling of concurrent requests"""
        tasks = []
        for _ in range(10):
            task = async_client.get("/api/stock/AAPL/price")
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        
        for response in responses:
            assert response.status_code in [200, 429]  # Success or rate limited

# WebSocket Tests
class TestWebSocket:
    @pytest.mark.asyncio
    async def test_websocket_connection(self):
        from fastapi.testclient import TestClient
        
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as websocket:
                websocket.send_json({"type": "subscribe", "symbols": ["AAPL"]})
                data = websocket.receive_json()
                assert data["type"] == "subscription_confirmed"
    
    @pytest.mark.asyncio
    async def test_websocket_price_updates(self):
        from fastapi.testclient import TestClient
        
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as websocket:
                websocket.send_json({"type": "subscribe", "symbols": ["AAPL"]})
                websocket.receive_json()  # Confirmation
                
                # Should receive price update
                data = websocket.receive_json(timeout=5)
                assert data["type"] == "price_update"
                assert "symbol" in data
                assert "price" in data

# Load Tests
class TestLoad:
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_load_handling(self, async_client):
        """Test system under load"""
        import aiohttp
        import asyncio
        
        async def make_request(session, url):
            try:
                async with session.get(url) as response:
                    return response.status
            except:
                return 500
        
        async def load_test():
            url = "http://localhost:8000/api/stock/AAPL/price"
            async with aiohttp.ClientSession() as session:
                tasks = []
                for _ in range(100):  # 100 concurrent requests
                    task = make_request(session, url)
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks)
                
                # Check success rate
                success_count = sum(1 for r in results if r == 200)
                success_rate = success_count / len(results)
                
                assert success_rate > 0.95, f"Success rate: {success_rate}"
        
        await load_test()

# End-to-End Tests
"""
// tests/e2e/test_e2e.js - Playwright E2E Tests
const { test, expect } = require('@playwright/test');

test.describe('StockAlert Pro E2E Tests', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto('https://app.stockalertpro.com');
    });
    
    test('User can register and login', async ({ page }) => {
        // Click sign up
        await page.click('text=Sign Up');
        
        // Fill registration form
        await page.fill('input[name="email"]', 'e2e@test.com');
        await page.fill('input[name="password"]', 'E2ETest123!');
        await page.fill('input[name="confirmPassword"]', 'E2ETest123!');
        
        // Submit
        await page.click('button[type="submit"]');
        
        // Should redirect to dashboard
        await expect(page).toHaveURL(/.*dashboard/);
        await expect(page.locator('text=Portfolio')).toBeVisible();
    });
    
    test('User can search and add stock to watchlist', async ({ page }) => {
        // Login first
        await login(page);
        
        // Search for stock
        await page.click('text=Search');
        await page.fill('input[placeholder="Search stocks..."]', 'AAPL');
        await page.press('input[placeholder="Search stocks..."]', 'Enter');
        
        // Add to watchlist
        await page.click('button:has-text("Add to Watchlist")');
        
        // Verify added
        await page.click('text=Watchlist');
        await expect(page.locator('text=AAPL')).toBeVisible();
    });
    
    test('User can add transaction to portfolio', async ({ page }) => {
        await login(page);
        
        // Go to portfolio
        await page.click('text=Portfolio');
        
        // Add transaction
        await page.click('button:has-text("Add Transaction")');
        await page.fill('input[name="symbol"]', 'AAPL');
        await page.fill('input[name="shares"]', '10');
        await page.fill('input[name="price"]', '150');
        await page.click('text=Buy');
        await page.click('button:has-text("Submit")');
        
        // Verify position appears
        await expect(page.locator('text=AAPL')).toBeVisible();
        await expect(page.locator('text=10 shares')).toBeVisible();
    });
    
    test('User can view technical analysis', async ({ page }) => {
        await login(page);
        
        // Navigate to technical analysis
        await page.click('text=Analysis');
        
        // Enter symbol
        await page.fill('input[placeholder="Enter symbol"]', 'AAPL');
        await page.click('button:has-text("Analyze")');
        
        // Wait for results
        await expect(page.locator('text=RSI')).toBeVisible();
        await expect(page.locator('text=MACD')).toBeVisible();
        await expect(page.locator('text=Trading Signal')).toBeVisible();
    });
    
    test('Real-time price updates work', async ({ page }) => {
        await login(page);
        
        // Go to watchlist
        await page.click('text=Watchlist');
        
        // Wait for initial price
        const initialPrice = await page.locator('.price').first().textContent();
        
        // Wait for update (max 30 seconds)
        await page.waitForTimeout(30000);
        
        // Check if price updated
        const updatedPrice = await page.locator('.price').first().textContent();
        expect(updatedPrice).not.toBe(initialPrice);
    });
});

async function login(page) {
    await page.fill('input[name="email"]', 'test@example.com');
    await page.fill('input[name="password"]', 'Test123!');
    await page.click('button:has-text("Sign In")');
    await page.waitForNavigation();
}
"""

# Mobile App Tests
"""
// test/widget_test.dart - Flutter Widget Tests
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:stockalert_pro/main.dart';
import 'package:mockito/mockito.dart';
import 'package:provider/provider.dart';

class MockAuthProvider extends Mock implements AuthProvider {}
class MockPortfolioProvider extends Mock implements PortfolioProvider {}
class MockWatchlistProvider extends Mock implements WatchlistProvider {}

void main() {
  group('Authentication Tests', () {
    testWidgets('Login screen displays correctly', (WidgetTester tester) async {
      await tester.pumpWidget(MaterialApp(home: LoginScreen()));
      
      expect(find.text('StockAlert Pro'), findsOneWidget);
      expect(find.byType(TextField), findsNWidgets(2));
      expect(find.text('Sign In'), findsOneWidget);
    });
    
    testWidgets('Login validation works', (WidgetTester tester) async {
      await tester.pumpWidget(MaterialApp(home: LoginScreen()));
      
      // Try to login without credentials
      await tester.tap(find.text('Sign In'));
      await tester.pump();
      
      expect(find.text('Please fill all fields'), findsOneWidget);
    });
  });
  
  group('Portfolio Tests', () {
    testWidgets('Portfolio screen shows positions', (WidgetTester tester) async {
      final mockPortfolioProvider = MockPortfolioProvider();
      when(mockPortfolioProvider.positions).thenReturn([
        Position(
          symbol: 'AAPL',
          shares: 10,
          averagePrice: 150,
          currentPrice: 155,
          totalValue: 1550,
          gainLoss: 50,
          gainLossPercent: 3.33,
        ),
      ]);
      
      await tester.pumpWidget(
        MultiProvider(
          providers: [
            ChangeNotifierProvider<PortfolioProvider>.value(
              value: mockPortfolioProvider,
            ),
          ],
          child: MaterialApp(home: PortfolioScreen()),
        ),
      );
      
      expect(find.text('AAPL'), findsOneWidget);
      expect(find.text('10 shares'), findsOneWidget);
    });
  });
  
  group('Technical Analysis Tests', () {
    testWidgets('Technical analysis screen loads', (WidgetTester tester) async {
      await tester.pumpWidget(MaterialApp(home: TechnicalAnalysisScreen()));
      
      expect(find.text('Technical Analysis'), findsOneWidget);
      expect(find.byType(TextField), findsOneWidget);
      expect(find.text('Analyze'), findsOneWidget);
    });
  });
}
"""