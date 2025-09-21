# main.py - Production Backend με Portfolio, Technical Analysis & Security
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, validator, Field
from typing import List, Optional, Dict, Any
import httpx
import asyncio
from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import credentials, firestore, auth, messaging
import yfinance as yf
from textblob import TextBlob
import numpy as np
import pandas as pd
import ta
import hashlib
import json
import os
from decimal import Decimal
import re
import bleach
from enum import Enum
import logging
import uvicorn
import ssl
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI with security headers
app = FastAPI(
    title="StockAlert Pro API",
    version="2.0.0",
    docs_url=None if os.getenv("ENVIRONMENT") == "production" else "/docs",
    redoc_url=None if os.getenv("ENVIRONMENT") == "production" else "/redoc"
)

# Security Configuration
ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
API_KEY = os.getenv("API_KEY", "")
JWT_SECRET = os.getenv("JWT_SECRET", "")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# Rate Limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Security Middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=ALLOWED_HOSTS
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    max_age=3600
)

# Firebase Initialization
firebase_config = json.loads(os.getenv("FIREBASE_CONFIG", "{}"))
if firebase_config:
    cred = credentials.Certificate(firebase_config)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
else:
    logger.warning("Firebase not configured - using mock data")
    db = None

# Authentication
security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify Firebase Auth token"""
    token = credentials.credentials
    try:
        decoded_token = auth.verify_id_token(token)
        return decoded_token
    except Exception as e:
        logger.error(f"Token verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )

# Input Sanitization
def sanitize_input(text: str) -> str:
    """Sanitize user input to prevent XSS and injection attacks"""
    if not text:
        return text
    # Remove HTML tags
    cleaned = bleach.clean(text, tags=[], strip=True)
    # Remove special characters that could be used for injection
    cleaned = re.sub(r'[<>\"\'%;()&+]', '', cleaned)
    return cleaned[:500]  # Limit length

# Models with validation
class PortfolioPosition(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=10)
    shares: float = Field(..., gt=0)
    average_price: float = Field(..., gt=0)
    current_price: Optional[float] = None
    total_value: Optional[float] = None
    gain_loss: Optional[float] = None
    gain_loss_percent: Optional[float] = None
    
    @validator('symbol')
    def validate_symbol(cls, v):
        return sanitize_input(v.upper())

class Transaction(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=10)
    type: str = Field(..., pattern="^(BUY|SELL)$")
    shares: float = Field(..., gt=0)
    price: float = Field(..., gt=0)
    timestamp: datetime = Field(default_factory=datetime.now)
    commission: float = Field(default=0, ge=0)
    
    @validator('symbol')
    def validate_symbol(cls, v):
        return sanitize_input(v.upper())

class TechnicalIndicators(BaseModel):
    symbol: str
    rsi: float
    macd: Dict[str, float]
    bollinger_bands: Dict[str, float]
    sma_20: float
    sma_50: float
    sma_200: float
    ema_12: float
    ema_26: float
    volume_avg: float
    atr: float
    stochastic: Dict[str, float]
    obv: float
    adx: float

class BacktestResult(BaseModel):
    strategy: str
    symbol: str
    period: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    avg_win: float
    avg_loss: float
    profit_factor: float

class SignalPerformance(BaseModel):
    signal_id: str
    symbol: str
    signal_type: str
    signal_date: datetime
    signal_price: float
    current_price: float
    performance_percent: float
    status: str  # 'active', 'closed', 'expired'
    outcome: Optional[str]  # 'success', 'failure', None

# Technical Analysis Service
class TechnicalAnalysisService:
    @staticmethod
    async def calculate_indicators(symbol: str, period: str = "3mo") -> TechnicalIndicators:
        """Calculate all technical indicators for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            
            if df.empty:
                raise ValueError(f"No data available for {symbol}")
            
            # Convert to numpy arrays
            close = df['Close'].values
            high = df['High'].values
            low = df['Low'].values
            volume = df['Volume'].values
            
            # Calculate indicators
            rsi = ta.momentum.RSIIndicator(pd.Series(close), window=14).rsi().iloc[-1]

            # MACD
            macd_ind = ta.trend.MACD(pd.Series(close), window_slow=26, window_fast=12, window_sign=9)
            macd = macd_ind.macd().iloc[-1]
            macd_signal = macd_ind.macd_signal().iloc[-1]
            macd_hist = macd_ind.macd_diff().iloc[-1]

            macd_data = {
                "macd": float(macd) if not np.isnan(macd) else 0,
                "signal": float(macd_signal) if not np.isnan(macd_signal) else 0,
                "histogram": float(macd_hist) if not np.isnan(macd_hist) else 0
            }
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(pd.Series(close), window=20, window_dev=2)
            upper = bb.bollinger_hband().iloc[-1]
            middle = bb.bollinger_mavg().iloc[-1]
            lower = bb.bollinger_lband().iloc[-1]

            bb_data = {
                "upper": float(upper) if not np.isnan(upper) else 0,
                "middle": float(middle) if not np.isnan(middle) else 0,
                "lower": float(lower) if not np.isnan(lower) else 0,
                "bandwidth": float((upper - lower) / middle) if not np.isnan(middle) and middle != 0 else 0
            }
            
            # Moving Averages
            sma_20 = ta.trend.SMAIndicator(pd.Series(close), window=20).sma_indicator().iloc[-1]
            sma_50 = ta.trend.SMAIndicator(pd.Series(close), window=50).sma_indicator().iloc[-1] if len(
                close) >= 50 else 0
            sma_200 = ta.trend.SMAIndicator(pd.Series(close), window=200).sma_indicator().iloc[-1] if len(
                close) >= 200 else 0

            ema_12 = ta.trend.EMAIndicator(pd.Series(close), window=12).ema_indicator().iloc[-1]
            ema_26 = ta.trend.EMAIndicator(pd.Series(close), window=26).ema_indicator().iloc[-1]

            volume_avg = ta.trend.SMAIndicator(pd.Series(volume), window=20).sma_indicator().iloc[-1]

            # ATR (Average True Range)
            atr = ta.volatility.AverageTrueRange(pd.Series(high), pd.Series(low), pd.Series(close), window=14).average_true_range().iloc[-1]
            
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(pd.Series(high), pd.Series(low), pd.Series(close), window=14,
                                                     smooth_window=3)
            slowk = stoch.stoch().iloc[-1]
            slowd = stoch.stoch_signal().iloc[-1]
            stoch_data = {
                "k": float(slowk) if not np.isnan(slowk) else 0,
                "d": float(slowd) if not np.isnan(slowd) else 0
            }
            
            # OBV (On Balance Volume)
            obv = ta.volume.OnBalanceVolumeIndicator(pd.Series(close), pd.Series(volume)).on_balance_volume().iloc[-1]

            # ADX (Average Directional Index)
            adx = ta.trend.ADXIndicator(pd.Series(high), pd.Series(low), pd.Series(close), window=14).adx().iloc[-1]

            return TechnicalIndicators(
                symbol=symbol,
                rsi=float(rsi) if not np.isnan(rsi) else 50,
                macd=macd_data,
                bollinger_bands=bb_data,
                sma_20=float(sma_20) if not np.isnan(sma_20) else 0,
                sma_50=float(sma_50) if not np.isnan(sma_50) else 0,
                sma_200=float(sma_200) if not np.isnan(sma_200) else 0,
                ema_12=float(ema_12) if not np.isnan(ema_12) else 0,
                ema_26=float(ema_26) if not np.isnan(ema_26) else 0,
                volume_avg=float(volume_avg) if not np.isnan(volume_avg) else 0,
                atr=float(atr) if not np.isnan(atr) else 0,
                stochastic=stoch_data,
                obv=float(obv) if not np.isnan(obv) else 0,
                adx=float(adx) if not np.isnan(adx) else 0
            )
            
        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# Portfolio Management Service
class PortfolioService:
    @staticmethod
    async def get_portfolio(user_id: str) -> Dict[str, Any]:
        """Get user's portfolio with current values"""
        try:
            if not db:
                return {"positions": [], "total_value": 0, "total_gain_loss": 0}
            
            # Get user's positions
            positions_ref = db.collection('portfolios').document(user_id).collection('positions')
            positions = positions_ref.stream()
            
            portfolio_data = []
            total_value = 0
            total_cost = 0
            
            for position in positions:
                pos_data = position.to_dict()
                symbol = position.id
                
                # Get current price
                ticker = yf.Ticker(symbol)
                current_price = ticker.info.get('currentPrice', ticker.info.get('regularMarketPrice', 0))
                
                # Calculate values
                shares = pos_data.get('shares', 0)
                avg_price = pos_data.get('average_price', 0)
                position_value = shares * current_price
                position_cost = shares * avg_price
                gain_loss = position_value - position_cost
                gain_loss_percent = (gain_loss / position_cost * 100) if position_cost > 0 else 0
                
                portfolio_data.append(PortfolioPosition(
                    symbol=symbol,
                    shares=shares,
                    average_price=avg_price,
                    current_price=current_price,
                    total_value=position_value,
                    gain_loss=gain_loss,
                    gain_loss_percent=gain_loss_percent
                ))
                
                total_value += position_value
                total_cost += position_cost
            
            total_gain_loss = total_value - total_cost
            total_gain_loss_percent = (total_gain_loss / total_cost * 100) if total_cost > 0 else 0
            
            return {
                "positions": portfolio_data,
                "total_value": total_value,
                "total_cost": total_cost,
                "total_gain_loss": total_gain_loss,
                "total_gain_loss_percent": total_gain_loss_percent,
                "position_count": len(portfolio_data)
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @staticmethod
    async def add_transaction(user_id: str, transaction: Transaction) -> Dict[str, Any]:
        """Add a buy/sell transaction"""
        try:
            if not db:
                return {"message": "Transaction recorded (demo mode)"}
            
            # Get current position
            position_ref = db.collection('portfolios').document(user_id).collection('positions').document(transaction.symbol)
            position_doc = position_ref.get()
            
            if position_doc.exists:
                current_data = position_doc.to_dict()
                current_shares = current_data.get('shares', 0)
                current_avg_price = current_data.get('average_price', 0)
            else:
                current_shares = 0
                current_avg_price = 0
            
            # Calculate new position
            if transaction.type == "BUY":
                new_shares = current_shares + transaction.shares
                new_avg_price = ((current_shares * current_avg_price) + (transaction.shares * transaction.price)) / new_shares
            else:  # SELL
                if current_shares < transaction.shares:
                    raise HTTPException(status_code=400, detail="Insufficient shares to sell")
                new_shares = current_shares - transaction.shares
                new_avg_price = current_avg_price  # Average price doesn't change on sell
            
            # Update position
            if new_shares > 0:
                position_ref.set({
                    'shares': new_shares,
                    'average_price': new_avg_price,
                    'last_updated': datetime.now()
                })
            else:
                position_ref.delete()  # Remove position if no shares left
            
            # Record transaction
            transaction_ref = db.collection('portfolios').document(user_id).collection('transactions').add({
                'symbol': transaction.symbol,
                'type': transaction.type,
                'shares': transaction.shares,
                'price': transaction.price,
                'timestamp': transaction.timestamp,
                'commission': transaction.commission,
                'total_value': transaction.shares * transaction.price + transaction.commission
            })
            
            return {
                "message": f"Transaction completed: {transaction.type} {transaction.shares} shares of {transaction.symbol}",
                "new_position": {
                    "shares": new_shares,
                    "average_price": new_avg_price
                }
            }
            
        except Exception as e:
            logger.error(f"Error adding transaction: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# Backtesting Service
class BacktestingService:
    @staticmethod
    async def backtest_strategy(symbol: str, strategy: str, period: str = "1y") -> BacktestResult:
        """Backtest a trading strategy"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            
            if df.empty:
                raise ValueError(f"No data available for {symbol}")
            
            # Implement different strategies
            if strategy == "RSI":
                results = BacktestingService._backtest_rsi(df)
            elif strategy == "MACD":
                results = BacktestingService._backtest_macd(df)
            elif strategy == "BOLLINGER":
                results = BacktestingService._backtest_bollinger(df)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            return BacktestResult(
                strategy=strategy,
                symbol=symbol,
                period=period,
                **results
            )
            
        except Exception as e:
            logger.error(f"Error backtesting {strategy} for {symbol}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @staticmethod
    def _backtest_rsi(df: pd.DataFrame) -> Dict:
        """Backtest RSI strategy"""
        close = df['Close'].values
        rsi = ta.momentum.RSIIndicator(pd.Series(close), window=14).rsi().values
        # Generate signals
        buy_signals = rsi < 30
        sell_signals = rsi > 70
        
        # Simulate trades
        position = 0
        trades = []
        entry_price = 0
        
        for i in range(14, len(close)):  # Start after RSI warmup
            if buy_signals[i] and position == 0:
                position = 1
                entry_price = close[i]
                trades.append(('BUY', close[i], i))
            elif sell_signals[i] and position == 1:
                position = 0
                exit_price = close[i]
                trades.append(('SELL', close[i], i))
                
        # Calculate performance
        return BacktestingService._calculate_performance(trades, close)
    
    @staticmethod
    def _backtest_macd(df: pd.DataFrame) -> Dict:
        """Backtest MACD strategy"""
        close = df['Close'].values
        # MACD
        macd_ind = ta.trend.MACD(pd.Series(close), window_slow=26, window_fast=12, window_sign=9)
        macd = macd_ind.macd().values
        signal = macd_ind.macd_signal().values
        hist = macd_ind.macd_diff().values
        # Generate signals (MACD crossover)
        buy_signals = (hist[:-1] < 0) & (hist[1:] > 0)
        sell_signals = (hist[:-1] > 0) & (hist[1:] < 0)
        
        # Simulate trades
        position = 0
        trades = []
        
        for i in range(26, len(close)-1):  # Start after MACD warmup
            if buy_signals[i-26] and position == 0:
                position = 1
                trades.append(('BUY', close[i], i))
            elif sell_signals[i-26] and position == 1:
                position = 0
                trades.append(('SELL', close[i], i))
        
        return BacktestingService._calculate_performance(trades, close)
    
    @staticmethod
    def _backtest_bollinger(df: pd.DataFrame) -> Dict:
        """Backtest Bollinger Bands strategy"""
        close = df['Close'].values
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(pd.Series(close), window=20, window_dev=2)
        upper = bb.bollinger_hband().values
        middle = bb.bollinger_mavg().values
        lower = bb.bollinger_lband().values
        # Generate signals
        buy_signals = close < lower
        sell_signals = close > upper
        
        # Simulate trades
        position = 0
        trades = []
        
        for i in range(20, len(close)):  # Start after BB warmup
            if buy_signals[i] and position == 0:
                position = 1
                trades.append(('BUY', close[i], i))
            elif sell_signals[i] and position == 1:
                position = 0
                trades.append(('SELL', close[i], i))
        
        return BacktestingService._calculate_performance(trades, close)
    
    @staticmethod
    def _calculate_performance(trades: List, prices: np.ndarray) -> Dict:
        """Calculate backtest performance metrics"""
        if len(trades) < 2:
            return {
                "total_trades": len(trades),
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "total_return": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "profit_factor": 0
            }
        
        # Calculate trade returns
        returns = []
        wins = []
        losses = []
        
        for i in range(0, len(trades)-1, 2):
            if i+1 < len(trades):
                buy_price = trades[i][1]
                sell_price = trades[i+1][1]
                ret = (sell_price - buy_price) / buy_price
                returns.append(ret)
                
                if ret > 0:
                    wins.append(ret)
                else:
                    losses.append(ret)
        
        if not returns:
            return {
                "total_trades": len(trades),
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "total_return": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "profit_factor": 0
            }
        
        # Calculate metrics
        total_return = np.prod([1 + r for r in returns]) - 1
        win_rate = len(wins) / len(returns) if returns else 0
        
        # Sharpe Ratio (annualized)
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Max Drawdown
        cumulative = np.cumprod([1 + r for r in returns])
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 0
        
        # Profit Factor
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 0
        profit_factor = (avg_win * len(wins)) / (avg_loss * len(losses)) if losses and avg_loss > 0 else 0
        
        return {
            "total_trades": len(trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": win_rate,
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor
        }

# API Endpoints

@app.get("/")
async def root():
    return {"message": "StockAlert Pro Production API", "version": "2.0.0", "status": "operational"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "environment": ENVIRONMENT,
        "firebase": "connected" if db else "not connected"
    }

# Protected Portfolio Endpoints
@app.get("/api/portfolio")
@limiter.limit("30/minute")
async def get_portfolio(request: Request, user_data: dict = Depends(verify_token)):
    """Get user's portfolio with current values"""
    user_id = user_data['uid']
    portfolio = await PortfolioService.get_portfolio(user_id)
    return portfolio

@app.post("/api/portfolio/transaction")
@limiter.limit("10/minute")
async def add_transaction(
    transaction: Transaction,
    request: Request,
    user_data: dict = Depends(verify_token)
):
    """Add a buy/sell transaction to portfolio"""
    user_id = user_data['uid']
    result = await PortfolioService.add_transaction(user_id, transaction)
    return result

@app.get("/api/portfolio/transactions/{symbol}")
@limiter.limit("30/minute")
async def get_transactions(
    symbol: str,
    request: Request,
    user_data: dict = Depends(verify_token),
    limit: int = 50
):
    """Get transaction history for a symbol"""
    user_id = user_data['uid']
    symbol = sanitize_input(symbol.upper())
    
    if not db:
        return {"transactions": []}
    
    transactions_ref = db.collection('portfolios').document(user_id).collection('transactions')
    query = transactions_ref.where('symbol', '==', symbol).order_by('timestamp', direction=firestore.Query.DESCENDING).limit(limit)
    transactions = query.stream()
    
    return {
        "transactions": [t.to_dict() for t in transactions]
    }

@app.get("/api/portfolio/performance")
@limiter.limit("20/minute")
async def get_portfolio_performance(
    request: Request,
    user_data: dict = Depends(verify_token),
    period: str = "1M"
):
    """Get portfolio performance over time"""
    user_id = user_data['uid']
    
    if not db:
        return {"performance": []}
    
    # Get historical snapshots
    snapshots_ref = db.collection('portfolios').document(user_id).collection('snapshots')
    
    # Calculate time range
    end_date = datetime.now()
    if period == "1D":
        start_date = end_date - timedelta(days=1)
    elif period == "1W":
        start_date = end_date - timedelta(weeks=1)
    elif period == "1M":
        start_date = end_date - timedelta(days=30)
    elif period == "3M":
        start_date = end_date - timedelta(days=90)
    elif period == "1Y":
        start_date = end_date - timedelta(days=365)
    else:
        start_date = end_date - timedelta(days=30)
    
    query = snapshots_ref.where('timestamp', '>=', start_date).order_by('timestamp')
    snapshots = query.stream()
    
    performance_data = []
    for snapshot in snapshots:
        data = snapshot.to_dict()
        performance_data.append({
            "timestamp": data.get('timestamp'),
            "total_value": data.get('total_value'),
            "daily_return": data.get('daily_return'),
            "cumulative_return": data.get('cumulative_return')
        })
    
    return {"performance": performance_data}

# Technical Analysis Endpoints
@app.get("/api/technical/{symbol}")
@limiter.limit("60/minute")
async def get_technical_indicators(
    symbol: str,
    request: Request,
    period: str = "3mo"
):
    """Get technical indicators for a symbol"""
    symbol = sanitize_input(symbol.upper())
    indicators = await TechnicalAnalysisService.calculate_indicators(symbol, period)
    return indicators

@app.get("/api/technical/{symbol}/signals")
@limiter.limit("30/minute")
async def get_technical_signals(symbol: str, request: Request):
    """Get trading signals based on technical indicators"""
    symbol = sanitize_input(symbol.upper())
    indicators = await TechnicalAnalysisService.calculate_indicators(symbol)
    
    signals = []
    confidence = 0
    
    # RSI Signal
    if indicators.rsi < 30:
        signals.append({"indicator": "RSI", "signal": "BUY", "value": indicators.rsi, "reason": "Oversold"})
        confidence += 0.2
    elif indicators.rsi > 70:
        signals.append({"indicator": "RSI", "signal": "SELL", "value": indicators.rsi, "reason": "Overbought"})
        confidence -= 0.2
    
    # MACD Signal
    if indicators.macd["histogram"] > 0 and indicators.macd["macd"] > indicators.macd["signal"]:
        signals.append({"indicator": "MACD", "signal": "BUY", "value": indicators.macd["histogram"], "reason": "Bullish crossover"})
        confidence += 0.25
    elif indicators.macd["histogram"] < 0 and indicators.macd["macd"] < indicators.macd["signal"]:
        signals.append({"indicator": "MACD", "signal": "SELL", "value": indicators.macd["histogram"], "reason": "Bearish crossover"})
        confidence -= 0.25
    
    # Bollinger Bands Signal
    ticker = yf.Ticker(symbol)
    current_price = ticker.info.get('currentPrice', ticker.info.get('regularMarketPrice', 0))
    
    if current_price < indicators.bollinger_bands["lower"]:
        signals.append({"indicator": "Bollinger", "signal": "BUY", "value": current_price, "reason": "Price below lower band"})
        confidence += 0.15
    elif current_price > indicators.bollinger_bands["upper"]:
        signals.append({"indicator": "Bollinger", "signal": "SELL", "value": current_price, "reason": "Price above upper band"})
        confidence -= 0.15
    
    # Moving Average Signal
    if indicators.sma_50 > 0 and indicators.sma_20 > indicators.sma_50:
        signals.append({"indicator": "MA", "signal": "BUY", "value": indicators.sma_20, "reason": "Golden cross pattern"})
        confidence += 0.2
    elif indicators.sma_50 > 0 and indicators.sma_20 < indicators.sma_50:
        signals.append({"indicator": "MA", "signal": "SELL", "value": indicators.sma_20, "reason": "Death cross pattern"})
        confidence -= 0.2
    
    # Stochastic Signal
    if indicators.stochastic["k"] < 20:
        signals.append({"indicator": "Stochastic", "signal": "BUY", "value": indicators.stochastic["k"], "reason": "Oversold"})
        confidence += 0.1
    elif indicators.stochastic["k"] > 80:
        signals.append({"indicator": "Stochastic", "signal": "SELL", "value": indicators.stochastic["k"], "reason": "Overbought"})
        confidence -= 0.1
    
    # Volume Signal
    if indicators.volume_avg > 0:
        ticker_history = ticker.history(period="1d")
        if not ticker_history.empty:
            current_volume = ticker_history['Volume'].iloc[-1]
            if current_volume > indicators.volume_avg * 1.5:
                signals.append({"indicator": "Volume", "signal": "STRONG", "value": current_volume, "reason": "High volume"})
                confidence = abs(confidence) * 1.2  # Amplify signal with volume
    
    # Determine overall signal
    overall_signal = "HOLD"
    if confidence > 0.3:
        overall_signal = "BUY"
    elif confidence < -0.3:
        overall_signal = "SELL"
    
    return {
        "symbol": symbol,
        "overall_signal": overall_signal,
        "confidence": min(abs(confidence), 1.0),
        "signals": signals,
        "current_price": current_price,
        "timestamp": datetime.now().isoformat()
    }

# Backtesting Endpoints
@app.get("/api/backtest/{symbol}")
@limiter.limit("10/minute")
async def backtest_strategy(
    symbol: str,
    request: Request,
    strategy: str = "RSI",
    period: str = "1y"
):
    """Backtest a trading strategy on historical data"""
    symbol = sanitize_input(symbol.upper())
    strategy = sanitize_input(strategy.upper())
    
    if strategy not in ["RSI", "MACD", "BOLLINGER"]:
        raise HTTPException(status_code=400, detail="Invalid strategy. Choose RSI, MACD, or BOLLINGER")
    
    result = await BacktestingService.backtest_strategy(symbol, strategy, period)
    return result

@app.get("/api/backtest/{symbol}/compare")
@limiter.limit("5/minute")
async def compare_strategies(
    symbol: str,
    request: Request,
    period: str = "1y"
):
    """Compare multiple strategies for a symbol"""
    symbol = sanitize_input(symbol.upper())
    strategies = ["RSI", "MACD", "BOLLINGER"]
    
    results = []
    for strategy in strategies:
        try:
            result = await BacktestingService.backtest_strategy(symbol, strategy, period)
            results.append(result)
        except Exception as e:
            logger.error(f"Error backtesting {strategy}: {e}")
    
    # Sort by total return
    results.sort(key=lambda x: x.total_return, reverse=True)
    
    return {
        "symbol": symbol,
        "period": period,
        "results": results,
        "best_strategy": results[0].strategy if results else None,
        "timestamp": datetime.now().isoformat()
    }

# Signal Performance Tracking
@app.get("/api/signals/performance")
@limiter.limit("20/minute")
async def get_signal_performance(
    request: Request,
    user_data: dict = Depends(verify_token),
    days: int = 30
):
    """Get historical performance of signals"""
    user_id = user_data['uid']
    
    if not db:
        return {"signals": [], "stats": {}}
    
    # Get historical signals
    start_date = datetime.now() - timedelta(days=days)
    signals_ref = db.collection('signals')
    query = signals_ref.where('timestamp', '>=', start_date).order_by('timestamp', direction=firestore.Query.DESCENDING)
    signals = query.stream()
    
    performance_data = []
    successful = 0
    failed = 0
    active = 0
    
    for signal in signals:
        data = signal.to_dict()
        signal_date = data.get('timestamp')
        signal_price = data.get('price')
        symbol = data.get('symbol')
        signal_type = data.get('signal_type')
        
        # Get current price
        ticker = yf.Ticker(symbol)
        current_price = ticker.info.get('currentPrice', ticker.info.get('regularMarketPrice', 0))
        
        # Calculate performance
        if signal_type == "BUY":
            performance_percent = ((current_price - signal_price) / signal_price) * 100
        else:  # SELL
            performance_percent = ((signal_price - current_price) / signal_price) * 100
        
        # Determine outcome
        status = "active"
        outcome = None
        
        if performance_percent > 5:  # 5% profit target
            status = "closed"
            outcome = "success"
            successful += 1
        elif performance_percent < -2:  # 2% stop loss
            status = "closed"
            outcome = "failure"
            failed += 1
        elif (datetime.now() - signal_date).days > 7:  # Expired after 7 days
            status = "expired"
            if performance_percent > 0:
                outcome = "success"
                successful += 1
            else:
                outcome = "failure"
                failed += 1
        else:
            active += 1
        
        performance_data.append(SignalPerformance(
            signal_id=signal.id,
            symbol=symbol,
            signal_type=signal_type,
            signal_date=signal_date,
            signal_price=signal_price,
            current_price=current_price,
            performance_percent=performance_percent,
            status=status,
            outcome=outcome
        ))
    
    # Calculate statistics
    total_signals = len(performance_data)
    win_rate = (successful / total_signals * 100) if total_signals > 0 else 0
    avg_return = np.mean([s.performance_percent for s in performance_data]) if performance_data else 0
    
    return {
        "signals": [s.dict() for s in performance_data[:50]],  # Return latest 50
        "stats": {
            "total_signals": total_signals,
            "successful": successful,
            "failed": failed,
            "active": active,
            "win_rate": win_rate,
            "average_return": avg_return
        }
    }

# Public endpoints (with rate limiting)
@app.get("/api/search/{query}")
@limiter.limit("60/minute")
async def search_stocks(query: str, request: Request):
    """Search for stock symbols"""
    query = sanitize_input(query)
    
    try:
        # Try to get info from yfinance
        ticker = yf.Ticker(query.upper())
        info = ticker.info
        
        if info and 'symbol' in info:
            return [{
                "symbol": info.get('symbol', query.upper()),
                "name": info.get('longName', info.get('shortName', '')),
                "exchange": info.get('exchange', ''),
                "sector": info.get('sector', ''),
                "industry": info.get('industry', '')
            }]
    except:
        pass
    
    # Fallback to predefined list for demo
    return []

@app.get("/api/stock/{symbol}/price")
@limiter.limit("120/minute")
async def get_stock_price(symbol: str, request: Request):
    """Get current stock price and info"""
    symbol = sanitize_input(symbol.upper())
    
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        history = ticker.history(period="1d")
        
        if history.empty:
            raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
        
        return {
            "symbol": symbol,
            "price": info.get('currentPrice', info.get('regularMarketPrice', 0)),
            "previousClose": info.get('previousClose', 0),
            "dayChange": info.get('regularMarketChange', 0),
            "dayChangePercent": info.get('regularMarketChangePercent', 0),
            "volume": int(history['Volume'].iloc[-1]) if not history.empty else 0,
            "marketCap": info.get('marketCap', 0),
            "pe": info.get('trailingPE', 0),
            "beta": info.get('beta', 0),
            "52WeekHigh": info.get('fiftyTwoWeekHigh', 0),
            "52WeekLow": info.get('fiftyTwoWeekLow', 0),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting price for {symbol}: {e}")
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")

# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# HTTPS Configuration for Production
if __name__ == "__main__":
    if ENVIRONMENT == "production":
        # SSL Configuration
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(
            os.getenv("SSL_CERT_PATH", "cert.pem"),
            os.getenv("SSL_KEY_PATH", "key.pem")
        )
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=int(os.getenv("PORT", 8000)),
            ssl_keyfile=os.getenv("SSL_KEY_PATH", "key.pem"),
            ssl_certfile=os.getenv("SSL_CERT_PATH", "cert.pem"),
            log_level="info"
        )
    else:
        # Development mode
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=int(os.getenv("PORT", 8000)),
            reload=True,
            log_level="debug"
        )