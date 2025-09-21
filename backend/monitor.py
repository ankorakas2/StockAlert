# monitor.py - Background Monitoring Service για StockAlert Pro
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import json
import os
from dataclasses import dataclass, asdict
import httpx
import yfinance as yf
from textblob import TextBlob
import firebase_admin
from firebase_admin import credentials, firestore, messaging
from collections import defaultdict
from enum import Enum
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Configuration
class Config:
    # Monitoring intervals (in seconds)
    PRICE_CHECK_INTERVAL = 60  # Check prices every minute
    NEWS_CHECK_INTERVAL = 300  # Check news every 5 minutes
    SIGNAL_GENERATION_INTERVAL = 600  # Generate signals every 10 minutes

    # Thresholds
    PRICE_CHANGE_THRESHOLD = 0.02  # 2% price change triggers notification
    VOLUME_SPIKE_THRESHOLD = 2.0  # 2x average volume
    SENTIMENT_CHANGE_THRESHOLD = 0.15  # 15% sentiment change

    # News sources
    TRUSTED_SOURCES = [
        'Reuters', 'Bloomberg', 'Yahoo Finance', 'CNBC',
        'Wall Street Journal', 'Financial Times', 'MarketWatch'
    ]

    # API Configuration
    NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')

    # Firebase Configuration
    FIREBASE_CONFIG = json.loads(os.getenv('FIREBASE_CONFIG', '{}'))


# Data Models
@dataclass
class PriceAlert:
    symbol: str
    current_price: float
    previous_price: float
    change_percent: float
    volume: int
    average_volume: int
    timestamp: datetime
    alert_type: str  # 'price_spike', 'volume_spike', 'breakout'


@dataclass
class NewsAlert:
    symbol: str
    title: str
    description: str
    source: str
    sentiment_score: float
    confidence: float
    url: str
    published_at: datetime
    impact_level: str  # 'high', 'medium', 'low'


@dataclass
class TradingSignal:
    symbol: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    reasons: List[str]
    price_target: Optional[float]
    stop_loss: Optional[float]
    timestamp: datetime


class AlertType(Enum):
    PRICE_SPIKE = "price_spike"
    VOLUME_SPIKE = "volume_spike"
    BREAKING_NEWS = "breaking_news"
    SENTIMENT_CHANGE = "sentiment_change"
    TRADING_SIGNAL = "trading_signal"


# Monitoring Service
class StockMonitor:
    def __init__(self):
        self.db = None
        self.watchlists: Dict[str, Set[str]] = {}  # user_id -> set of symbols
        self.stock_cache: Dict[str, Dict] = {}
        self.news_cache: Dict[str, List[NewsAlert]] = defaultdict(list)
        self.signal_history: Dict[str, TradingSignal] = {}
        self.user_preferences: Dict[str, Dict] = {}
        self.initialize_firebase()

    def initialize_firebase(self):
        """Initialize Firebase Admin SDK"""
        try:
            if Config.FIREBASE_CONFIG and not firebase_admin._apps:
                cred = credentials.Certificate(Config.FIREBASE_CONFIG)
                firebase_admin.initialize_app(cred)
                self.db = firestore.client()
                logger.info("Firebase initialized successfully")
            else:
                logger.warning("Firebase not configured - using demo mode")
                self.db = None
        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {e}")
            self.db = None

    async def load_watchlists(self):
        """Load all user watchlists from Firestore"""
        try:
            if not self.db:
                # Demo data
                self.watchlists = {"demo_user": {"AAPL", "GOOGL", "MSFT"}}
                return

            users_ref = self.db.collection('users')
            users = users_ref.stream()

            for user in users:
                user_id = user.id
                user_data = user.to_dict()

                # Load user preferences
                self.user_preferences[user_id] = user_data.get('preferences', {
                    'notifications_enabled': True,
                    'price_alerts': True,
                    'news_alerts': True,
                    'signal_alerts': True
                })

                # Load watchlist
                watchlist_ref = self.db.collection('watchlists').document(user_id).collection('stocks')
                stocks = watchlist_ref.stream()

                self.watchlists[user_id] = set()
                for stock in stocks:
                    self.watchlists[user_id].add(stock.id)

            logger.info(f"Loaded {len(self.watchlists)} user watchlists")

        except Exception as e:
            logger.error(f"Error loading watchlists: {e}")

    def get_all_symbols(self) -> Set[str]:
        """Get unique symbols from all watchlists"""
        all_symbols = set()
        for symbols in self.watchlists.values():
            all_symbols.update(symbols)
        return all_symbols

    async def check_prices(self):
        """Check prices for all watched stocks"""
        symbols = self.get_all_symbols()
        if not symbols:
            logger.info("No symbols to monitor")
            return

        logger.info(f"Checking prices for {len(symbols)} stocks")

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                history = ticker.history(period="5d")

                if history.empty:
                    continue

                current_price = history['Close'].iloc[-1]
                previous_close = history['Close'].iloc[-2] if len(history) > 1 else current_price
                volume = history['Volume'].iloc[-1]
                avg_volume = history['Volume'].mean()

                change_percent = ((current_price - previous_close) / previous_close) * 100

                # Cache current data
                self.stock_cache[symbol] = {
                    'price': current_price,
                    'previous_close': previous_close,
                    'change_percent': change_percent,
                    'volume': volume,
                    'avg_volume': avg_volume,
                    'last_updated': datetime.now()
                }

                # Check for alerts
                await self.check_price_alerts(symbol, current_price, previous_close, change_percent, volume, avg_volume)

            except Exception as e:
                logger.error(f"Error checking price for {symbol}: {e}")

    async def check_price_alerts(self, symbol: str, current_price: float, previous_close: float,
                                 change_percent: float, volume: int, avg_volume: float):
        """Check if price changes warrant an alert"""
        alerts = []

        # Price spike alert
        if abs(change_percent) >= Config.PRICE_CHANGE_THRESHOLD * 100:
            alert = PriceAlert(
                symbol=symbol,
                current_price=current_price,
                previous_price=previous_close,
                change_percent=change_percent,
                volume=volume,
                average_volume=int(avg_volume),
                timestamp=datetime.now(),
                alert_type='price_spike'
            )
            alerts.append(alert)
            logger.info(f"Price alert for {symbol}: {change_percent:.2f}% change")

        # Volume spike alert
        if volume > avg_volume * Config.VOLUME_SPIKE_THRESHOLD:
            alert = PriceAlert(
                symbol=symbol,
                current_price=current_price,
                previous_price=previous_close,
                change_percent=change_percent,
                volume=volume,
                average_volume=int(avg_volume),
                timestamp=datetime.now(),
                alert_type='volume_spike'
            )
            alerts.append(alert)
            logger.info(f"Volume alert for {symbol}: {volume / avg_volume:.1f}x average")

        # Send notifications for alerts
        for alert in alerts:
            await self.send_price_alert(alert)

    async def check_news(self):
        """Check for news on all watched stocks"""
        symbols = self.get_all_symbols()
        if not symbols:
            return

        logger.info(f"Checking news for {len(symbols)} stocks")

        for symbol in symbols:
            try:
                news_items = await self.fetch_news(symbol)

                # Analyze sentiment for each news item
                for news in news_items:
                    sentiment = self.analyze_sentiment(news['title'] + ' ' + news['description'])

                    news_alert = NewsAlert(
                        symbol=symbol,
                        title=news['title'],
                        description=news['description'],
                        source=news['source'],
                        sentiment_score=sentiment['score'],
                        confidence=sentiment['confidence'],
                        url=news['url'],
                        published_at=news['published_at'],
                        impact_level=self.assess_impact_level(sentiment['score'], sentiment['confidence'])
                    )

                    # Check if this is significant news
                    if self.is_significant_news(news_alert):
                        await self.send_news_alert(news_alert)

                    # Cache news
                    self.news_cache[symbol].append(news_alert)

            except Exception as e:
                logger.error(f"Error checking news for {symbol}: {e}")

    async def fetch_news(self, symbol: str) -> List[Dict]:
        """Fetch news from multiple sources"""
        news_items = []

        # Fetch from NewsAPI
        if Config.NEWSAPI_KEY:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        'https://newsapi.org/v2/everything',
                        params={
                            'q': symbol,
                            'apiKey': Config.NEWSAPI_KEY,
                            'language': 'en',
                            'sortBy': 'publishedAt',
                            'pageSize': 5
                        }
                    )

                    if response.status_code == 200:
                        data = response.json()
                        for article in data.get('articles', []):
                            # Filter by trusted sources
                            source_name = article.get('source', {}).get('name', '')
                            news_items.append({
                                'title': article.get('title', ''),
                                'description': article.get('description', '')[:200],
                                'url': article.get('url', ''),
                                'source': source_name,
                                'published_at': datetime.fromisoformat(
                                    article.get('publishedAt', '').replace('Z', '+00:00'))
                            })
            except Exception as e:
                logger.error(f"Error fetching news from NewsAPI: {e}")

        # Fetch from Yahoo Finance
        try:
            ticker = yf.Ticker(symbol)
            yahoo_news = ticker.news[:3] if hasattr(ticker, 'news') else []

            for article in yahoo_news:
                news_items.append({
                    'title': article.get('title', ''),
                    'description': article.get('summary', '')[:200],
                    'url': article.get('link', ''),
                    'source': 'Yahoo Finance',
                    'published_at': datetime.fromtimestamp(article.get('providerPublishTime', 0))
                })
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance news: {e}")

        return news_items

    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of text using TextBlob"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            # Normalize to 0-1 scale
            sentiment_score = (polarity + 1) / 2

            # Calculate confidence based on subjectivity
            confidence = 1 - subjectivity

            return {
                'score': sentiment_score,
                'confidence': confidence,
                'category': 'positive' if sentiment_score > 0.6 else 'negative' if sentiment_score < 0.4 else 'neutral'
            }
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {'score': 0.5, 'confidence': 0.5, 'category': 'neutral'}

    def assess_impact_level(self, sentiment_score: float, confidence: float) -> str:
        """Assess the impact level of news based on sentiment and confidence"""
        combined_score = abs(sentiment_score - 0.5) * confidence

        if combined_score > 0.3:
            return 'high'
        elif combined_score > 0.15:
            return 'medium'
        else:
            return 'low'

    def is_significant_news(self, news_alert: NewsAlert) -> bool:
        """Determine if news is significant enough to send alert"""
        # Check if news is recent (within last 2 hours)
        if datetime.now() - news_alert.published_at > timedelta(hours=2):
            return False

        # Check impact level
        if news_alert.impact_level != 'high':
            return False

        return True

    async def generate_signals(self):
        """Generate trading signals based on collected data"""
        symbols = self.get_all_symbols()

        for symbol in symbols:
            try:
                # Get cached data
                stock_data = self.stock_cache.get(symbol)
                news_alerts = self.news_cache.get(symbol, [])

                if not stock_data:
                    continue

                # Analyze recent news sentiment
                recent_news = [n for n in news_alerts if datetime.now() - n.published_at < timedelta(days=1)]

                if not recent_news:
                    continue

                avg_sentiment = sum(n.sentiment_score for n in recent_news) / len(recent_news)
                avg_confidence = sum(n.confidence for n in recent_news) / len(recent_news)

                # Generate signal
                signal = self.calculate_signal(symbol, stock_data, avg_sentiment, avg_confidence)

                if signal and signal.confidence > 0.6:
                    await self.send_signal_alert(signal)
                    self.signal_history[symbol] = signal

            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")

    def calculate_signal(self, symbol: str, stock_data: Dict, sentiment: float, confidence: float) -> Optional[
        TradingSignal]:
        """Calculate trading signal based on multiple factors"""
        reasons = []
        signal_type = 'HOLD'
        signal_confidence = confidence

        # Price momentum
        change_percent = stock_data.get('change_percent', 0)
        if abs(change_percent) > 2:
            if change_percent > 0:
                reasons.append(f"Strong price momentum (+{change_percent:.2f}%)")
            else:
                reasons.append(f"Negative price momentum ({change_percent:.2f}%)")

        # Volume analysis
        volume = stock_data.get('volume', 0)
        avg_volume = stock_data.get('avg_volume', 0)
        if avg_volume > 0:
            volume_ratio = volume / avg_volume
            if volume_ratio > 1.5:
                reasons.append(f"High trading volume ({volume_ratio:.1f}x average)")

        # Sentiment analysis
        if sentiment > 0.65 and confidence > 0.6:
            signal_type = 'BUY'
            reasons.append(f"Positive sentiment ({sentiment:.2%})")
        elif sentiment < 0.35 and confidence > 0.6:
            signal_type = 'SELL'
            reasons.append(f"Negative sentiment ({sentiment:.2%})")

        # Calculate price targets
        current_price = stock_data.get('price', 0)
        price_target = None
        stop_loss = None

        if signal_type == 'BUY':
            price_target = current_price * 1.05  # 5% profit target
            stop_loss = current_price * 0.98  # 2% stop loss
        elif signal_type == 'SELL':
            price_target = current_price * 0.95  # 5% decline target
            stop_loss = current_price * 1.02  # 2% stop loss

        if signal_type != 'HOLD':
            return TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=signal_confidence,
                reasons=reasons,
                price_target=price_target,
                stop_loss=stop_loss,
                timestamp=datetime.now()
            )

        return None

    async def send_price_alert(self, alert: PriceAlert):
        """Send price alert notifications to users"""
        users_to_notify = [uid for uid, symbols in self.watchlists.items() if alert.symbol in symbols]

        for user_id in users_to_notify:
            if not self.user_preferences.get(user_id, {}).get('price_alerts', True):
                continue

            logger.info(f"Sending price alert for {alert.symbol} to {user_id}")
            # In production, send via Firebase Cloud Messaging
            # For now, just log it

    async def send_news_alert(self, alert: NewsAlert):
        """Send news alert notifications"""
        users_to_notify = [uid for uid, symbols in self.watchlists.items() if alert.symbol in symbols]

        for user_id in users_to_notify:
            if not self.user_preferences.get(user_id, {}).get('news_alerts', True):
                continue

            logger.info(f"Sending news alert for {alert.symbol} to {user_id}")
            # In production, send via Firebase Cloud Messaging

    async def send_signal_alert(self, signal: TradingSignal):
        """Send trading signal notifications"""
        users_to_notify = [uid for uid, symbols in self.watchlists.items() if signal.symbol in symbols]

        for user_id in users_to_notify:
            if not self.user_preferences.get(user_id, {}).get('signal_alerts', True):
                continue

            logger.info(f"Sending {signal.signal_type} signal for {signal.symbol} to {user_id}")
            # In production, send via Firebase Cloud Messaging

    async def run_monitoring_loop(self):
        """Main monitoring loop"""
        logger.info("Starting monitoring service...")

        # Initial load
        await self.load_watchlists()

        # Create tasks for different monitoring intervals
        async def price_monitor():
            while True:
                await self.check_prices()
                await asyncio.sleep(Config.PRICE_CHECK_INTERVAL)

        async def news_monitor():
            while True:
                await self.check_news()
                await asyncio.sleep(Config.NEWS_CHECK_INTERVAL)

        async def signal_generator():
            while True:
                await self.generate_signals()
                await asyncio.sleep(Config.SIGNAL_GENERATION_INTERVAL)

        async def watchlist_reloader():
            while True:
                await asyncio.sleep(300)
                await self.load_watchlists()

        # Run all tasks
        try:
            await asyncio.gather(
                price_monitor(),
                news_monitor(),
                signal_generator(),
                watchlist_reloader()
            )
        except KeyboardInterrupt:
            logger.info("Monitoring service stopped by user")
        except Exception as e:
            logger.error(f"Monitoring service error: {e}")

    def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up monitoring service...")


# Main execution
async def main():
    monitor = StockMonitor()
    try:
        await monitor.run_monitoring_loop()
    finally:
        monitor.cleanup()


if __name__ == "__main__":
    # Run the monitoring service
    print("Starting StockAlert Pro Monitoring Service...")
    print("Press Ctrl+C to stop")
    asyncio.run(main())