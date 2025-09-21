# monitoring.py - Complete Monitoring & Observability System
import time
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import aiohttp
from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest
from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
import structlog
from datadog import initialize, statsd
import psutil
import redis
from sqlalchemy import create_engine, text
from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
import boto3

# Configure Structured Logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Metrics Definitions
class MetricsCollector:
    # Request Metrics
    http_requests_total = Counter(
        'http_requests_total',
        'Total HTTP requests',
        ['method', 'endpoint', 'status']
    )
    
    http_request_duration_seconds = Histogram(
        'http_request_duration_seconds',
        'HTTP request duration in seconds',
        ['method', 'endpoint']
    )
    
    http_request_size_bytes = Histogram(
        'http_request_size_bytes',
        'HTTP request size in bytes',
        ['method', 'endpoint']
    )
    
    http_response_size_bytes = Histogram(
        'http_response_size_bytes',
        'HTTP response size in bytes',
        ['method', 'endpoint']
    )
    
    # Business Metrics
    portfolio_value_total = Gauge(
        'portfolio_value_total',
        'Total portfolio value by user',
        ['user_id']
    )
    
    trades_total = Counter(
        'trades_total',
        'Total number of trades',
        ['user_id', 'symbol', 'type']
    )
    
    active_users_total = Gauge(
        'active_users_total',
        'Total number of active users'
    )
    
    watchlist_size = Gauge(
        'watchlist_size',
        'Size of user watchlists',
        ['user_id']
    )
    
    signals_generated_total = Counter(
        'signals_generated_total',
        'Total trading signals generated',
        ['symbol', 'signal_type']
    )
    
    alerts_sent_total = Counter(
        'alerts_sent_total',
        'Total alerts sent',
        ['alert_type', 'channel']
    )
    
    # Technical Metrics
    database_connections_active = Gauge(
        'database_connections_active',
        'Active database connections'
    )
    
    redis_connections_active = Gauge(
        'redis_connections_active',
        'Active Redis connections'
    )
    
    cache_hits_total = Counter(
        'cache_hits_total',
        'Total cache hits',
        ['cache_type']
    )
    
    cache_misses_total = Counter(
        'cache_misses_total',
        'Total cache misses',
        ['cache_type']
    )
    
    external_api_calls_total = Counter(
        'external_api_calls_total',
        'Total external API calls',
        ['api_name', 'status']
    )
    
    external_api_latency_seconds = Histogram(
        'external_api_latency_seconds',
        'External API latency',
        ['api_name']
    )
    
    # System Metrics
    system_cpu_usage_percent = Gauge(
        'system_cpu_usage_percent',
        'System CPU usage percentage'
    )
    
    system_memory_usage_percent = Gauge(
        'system_memory_usage_percent',
        'System memory usage percentage'
    )
    
    system_disk_usage_percent = Gauge(
        'system_disk_usage_percent',
        'System disk usage percentage'
    )

# Health Check System
class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@dataclass
class ComponentHealth:
    name: str
    status: HealthStatus
    latency_ms: float
    message: Optional[str] = None
    metadata: Optional[Dict] = None

class HealthChecker:
    def __init__(self, db_url: str, redis_host: str, redis_port: int):
        self.db_url = db_url
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.components = {}
        
    async def check_database(self) -> ComponentHealth:
        """Check database health"""
        start = time.time()
        try:
            engine = create_engine(self.db_url)
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                latency = (time.time() - start) * 1000
                
                # Check connection pool
                pool_size = engine.pool.size()
                pool_checked_out = engine.pool.checked_out()
                
                return ComponentHealth(
                    name="database",
                    status=HealthStatus.HEALTHY,
                    latency_ms=latency,
                    metadata={
                        "pool_size": pool_size,
                        "connections_active": pool_checked_out
                    }
                )
        except Exception as e:
            return ComponentHealth(
                name="database",
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start) * 1000,
                message=str(e)
            )
    
    async def check_redis(self) -> ComponentHealth:
        """Check Redis health"""
        start = time.time()
        try:
            r = redis.Redis(host=self.redis_host, port=self.redis_port)
            r.ping()
            latency = (time.time() - start) * 1000
            
            # Get Redis info
            info = r.info()
            
            return ComponentHealth(
                name="redis",
                status=HealthStatus.HEALTHY,
                latency_ms=latency,
                metadata={
                    "connected_clients": info.get("connected_clients"),
                    "used_memory_human": info.get("used_memory_human"),
                    "uptime_in_seconds": info.get("uptime_in_seconds")
                }
            )
        except Exception as e:
            return ComponentHealth(
                name="redis",
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start) * 1000,
                message=str(e)
            )
    
    async def check_external_apis(self) -> List[ComponentHealth]:
        """Check external API health"""
        apis = [
            ("NewsAPI", "https://newsapi.org/v2/status"),
            ("Yahoo Finance", "https://query1.finance.yahoo.com/v1/test"),
            ("Firebase", "https://firebase.googleapis.com/v1/projects/test")
        ]
        
        results = []
        async with aiohttp.ClientSession() as session:
            for api_name, url in apis:
                start = time.time()
                try:
                    async with session.get(url, timeout=5) as response:
                        latency = (time.time() - start) * 1000
                        status = HealthStatus.HEALTHY if response.status == 200 else HealthStatus.DEGRADED
                        
                        results.append(ComponentHealth(
                            name=f"api_{api_name.lower().replace(' ', '_')}",
                            status=status,
                            latency_ms=latency,
                            metadata={"status_code": response.status}
                        ))
                except Exception as e:
                    results.append(ComponentHealth(
                        name=f"api_{api_name.lower().replace(' ', '_')}",
                        status=HealthStatus.UNHEALTHY,
                        latency_ms=(time.time() - start) * 1000,
                        message=str(e)
                    ))
        
        return results
    
    async def check_system_resources(self) -> ComponentHealth:
        """Check system resource usage"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Determine health status based on resource usage
        status = HealthStatus.HEALTHY
        warnings = []
        
        if cpu_percent > 80:
            status = HealthStatus.DEGRADED
            warnings.append(f"High CPU usage: {cpu_percent}%")
        
        if memory.percent > 85:
            status = HealthStatus.DEGRADED
            warnings.append(f"High memory usage: {memory.percent}%")
        
        if disk.percent > 90:
            status = HealthStatus.UNHEALTHY
            warnings.append(f"Critical disk usage: {disk.percent}%")
        
        return ComponentHealth(
            name="system_resources",
            status=status,
            latency_ms=0,
            message="; ".join(warnings) if warnings else None,
            metadata={
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "load_average": psutil.getloadavg()
            }
        )
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        # Run all health checks
        db_health = await self.check_database()
        redis_health = await self.check_redis()
        api_health = await self.check_external_apis()
        system_health = await self.check_system_resources()
        
        # Combine all component health
        components = [db_health, redis_health, system_health] + api_health
        
        # Determine overall status
        if any(c.status == HealthStatus.UNHEALTHY for c in components):
            overall_status = HealthStatus.UNHEALTHY
        elif any(c.status == HealthStatus.DEGRADED for c in components):
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
        
        return {
            "status": overall_status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "components": {c.name: asdict(c) for c in components},
            "version": "2.0.0",
            "uptime": self._get_uptime()
        }
    
    def _get_uptime(self) -> str:
        """Get application uptime"""
        boot_time = datetime.fromtimestamp(psutil.boot_time())
        uptime = datetime.now() - boot_time
        return str(uptime)

# Distributed Tracing
class TracingConfig:
    @staticmethod
    def setup_tracing(service_name: str, otlp_endpoint: str):
        """Setup OpenTelemetry tracing"""
        # Create tracer provider
        tracer_provider = TracerProvider()
        trace.set_tracer_provider(tracer_provider)
        
        # Configure OTLP exporter
        otlp_exporter = OTLPSpanExporter(
            endpoint=otlp_endpoint,
            insecure=True
        )
        
        # Add span processor
        span_processor = BatchSpanProcessor(otlp_exporter)
        tracer_provider.add_span_processor(span_processor)
        
        # Instrument FastAPI
        FastAPIInstrumentor.instrument_app(app)
        
        # Instrument requests library
        RequestsInstrumentor().instrument()
        
        return trace.get_tracer(service_name)

# Error Tracking
class ErrorTracker:
    @staticmethod
    def setup_sentry(dsn: str, environment: str):
        """Setup Sentry error tracking"""
        sentry_sdk.init(
            dsn=dsn,
            environment=environment,
            integrations=[
                FastApiIntegration(transaction_style="endpoint"),
                SqlalchemyIntegration()
            ],
            traces_sample_rate=0.1,
            profiles_sample_rate=0.1,
            attach_stacktrace=True,
            send_default_pii=False,
            before_send=ErrorTracker._before_send
        )
    
    @staticmethod
    def _before_send(event, hint):
        """Filter sensitive data before sending to Sentry"""
        # Remove sensitive data
        if 'request' in event and 'headers' in event['request']:
            event['request']['headers'] = {
                k: v for k, v in event['request']['headers'].items()
                if k.lower() not in ['authorization', 'cookie', 'x-api-key']
            }
        
        # Remove password fields
        if 'extra' in event:
            event['extra'] = {
                k: '***' if 'password' in k.lower() else v
                for k, v in event['extra'].items()
            }
        
        return event

# Custom Monitoring Middleware
class MonitoringMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, metrics_collector: MetricsCollector):
        super().__init__(app)
        self.metrics = metrics_collector
        
    async def dispatch(self, request: Request, call_next):
        # Start timing
        start_time = time.time()
        
        # Get request size
        request_size = int(request.headers.get('content-length', 0))
        
        # Process request
        response = await call_next(request)
        
        # Calculate metrics
        duration = time.time() - start_time
        response_size = int(response.headers.get('content-length', 0))
        
        # Update metrics
        endpoint = request.url.path
        method = request.method
        status = response.status_code
        
        self.metrics.http_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status=status
        ).inc()
        
        self.metrics.http_request_duration_seconds.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
        
        self.metrics.http_request_size_bytes.labels(
            method=method,
            endpoint=endpoint
        ).observe(request_size)
        
        self.metrics.http_response_size_bytes.labels(
            method=method,
            endpoint=endpoint
        ).observe(response_size)
        
        # Log request
        logger.info(
            "http_request",
            method=method,
            path=endpoint,
            status=status,
            duration_ms=duration * 1000,
            request_size=request_size,
            response_size=response_size,
            user_agent=request.headers.get('user-agent'),
            ip=request.client.host
        )
        
        # Send to DataDog
        statsd.increment('api.requests', tags=[
            f'method:{method}',
            f'endpoint:{endpoint}',
            f'status:{status}'
        ])
        statsd.histogram('api.latency', duration * 1000, tags=[
            f'method:{method}',
            f'endpoint:{endpoint}'
        ])
        
        return response

# Alerting System
class AlertManager:
    def __init__(self):
        self.sns_client = boto3.client('sns')
        self.pagerduty_key = os.getenv('PAGERDUTY_KEY')
        self.slack_webhook = os.getenv('SLACK_WEBHOOK')
        
    async def send_alert(self, severity: str, title: str, description: str, metadata: Dict = None):
        """Send alert through multiple channels"""
        alert_data = {
            "severity": severity,
            "title": title,
            "description": description,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        
        # Send to different channels based on severity
        if severity == "CRITICAL":
            await self._send_pagerduty(alert_data)
            await self._send_sms(alert_data)
        
        if severity in ["CRITICAL", "HIGH"]:
            await self._send_email(alert_data)
        
        # Always send to Slack
        await self._send_slack(alert_data)
        
        # Log alert
        logger.warning("alert_triggered", **alert_data)
    
    async def _send_pagerduty(self, alert_data: Dict):
        """Send alert to PagerDuty"""
        if not self.pagerduty_key:
            return
        
        async with aiohttp.ClientSession() as session:
            payload = {
                "routing_key": self.pagerduty_key,
                "event_action": "trigger",
                "payload": {
                    "summary": alert_data["title"],
                    "source": "stockalert-api",
                    "severity": alert_data["severity"].lower(),
                    "custom_details": alert_data
                }
            }
            
            await session.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=payload
            )
    
    async def _send_slack(self, alert_data: Dict):
        """Send alert to Slack"""
        if not self.slack_webhook:
            return
        
        emoji = {
            "CRITICAL": "🔴",
            "HIGH": "🟠",
            "MEDIUM": "🟡",
            "LOW": "🔵"
        }.get(alert_data["severity"], "⚪")
        
        async with aiohttp.ClientSession() as session:
            payload = {
                "text": f"{emoji} *{alert_data['title']}*\n{alert_data['description']}",
                "attachments": [{
                    "color": {
                        "CRITICAL": "danger",
                        "HIGH": "warning",
                        "MEDIUM": "warning",
                        "LOW": "good"
                    }.get(alert_data["severity"], ""),
                    "fields": [
                        {"title": k, "value": str(v), "short": True}
                        for k, v in alert_data.get("metadata", {}).items()
                    ],
                    "ts": int(datetime.utcnow().timestamp())
                }]
            }
            
            await session.post(self.slack_webhook, json=payload)
    
    async def _send_email(self, alert_data: Dict):
        """Send alert via email using AWS SES"""
        ses_client = boto3.client('ses')
        
        try:
            ses_client.send_email(
                Source='alerts@stockalertpro.com',
                Destination={'ToAddresses': ['devops@stockalertpro.com']},
                Message={
                    'Subject': {'Data': f"[{alert_data['severity']}] {alert_data['title']}"},
                    'Body': {
                        'Text': {'Data': json.dumps(alert_data, indent=2)},
                        'Html': {'Data': self._format_email_html(alert_data)}
                    }
                }
            )
        except Exception as e:
            logger.error("Failed to send email alert", error=str(e))
    
    async def _send_sms(self, alert_data: Dict):
        """Send SMS alert for critical issues"""
        try:
            self.sns_client.publish(
                PhoneNumber=os.getenv('ONCALL_PHONE'),
                Message=f"CRITICAL: {alert_data['title']}\n{alert_data['description'][:100]}"
            )
        except Exception as e:
            logger.error("Failed to send SMS alert", error=str(e))
    
    def _format_email_html(self, alert_data: Dict) -> str:
        """Format alert as HTML for email"""
        return f"""
        <html>
            <body>
                <h2 style="color: {'red' if alert_data['severity'] == 'CRITICAL' else 'orange'}">
                    {alert_data['title']}
                </h2>
                <p>{alert_data['description']}</p>
                <h3>Details:</h3>
                <table border="1">
                    {''.join([f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in alert_data.get('metadata', {}).items()])}
                </table>
                <p><small>Time: {alert_data['timestamp']}</small></p>
            </body>
        </html>
        """

# Custom Metrics Dashboard
class MetricsDashboard:
    @staticmethod
    def setup_custom_dashboards():
        """Setup custom Grafana dashboards"""
        dashboards = {
            "business_metrics": {
                "title": "Business Metrics",
                "panels": [
                    {
                        "title": "Active Users",
                        "query": "active_users_total",
                        "type": "stat"
                    },
                    {
                        "title": "Portfolio Values",
                        "query": "sum(portfolio_value_total)",
                        "type": "timeseries"
                    },
                    {
                        "title": "Trading Volume",
                        "query": "rate(trades_total[5m])",
                        "type": "graph"
                    },
                    {
                        "title": "Top Traded Stocks",
                        "query": "topk(10, sum by(symbol) (rate(trades_total[1h])))",
                        "type": "table"
                    }
                ]
            },
            "technical_metrics": {
                "title": "Technical Metrics",
                "panels": [
                    {
                        "title": "API Latency P95",
                        "query": "histogram_quantile(0.95, http_request_duration_seconds_bucket)",
                        "type": "graph"
                    },
                    {
                        "title": "Cache Hit Rate",
                        "query": "rate(cache_hits_total[5m]) / (rate(cache_hits_total[5m]) + rate(cache_misses_total[5m]))",
                        "type": "gauge"
                    },
                    {
                        "title": "Database Connections",
                        "query": "database_connections_active",
                        "type": "stat"
                    },
                    {
                        "title": "External API Performance",
                        "query": "histogram_quantile(0.99, external_api_latency_seconds_bucket)",
                        "type": "heatmap"
                    }
                ]
            },
            "alerts_dashboard": {
                "title": "Alerts & Incidents",
                "panels": [
                    {
                        "title": "Alert Rate",
                        "query": "rate(alerts_sent_total[5m])",
                        "type": "graph"
                    },
                    {
                        "title": "Error Rate",
                        "query": "rate(http_requests_total{status=~'5..'}[5m])",
                        "type": "graph"
                    },
                    {
                        "title": "System Health Score",
                        "query": "(1 - (rate(http_requests_total{status=~'5..'}[5m]) / rate(http_requests_total[5m]))) * 100",
                        "type": "gauge"
                    }
                ]
            }
        }
        return dashboards

# Application Performance Monitoring
class APMCollector:
    def __init__(self):
        self.transaction_times = []
        self.slow_queries = []
        self.memory_leaks = []
        
    async def collect_performance_metrics(self):
        """Collect detailed performance metrics"""
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "cpu": {
                "percent": psutil.cpu_percent(interval=1),
                "count": psutil.cpu_count(),
                "freq": psutil.cpu_freq().current if psutil.cpu_freq() else 0
            },
            "memory": {
                "percent": psutil.virtual_memory().percent,
                "available": psutil.virtual_memory().available,
                "used": psutil.virtual_memory().used,
                "swap_percent": psutil.swap_memory().percent
            },
            "disk": {
                "percent": psutil.disk_usage('/').percent,
                "read_bytes": psutil.disk_io_counters().read_bytes,
                "write_bytes": psutil.disk_io_counters().write_bytes
            },
            "network": {
                "bytes_sent": psutil.net_io_counters().bytes_sent,
                "bytes_recv": psutil.net_io_counters().bytes_recv,
                "packets_sent": psutil.net_io_counters().packets_sent,
                "packets_recv": psutil.net_io_counters().packets_recv
            },
            "processes": {
                "total": len(psutil.pids()),
                "python_threads": len(psutil.Process().threads())
            }
        }
        
        # Check for issues
        if metrics["memory"]["percent"] > 90:
            await self._check_memory_leak()
        
        if metrics["cpu"]["percent"] > 90:
            await self._profile_cpu()
        
        return metrics
    
    async def _check_memory_leak(self):
        """Check for potential memory leaks"""
        import tracemalloc
        tracemalloc.start()
        
        # Take snapshot
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        # Log top memory consumers
        logger.warning("High memory usage detected", 
                      top_consumers=[str(stat) for stat in top_stats[:10]])
    
    async def _profile_cpu(self):
        """Profile CPU usage"""
        import cProfile
        import pstats
        from io import StringIO
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        # Profile for 5 seconds
        await asyncio.sleep(5)
        
        profiler.disable()
        
        # Get stats
        s = StringIO()
        stats = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        stats.print_stats(10)
        
        logger.warning("High CPU usage detected", profile=s.getvalue())

# Log Aggregation
class LogAggregator:
    def __init__(self):
        self.elasticsearch_host = os.getenv('ELASTICSEARCH_HOST', 'localhost:9200')
        
    async def setup_log_shipping(self):
        """Setup log shipping to Elasticsearch"""
        config = {
            "version": 1,
            "handlers": {
                "elasticsearch": {
                    "class": "CMRESHandler.CMRESHandler",
                    "hosts": [{"host": self.elasticsearch_host.split(':')[0], 
                              "port": int(self.elasticsearch_host.split(':')[1])}],
                    "auth_type": "NO_AUTH",
                    "index_name_frequency": "daily",
                    "es_index_name": "stockalert-logs",
                    "es_doc_type": "log",
                    "buffer_size": 1000,
                    "flush_frequency_in_sec": 5
                }
            },
            "root": {
                "level": "INFO",
                "handlers": ["elasticsearch"]
            }
        }
        
        logging.config.dictConfig(config)

# Service Level Objectives (SLOs)
class SLOMonitor:
    def __init__(self):
        self.slos = {
            "availability": {
                "target": 99.9,
                "window": "30d",
                "query": "avg_over_time(up[30d])"
            },
            "latency_p99": {
                "target": 500,  # milliseconds
                "window": "1h",
                "query": "histogram_quantile(0.99, http_request_duration_seconds_bucket[1h])"
            },
            "error_rate": {
                "target": 0.1,  # 0.1%
                "window": "1h",
                "query": "rate(http_requests_total{status=~'5..'}[1h]) / rate(http_requests_total[1h])"
            }
        }
        
    async def check_slos(self) -> Dict[str, Dict]:
        """Check if SLOs are being met"""
        results = {}
        
        for slo_name, slo_config in self.slos.items():
            # Query Prometheus for actual value
            actual_value = await self._query_prometheus(slo_config["query"])
            
            # Check if SLO is met
            if slo_name == "error_rate":
                is_met = actual_value * 100 <= slo_config["target"]
            elif slo_name == "latency_p99":
                is_met = actual_value * 1000 <= slo_config["target"]
            else:
                is_met = actual_value >= slo_config["target"] / 100
            
            results[slo_name] = {
                "target": slo_config["target"],
                "actual": actual_value,
                "is_met": is_met,
                "error_budget_remaining": self._calculate_error_budget(slo_config, actual_value)
            }
        
        return results
    
    def _calculate_error_budget(self, slo_config: Dict, actual_value: float) -> float:
        """Calculate remaining error budget"""
        if "availability" in slo_config:
            allowed_downtime = (100 - slo_config["target"]) / 100
            actual_downtime = 1 - actual_value
            return max(0, (allowed_downtime - actual_downtime) / allowed_downtime * 100)
        return 100.0
    
    async def _query_prometheus(self, query: str) -> float:
        """Query Prometheus for metrics"""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"http://localhost:9090/api/v1/query",
                params={"query": query}
            ) as response:
                data = await response.json()
                if data["status"] == "success" and data["data"]["result"]:
                    return float(data["data"]["result"][0]["value"][1])
                return 0.0