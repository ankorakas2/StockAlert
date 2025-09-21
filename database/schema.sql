-- database_optimization.sql - Complete Database Optimization & Scaling Configuration

-- =====================================================
-- 1. DATABASE SCHEMA OPTIMIZATION
-- =====================================================

-- Create optimized schema with partitioning
CREATE SCHEMA IF NOT EXISTS stockalert;
SET search_path TO stockalert;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "btree_gist";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "timescaledb";

-- =====================================================
-- 2. OPTIMIZED TABLES WITH PARTITIONING
-- =====================================================

-- Users table with optimized indexes
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT true,
    is_premium BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login_at TIMESTAMP WITH TIME ZONE,
    email_verified BOOLEAN DEFAULT false,
    two_factor_enabled BOOLEAN DEFAULT false,
    api_key_hash VARCHAR(255),
    notification_preferences JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}'
) WITH (fillfactor = 90);

CREATE INDEX idx_users_email_trgm ON users USING gin(email gin_trgm_ops);
CREATE INDEX idx_users_active ON users(is_active) WHERE is_active = true;
CREATE INDEX idx_users_created_at ON users(created_at DESC);
CREATE INDEX idx_users_last_login ON users(last_login_at DESC) WHERE last_login_at IS NOT NULL;

-- Portfolio positions table
CREATE TABLE IF NOT EXISTS portfolio_positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    symbol VARCHAR(10) NOT NULL,
    shares DECIMAL(15, 4) NOT NULL CHECK (shares >= 0),
    average_price DECIMAL(15, 4) NOT NULL CHECK (average_price > 0),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, symbol)
) WITH (fillfactor = 85);

CREATE INDEX idx_positions_user_id ON portfolio_positions(user_id);
CREATE INDEX idx_positions_symbol ON portfolio_positions(symbol);
CREATE INDEX idx_positions_value ON portfolio_positions((shares * average_price) DESC);

-- Transactions table (partitioned by month)
CREATE TABLE IF NOT EXISTS transactions (
    id UUID DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    transaction_type VARCHAR(10) NOT NULL CHECK (transaction_type IN ('BUY', 'SELL')),
    shares DECIMAL(15, 4) NOT NULL CHECK (shares > 0),
    price DECIMAL(15, 4) NOT NULL CHECK (price > 0),
    commission DECIMAL(10, 4) DEFAULT 0,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    notes TEXT,
    metadata JSONB DEFAULT '{}',
    PRIMARY KEY (id, timestamp)
) PARTITION BY RANGE (timestamp);

-- Create partitions for the next 12 months
DO $$
DECLARE
    start_date DATE := DATE_TRUNC('month', CURRENT_DATE);
    end_date DATE;
    partition_name TEXT;
BEGIN
    FOR i IN 0..11 LOOP
        end_date := start_date + INTERVAL '1 month';
        partition_name := 'transactions_' || TO_CHAR(start_date, 'YYYY_MM');
        
        EXECUTE format('
            CREATE TABLE IF NOT EXISTS %I PARTITION OF transactions
            FOR VALUES FROM (%L) TO (%L)',
            partition_name, start_date, end_date
        );
        
        -- Create indexes on partition
        EXECUTE format('
            CREATE INDEX IF NOT EXISTS idx_%I_user_id ON %I(user_id);
            CREATE INDEX IF NOT EXISTS idx_%I_symbol ON %I(symbol);
            CREATE INDEX IF NOT EXISTS idx_%I_timestamp ON %I(timestamp DESC);',
            partition_name, partition_name,
            partition_name, partition_name,
            partition_name, partition_name
        );
        
        start_date := end_date;
    END LOOP;
END $$;

-- Price history table (using TimescaleDB hypertable)
CREATE TABLE IF NOT EXISTS price_history (
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    open DECIMAL(15, 4),
    high DECIMAL(15, 4),
    low DECIMAL(15, 4),
    close DECIMAL(15, 4) NOT NULL,
    volume BIGINT,
    PRIMARY KEY (symbol, timestamp)
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('price_history', 'timestamp', 
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

-- Create continuous aggregate for 1-minute candles
CREATE MATERIALIZED VIEW IF NOT EXISTS price_1min
WITH (timescaledb.continuous) AS
SELECT 
    symbol,
    time_bucket('1 minute', timestamp) AS bucket,
    FIRST(open, timestamp) AS open,
    MAX(high) AS high,
    MIN(low) AS low,
    LAST(close, timestamp) AS close,
    SUM(volume) AS volume
FROM price_history
GROUP BY symbol, bucket
WITH NO DATA;

-- Add refresh policy
SELECT add_continuous_aggregate_policy('price_1min',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute',
    if_not_exists => TRUE);

-- Watchlist table with covering index
CREATE TABLE IF NOT EXISTS watchlists (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    symbol VARCHAR(10) NOT NULL,
    target_price DECIMAL(15, 4),
    stop_loss DECIMAL(15, 4),
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, symbol)
) WITH (fillfactor = 90);

CREATE INDEX idx_watchlist_user_symbol ON watchlists(user_id, symbol) 
    INCLUDE (target_price, stop_loss);

-- Alerts table (partitioned by status)
CREATE TABLE IF NOT EXISTS alerts (
    id UUID DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    symbol VARCHAR(10),
    alert_type VARCHAR(50) NOT NULL,
    condition JSONB NOT NULL,
    triggered_at TIMESTAMP WITH TIME ZONE,
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    message TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id, status)
) PARTITION BY LIST (status);

-- Create status partitions
CREATE TABLE alerts_pending PARTITION OF alerts FOR VALUES IN ('pending');
CREATE TABLE alerts_triggered PARTITION OF alerts FOR VALUES IN ('triggered');
CREATE TABLE alerts_acknowledged PARTITION OF alerts FOR VALUES IN ('acknowledged');
CREATE TABLE alerts_cancelled PARTITION OF alerts FOR VALUES IN ('cancelled');

-- Create indexes on each partition
DO $$
DECLARE
    partition_name TEXT;
BEGIN
    FOREACH partition_name IN ARRAY ARRAY['alerts_pending', 'alerts_triggered', 
                                          'alerts_acknowledged', 'alerts_cancelled'] LOOP
        EXECUTE format('
            CREATE INDEX idx_%I_user_id ON %I(user_id);
            CREATE INDEX idx_%I_symbol ON %I(symbol);
            CREATE INDEX idx_%I_created_at ON %I(created_at DESC);',
            partition_name, partition_name,
            partition_name, partition_name,
            partition_name, partition_name
        );
    END LOOP;
END $$;

-- Trading signals table with compression
CREATE TABLE IF NOT EXISTS trading_signals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(10) NOT NULL,
    signal_type VARCHAR(10) NOT NULL CHECK (signal_type IN ('BUY', 'SELL', 'HOLD')),
    confidence DECIMAL(5, 4) CHECK (confidence >= 0 AND confidence <= 1),
    price DECIMAL(15, 4),
    target_price DECIMAL(15, 4),
    stop_loss DECIMAL(15, 4),
    reasoning TEXT,
    indicators JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE
);

-- Enable compression for older signals
ALTER TABLE trading_signals SET (
    autovacuum_vacuum_scale_factor = 0.01,
    autovacuum_analyze_scale_factor = 0.005
);

-- News sentiment table
CREATE TABLE IF NOT EXISTS news_sentiment (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(10) NOT NULL,
    headline TEXT NOT NULL,
    source VARCHAR(100),
    url TEXT,
    sentiment_score DECIMAL(5, 4),
    confidence DECIMAL(5, 4),
    published_at TIMESTAMP WITH TIME ZONE,
    analyzed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_news_symbol_date ON news_sentiment(symbol, published_at DESC);
CREATE INDEX idx_news_sentiment ON news_sentiment(sentiment_score);

-- =====================================================
-- 3. MATERIALIZED VIEWS FOR PERFORMANCE
-- =====================================================

-- Portfolio summary view
CREATE MATERIALIZED VIEW IF NOT EXISTS portfolio_summary AS
SELECT 
    pp.user_id,
    COUNT(DISTINCT pp.symbol) AS position_count,
    SUM(pp.shares * ph.latest_price) AS total_value,
    SUM(pp.shares * ph.latest_price - pp.shares * pp.average_price) AS total_gain_loss,
    SUM(CASE 
        WHEN pp.shares * ph.latest_price > pp.shares * pp.average_price 
        THEN 1 ELSE 0 
    END)::FLOAT / NULLIF(COUNT(*), 0) AS win_rate,
    MAX(pp.updated_at) AS last_updated
FROM portfolio_positions pp
JOIN LATERAL (
    SELECT close AS latest_price
    FROM price_history
    WHERE symbol = pp.symbol
    ORDER BY timestamp DESC
    LIMIT 1
) ph ON true
GROUP BY pp.user_id;

CREATE UNIQUE INDEX idx_portfolio_summary_user ON portfolio_summary(user_id);

-- Top movers view
CREATE MATERIALIZED VIEW IF NOT EXISTS top_movers AS
WITH price_changes AS (
    SELECT 
        symbol,
        close AS current_price,
        LAG(close, 1) OVER (PARTITION BY symbol ORDER BY timestamp DESC) AS prev_price,
        timestamp
    FROM price_history
    WHERE timestamp > CURRENT_TIMESTAMP - INTERVAL '24 hours'
)
SELECT 
    symbol,
    current_price,
    (current_price - prev_price) / NULLIF(prev_price, 0) * 100 AS change_percent,
    timestamp
FROM price_changes
WHERE prev_price IS NOT NULL
ORDER BY ABS((current_price - prev_price) / NULLIF(prev_price, 0)) DESC
LIMIT 20;

-- =====================================================
-- 4. PERFORMANCE OPTIMIZATION FUNCTIONS
-- =====================================================

-- Function to automatically create new partitions
CREATE OR REPLACE FUNCTION create_monthly_partitions()
RETURNS void AS $$
DECLARE
    start_date DATE;
    end_date DATE;
    partition_name TEXT;
BEGIN
    -- Get the last partition's end date
    SELECT MAX(partition_end) INTO start_date
    FROM (
        SELECT 
            pg_get_expr(c.relpartbound, c.oid) AS partition_bound,
            split_part(
                split_part(pg_get_expr(c.relpartbound, c.oid), 'TO (''', 2),
                '''', 1
            )::DATE AS partition_end
        FROM pg_class c
        JOIN pg_inherits i ON i.inhrelid = c.oid
        WHERE i.inhparent = 'transactions'::regclass
    ) AS partitions;
    
    -- Create next 3 months of partitions
    FOR i IN 1..3 LOOP
        end_date := start_date + INTERVAL '1 month';
        partition_name := 'transactions_' || TO_CHAR(start_date, 'YYYY_MM');
        
        EXECUTE format('
            CREATE TABLE IF NOT EXISTS %I PARTITION OF transactions
            FOR VALUES FROM (%L) TO (%L)',
            partition_name, start_date, end_date
        );
        
        start_date := end_date;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Function to update portfolio summary
CREATE OR REPLACE FUNCTION update_portfolio_summary()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY portfolio_summary;
END;
$$ LANGUAGE plpgsql;

-- Function to archive old data
CREATE OR REPLACE FUNCTION archive_old_data()
RETURNS void AS $$
BEGIN
    -- Archive transactions older than 2 years
    INSERT INTO transactions_archive
    SELECT * FROM transactions
    WHERE timestamp < CURRENT_TIMESTAMP - INTERVAL '2 years';
    
    -- Delete archived data
    DELETE FROM transactions
    WHERE timestamp < CURRENT_TIMESTAMP - INTERVAL '2 years';
    
    -- Archive old alerts
    INSERT INTO alerts_archive
    SELECT * FROM alerts
    WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '6 months'
    AND status IN ('acknowledged', 'cancelled');
    
    DELETE FROM alerts
    WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '6 months'
    AND status IN ('acknowledged', 'cancelled');
    
    RAISE NOTICE 'Archival completed at %', CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- 5. TRIGGERS FOR AUTOMATION
-- =====================================================

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_positions_updated_at BEFORE UPDATE ON portfolio_positions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- Trigger to update portfolio on transaction
CREATE OR REPLACE FUNCTION update_portfolio_on_transaction()
RETURNS TRIGGER AS $$
DECLARE
    current_shares DECIMAL(15, 4);
    current_avg_price DECIMAL(15, 4);
BEGIN
    -- Get current position
    SELECT shares, average_price INTO current_shares, current_avg_price
    FROM portfolio_positions
    WHERE user_id = NEW.user_id AND symbol = NEW.symbol;
    
    IF NOT FOUND THEN
        -- Create new position for BUY
        IF NEW.transaction_type = 'BUY' THEN
            INSERT INTO portfolio_positions (user_id, symbol, shares, average_price)
            VALUES (NEW.user_id, NEW.symbol, NEW.shares, NEW.price);
        END IF;
    ELSE
        IF NEW.transaction_type = 'BUY' THEN
            -- Update position for BUY
            UPDATE portfolio_positions
            SET shares = current_shares + NEW.shares,
                average_price = ((current_shares * current_avg_price) + 
                                (NEW.shares * NEW.price)) / 
                               (current_shares + NEW.shares)
            WHERE user_id = NEW.user_id AND symbol = NEW.symbol;
        ELSIF NEW.transaction_type = 'SELL' THEN
            -- Update position for SELL
            IF current_shares >= NEW.shares THEN
                UPDATE portfolio_positions
                SET shares = current_shares - NEW.shares
                WHERE user_id = NEW.user_id AND symbol = NEW.symbol;
                
                -- Delete position if no shares left
                DELETE FROM portfolio_positions
                WHERE user_id = NEW.user_id 
                AND symbol = NEW.symbol 
                AND shares = 0;
            ELSE
                RAISE EXCEPTION 'Insufficient shares to sell';
            END IF;
        END IF;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_portfolio
    AFTER INSERT ON transactions
    FOR EACH ROW
    EXECUTE FUNCTION update_portfolio_on_transaction();

-- =====================================================
-- 6. QUERY OPTIMIZATION & STATISTICS
-- =====================================================

-- Update table statistics
ANALYZE users;
ANALYZE portfolio_positions;
ANALYZE transactions;
ANALYZE price_history;
ANALYZE watchlists;
ANALYZE alerts;
ANALYZE trading_signals;
ANALYZE news_sentiment;

-- Set appropriate autovacuum settings
ALTER TABLE transactions SET (
    autovacuum_vacuum_scale_factor = 0.05,
    autovacuum_analyze_scale_factor = 0.02,
    autovacuum_vacuum_cost_delay = 10
);

ALTER TABLE price_history SET (
    autovacuum_vacuum_scale_factor = 0.01,
    autovacuum_analyze_scale_factor = 0.01
);

-- =====================================================
-- 7. READ REPLICA CONFIGURATION
-- =====================================================

-- Create publication for logical replication
CREATE PUBLICATION stockalert_pub FOR ALL TABLES;

-- Configuration for read replicas (run on replica)
/*
-- On replica server:
CREATE SUBSCRIPTION stockalert_sub
    CONNECTION 'host=primary-db.stockalertpro.com dbname=stockalert user=replicator password=xxx'
    PUBLICATION stockalert_pub;
*/

-- =====================================================
-- 8. CONNECTION POOLING CONFIGURATION
-- =====================================================

-- PgBouncer configuration (pgbouncer.ini)
/*
[databases]
stockalert = host=localhost port=5432 dbname=stockalert

[pgbouncer]
listen_port = 6432
listen_addr = *
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 25
min_pool_size = 5
reserve_pool_size = 5
reserve_pool_timeout = 3
max_db_connections = 100
max_user_connections = 100
server_lifetime = 3600
server_idle_timeout = 600
server_connect_timeout = 15
server_login_retry = 15
query_timeout = 0
query_wait_timeout = 120
client_idle_timeout = 0
client_login_timeout = 60
*/

-- =====================================================
-- 9. BACKUP & RECOVERY PROCEDURES
-- =====================================================

-- Backup script (backup.sh)
/*
#!/bin/bash
# Full backup
pg_dump -h localhost -U stockalert -d stockalert -F custom -b -v -f "backup_$(date +%Y%m%d_%H%M%S).dump"

# Incremental backup using pg_basebackup
pg_basebackup -h localhost -U replicator -D /backup/base -Ft -z -P

# Point-in-time recovery setup
# In postgresql.conf:
# wal_level = replica
# archive_mode = on
# archive_command = 'test ! -f /archive/%f && cp %p /archive/%f'
# max_wal_senders = 5
# wal_keep_segments = 64
*/

-- =====================================================
-- 10. MONITORING QUERIES
-- =====================================================

-- Check slow queries
CREATE OR REPLACE VIEW slow_queries AS
SELECT 
    query,
    calls,
    mean_exec_time,
    total_exec_time,
    min_exec_time,
    max_exec_time,
    stddev_exec_time,
    rows
FROM pg_stat_statements
WHERE mean_exec_time > 100  -- queries slower than 100ms
ORDER BY mean_exec_time DESC
LIMIT 20;

-- Check table bloat
CREATE OR REPLACE VIEW table_bloat AS
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) AS table_size,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS total_size,
    round(100 * pg_relation_size(schemaname||'.'||tablename) / 
          NULLIF(pg_total_relation_size(schemaname||'.'||tablename), 0), 2) AS table_percent
FROM pg_tables
WHERE schemaname = 'stockalert'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Check index usage
CREATE OR REPLACE VIEW index_usage AS
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
WHERE schemaname = 'stockalert'
ORDER BY idx_scan;

-- Check lock conflicts
CREATE OR REPLACE VIEW lock_conflicts AS
SELECT
    blocked_locks.pid AS blocked_pid,
    blocked_activity.usename AS blocked_user,
    blocking_locks.pid AS blocking_pid,
    blocking_activity.usename AS blocking_user,
    blocked_activity.query AS blocked_statement,
    blocking_activity.query AS blocking_statement
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
JOIN pg_catalog.pg_locks blocking_locks 
    ON blocking_locks.locktype = blocked_locks.locktype
    AND blocking_locks.database IS NOT DISTINCT FROM blocked_locks.database
    AND blocking_locks.relation IS NOT DISTINCT FROM blocked_locks.relation
    AND blocking_locks.page IS NOT DISTINCT FROM blocked_locks.page
    AND blocking_locks.tuple IS NOT DISTINCT FROM blocked_locks.tuple
    AND blocking_locks.virtualxid IS NOT DISTINCT FROM blocked_locks.virtualxid
    AND blocking_locks.transactionid IS NOT DISTINCT FROM blocked_locks.transactionid
    AND blocking_locks.classid IS NOT DISTINCT FROM blocked_locks.classid
    AND blocking_locks.objid IS NOT DISTINCT FROM blocked_locks.objid
    AND blocking_locks.objsubid IS NOT DISTINCT FROM blocked_locks.objsubid
    AND blocking_locks.pid != blocked_locks.pid
JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
WHERE NOT blocked_locks.granted;

-- =====================================================
-- 11. SCHEDULED MAINTENANCE JOBS
-- =====================================================

-- Create pg_cron extension for scheduled jobs
CREATE EXTENSION IF NOT EXISTS pg_cron;

-- Schedule partition creation (monthly)
SELECT cron.schedule('create-partitions', '0 0 1 * *', 
    'SELECT create_monthly_partitions()');

-- Schedule portfolio summary refresh (every 5 minutes)
SELECT cron.schedule('refresh-portfolio-summary', '*/5 * * * *',
    'REFRESH MATERIALIZED VIEW CONCURRENTLY portfolio_summary');

-- Schedule top movers refresh (every minute)
SELECT cron.schedule('refresh-top-movers', '* * * * *',
    'REFRESH MATERIALIZED VIEW CONCURRENTLY top_movers');

-- Schedule old data archival (weekly)
SELECT cron.schedule('archive-old-data', '0 2 * * 0',
    'SELECT archive_old_data()');

-- Schedule statistics update (daily)
SELECT cron.schedule('update-statistics', '0 3 * * *',
    'ANALYZE;');

-- Schedule vacuum (daily)
SELECT cron.schedule('vacuum-tables', '0 4 * * *',
    'VACUUM ANALYZE;');

-- =====================================================
-- 12. PERFORMANCE TUNING PARAMETERS
-- =====================================================

-- PostgreSQL configuration (postgresql.conf)
/*
# Memory
shared_buffers = 4GB
effective_cache_size = 12GB
maintenance_work_mem = 1GB
work_mem = 32MB

# Checkpoint
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1

# Connections
max_connections = 200
max_prepared_transactions = 100

# Parallel query
max_parallel_workers_per_gather = 4
max_parallel_workers = 8
max_parallel_maintenance_workers = 4

# Write performance
synchronous_commit = off
wal_writer_delay = 200ms
commit_delay = 100

# Query planning
enable_partitionwise_join = on
enable_partitionwise_aggregate = on

# Monitoring
shared_preload_libraries = 'pg_stat_statements,auto_explain,pg_cron'
pg_stat_statements.track = all
auto_explain.log_min_duration = '100ms'
*/

-- =====================================================
-- 13. SECURITY & ACCESS CONTROL
-- =====================================================

-- Create read-only user for analytics
CREATE USER analytics_user WITH PASSWORD 'secure_password';
GRANT CONNECT ON DATABASE stockalert TO analytics_user;
GRANT USAGE ON SCHEMA stockalert TO analytics_user;
GRANT SELECT ON ALL TABLES IN SCHEMA stockalert TO analytics_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA stockalert GRANT SELECT ON TABLES TO analytics_user;

-- Create application user with limited privileges
CREATE USER app_user WITH PASSWORD 'secure_password';
GRANT CONNECT ON DATABASE stockalert TO app_user;
GRANT USAGE ON SCHEMA stockalert TO app_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA stockalert TO app_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA stockalert TO app_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA stockalert GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO app_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA stockalert GRANT USAGE, SELECT ON SEQUENCES TO app_user;

-- Row-level security for multi-tenancy
ALTER TABLE portfolio_positions ENABLE ROW LEVEL SECURITY;
ALTER TABLE transactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE watchlists ENABLE ROW LEVEL SECURITY;
ALTER TABLE alerts ENABLE ROW LEVEL SECURITY;

-- Create policies
CREATE POLICY portfolio_isolation ON portfolio_positions
    FOR ALL
    TO app_user
    USING (user_id = current_setting('app.current_user_id')::UUID);

CREATE POLICY transaction_isolation ON transactions
    FOR ALL
    TO app_user
    USING (user_id = current_setting('app.current_user_id')::UUID);

CREATE POLICY watchlist_isolation ON watchlists
    FOR ALL
    TO app_user
    USING (user_id = current_setting('app.current_user_id')::UUID);

CREATE POLICY alert_isolation ON alerts
    FOR ALL
    TO app_user
    USING (user_id = current_setting('app.current_user_id')::UUID);