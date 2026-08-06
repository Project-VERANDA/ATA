-- =============================================================================
-- PostgreSQL Initialization Script - ATA Speech Anonymizer
-- Job metadata and tracking database
-- =============================================================================

-- Create database (if not already created by POSTGRES_DB env var)
-- NOTE: This runs in a fresh container, database already exists from env

-- Enable UUID extension for job IDs
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Jobs table (tracks processing status)
CREATE TABLE IF NOT EXISTS jobs (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    user_id VARCHAR(255) NOT NULL,  -- Keycloak subject ID
    filename VARCHAR(512) NOT NULL,
    file_size BIGINT NOT NULL,
    
    -- Processing configuration
    language VARCHAR(10) DEFAULT 'auto',
    whisper_model VARCHAR(50) DEFAULT 'large-v3',
    enable_diarization BOOLEAN DEFAULT TRUE,
    enable_anonymization BOOLEAN DEFAULT TRUE,
    llm_enabled BOOLEAN DEFAULT FALSE,
    llm_model VARCHAR(50) DEFAULT 'medgemma',
    tts_enabled BOOLEAN DEFAULT FALSE,
    pii_tags TEXT[] DEFAULT ARRAY['person', 'organization']::TEXT[],
    
    -- Status tracking
    status VARCHAR(50) DEFAULT 'queued',  -- queued, processing, completed, failed
    progress INTEGER DEFAULT 0,           -- 0-100 percentage
    error_message TEXT,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Output files
    transcript_path VARCHAR(1024),
    anonymized_path VARCHAR(1024),
    llm_rewritten_path VARCHAR(1024),
    tts_audio_path VARCHAR(1024),
    diarization_path VARCHAR(1024),
    
    -- Metadata
    duration_seconds FLOAT,
    word_count INTEGER,
    
    -- Compliance
    data_retention_until TIMESTAMP WITH TIME ZONE,
    deleted BOOLEAN DEFAULT FALSE,
    
    INDEX idx_jobs_user_id (user_id),
    INDEX idx_jobs_status (status),
    INDEX idx_jobs_created_at (created_at),
    INDEX idx_jobs_uuid (uuid)
);

-- Job sessions table (for batch processing)
CREATE TABLE IF NOT EXISTS job_sessions (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    name VARCHAR(255),
    total_jobs INTEGER DEFAULT 0,
    completed_jobs INTEGER DEFAULT 0,
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_sessions_user_id (user_id),
    INDEX idx_sessions_status (status)
);

-- Audit log table (HIPAA compliance)
CREATE TABLE IF NOT EXISTS audit_log (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100),
    resource_id UUID,
    ip_address INET,
    user_agent TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_audit_user_id (user_id),
    INDEX idx_audit_timestamp (timestamp),
    INDEX idx_audit_action (action)
);

-- User preferences table
CREATE TABLE IF NOT EXISTS user_preferences (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) UNIQUE NOT NULL,
    default_language VARCHAR(10) DEFAULT 'auto',
    default_whisper_model VARCHAR(50) DEFAULT 'large-v3',
    default_llm_model VARCHAR(50) DEFAULT 'medgemma',
    default_tts_voice VARCHAR(50) DEFAULT 'en_US-amy-medium',
    enable_diarization_by_default BOOLEAN DEFAULT TRUE,
    enable_anonymization_by_default BOOLEAN DEFAULT TRUE,
    auto_delete_after_days INTEGER DEFAULT 30,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE (user_id)
);

-- Grant permissions (if using separate read/write users later)
-- GRANT SELECT, INSERT, UPDATE ON jobs TO ata_user;
-- GRANT SELECT, INSERT, UPDATE ON audit_log TO ata_user;
-- GRANT USAGE, SELECT ON SEQUENCE jobs_id_seq TO ata_user;
-- GRANT USAGE, SELECT ON SEQUENCE job_sessions_id_seq TO ata_user;
-- GRANT USAGE, SELECT ON SEQUENCE audit_log_id_seq TO ata_user;
-- GRANT USAGE, SELECT ON SEQUENCE user_preferences_id_seq TO ata_user;

-- Vacuum and analyze (optimize initial state)
VACUUM ANALYZE jobs;
VACUUM ANALYZE job_sessions;
VACUUM ANALYZE audit_log;
VACUUM ANALYZE user_preferences;
