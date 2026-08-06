#!/bin/bash
# =============================================================================
# Health Check Script for Medical Data Processing Pipeline
# =============================================================================

set -e

BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== ATA Pipeline Health Check ===${NC}"
echo "Timestamp: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo ""

# Check backend
echo -e "${BLUE}Checking Backend...${NC}"
if curl -sf http://localhost:5001/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Backend healthy${NC}"
else
    echo -e "${RED}✗ Backend unhealthy${NC}"
fi

# Check Keycloak
echo -e "${BLUE}Checking Keycloak...${NC}"
if curl -sf http://localhost:8080/health/started > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Keycloak healthy${NC}"
else
    echo -e "${RED}✗ Keycloak unhealthy${NC}"
fi

# Check Redis
echo -e "${BLUE}Checking Redis...${NC}"
if docker exec ata-redis redis-cli ping | grep -q PONG; then
    echo -e "${GREEN}✓ Redis healthy${NC}"
else
    echo -e "${RED}✗ Redis unhealthy${NC}"
fi

# Check PostgreSQL
echo -e "${BLUE}Checking PostgreSQL...${NC}"
if docker exec ata-postgres pg_isready -U ata_user -d ata_db > /dev/null 2>&1; then
    echo -e "${GREEN}✓ PostgreSQL healthy${NC}"
else
    echo -e "${RED}✗ PostgreSQL unhealthy${NC}"
fi

# Check GPU availability
echo -e "${BLUE}Checking GPU Resources...${NC}"
if nvidia-smi > /dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    echo -e "${GREEN}✓ GPU Available: ${GPU_NAME} (${GPU_MEM})${NC}"
else
    echo -e "${RED}✗ GPU not available (CPU-only mode)${NC}"
fi

# Check disk space
echo -e "${BLUE}Checking Disk Space...${NC}"
DISK_AVAIL=$(df -BG / | tail -1 | awk '{print $4}' | tr -d 'G')
if [ "$DISK_AVAIL" -gt 20 ]; then
    echo -e "${GREEN}✓ Disk space adequate: ${DISK_AVAIL}GB${NC}"
else
    echo -e "${RED}✗ Low disk space: ${DISK_AVAIL}GB remaining${NC}"
fi

echo ""
echo -e "${BLUE}=== Health Check Complete ===${NC}"
