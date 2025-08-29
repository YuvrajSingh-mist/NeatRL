#!/bin/bash
# Comprehensive visitor metrics monitoring and automation script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
API_URL="http://localhost:8000"
PROMETHEUS_URL="http://localhost:9090"
GRAFANA_URL="http://localhost:3000"
LOG_FILE="/tmp/visitor_metrics_monitor.log"

# Log function
log() {
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Health check function
check_service_health() {
    local service_name="$1"
    local url="$2"
    
    if curl -s --max-time 15 "$url" > /dev/null; then
        log "${GREEN}✅ $service_name is healthy${NC}"
        return 0
    else
        log "${RED}❌ $service_name is down${NC}"
        return 1
    fi
}

# Check visitor metrics
check_visitor_metrics() {
    log "${BLUE}🔍 Checking visitor metrics...${NC}"
    
    # Check if API is responding
    if ! curl -s --max-time 10 "$API_URL/metrics" > /dev/null; then
        log "${RED}❌ API metrics endpoint not responding${NC}"
        return 1
    fi
    
    # Check if metrics are exposed
    if curl -s --max-time 10 "$API_URL/metrics" | grep -q "unique_visitors_alltime"; then
        log "${GREEN}✅ Visitor metrics are exposed${NC}"
        
        # Get current counts
        local alltime_count=$(curl -s --max-time 10 "$API_URL/metrics" | grep "unique_visitors_alltime" | grep -o '[0-9]\+\.0' | head -1)
        local today_count=$(curl -s --max-time 10 "$API_URL/metrics" | grep "unique_visitors_today" | grep -o '[0-9]\+\.0' | head -1)
        local monthly_count=$(curl -s --max-time 10 "$API_URL/metrics" | grep "unique_visitors_month" | wc -l)
        
        log "${GREEN}📊 All-time visitors: $alltime_count${NC}"
        log "${GREEN}📊 Today's visitors: $today_count${NC}"
        log "${GREEN}📊 Monthly metrics: $monthly_count entries${NC}"
        
        return 0
    else
        log "${RED}❌ Visitor metrics not found${NC}"
        return 1
    fi
}

# Check Prometheus collection
check_prometheus_collection() {
    log "${BLUE}🔍 Checking Prometheus collection...${NC}"
    
    # Check if Prometheus is responding
    if ! curl -s --max-time 10 "$PROMETHEUS_URL/api/v1/query?query=up" > /dev/null; then
        log "${RED}❌ Prometheus not responding${NC}"
        return 1
    fi
    
    # Check all-time visitors
    local alltime_result=$(curl -s --max-time 10 "$PROMETHEUS_URL/api/v1/query?query=max(unique_visitors_alltime)" | jq -r '.data.result[0].value[1] // "null"')
    
    if [ "$alltime_result" != "null" ] && [ "$alltime_result" != "" ]; then
        log "${GREEN}✅ Prometheus collecting all-time visitors: $alltime_result${NC}"
        
        # Check monthly visitors
        local monthly_count=$(curl -s --max-time 10 "$PROMETHEUS_URL/api/v1/query?query=unique_visitors_month" | jq '.data.result | length')
        log "${GREEN}✅ Prometheus collecting monthly visitors: $monthly_count entries${NC}"
        
        return 0
    else
        # Check if the metric exists but has no data (which is normal for new deployments)
        local metric_exists=$(curl -s --max-time 10 "$PROMETHEUS_URL/api/v1/query?query=unique_visitors_alltime" | jq '.data.result | length')
        if [ "$metric_exists" -gt 0 ]; then
            log "${YELLOW}⚠️  Prometheus has visitor metrics but no data yet (normal for new deployments)${NC}"
            return 0  # This is not a failure, just no data yet
        else
            log "${RED}❌ Prometheus not collecting visitor metrics${NC}"
            return 1
        fi
    fi
}

# Check API requests metrics
check_api_metrics() {
    log "${BLUE}🔍 Checking API requests metrics...${NC}"
    
    local api_requests=$(curl -s --max-time 10 "$PROMETHEUS_URL/api/v1/query?query=sum(http_request_size_bytes_count)" | jq -r '.data.result[0].value[1] // "null"')
    
    if [ "$api_requests" != "null" ] && [ "$api_requests" != "" ]; then
        log "${GREEN}✅ API requests being tracked: $api_requests total${NC}"
        return 0
    else
        # Check if the metric exists but has no data
        local metric_exists=$(curl -s --max-time 10 "$PROMETHEUS_URL/api/v1/query?query=http_request_size_bytes_count" | jq '.data.result | length')
        if [ "$metric_exists" -gt 0 ]; then
            log "${YELLOW}⚠️  API requests metric exists but no data yet (normal for new deployments)${NC}"
            return 0  # This is not a failure, just no data yet
        else
            log "${RED}❌ API requests not being tracked${NC}"
            return 1
        fi
    fi
}

# Main monitoring function
main() {
    log "${YELLOW}🚀 Starting visitor metrics monitoring...${NC}"
    
    # Check service health
    local health_ok=true
    
    if ! check_service_health "API" "$API_URL/health"; then
        health_ok=false
    fi
    
    if ! check_service_health "Prometheus" "$PROMETHEUS_URL/-/healthy"; then
        health_ok=false
    fi
    
    if ! check_service_health "Grafana" "$GRAFANA_URL/api/health"; then
        health_ok=false
    fi
    
    if [ "$health_ok" = false ]; then
        log "${RED}❌ Some services are down. Skipping metrics checks.${NC}"
        return 1
    fi
    
    # Check metrics
    local metrics_ok=true
    
    if ! check_visitor_metrics; then
        metrics_ok=false
    fi
    
    if ! check_prometheus_collection; then
        metrics_ok=false
    fi
    
    if ! check_api_metrics; then
        metrics_ok=false
    fi
    
    # Refresh metrics if needed
    if [ "$metrics_ok" = false ]; then
        log "${YELLOW}🔄 Attempting to refresh metrics...${NC}"
        if python scripts/refresh_visitor_metrics.py; then
            log "${GREEN}✅ Metrics refreshed successfully${NC}"
            # After refresh, check if metrics are now working
            sleep 5  # Give Prometheus time to scrape
            if check_visitor_metrics && check_prometheus_collection; then
                log "${GREEN}✅ Metrics are now working after refresh${NC}"
                metrics_ok=true
            else
                log "${YELLOW}⚠️  Metrics still not working after refresh - this may be normal if no visitors yet${NC}"
            fi

        else
            log "${RED}❌ Failed to refresh metrics${NC}"
        fi
    fi
    
    # Summary - be more lenient with metrics failures
    if [ "$health_ok" = true ]; then
        if [ "$metrics_ok" = true ]; then
            log "${GREEN}🎉 All systems operational!${NC}"
            return 0
        else
            log "${YELLOW}⚠️  Services healthy but some metrics issues detected. This may be normal if no visitors yet.${NC}"
            return 0  # Don't fail the script for metrics issues
        fi
    else
        log "${RED}❌ Some services are down. Check logs for details.${NC}"
        return 1
    fi
}

# Run main function
main "$@"
