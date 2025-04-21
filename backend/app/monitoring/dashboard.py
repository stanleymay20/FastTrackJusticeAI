from fastapi import APIRouter, HTTPException
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
from collections import defaultdict
import re

from app.utils.logger import get_logger
from app.utils.scroll_time import get_scroll_time

router = APIRouter()
log = get_logger(__name__)

class ScrollMetricsCollector:
    def __init__(self):
        self.log_dir = Path("logs")
        self.cache_duration = timedelta(minutes=5)
        self.last_cache_update = datetime.min
        self.metrics_cache = {}

    def parse_log_line(self, line: str) -> Optional[Dict]:
        """Parse a log line into structured data."""
        try:
            pattern = r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| (?P<level>\w+) \| (?P<phase>\w+) \| Gate: (?P<gate>\d+) \| (?P<message>.*)"
            match = re.match(pattern, line)
            if match:
                return match.groupdict()
            return None
        except Exception as e:
            log.error(f"Error parsing log line: {e}")
            return None

    def collect_metrics(self) -> Dict:
        """Collect metrics from log files with caching."""
        now = datetime.now()
        if (now - self.last_cache_update) < self.cache_duration:
            return self.metrics_cache

        metrics = {
            "scroll_phases": defaultdict(int),
            "gates": defaultdict(int),
            "judgment_counts": defaultdict(int),
            "error_rates": defaultdict(float),
            "response_times": defaultdict(list)
        }

        try:
            # Process scroll phase specific logs
            scroll_log_path = self.log_dir / "scroll_phases.log"
            if scroll_log_path.exists():
                with open(scroll_log_path) as f:
                    for line in f:
                        data = self.parse_log_line(line)
                        if data:
                            metrics["scroll_phases"][data["phase"]] += 1
                            metrics["gates"][data["gate"]] += 1

            # Process error logs
            error_log_path = self.log_dir / "errors.log"
            if error_log_path.exists():
                with open(error_log_path) as f:
                    error_count = sum(1 for _ in f)
                metrics["error_count"] = error_count

            # Calculate error rates per phase
            total_requests = sum(metrics["scroll_phases"].values())
            if total_requests > 0:
                for phase in metrics["scroll_phases"]:
                    phase_requests = metrics["scroll_phases"][phase]
                    metrics["error_rates"][phase] = (metrics.get("error_count", 0) / total_requests) * 100

            self.metrics_cache = metrics
            self.last_cache_update = now
            return metrics

        except Exception as e:
            log.error(f"Error collecting metrics: {e}")
            return metrics

metrics_collector = ScrollMetricsCollector()

@router.get("/api/monitoring/summary")
async def get_monitoring_summary():
    """Get a summary of system metrics and scroll phase statistics."""
    try:
        metrics = metrics_collector.collect_metrics()
        current_scroll = get_scroll_time()

        return {
            "status": "active",
            "current_scroll_phase": current_scroll["phase"],
            "current_gate": current_scroll["gate"],
            "metrics": {
                "total_judgments": sum(metrics["scroll_phases"].values()),
                "phase_distribution": dict(metrics["scroll_phases"]),
                "gate_distribution": dict(metrics["gates"]),
                "error_rates": dict(metrics["error_rates"]),
                "health": {
                    "error_rate": metrics.get("error_count", 0) / max(sum(metrics["scroll_phases"].values()), 1) * 100,
                    "active_phases": len(metrics["scroll_phases"])
                }
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        log.error(f"Error generating monitoring summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/monitoring/logs/{phase}")
async def get_phase_logs(phase: str, limit: int = 100):
    """Get recent logs for a specific scroll phase."""
    try:
        log_path = Path("logs") / "scroll_phases.log"
        if not log_path.exists():
            return {"logs": [], "count": 0}

        phase_logs = []
        with open(log_path) as f:
            for line in f:
                if phase.lower() in line.lower():
                    phase_logs.append(metrics_collector.parse_log_line(line))
                if len(phase_logs) >= limit:
                    break

        return {
            "logs": phase_logs,
            "count": len(phase_logs)
        }
    except Exception as e:
        log.error(f"Error retrieving phase logs: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 