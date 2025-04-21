import json
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from collections import defaultdict

# Configure logging
logger = logging.getLogger(__name__)

class ScrollIntelligence:
    """Generates AI-powered insights from scroll metrics and judgment data"""
    
    def __init__(self):
        self.logs_dir = Path("logs")
        self.metrics_dir = self.logs_dir / "metrics"
        self.judgments_dir = self.logs_dir / "judgments"
        self.insights_dir = self.logs_dir / "insights"
        self.insights_dir.mkdir(parents=True, exist_ok=True)
        
        # Sacred emojis for different insights
        self.EMOJI_MAP = {
            "dawn": "🌅",
            "noon": "☀️",
            "dusk": "🌆",
            "night": "🌙",
            "success": "✅",
            "warning": "⚠️",
            "error": "❌",
            "insight": "🔮",
            "trend": "📈",
            "anomaly": "🔍",
            "gate": "⚔️",
            "judgment": "⚖️",
            "scroll": "📜"
        }
    
    def load_metrics(self, days: int = 7) -> pd.DataFrame:
        """Load metrics from the last N days"""
        metrics_data = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            metrics_file = self.metrics_dir / f"metrics_{date_str}.json"
            
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    daily_metrics = json.load(f)
                    # Add date to each record
                    for metric in daily_metrics:
                        metric["date"] = date_str
                    metrics_data.extend(daily_metrics)
            
            current_date += timedelta(days=1)
        
        if not metrics_data:
            return pd.DataFrame()
        
        return pd.DataFrame(metrics_data)
    
    def load_judgments(self, days: int = 7) -> pd.DataFrame:
        """Load judgment data from the last N days"""
        judgments_data = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            judgments_file = self.judgments_dir / f"judgments_{date_str}.json"
            
            if judgments_file.exists():
                with open(judgments_file, "r") as f:
                    daily_judgments = json.load(f)
                    # Add date to each record
                    for judgment in daily_judgments:
                        judgment["date"] = date_str
                    judgments_data.extend(daily_judgments)
            
            current_date += timedelta(days=1)
        
        if not judgments_data:
            return pd.DataFrame()
        
        return pd.DataFrame(judgments_data)
    
    def analyze_phase_distribution(self, metrics_df: pd.DataFrame) -> List[Dict]:
        """Analyze distribution of judgments across scroll phases"""
        if metrics_df.empty:
            return []
        
        insights = []
        
        # Group by phase and count
        phase_counts = metrics_df.groupby("scroll_phase").size().to_dict()
        total_judgments = sum(phase_counts.values())
        
        if total_judgments == 0:
            return []
        
        # Calculate percentages
        phase_percentages = {phase: (count / total_judgments) * 100 for phase, count in phase_counts.items()}
        
        # Find dominant phase
        dominant_phase = max(phase_percentages.items(), key=lambda x: x[1])
        
        # Check for unusual distribution
        expected_percentage = 25  # Assuming equal distribution across 4 phases
        unusual_phases = [phase for phase, pct in phase_percentages.items() 
                         if abs(pct - expected_percentage) > 15]
        
        if unusual_phases:
            for phase in unusual_phases:
                pct = phase_percentages[phase]
                emoji = self.EMOJI_MAP.get(phase, self.EMOJI_MAP["insight"])
                
                if pct > expected_percentage + 15:
                    insights.append({
                        "type": "phase_distribution",
                        "phase": phase,
                        "message": f"{emoji} {phase.capitalize()} phase had {pct:.1f}% of judgments, significantly above average",
                        "severity": "info",
                        "emoji": emoji
                    })
                elif pct < expected_percentage - 15:
                    insights.append({
                        "type": "phase_distribution",
                        "phase": phase,
                        "message": f"{emoji} {phase.capitalize()} phase had only {pct:.1f}% of judgments, significantly below average",
                        "severity": "warning",
                        "emoji": emoji
                    })
        
        # Add dominant phase insight
        if dominant_phase[1] > 40:  # If any phase has more than 40% of judgments
            insights.append({
                "type": "dominant_phase",
                "phase": dominant_phase[0],
                "message": f"{self.EMOJI_MAP.get(dominant_phase[0], self.EMOJI_MAP['insight'])} {dominant_phase[0].capitalize()} phase dominated with {dominant_phase[1]:.1f}% of all judgments",
                "severity": "info",
                "emoji": self.EMOJI_MAP.get(dominant_phase[0], self.EMOJI_MAP["insight"])
            })
        
        return insights
    
    def analyze_gate_activity(self, metrics_df: pd.DataFrame) -> List[Dict]:
        """Analyze activity patterns across scroll gates"""
        if metrics_df.empty or "gate" not in metrics_df.columns:
            return []
        
        insights = []
        
        # Group by gate and count
        gate_counts = metrics_df.groupby("gate").size().to_dict()
        total_judgments = sum(gate_counts.values())
        
        if total_judgments == 0:
            return []
        
        # Calculate percentages
        gate_percentages = {gate: (count / total_judgments) * 100 for gate, count in gate_counts.items()}
        
        # Find most active gate
        most_active_gate = max(gate_percentages.items(), key=lambda x: x[1])
        
        # Check for unusual gate activity
        expected_percentage = 100 / len(gate_percentages)  # Assuming equal distribution
        unusual_gates = [gate for gate, pct in gate_percentages.items() 
                        if abs(pct - expected_percentage) > 20]
        
        if unusual_gates:
            for gate in unusual_gates:
                pct = gate_percentages[gate]
                
                if pct > expected_percentage + 20:
                    insights.append({
                        "type": "gate_activity",
                        "gate": gate,
                        "message": f"{self.EMOJI_MAP['gate']} Gate {gate} was unusually active with {pct:.1f}% of judgments",
                        "severity": "info",
                        "emoji": self.EMOJI_MAP["gate"]
                    })
                elif pct < expected_percentage - 20:
                    insights.append({
                        "type": "gate_activity",
                        "gate": gate,
                        "message": f"{self.EMOJI_MAP['gate']} Gate {gate} had low activity with only {pct:.1f}% of judgments",
                        "severity": "warning",
                        "emoji": self.EMOJI_MAP["gate"]
                    })
        
        # Add most active gate insight
        if most_active_gate[1] > 40:  # If any gate has more than 40% of judgments
            insights.append({
                "type": "most_active_gate",
                "gate": most_active_gate[0],
                "message": f"{self.EMOJI_MAP['gate']} Gate {most_active_gate[0]} was the most active with {most_active_gate[1]:.1f}% of all judgments",
                "severity": "info",
                "emoji": self.EMOJI_MAP["gate"]
            })
        
        return insights
    
    def analyze_error_rates(self, metrics_df: pd.DataFrame) -> List[Dict]:
        """Analyze error rates and identify potential issues"""
        if metrics_df.empty or "error" not in metrics_df.columns:
            return []
        
        insights = []
        
        # Calculate overall error rate
        total_requests = len(metrics_df)
        error_count = metrics_df["error"].sum() if "error" in metrics_df.columns else 0
        error_rate = (error_count / total_requests) * 100 if total_requests > 0 else 0
        
        # Check for high error rates
        if error_rate > 10:
            insights.append({
                "type": "error_rate",
                "message": f"{self.EMOJI_MAP['error']} High error rate detected: {error_rate:.1f}% of requests failed",
                "severity": "warning",
                "emoji": self.EMOJI_MAP["error"]
            })
        
        # Check for phase-specific error rates
        if "scroll_phase" in metrics_df.columns:
            phase_errors = metrics_df.groupby("scroll_phase").agg({
                "error": "sum",
                "scroll_phase": "count"
            }).rename(columns={"scroll_phase": "total"})
            
            phase_errors["error_rate"] = (phase_errors["error"] / phase_errors["total"]) * 100
            
            # Find phases with unusually high error rates
            high_error_phases = phase_errors[phase_errors["error_rate"] > 15]
            
            for phase, row in high_error_phases.iterrows():
                insights.append({
                    "type": "phase_error_rate",
                    "phase": phase,
                    "message": f"{self.EMOJI_MAP.get(phase, self.EMOJI_MAP['error'])} {phase.capitalize()} phase had a high error rate of {row['error_rate']:.1f}%",
                    "severity": "warning",
                    "emoji": self.EMOJI_MAP.get(phase, self.EMOJI_MAP["error"])
                })
        
        return insights
    
    def analyze_judgment_tone(self, judgments_df: pd.DataFrame) -> List[Dict]:
        """Analyze the tone and characteristics of judgments"""
        if judgments_df.empty:
            return []
        
        insights = []
        
        # Check for phase-specific judgment characteristics
        if "scroll_phase" in judgments_df.columns and "category" in judgments_df.columns:
            # Group by phase and category
            phase_category = judgments_df.groupby(["scroll_phase", "category"]).size().reset_index(name="count")
            
            # Find dominant categories for each phase
            for phase in judgments_df["scroll_phase"].unique():
                phase_data = phase_category[phase_category["scroll_phase"] == phase]
                
                if not phase_data.empty:
                    # Find the most common category in this phase
                    dominant_category = phase_data.loc[phase_data["count"].idxmax()]
                    
                    # Calculate percentage of this category in the phase
                    total_in_phase = phase_data["count"].sum()
                    category_percentage = (dominant_category["count"] / total_in_phase) * 100
                    
                    if category_percentage > 50:  # If a category dominates (>50%)
                        insights.append({
                            "type": "phase_category_dominance",
                            "phase": phase,
                            "category": dominant_category["category"],
                            "message": f"{self.EMOJI_MAP.get(phase, self.EMOJI_MAP['insight'])} {phase.capitalize()} phase was dominated by {dominant_category['category']} cases ({category_percentage:.1f}%)",
                            "severity": "info",
                            "emoji": self.EMOJI_MAP.get(phase, self.EMOJI_MAP["insight"])
                        })
        
        # Check for judgment length patterns
        if "text_length" in judgments_df.columns:
            avg_length_by_phase = judgments_df.groupby("scroll_phase")["text_length"].mean()
            
            # Find phases with unusually long or short judgments
            overall_avg = judgments_df["text_length"].mean()
            
            for phase, avg_length in avg_length_by_phase.items():
                if avg_length > overall_avg * 1.3:  # 30% longer than average
                    insights.append({
                        "type": "judgment_length",
                        "phase": phase,
                        "message": f"{self.EMOJI_MAP.get(phase, self.EMOJI_MAP['judgment'])} {phase.capitalize()} phase produced {avg_length/overall_avg:.1f}x longer judgments than average",
                        "severity": "info",
                        "emoji": self.EMOJI_MAP.get(phase, self.EMOJI_MAP["judgment"])
                    })
                elif avg_length < overall_avg * 0.7:  # 30% shorter than average
                    insights.append({
                        "type": "judgment_length",
                        "phase": phase,
                        "message": f"{self.EMOJI_MAP.get(phase, self.EMOJI_MAP['judgment'])} {phase.capitalize()} phase produced {avg_length/overall_avg:.1f}x shorter judgments than average",
                        "severity": "info",
                        "emoji": self.EMOJI_MAP.get(phase, self.EMOJI_MAP["judgment"])
                    })
        
        return insights
    
    def analyze_trends(self, metrics_df: pd.DataFrame, judgments_df: pd.DataFrame) -> List[Dict]:
        """Analyze trends over time"""
        insights = []
        
        if metrics_df.empty or "date" not in metrics_df.columns:
            return insights
        
        # Check for increasing/decreasing judgment volume
        daily_counts = metrics_df.groupby("date").size().reset_index(name="count")
        
        if len(daily_counts) >= 3:
            # Calculate daily change
            daily_counts["change"] = daily_counts["count"].pct_change() * 100
            
            # Check for significant changes
            recent_changes = daily_counts.tail(3)["change"].dropna()
            
            if not recent_changes.empty:
                avg_change = recent_changes.mean()
                
                if avg_change > 20:  # More than 20% increase
                    insights.append({
                        "type": "volume_trend",
                        "message": f"{self.EMOJI_MAP['trend']} Judgment volume has increased by {avg_change:.1f}% over the last 3 days",
                        "severity": "info",
                        "emoji": self.EMOJI_MAP["trend"]
                    })
                elif avg_change < -20:  # More than 20% decrease
                    insights.append({
                        "type": "volume_trend",
                        "message": f"{self.EMOJI_MAP['trend']} Judgment volume has decreased by {abs(avg_change):.1f}% over the last 3 days",
                        "severity": "warning",
                        "emoji": self.EMOJI_MAP["trend"]
                    })
        
        # Check for phase transition patterns
        if "scroll_phase" in metrics_df.columns and "date" in metrics_df.columns:
            phase_transitions = metrics_df.groupby(["date", "scroll_phase"]).size().reset_index(name="count")
            
            # Pivot to get phase counts by date
            phase_pivot = phase_transitions.pivot(index="date", columns="scroll_phase", values="count").fillna(0)
            
            # Calculate phase ratios
            for phase in phase_pivot.columns:
                phase_pivot[f"{phase}_ratio"] = phase_pivot[phase] / phase_pivot.sum(axis=1)
            
            # Check for increasing phase dominance
            for phase in phase_pivot.columns:
                if phase.endswith("_ratio"):
                    continue
                
                phase_ratio_col = f"{phase}_ratio"
                if phase_ratio_col in phase_pivot.columns:
                    recent_ratios = phase_pivot.tail(3)[phase_ratio_col]
                    
                    if len(recent_ratios) >= 2 and recent_ratios.iloc[-1] > recent_ratios.iloc[0] * 1.5:
                        # Phase has become 50% more dominant
                        insights.append({
                            "type": "phase_trend",
                            "phase": phase,
                            "message": f"{self.EMOJI_MAP.get(phase, self.EMOJI_MAP['trend'])} {phase.capitalize()} phase has become increasingly dominant, up {((recent_ratios.iloc[-1]/recent_ratios.iloc[0])-1)*100:.1f}% in the last 3 days",
                            "severity": "info",
                            "emoji": self.EMOJI_MAP.get(phase, self.EMOJI_MAP["trend"])
                        })
        
        return insights
    
    def generate_daily_summary(self) -> Dict:
        """Generate a comprehensive daily summary of scroll insights"""
        # Load data
        metrics_df = self.load_metrics(days=7)
        judgments_df = self.load_judgments(days=7)
        
        # Generate insights
        phase_insights = self.analyze_phase_distribution(metrics_df)
        gate_insights = self.analyze_gate_activity(metrics_df)
        error_insights = self.analyze_error_rates(metrics_df)
        tone_insights = self.analyze_judgment_tone(judgments_df)
        trend_insights = self.analyze_trends(metrics_df, judgments_df)
        
        # Combine all insights
        all_insights = phase_insights + gate_insights + error_insights + tone_insights + trend_insights
        
        # Generate summary
        today = datetime.now().strftime("%Y-%m-%d")
        summary = {
            "date": today,
            "insights": all_insights,
            "insight_count": len(all_insights),
            "metrics_summary": {
                "total_judgments": len(metrics_df) if not metrics_df.empty else 0,
                "error_rate": (metrics_df["error"].sum() / len(metrics_df) * 100) if not metrics_df.empty and "error" in metrics_df.columns else 0,
                "phase_distribution": metrics_df["scroll_phase"].value_counts().to_dict() if not metrics_df.empty and "scroll_phase" in metrics_df.columns else {}
            }
        }
        
        # Save summary to file
        summary_file = self.insights_dir / f"summary_{today}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Generated scroll intelligence summary for {today} with {len(all_insights)} insights")
        return summary
    
    def get_latest_summary(self) -> Dict:
        """Get the most recent summary"""
        today = datetime.now().strftime("%Y-%m-%d")
        summary_file = self.insights_dir / f"summary_{today}.json"
        
        if summary_file.exists():
            with open(summary_file, "r") as f:
                return json.load(f)
        
        # If today's summary doesn't exist, generate it
        return self.generate_daily_summary()

# Create a singleton instance
scroll_intelligence = ScrollIntelligence() 