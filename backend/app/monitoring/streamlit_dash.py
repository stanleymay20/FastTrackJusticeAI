import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import json
from pathlib import Path
from typing import Dict, List
import os

# Set page config
st.set_page_config(
    page_title="FastTrackJustice Monitoring",
    page_icon="⚖️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Constants
INSIGHTS_DIR = Path("backend/logs/insights")
API_BASE_URL = "http://localhost:8000/api"

def setup_page():
    st.title("⚖️ FastTrackJustice Monitoring Dashboard")

def fetch_metrics():
    try:
        response = requests.get("http://localhost:8000/api/monitoring/summary")
        return response.json()
    except Exception as e:
        st.error(f"Failed to fetch metrics: {e}")
        return None

def render_current_status(metrics):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Current Scroll Phase",
            metrics["current_scroll_phase"].upper(),
            delta="Active" if metrics["status"] == "active" else "Inactive"
        )
    
    with col2:
        st.metric(
            "Current Gate",
            f"Gate {metrics['current_gate']:.0f}",
            delta=f"Phase: {metrics['current_scroll_phase']}"
        )
    
    with col3:
        error_rate = metrics["metrics"]["health"]["error_rate"]
        st.metric(
            "System Health",
            f"{100 - error_rate:.1f}%",
            delta=f"-{error_rate:.1f}%" if error_rate > 5 else f"+{100-error_rate:.1f}%"
        )

def plot_phase_distribution(metrics):
    phase_data = pd.DataFrame(
        metrics["metrics"]["phase_distribution"].items(),
        columns=["Phase", "Count"]
    )
    
    fig = px.bar(
        phase_data,
        x="Phase",
        y="Count",
        title="Judgments by Scroll Phase",
        color="Phase",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_gate_distribution(metrics):
    gate_data = pd.DataFrame(
        metrics["metrics"]["gate_distribution"].items(),
        columns=["Gate", "Count"]
    )
    gate_data["Gate"] = pd.to_numeric(gate_data["Gate"])
    gate_data = gate_data.sort_values("Gate")
    
    fig = px.line(
        gate_data,
        x="Gate",
        y="Count",
        title="Activity by Gate Number",
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)

def display_error_rates(metrics):
    st.subheader("Error Rates by Phase")
    error_data = pd.DataFrame(
        metrics["metrics"]["error_rates"].items(),
        columns=["Phase", "Error Rate"]
    )
    error_data["Error Rate"] = error_data["Error Rate"].round(2)
    
    fig = px.scatter(
        error_data,
        x="Phase",
        y="Error Rate",
        size="Error Rate",
        color="Phase",
        title="Error Distribution Across Phases"
    )
    st.plotly_chart(fig, use_container_width=True)

def display_phase_logs(phase):
    try:
        response = requests.get(f"http://localhost:8000/api/monitoring/logs/{phase}")
        logs = response.json()
        
        if logs["count"] > 0:
            log_df = pd.DataFrame(logs["logs"])
            st.dataframe(log_df)
        else:
            st.info(f"No logs found for phase: {phase}")
    except Exception as e:
        st.error(f"Failed to fetch logs: {e}")

def load_metrics():
    """Load metrics from log files"""
    metrics_path = Path("logs/metrics")
    if not metrics_path.exists():
        return pd.DataFrame()
    
    data = []
    for file in metrics_path.glob("*.json"):
        with open(file) as f:
            metrics = json.load(f)
            data.append(metrics)
    
    return pd.DataFrame(data)

def load_judgments():
    """Load judgment data from logs"""
    judgments_path = Path("logs/judgments")
    if not judgments_path.exists():
        return pd.DataFrame()
    
    data = []
    for file in judgments_path.glob("*.json"):
        with open(file) as f:
            judgment = json.load(f)
            data.append(judgment)
    
    return pd.DataFrame(data)

def load_daily_summary() -> Dict:
    """Load the latest daily summary from the API"""
    try:
        response = requests.get(f"{API_BASE_URL}/insights/daily-summary")
        if response.status_code == 200:
            return response.json()
        return {}
    except Exception as e:
        st.error(f"Error loading daily summary: {str(e)}")
        return {}

def load_historical_insights(days: int = 7) -> List[Dict]:
    """Load historical insights from the API"""
    try:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        response = requests.get(
            f"{API_BASE_URL}/insights/historical",
            params={"start_date": start_date, "end_date": end_date}
        )
        if response.status_code == 200:
            return response.json().get("historical_insights", [])
        return []
    except Exception as e:
        st.error(f"Error loading historical insights: {str(e)}")
        return []

def plot_phase_distribution(summary: Dict):
    """Plot phase distribution as a pie chart"""
    if not summary.get("phase_distribution"):
        return
    
    df = pd.DataFrame({
        "Phase": list(summary["phase_distribution"].keys()),
        "Percentage": list(summary["phase_distribution"].values())
    })
    
    fig = px.pie(
        df,
        values="Percentage",
        names="Phase",
        title="Scroll Phase Distribution",
        color="Phase",
        color_discrete_map={
            "dawn": "#FFB6C1",
            "noon": "#FFD700",
            "dusk": "#FFA500",
            "night": "#4B0082"
        }
    )
    
    st.plotly_chart(fig)

def plot_severity_distribution(summary: Dict):
    """Plot severity distribution as a bar chart"""
    if not summary.get("severity_distribution"):
        return
    
    df = pd.DataFrame({
        "Severity": list(summary["severity_distribution"].keys()),
        "Count": list(summary["severity_distribution"].values())
    })
    
    fig = px.bar(
        df,
        x="Severity",
        y="Count",
        title="Insight Severity Distribution",
        color="Severity",
        color_discrete_map={
            "info": "#00FF00",
            "warning": "#FFA500",
            "error": "#FF0000"
        }
    )
    
    st.plotly_chart(fig)

def display_top_insights(insights: List[Dict]):
    """Display top insights in an expandable section"""
    if not insights:
        return
    
    st.subheader("🔮 Top Insights")
    for insight in insights:
        with st.expander(f"{insight.get('emoji', '🔮')} {insight.get('message', '')}"):
            st.write(f"**Type:** {insight.get('type', 'N/A')}")
            st.write(f"**Phase:** {insight.get('phase', 'N/A')}")
            st.write(f"**Severity:** {insight.get('severity', 'N/A')}")
            st.write(f"**Timestamp:** {insight.get('timestamp', 'N/A')}")

def plot_historical_trends(historical_insights: List[Dict]):
    """Plot historical trends over time"""
    if not historical_insights:
        return
    
    # Convert insights to DataFrame
    df = pd.DataFrame(historical_insights)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    
    # Count insights by date and phase
    phase_counts = df.groupby(["date", "phase"]).size().unstack(fill_value=0)
    
    fig = go.Figure()
    for phase in phase_counts.columns:
        fig.add_trace(go.Scatter(
            x=phase_counts.index,
            y=phase_counts[phase],
            name=phase.capitalize(),
            mode="lines+markers"
        ))
    
    fig.update_layout(
        title="Historical Phase Distribution",
        xaxis_title="Date",
        yaxis_title="Number of Insights",
        hovermode="x unified"
    )
    
    st.plotly_chart(fig)

def main():
    setup_page()
    
    # Add auto-refresh option
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
    refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 5, 60, 30)
    
    while True:
        metrics = fetch_metrics()
        if metrics:
            render_current_status(metrics)
            
            # Create two columns for charts
            col1, col2 = st.columns(2)
            
            with col1:
                plot_phase_distribution(metrics)
            
            with col2:
                plot_gate_distribution(metrics)
            
            display_error_rates(metrics)
            
            # Phase-specific logs
            st.subheader("Phase Logs")
            selected_phase = st.selectbox(
                "Select Scroll Phase",
                options=list(metrics["metrics"]["phase_distribution"].keys())
            )
            display_phase_logs(selected_phase)
        
        # Auto-refresh logic
        if auto_refresh:
            time.sleep(refresh_interval)
            st.experimental_rerun()
        else:
            break

    # Load data
    metrics_df = load_metrics()
    judgments_df = load_judgments()

    # Sidebar filters
    st.sidebar.header("Filters")
    time_range = st.sidebar.selectbox(
        "Time Range",
        ["Last 24 Hours", "Last Week", "Last Month", "All Time"]
    )

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Judgments",
            len(judgments_df) if not judgments_df.empty else 0
        )

    with col2:
        st.metric(
            "Avg Processing Time",
            f"{metrics_df['processing_time'].mean():.2f}s" if not metrics_df.empty else "N/A"
        )

    with col3:
        st.metric(
            "Success Rate",
            f"{(metrics_df['success'].mean() * 100):.1f}%" if not metrics_df.empty else "N/A"
        )

    with col4:
        st.metric(
            "Active Cases",
            len(judgments_df[judgments_df['status'] == 'active']) if not judgments_df.empty else 0
        )

    # Judgment Distribution
    st.header("Judgment Distribution")
    if not judgments_df.empty:
        fig = px.pie(
            judgments_df,
            names='category',
            title='Cases by Category'
        )
        st.plotly_chart(fig)
    else:
        st.info("No judgment data available")

    # Processing Time Trends
    st.header("Processing Time Trends")
    if not metrics_df.empty:
        fig = px.line(
            metrics_df,
            x='timestamp',
            y='processing_time',
            title='Processing Time Over Time'
        )
        st.plotly_chart(fig)
    else:
        st.info("No metrics data available")

    # Scroll Phase Analysis
    st.header("Scroll Phase Analysis")
    if not judgments_df.empty:
        phase_stats = judgments_df.groupby('scroll_phase').agg({
            'case_id': 'count',
            'processing_time': 'mean'
        }).reset_index()
        
        fig = px.bar(
            phase_stats,
            x='scroll_phase',
            y='case_id',
            title='Cases by Scroll Phase'
        )
        st.plotly_chart(fig)
    else:
        st.info("No scroll phase data available")

    # Recent Activity
    st.header("Recent Activity")
    if not judgments_df.empty:
        recent = judgments_df.sort_values('timestamp', ascending=False).head(5)
        for _, row in recent.iterrows():
            st.write(f"**Case {row['case_id']}** - {row['category']} ({row['scroll_phase']})")
            st.write(f"Status: {row['status']}")
            st.write("---")
    else:
        st.info("No recent activity to display")

    # System Health
    st.header("System Health")
    if not metrics_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(
                metrics_df,
                x='timestamp',
                y='memory_usage',
                title='Memory Usage'
            )
            st.plotly_chart(fig)
        
        with col2:
            fig = px.line(
                metrics_df,
                x='timestamp',
                y='cpu_usage',
                title='CPU Usage'
            )
            st.plotly_chart(fig)
    else:
        st.info("No system health data available")

    # Load daily summary
    summary = load_daily_summary()
    historical_insights = load_historical_insights()
    
    # Display summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Insights", summary.get("total_insights", 0))
    with col2:
        st.metric("Active Phases", len(summary.get("phase_distribution", {})))
    with col3:
        st.metric("Latest Update", datetime.now().strftime("%Y-%m-%d %H:%M"))
    
    # Plot distributions
    col1, col2 = st.columns(2)
    with col1:
        plot_phase_distribution(summary)
    with col2:
        plot_severity_distribution(summary)
    
    # Display historical trends
    st.subheader("📈 Historical Trends")
    plot_historical_trends(historical_insights)
    
    # Display top insights
    display_top_insights(summary.get("top_insights", []))
    
    # Add refresh button
    if st.button("🔄 Refresh Data"):
        st.experimental_rerun()

if __name__ == "__main__":
    main() 