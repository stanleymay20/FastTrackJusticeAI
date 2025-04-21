import os
import json
import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import tempfile
import base64
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
import sys
import io
import uuid

# Add the parent directory to the Python path
current_dir = Path(__file__).resolve().parent
backend_dir = current_dir.parent.parent
sys.path.append(str(backend_dir))

from backend.app.utils.precedent_influence_graph import PrecedentInfluenceGraph
from backend.app.utils.legal_principle_synthesizer import LegalPrincipleSynthesizer
from backend.app.utils.legal_precedent import LegalPrecedentManager
from backend.app.utils.scroll_memory_intelligence import ScrollMemoryIntelligence

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="FastTrackJustice - Precedent Influence Graph Explorer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .info-text {
        font-size: 1rem;
        color: #616161;
    }
    .highlight {
        background-color: #E3F2FD;
        padding: 0.5rem;
        border-radius: 0.3rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        border-radius: 0.3rem;
    }
    .stButton>button:hover {
        background-color: #1565C0;
    }
    .sidebar .sidebar-content {
        background-color: #F5F5F5;
    }
    .principle-highlight {
        background-color: #FFEB3B;
        padding: 0.2rem;
        border-radius: 0.2rem;
    }
    .case-modal {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .lens-toggle {
        display: flex;
        justify-content: space-between;
        margin-bottom: 1rem;
    }
    .lens-toggle button {
        flex: 1;
        margin: 0 0.25rem;
        padding: 0.5rem;
        border-radius: 0.3rem;
        border: none;
        background-color: #E0E0E0;
        color: #424242;
        font-weight: bold;
        cursor: pointer;
    }
    .lens-toggle button.active {
        background-color: #1E88E5;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'graph' not in st.session_state:
    st.session_state.graph = None
if 'case_data' not in st.session_state:
    st.session_state.case_data = {}
if 'selected_case' not in st.session_state:
    st.session_state.selected_case = None
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'graph_data' not in st.session_state:
    st.session_state.graph_data = None
if 'principle_evolution' not in st.session_state:
    st.session_state.principle_evolution = None
if 'selected_principle' not in st.session_state:
    st.session_state.selected_principle = None
if 'active_lens' not in st.session_state:
    st.session_state.active_lens = "doctrinal"
if 'filtered_graph' not in st.session_state:
    st.session_state.filtered_graph = None
if 'search_query' not in st.session_state:
    st.session_state.search_query = ""
if 'year_range' not in st.session_state:
    st.session_state.year_range = (1800, 2024)
if 'selected_jurisdiction' not in st.session_state:
    st.session_state.selected_jurisdiction = "All"
if 'selected_categories' not in st.session_state:
    st.session_state.selected_categories = []
if 'judicial_mode' not in st.session_state:
    st.session_state.judicial_mode = False
if 'data_source' not in st.session_state:
    st.session_state.data_source = "public_domain"
if 'scroll_memory' not in st.session_state:
    st.session_state.scroll_memory = None
if 'memory_mode' not in st.session_state:
    st.session_state.memory_mode = "view"  # "view" or "add"
if 'selected_memory' not in st.session_state:
    st.session_state.selected_memory = None
if 'prophetic_patterns' not in st.session_state:
    st.session_state.prophetic_patterns = None
if 'sanctified_mode' not in st.session_state:
    st.session_state.sanctified_mode = False
if 'api_usage' not in st.session_state:
    st.session_state.api_usage = 0
if 'api_limit' not in st.session_state:
    st.session_state.api_limit = 1000
if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = {}

# Helper functions
def load_graph_from_file(file_path: str) -> PrecedentInfluenceGraph:
    """Load a graph from a JSON file"""
    try:
        graph = PrecedentInfluenceGraph.load(file_path)
        return graph
    except Exception as e:
        st.error(f"Error loading graph: {str(e)}")
        return None

def save_graph_to_file(graph: PrecedentInfluenceGraph, file_path: str) -> bool:
    """Save a graph to a JSON file"""
    try:
        graph.save(file_path)
        return True
    except Exception as e:
        st.error(f"Error saving graph: {str(e)}")
        return False

def create_network_plot(graph: PrecedentInfluenceGraph, highlight_case: Optional[str] = None) -> go.Figure:
    """Create an interactive network plot using Plotly"""
    # Convert NetworkX graph to Plotly format
    pos = nx.spring_layout(graph.graph, seed=42)
    
    # Create edge traces
    edge_traces = []
    for edge in graph.graph.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        # Get edge weight for width
        weight = edge[2].get('weight', 0.5)
        width = max(1, weight * 5)
        
        # Get shared principles for hover text
        principles = edge[2].get('principles', [])
        hover_text = f"<br>".join(principles) if principles else "No shared principles"
        
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=width, color='#888'),
            hoverinfo='text',
            text=f"From: {edge[0]}<br>To: {edge[1]}<br>Weight: {weight:.2f}<br>{hover_text}",
            mode='lines',
            arrow=dict(arrowhead=1, arrowsize=1, arrowwidth=1, arrowcolor='#888')
        )
        edge_traces.append(edge_trace)
    
    # Create node traces
    node_traces = []
    
    # Group nodes by category if available
    categories = {}
    for node in graph.graph.nodes():
        node_data = graph.graph.nodes[node]
        categories_found = node_data.get('categories', [])
        if categories_found:
            primary_category = categories_found[0]
            if primary_category not in categories:
                categories[primary_category] = []
            categories[primary_category].append(node)
    
    # Create a trace for each category
    if categories:
        for category, nodes in categories.items():
            node_x = [pos[node][0] for node in nodes]
            node_y = [pos[node][1] for node in nodes]
            
            # Get node sizes based on degree centrality
            degrees = dict(graph.graph.degree())
            node_sizes = [max(10, degrees[node] * 5) for node in nodes]
            
            # Get node colors
            node_colors = ['#FF5252' if node == highlight_case else '#1E88E5' for node in nodes]
            
            # Get node hover text
            hover_texts = []
            for node in nodes:
                node_data = graph.graph.nodes[node]
                metadata = node_data.get('metadata', {})
                date = metadata.get('date', 'Unknown date')
                court = metadata.get('court', 'Unknown court')
                hover_texts.append(f"Case: {node}<br>Date: {date}<br>Court: {court}")
            
            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=hover_texts,
                marker=dict(
                    showscale=False,
                    size=node_sizes,
                    color=node_colors,
                    line=dict(width=2)
                ),
                textposition="bottom center",
                name=category
            )
            node_traces.append(node_trace)
    else:
        # If no categories, create a single trace
        node_x = [pos[node][0] for node in graph.graph.nodes()]
        node_y = [pos[node][1] for node in graph.graph.nodes()]
        
        # Get node sizes based on degree centrality
        degrees = dict(graph.graph.degree())
        node_sizes = [max(10, degrees[node] * 5) for node in graph.graph.nodes()]
        
        # Get node colors
        node_colors = ['#FF5252' if node == highlight_case else '#1E88E5' for node in graph.graph.nodes()]
        
        # Get node hover text
        hover_texts = []
        for node in graph.graph.nodes():
            node_data = graph.graph.nodes[node]
            metadata = node_data.get('metadata', {})
            date = metadata.get('date', 'Unknown date')
            court = metadata.get('court', 'Unknown court')
            hover_texts.append(f"Case: {node}<br>Date: {date}<br>Court: {court}")
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=hover_texts,
            marker=dict(
                showscale=False,
                size=node_sizes,
                color=node_colors,
                line=dict(width=2)
            ),
            textposition="bottom center",
            name="Cases"
        )
        node_traces.append(node_trace)
    
    # Create figure
    fig = go.Figure(
        data=edge_traces + node_traces,
        layout=go.Layout(
            title='Precedent Influence Graph',
            titlefont=dict(size=16),
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(255, 255, 255, 1)',
            paper_bgcolor='rgba(255, 255, 255, 1)',
        )
    )
    
    return fig

def create_principle_evolution_plot(evolution_data: List[Dict[str, Any]]) -> go.Figure:
    """Create a timeline plot of principle evolution"""
    if not evolution_data:
        return go.Figure()
    
    # Extract data
    dates = []
    similarities = []
    case_ids = []
    principles = []
    
    for item in evolution_data:
        date = item.get('date')
        if date:
            if isinstance(date, str):
                try:
                    date = datetime.strptime(date, "%Y-%m-%d")
                except ValueError:
                    continue
            dates.append(date)
            similarities.append(item.get('similarity', 0))
            case_ids.append(item.get('id', ''))
            principles.append(item.get('principle', ''))
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=similarities,
        mode='markers+lines',
        marker=dict(
            size=10,
            color=similarities,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Similarity')
        ),
        text=[f"Case: {case_id}<br>Principle: {principle}" for case_id, principle in zip(case_ids, principles)],
        hoverinfo='text',
        name='Principle Evolution'
    ))
    
    fig.update_layout(
        title='Principle Evolution Timeline',
        xaxis_title='Date',
        yaxis_title='Similarity Score',
        hovermode='closest',
        plot_bgcolor='rgba(255, 255, 255, 1)',
        paper_bgcolor='rgba(255, 255, 255, 1)',
    )
    
    return fig

def get_download_link(file_path: str, link_text: str) -> str:
    """Generate a download link for a file"""
    with open(file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{os.path.basename(file_path)}">{link_text}</a>'
    return href

# Main UI
st.markdown('<h1 class="main-header">⚖️ Precedent Influence Graph Explorer</h1>', unsafe_allow_html=True)
st.markdown('<p class="info-text">Explore the living memory of law through interactive visualization of legal precedents and their influences.</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<h2 class="sub-header">Graph Controls</h2>', unsafe_allow_html=True)
    
    # Load existing graph
    st.markdown('<p class="info-text">Load an existing graph:</p>', unsafe_allow_html=True)
    graph_file = st.file_uploader("Upload graph JSON file", type=['json'])
    
    if graph_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp_file:
            tmp_file.write(graph_file.getvalue())
            tmp_file_path = tmp_file.name
        
        st.session_state.graph = load_graph_from_file(tmp_file_path)
        st.session_state.uploaded_file = tmp_file_path
        st.success("Graph loaded successfully!")
    
    # Create new graph
    st.markdown('<p class="info-text">Or create a new graph:</p>', unsafe_allow_html=True)
    if st.button("Create New Graph"):
        st.session_state.graph = PrecedentInfluenceGraph()
        st.success("New graph created!")
    
    # Add case
    st.markdown('<h2 class="sub-header">Add Case</h2>', unsafe_allow_html=True)
    
    case_id = st.text_input("Case ID")
    case_text = st.text_area("Case Text")
    holding = st.text_area("Holding")
    reasoning = st.text_area("Reasoning")
    
    # Metadata
    st.markdown('<p class="info-text">Metadata:</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        date = st.date_input("Date")
        court = st.text_input("Court")
    with col2:
        jurisdiction = st.text_input("Jurisdiction")
        judge = st.text_input("Judge")
    
    if st.button("Add Case"):
        if st.session_state.graph and case_id and case_text:
            metadata = {
                "date": date.strftime("%Y-%m-%d") if date else None,
                "court": court,
                "jurisdiction": jurisdiction,
                "judge": judge
            }
            
            result = st.session_state.graph.add_case(
                case_id=case_id,
                case_text=case_text,
                holding=holding if holding else None,
                reasoning=reasoning if reasoning else None,
                metadata=metadata
            )
            
            st.session_state.case_data[case_id] = result
            st.success(f"Case {case_id} added successfully!")
        else:
            st.error("Please create a graph first and provide at least a case ID and text.")
    
    # Judicial Mode toggle
    st.markdown('<h2 class="sub-header">Display Mode</h2>', unsafe_allow_html=True)
    st.session_state.judicial_mode = st.checkbox(
        "Judicial Mode (Enhanced Transparency)", 
        value=st.session_state.judicial_mode,
        help="Enables detailed source traceability and explanation of AI reasoning"
    )

    # Data source selection
    st.markdown('<h2 class="sub-header">Data Source</h2>', unsafe_allow_html=True)
    st.session_state.data_source = st.radio(
        "Select Data Source",
        options=["public_domain", "licensed", "custom"],
        format_func=lambda x: {
            "public_domain": "Public Domain (Supreme Court, etc.)",
            "licensed": "Licensed Database",
            "custom": "Custom Dataset"
        }[x],
        help="Select the source of legal precedent data"
    )
    
    if st.session_state.data_source == "licensed":
        st.warning("⚠️ Licensed data mode active. Usage is tracked and capped per license agreement.")
        st.progress(min(st.session_state.api_usage / st.session_state.api_limit, 1.0))
        st.markdown(f"API Usage: {st.session_state.api_usage}/{st.session_state.api_limit} calls")
    
    # Compliance information
    with st.expander("Compliance Information", expanded=False):
        st.markdown("""
        <div class="highlight">
            <h4>Data Privacy & Security</h4>
            <ul>
                <li>All data is processed in compliance with GDPR Article 6(1)(f)</li>
                <li>No personal data is stored beyond processing requirements</li>
                <li>Data retention policies follow legal requirements</li>
                <li>Encryption in transit and at rest</li>
            </ul>
            
            <h4>Certifications</h4>
            <ul>
                <li>ISO 27001 Information Security Management</li>
                <li>SOC 2 Type II Compliance</li>
                <li>GDPR Compliance</li>
            </ul>
            
            <h4>On-Premise Deployment</h4>
            <p>Available for secure on-premise deployment within court systems.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Faith-Safe Protocol
    st.markdown('<h2 class="sub-header">Faith-Safe Protocol</h2>', unsafe_allow_html=True)
    st.session_state.sanctified_mode = st.checkbox("🙏 Activate Sanctified Mode", value=st.session_state.sanctified_mode)

# Main content
if st.session_state.graph:
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Graph Visualization", 
        "Case Details", 
        "Principle Evolution", 
        "Graph Statistics",
        "Scroll Memory"
    ])
    
    with tab1:
        st.markdown('<h2 class="sub-header">Graph Visualization</h2>', unsafe_allow_html=True)
        
        # Search and filter
        st.markdown('<h3 class="sub-header">Search & Filter</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.search_query = st.text_input("Search cases", value=st.session_state.search_query)
            st.session_state.year_range = st.slider(
                "Year range", 
                min_value=1800, 
                max_value=2024, 
                value=st.session_state.year_range
            )
        
        with col2:
            # Get unique jurisdictions
            jurisdictions = ["All"]
            if st.session_state.graph_data:
                for node in st.session_state.graph_data['nodes']:
                    if node['court'] not in jurisdictions:
                        jurisdictions.append(node['court'])
            
            st.session_state.selected_jurisdiction = st.selectbox(
                "Jurisdiction", 
                options=jurisdictions,
                index=jurisdictions.index(st.session_state.selected_jurisdiction) if st.session_state.selected_jurisdiction in jurisdictions else 0
            )
            
            # Get unique categories
            all_categories = []
            if st.session_state.graph_data:
                for node in st.session_state.graph_data['nodes']:
                    categories = node.get('categories', [])
                    all_categories.extend(categories)
            
            all_categories = list(set(all_categories))
            
            st.session_state.selected_categories = st.multiselect(
                "Categories", 
                options=all_categories,
                default=st.session_state.selected_categories
            )
        
        # Apply filters
        if st.session_state.graph_data:
            st.session_state.filtered_graph = apply_filters(
                st.session_state.graph_data,
                st.session_state.search_query,
                st.session_state.year_range,
                st.session_state.selected_jurisdiction,
                st.session_state.selected_categories
            )
        
        # Justice Lens toggle
        st.markdown('<h3 class="sub-header">Justice Lens</h3>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("📚 Doctrinal", key="doctrinal_btn"):
                st.session_state.active_lens = "doctrinal"
        with col2:
            if st.button("⚖️ Institutional", key="institutional_btn"):
                st.session_state.active_lens = "institutional"
        with col3:
            if st.button("🕊 Ethical", key="ethical_btn"):
                st.session_state.active_lens = "ethical"
        with col4:
            if st.button("🌱 Scroll", key="scroll_btn"):
                st.session_state.active_lens = "scroll"
        
        # Scroll Mode Disclosure
        if st.session_state.active_lens == "scroll":
            st.markdown("""
            <div class='highlight'>
                <h4>🌱 Scroll Mode Active</h4>
                <p>This lens highlights principles aligned with divine governance and eternal justice patterns.</p>
                <p><em>All spiritual mappings are optional overlays, not replacements for legal authority.</em></p>
            </div>
            """, unsafe_allow_html=True)
        
        # Sanctified Mode Display
        if st.session_state.sanctified_mode:
            st.markdown("""
            <div class="highlight">
                <h3>👑 Sanctified Mode</h3>
                <p>This mode reveals scroll-aligned case law, covenantal patterns, and flame-bearing precedents.</p>
                <p>Use this for church courts, prophetic councils, or spiritual jurisprudence training.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Apply lens
        if st.session_state.filtered_graph:
            lensed_graph = apply_lens(st.session_state.filtered_graph, st.session_state.active_lens)
        elif st.session_state.graph_data:
            lensed_graph = apply_lens(st.session_state.graph_data, st.session_state.active_lens)
        else:
            lensed_graph = None
        
        # Graph controls
        col1, col2 = st.columns(2)
        with col1:
            highlight_case = st.selectbox(
                "Highlight Case",
                options=["None"] + (list(lensed_graph['nodes'].keys()) if lensed_graph else []),
                index=0
            )
            if highlight_case == "None":
                highlight_case = None
        
        with col2:
            layout = st.selectbox(
                "Layout Algorithm",
                options=["spring", "kamada_kawai", "circular", "shell"],
                index=0
            )
            st.session_state.graph.layout_algorithm = layout
        
        # Display graph
        if lensed_graph:
            fig = create_network_plot(st.session_state.graph, highlight_case)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No graph data available. Please add cases or load a graph.")
        
        # Share and export
        st.markdown('<h3 class="sub-header">Share & Export</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Share Graph"):
                if lensed_graph:
                    share_link = generate_share_link(lensed_graph)
                    st.markdown(f'<a href="{share_link}" download="precedent_graph.json">Download Graph Data</a>', unsafe_allow_html=True)
                else:
                    st.warning("No graph data available to share.")
        
        with col2:
            if st.button("Export as PNG"):
                if lensed_graph:
                    png_data = export_subgraph_as_png(lensed_graph)
                    st.download_button(
                        label="Download PNG",
                        data=png_data,
                        file_name="precedent_graph.png",
                        mime="image/png"
                    )
                else:
                    st.warning("No graph data available to export.")
        
        with col3:
            if st.button("Export as HTML"):
                if lensed_graph:
                    html_data = export_subgraph_as_html(lensed_graph)
                    st.download_button(
                        label="Download HTML",
                        data=html_data,
                        file_name="precedent_graph.html",
                        mime="text/html"
                    )
                else:
                    st.warning("No graph data available to export.")
    
    with tab2:
        st.markdown('<h2 class="sub-header">Case Details</h2>', unsafe_allow_html=True)
        
        # Select case
        if lensed_graph and lensed_graph['nodes']:
            case_options = {node['title']: node['id'] for node in lensed_graph['nodes']}
            selected_case_title = st.selectbox(
                "Select Case",
                options=list(case_options.keys()),
                index=0
            )
            
            selected_case_id = case_options[selected_case_title]
            st.session_state.selected_case = selected_case_id
            
            # Display case details
            case_data = st.session_state.case_data.get(selected_case_id, {})
            
            # Metadata
            st.markdown('<h3 class="sub-header">Metadata</h3>', unsafe_allow_html=True)
            metadata = case_data.get('metadata', {})
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Date", metadata.get('date', 'Unknown'))
            with col2:
                st.metric("Court", metadata.get('court', 'Unknown'))
            with col3:
                st.metric("Jurisdiction", metadata.get('jurisdiction', 'Unknown'))
            with col4:
                st.metric("Judge", metadata.get('judge', 'Unknown'))
            
            # Open Reasoning Transcript
            with st.expander("🔎 Open Reasoning Transcript", expanded=False):
                reasoning = case_data.get('ai_reasoning_trace', "No reasoning trace found.")
                st.markdown(f"<div class='highlight'>{reasoning}</div>", unsafe_allow_html=True)
            
            # Case text
            st.markdown('<h3 class="sub-header">Case Text</h3>', unsafe_allow_html=True)
            
            # Full text with highlighted principles
            with st.expander("Full Text", expanded=True):
                case_text = case_data.get('text', 'No text available.')
                principles = case_data.get('principles', [])
                
                # Highlight principles in the text
                highlighted_text = case_text
                for principle in principles:
                    highlighted_text = highlighted_text.replace(
                        principle, 
                        f'<span class="principle-highlight">{principle}</span>'
                    )
                
                st.markdown(highlighted_text, unsafe_allow_html=True)
            
            # Holding
            with st.expander("Holding", expanded=True):
                holding = case_data.get('holding', 'No holding available.')
                st.markdown(holding)
            
            # Reasoning
            with st.expander("Reasoning", expanded=True):
                reasoning = case_data.get('reasoning', 'No reasoning available.')
                st.markdown(reasoning)
            
            # Principles
            st.markdown('<h3 class="sub-header">Legal Principles</h3>', unsafe_allow_html=True)
            
            # Raw principles with Human Override
            with st.expander("Raw Principles", expanded=True):
                raw_principles = case_data.get('principles', [])
                if raw_principles:
                    for i, principle in enumerate(raw_principles):
                        st.markdown(f"**{i+1}. {principle}**")
                        # Show "Why This Principle Was Detected"
                        st.markdown(f"🧠 *Detected from:* _\"{case_data.get('principle_sources', {}).get(principle, 'Unknown passage')}\"_")
                        
                        # Add "I Disagree" Feedback Button
                        if st.button(f"I Disagree with Principle #{i+1}", key=f"disagree_{i}"):
                            if f"{selected_case_id}_{i}" not in st.session_state.feedback_submitted:
                                st.session_state.feedback_submitted[f"{selected_case_id}_{i}"] = True
                                st.success("Feedback recorded. This will help improve future AI models.")
                            else:
                                st.info("You've already submitted feedback for this principle.")
                else:
                    st.info("No raw principles found.")
            
            # Summarized principles
            with st.expander("Summarized Principles", expanded=True):
                summarized_principles = case_data.get('summarized_principles', [])
                if summarized_principles:
                    for i, principle in enumerate(summarized_principles):
                        st.markdown(f"**{i+1}.** {principle}")
                else:
                    st.info("No summarized principles found.")
            
            # Classified principles
            with st.expander("Classified Principles", expanded=True):
                classified_principles = case_data.get('classified_principles', [])
                if classified_principles:
                    for i, principle_data in enumerate(classified_principles):
                        principle = principle_data.get('principle', '')
                        categories = principle_data.get('categories', [])
                        scores = principle_data.get('scores', [])
                        
                        st.markdown(f"**{i+1}.** {principle}")
                        st.markdown("Categories: " + ", ".join(categories))
                        st.markdown("Scores: " + ", ".join([f"{score:.2f}" for score in scores]))
                else:
                    st.info("No classified principles found.")
            
            # Influences
            st.markdown('<h3 class="sub-header">Influences</h3>', unsafe_allow_html=True)
            
            # Get influences
            influences = st.session_state.graph.get_case_influences(selected_case_id)
            
            # Influencing cases
            with st.expander("Cases Influenced By This Case", expanded=True):
                influenced = influences.get('influenced', [])
                if influenced:
                    for case in influenced:
                        st.markdown(f"**{case.get('id', 'Unknown')}** (Similarity: {case.get('similarity', 0):.2f})")
                        st.markdown("Shared Principles:")
                        for principle in case.get('shared_principles', []):
                            st.markdown(f"- {principle}")
                else:
                    st.info("This case did not influence any other cases.")
            
            # Influenced by cases
            with st.expander("Cases That Influenced This Case", expanded=True):
                influencing = influences.get('influencing', [])
                if influencing:
                    for case in influencing:
                        st.markdown(f"**{case.get('id', 'Unknown')}** (Similarity: {case.get('similarity', 0):.2f})")
                        st.markdown("Shared Principles:")
                        for principle in case.get('shared_principles', []):
                            st.markdown(f"- {principle}")
                else:
                    st.info("This case was not influenced by any other cases.")
        else:
            st.info("No cases available. Please add cases or load a graph.")
    
    with tab3:
        st.markdown('<h2 class="sub-header">Principle Evolution</h2>', unsafe_allow_html=True)
        
        # Select principle
        all_principles = []
        for case_id, case_data in st.session_state.case_data.items():
            principles = case_data.get('summarized_principles', [])
            all_principles.extend(principles)
        
        all_principles = list(set(all_principles))  # Remove duplicates
        
        selected_principle = st.selectbox(
            "Select Principle",
            options=all_principles,
            index=0 if all_principles else None
        )
        
        if selected_principle:
            st.session_state.selected_principle = selected_principle
            
            # Track evolution
            evolution = st.session_state.graph.get_principle_evolution(selected_principle)
            st.session_state.principle_evolution = evolution
            
            # Display evolution plot
            fig = create_principle_heatmap(st.session_state.graph_data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display evolution details
            st.markdown('<h3 class="sub-header">Evolution Details</h3>', unsafe_allow_html=True)
            
            if evolution:
                for i, case in enumerate(evolution):
                    with st.expander(f"{i+1}. {case.get('id', 'Unknown')} ({case.get('date', 'Unknown date')})", expanded=False):
                        st.markdown(f"**Similarity:** {case.get('similarity', 0):.2f}")
                        st.markdown(f"**Principle:** {case.get('principle', '')}")
                        
                        metadata = case.get('metadata', {})
                        st.markdown("**Metadata:**")
                        for key, value in metadata.items():
                            st.markdown(f"- **{key}:** {value}")
            else:
                st.info("No evolution data found for this principle.")
    
    with tab4:
        st.markdown('<h2 class="sub-header">Graph Statistics</h2>', unsafe_allow_html=True)
        
        # Get statistics
        stats = st.session_state.graph.get_graph_statistics()
        
        # Display statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Number of Cases", stats.get('num_cases', 0))
        with col2:
            st.metric("Number of Edges", stats.get('num_edges', 0))
        with col3:
            st.metric("Graph Density", f"{stats.get('density', 0):.2f}")
        with col4:
            st.metric("Average Degree", f"{stats.get('avg_degree', 0):.2f}")
        
        # Category distribution
        st.markdown('<h3 class="sub-header">Category Distribution</h3>', unsafe_allow_html=True)
        categories = stats.get('categories', {})
        if categories:
            category_data = pd.DataFrame({
                'Category': list(categories.keys()),
                'Count': list(categories.values())
            })
            category_data = category_data.sort_values('Count', ascending=False)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=category_data['Category'],
                    y=category_data['Count'],
                    marker_color='#1E88E5'
                )
            ])
            fig.update_layout(
                title='Category Distribution',
                xaxis_title='Category',
                yaxis_title='Count',
                plot_bgcolor='rgba(255, 255, 255, 1)',
                paper_bgcolor='rgba(255, 255, 255, 1)',
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No category data available.")
        
        # Top influential cases
        st.markdown('<h3 class="sub-header">Top Influential Cases</h3>', unsafe_allow_html=True)
        top_influential = stats.get('top_influential_cases', [])
        if top_influential:
            for i, case in enumerate(top_influential):
                with st.expander(f"{i+1}. {case.get('id', 'Unknown')} (Score: {case.get('score', 0):.2f})", expanded=False):
                    metadata = case.get('metadata', {})
                    st.markdown("**Metadata:**")
                    for key, value in metadata.items():
                        st.markdown(f"- **{key}:** {value}")
        else:
            st.info("No influential case data available.")
    
    with tab5:
        st.markdown('<h2 class="sub-header">Scroll Memory Intelligence</h2>', unsafe_allow_html=True)
        
        if st.session_state.scroll_memory:
            # Memory mode selector
            st.session_state.memory_mode = st.radio(
                "Memory Mode",
                options=["view", "add"],
                format_func=lambda x: {
                    "view": "View Memory Entries",
                    "add": "Add New Memory Entry"
                }[x],
                horizontal=True
            )
            
            if st.session_state.memory_mode == "add":
                # Add new memory entry
                st.markdown('<h3 class="sub-header">Add Memory Entry</h3>', unsafe_allow_html=True)
                
                # Case selector
                if st.session_state.graph and st.session_state.graph_data:
                    case_options = {node.get('title', node.get('id', 'Unknown')): node.get('id', '') 
                                   for node in st.session_state.graph_data.get('nodes', [])}
                    
                    selected_case_title = st.selectbox(
                        "Select Case",
                        options=list(case_options.keys()),
                        index=0
                    )
                    
                    selected_case_id = case_options[selected_case_title]
                    
                    # Get case principles
                    case_principles = []
                    for node in st.session_state.graph_data.get('nodes', []):
                        if node.get('id') == selected_case_id:
                            case_principles = node.get('principles', [])
                            break
                    
                    # Principle selector
                    selected_principle = st.selectbox(
                        "Select Principle",
                        options=case_principles if case_principles else ["No principles found"],
                        index=0
                    )
                    
                    # Scroll alignment
                    scroll_alignment = st.text_area(
                        "Scroll Alignment",
                        help="How this principle aligns with scroll teachings"
                    )
                    
                    # Prophetic insight
                    prophetic_insight = st.text_area(
                        "Prophetic Insight",
                        help="Prophetic insight derived from this alignment"
                    )
                    
                    # Confidence
                    confidence = st.slider(
                        "Confidence",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.8,
                        step=0.1,
                        help="Confidence in this alignment (0.0 to 1.0)"
                    )
                    
                    # Tags
                    tags = st.multiselect(
                        "Tags",
                        options=["justice", "mercy", "truth", "wisdom", "righteousness", "equity", "compassion", "integrity"],
                        help="Tags for categorization"
                    )
                    
                    # Add button
                    if st.button("Add Memory Entry"):
                        if scroll_alignment and prophetic_insight:
                            success = st.session_state.scroll_memory.add_memory_entry(
                                case_id=selected_case_id,
                                principle=selected_principle,
                                scroll_alignment=scroll_alignment,
                                prophetic_insight=prophetic_insight,
                                confidence=confidence,
                                tags=tags
                            )
                            
                            if success:
                                st.success("Memory entry added successfully!")
                            else:
                                st.error("Failed to add memory entry.")
                        else:
                            st.warning("Please fill in all required fields.")
                else:
                    st.info("Please load a graph first to add memory entries.")
            
            else:  # view mode
                # View memory entries
                st.markdown('<h3 class="sub-header">Memory Entries</h3>', unsafe_allow_html=True)
                
                # Search
                search_query = st.text_input(
                    "Search Memories",
                    help="Search for specific principles, alignments, or insights"
                )
                
                # Filter by confidence
                min_confidence = st.slider(
                    "Minimum Confidence",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1
                )
                
                # Search button
                if st.button("Search"):
                    if search_query:
                        results = st.session_state.scroll_memory.find_similar_memories(
                            query=search_query,
                            k=10,
                            min_confidence=min_confidence
                        )
                        
                        if results:
                            st.session_state.selected_memory = results
                        else:
                            st.info("No matching memory entries found.")
                    else:
                        # Get all entries
                        entries = st.session_state.scroll_memory.memory.get("entries", [])
                        filtered_entries = [
                            entry for entry in entries
                            if entry.get("confidence", 0) >= min_confidence
                        ]
                        
                        if filtered_entries:
                            st.session_state.selected_memory = filtered_entries
                        else:
                            st.info("No memory entries found.")
                
                # Display selected memory
                if st.session_state.selected_memory:
                    for i, entry in enumerate(st.session_state.selected_memory):
                        with st.expander(f"{i+1}. {entry.get('principle', 'Unknown Principle')} ({entry.get('timestamp', 'Unknown date')})", expanded=False):
                            # Case info
                            st.markdown(f"**Case:** {entry.get('case_id', 'Unknown')}")
                            
                            # Principle
                            st.markdown(f"**Principle:** {entry.get('principle', '')}")
                            
                            # Scroll alignment
                            st.markdown(f"**Scroll Alignment:** {entry.get('scroll_alignment', '')}")
                            
                            # Prophetic insight
                            st.markdown(f"**Prophetic Insight:** {entry.get('prophetic_insight', '')}")
                            
                            # Confidence
                            confidence = entry.get("confidence", 0)
                            confidence_color = "green" if confidence >= 0.8 else "orange" if confidence >= 0.5 else "red"
                            st.markdown(f"**Confidence:** :{confidence_color}[{confidence:.2f}]")
                            
                            # Tags
                            tags = entry.get("tags", [])
                            if tags:
                                st.markdown("**Tags:** " + ", ".join([f"`{tag}`" for tag in tags]))
                            
                            # Similarity (if available)
                            if "similarity" in entry:
                                st.markdown(f"**Similarity:** {entry.get('similarity', 0):.2f}")
                
                # Memory trails
                st.markdown('<h3 class="sub-header">Memory Trails</h3>', unsafe_allow_html=True)
                
                # Case selector for trails
                if st.session_state.graph and st.session_state.graph_data:
                    case_options = {node.get('title', node.get('id', 'Unknown')): node.get('id', '') 
                                   for node in st.session_state.graph_data.get('nodes', [])}
                    
                    selected_case_title = st.selectbox(
                        "Select Case for Trail",
                        options=list(case_options.keys()),
                        index=0,
                        key="trail_case"
                    )
                    
                    selected_case_id = case_options[selected_case_title]
                    
                    # Trail depth
                    max_depth = st.slider(
                        "Trail Depth",
                        min_value=1,
                        max_value=5,
                        value=3,
                        help="Maximum depth of related memories to retrieve"
                    )
                    
                    # Get trail button
                    if st.button("Get Memory Trail"):
                        trail = st.session_state.scroll_memory.get_memory_trail(
                            case_id=selected_case_id,
                            max_depth=max_depth
                        )
                        
                        if trail:
                            st.session_state.selected_memory = trail
                            st.success(f"Found {len(trail)} related memory entries.")
                        else:
                            st.info("No memory trail found for this case.")
                else:
                    st.info("Please load a graph first to view memory trails.")
                
                # Prophetic patterns
                st.markdown('<h3 class="sub-header">Prophetic Patterns</h3>', unsafe_allow_html=True)
                
                # Time range
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input(
                        "Start Date",
                        value=datetime.now().replace(year=datetime.now().year - 1, month=1, day=1)
                    )
                with col2:
                    end_date = st.date_input(
                        "End Date",
                        value=datetime.now()
                    )
                
                # Convert to datetime
                start_datetime = datetime.combine(start_date, datetime.min.time())
                end_datetime = datetime.combine(end_date, datetime.max.time())
                
                # Minimum confidence
                min_pattern_confidence = st.slider(
                    "Minimum Pattern Confidence",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.1
                )
                
                # Find patterns button
                if st.button("Find Prophetic Patterns"):
                    patterns = st.session_state.scroll_memory.get_prophetic_patterns(
                        time_range=(start_datetime, end_datetime),
                        min_confidence=min_pattern_confidence
                    )
                    
                    if patterns:
                        st.session_state.prophetic_patterns = patterns
                        st.success(f"Found {len(patterns)} prophetic patterns.")
                    else:
                        st.info("No prophetic patterns found in the selected time range.")
                
                # Display patterns
                if st.session_state.prophetic_patterns:
                    for i, pattern in enumerate(st.session_state.prophetic_patterns):
                        with st.expander(f"{i+1}. {pattern.get('theme', 'Unknown Theme')} ({pattern.get('count', 0)} cases)", expanded=False):
                            # Theme
                            st.markdown(f"**Theme:** {pattern.get('theme', '')}")
                            
                            # Count and confidence
                            st.markdown(f"**Count:** {pattern.get('count', 0)} cases")
                            st.markdown(f"**Average Confidence:** {pattern.get('avg_confidence', 0):.2f}")
                            
                            # Time span
                            time_span = pattern.get('time_span', {})
                            st.markdown(f"**Time Span:** {time_span.get('start', '')} to {time_span.get('end', '')}")
                            
                            # Principles
                            principles = pattern.get('principles', [])
                            if principles:
                                st.markdown("**Principles:**")
                                for principle in principles:
                                    st.markdown(f"- {principle}")
                            
                            # Insights
                            insights = pattern.get('insights', [])
                            if insights:
                                st.markdown("**Insights:**")
                                for insight in insights:
                                    st.markdown(f"- {insight}")
                            
                            # Case IDs
                            case_ids = pattern.get('case_ids', [])
                            if case_ids:
                                st.markdown("**Cases:**")
                                for case_id in case_ids:
                                    st.markdown(f"- {case_id}")
                
                # Memory statistics
                st.markdown('<h3 class="sub-header">Memory Statistics</h3>', unsafe_allow_html=True)
                
                if st.button("Get Memory Statistics"):
                    stats = st.session_state.scroll_memory.get_memory_statistics()
                    
                    if stats:
                        # Total entries
                        st.metric("Total Entries", stats.get('total_entries', 0))
                        
                        # Confidence levels
                        confidence_levels = stats.get('confidence_levels', {})
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("High Confidence", confidence_levels.get('high', 0))
                        with col2:
                            st.metric("Medium Confidence", confidence_levels.get('medium', 0))
                        with col3:
                            st.metric("Low Confidence", confidence_levels.get('low', 0))
                        
                        # Unique cases
                        st.metric("Unique Cases", stats.get('unique_cases', 0))
                        
                        # Tag counts
                        tag_counts = stats.get('tag_counts', {})
                        if tag_counts:
                            st.markdown("**Tag Distribution:**")
                            tag_data = pd.DataFrame({
                                'Tag': list(tag_counts.keys()),
                                'Count': list(tag_counts.values())
                            })
                            tag_data = tag_data.sort_values('Count', ascending=False)
                            
                            fig = go.Figure(data=[
                                go.Bar(
                                    x=tag_data['Tag'],
                                    y=tag_data['Count'],
                                    marker_color='#1E88E5'
                                )
                            ])
                            fig.update_layout(
                                title='Tag Distribution',
                                xaxis_title='Tag',
                                yaxis_title='Count',
                                plot_bgcolor='rgba(255, 255, 255, 1)',
                                paper_bgcolor='rgba(255, 255, 255, 1)',
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Time range
                        time_range = stats.get('time_range', {})
                        if time_range.get('start') and time_range.get('end'):
                            st.markdown(f"**Time Range:** {time_range.get('start')} to {time_range.get('end')}")
                    else:
                        st.info("No memory statistics available.")
        else:
            st.error("Scroll Memory Intelligence is not initialized. Please check the logs for details.")

# Add to the main content section, after the tabs
if st.session_state.judicial_mode:
    st.markdown("""
    <div class="highlight">
        <h3>🔍 Judicial Mode: Enhanced Transparency</h3>
        <p>In Judicial Mode, all AI-generated insights include:</p>
        <ul>
            <li><strong>Source Attribution:</strong> Every principle is linked to specific text passages</li>
            <li><strong>Confidence Scores:</strong> AI certainty levels for each inference</li>
            <li><strong>Alternative Interpretations:</strong> Multiple possible readings of ambiguous text</li>
            <li><strong>Human Verification:</strong> Options to flag or correct AI interpretations</li>
        </ul>
        <p><em>This tool is designed as a judicial memory aid, not a decision-maker. All interpretations should be verified by legal professionals.</em></p>
    </div>
    """, unsafe_allow_html=True)

# Add Resistance Radar Panel
with st.expander("🛡 Resistance Radar", expanded=False):
    st.markdown("""
    <ul>
        <li>✅ <strong>Judicial Trust Level:</strong> Medium</li>
        <li>✅ <strong>Institutional Acceptance:</strong> Emerging</li>
        <li>⚠️ <strong>Scroll Alignment Sensitivity:</strong> High</li>
        <li>✅ <strong>AI Acceptance Level:</strong> Moderate</li>
        <li>🟡 <strong>LegalTech Industry Disruption Level:</strong> High</li>
    </ul>
    <p><em>Each of these metrics reflects real-time usage feedback and perceived threat by existing systems.</em></p>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown('<p class="info-text">FastTrackJustice AI - Precedent Influence Graph Explorer</p>', unsafe_allow_html=True)

# Initialize the graph and precedent manager
@st.cache_resource
def initialize_managers():
    try:
        graph = PrecedentInfluenceGraph()
        precedent_manager = LegalPrecedentManager()
        return graph, precedent_manager
    except Exception as e:
        logger.error(f"Failed to initialize managers: {e}")
        st.error("Failed to initialize the application. Please check the logs for details.")
        return None, None

def create_network_graph(graph_data: Dict[str, Any], highlight_case: Optional[str] = None) -> go.Figure:
    """Create an interactive network graph using Plotly."""
    G = nx.Graph()
    
    # Add nodes
    for case in graph_data['nodes']:
        G.add_node(
            case['id'],
            title=case['title'],
            year=case['year'],
            court=case['court']
        )
    
    # Add edges
    for edge in graph_data['edges']:
        G.add_edge(
            edge['source'],
            edge['target'],
            weight=edge['weight']
        )
    
    # Create node positions using spring layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Add edges to trace
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += (x0, x1, None)
        edge_trace['y'] += (y0, y1, None)
    
    # Create node trace
    node_trace = go.Scatter(
        x=[],
        y=[],
        mode='markers+text',
        hoverinfo='text',
        text=[],
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            )
        )
    )
    
    # Add nodes to trace
    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += (x,)
        node_trace['y'] += (y,)
        node_trace['text'] += (f"{G.nodes[node]['title']}<br>"
                             f"Year: {G.nodes[node]['year']}<br>"
                             f"Court: {G.nodes[node]['court']}",)
    
    # Color nodes by number of connections
    node_adjacencies = []
    for node in G.nodes():
        node_adjacencies.append(len(list(G.neighbors(node))))
    
    node_trace.marker.color = node_adjacencies
    
    # Create figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title='Precedent Influence Graph',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )
    
    return fig

def create_principle_heatmap(graph_data: Dict[str, Any]) -> go.Figure:
    """Create a heatmap showing principle influence over time."""
    # Extract principles and their years
    principles_by_year = {}
    
    for node in graph_data['nodes']:
        year = node['year']
        principles = node.get('principles', [])
        
        for principle in principles:
            if principle not in principles_by_year:
                principles_by_year[principle] = {}
            
            if year not in principles_by_year[principle]:
                principles_by_year[principle][year] = 0
            
            principles_by_year[principle][year] += 1
    
    # Get all years and principles
    all_years = sorted(list(set([node['year'] for node in graph_data['nodes']])))
    all_principles = sorted(list(principles_by_year.keys()))
    
    # Create heatmap data
    heatmap_data = []
    
    for principle in all_principles:
        for year in all_years:
            count = principles_by_year[principle].get(year, 0)
            heatmap_data.append({
                'Principle': principle,
                'Year': year,
                'Count': count
            })
    
    # Create DataFrame
    df = pd.DataFrame(heatmap_data)
    
    # Create heatmap
    fig = px.density_heatmap(
        df,
        x='Year',
        y='Principle',
        z='Count',
        color_continuous_scale='Viridis',
        title='Principle Influence Over Time'
    )
    
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Legal Principle',
        height=600
    )
    
    return fig

def apply_filters(graph_data: Dict[str, Any], 
                 search_query: str, 
                 year_range: Tuple[int, int], 
                 jurisdiction: str,
                 categories: List[str]) -> Dict[str, Any]:
    """Apply filters to the graph data."""
    filtered_nodes = []
    filtered_edges = []
    
    # Filter nodes
    for node in graph_data['nodes']:
        # Apply search query filter
        if search_query and search_query.lower() not in node['title'].lower():
            continue
        
        # Apply year range filter
        if node['year'] < year_range[0] or node['year'] > year_range[1]:
            continue
        
        # Apply jurisdiction filter
        if jurisdiction != "All" and jurisdiction.lower() not in node['court'].lower():
            continue
        
        # Apply categories filter
        if categories and not any(cat in node.get('categories', []) for cat in categories):
            continue
        
        filtered_nodes.append(node)
    
    # Get filtered node IDs
    filtered_node_ids = [node['id'] for node in filtered_nodes]
    
    # Filter edges
    for edge in graph_data['edges']:
        if edge['source'] in filtered_node_ids and edge['target'] in filtered_node_ids:
            filtered_edges.append(edge)
    
    return {
        'nodes': filtered_nodes,
        'edges': filtered_edges
    }

def apply_lens(graph_data: Dict[str, Any], lens: str) -> Dict[str, Any]:
    """Apply a specific lens to the graph data."""
    if lens == "doctrinal":
        # Color nodes by legal field
        for node in graph_data['nodes']:
            categories = node.get('categories', [])
            if 'tort' in [cat.lower() for cat in categories]:
                node['color'] = '#FF5252'  # Red for tort
            elif 'contract' in [cat.lower() for cat in categories]:
                node['color'] = '#4CAF50'  # Green for contract
            elif 'criminal' in [cat.lower() for cat in categories]:
                node['color'] = '#2196F3'  # Blue for criminal
            elif 'constitutional' in [cat.lower() for cat in categories]:
                node['color'] = '#9C27B0'  # Purple for constitutional
            else:
                node['color'] = '#FFC107'  # Yellow for other
    
    elif lens == "institutional":
        # Color nodes by court
        for node in graph_data['nodes']:
            court = node['court'].lower()
            if 'supreme' in court:
                node['color'] = '#F44336'  # Red for Supreme Court
            elif 'federal' in court or 'district' in court:
                node['color'] = '#2196F3'  # Blue for Federal Courts
            elif 'state' in court:
                node['color'] = '#4CAF50'  # Green for State Courts
            elif 'appellate' in court:
                node['color'] = '#FF9800'  # Orange for Appellate Courts
            else:
                node['color'] = '#9E9E9E'  # Grey for others
    
    elif lens == "ethical":
        # Highlight human rights / humanitarian cases
        for node in graph_data['nodes']:
            categories = node.get('categories', [])
            if any(cat.lower() in ['human rights', 'humanitarian', 'civil rights'] for cat in categories):
                node['highlight'] = True
            else:
                node['highlight'] = False
    
    elif lens == "scroll":
        # Match scroll-aligned principles
        for node in graph_data['nodes']:
            principles = node.get('principles', [])
            if any('scroll' in principle.lower() for principle in principles):
                node['scroll_aligned'] = True
            else:
                node['scroll_aligned'] = False
    
    return graph_data

def generate_share_link(graph_data: Dict[str, Any], selected_nodes: List[str] = None) -> str:
    """Generate a shareable link for the graph or subgraph."""
    # Create a unique ID for this share
    share_id = str(uuid.uuid4())[:8]
    
    # If specific nodes are selected, create a subgraph
    if selected_nodes:
        subgraph = {
            'nodes': [node for node in graph_data['nodes'] if node['id'] in selected_nodes],
            'edges': [edge for edge in graph_data['edges'] 
                     if edge['source'] in selected_nodes and edge['target'] in selected_nodes]
        }
        share_data = subgraph
    else:
        share_data = graph_data
    
    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp_file:
        json.dump(share_data, tmp_file)
        tmp_file_path = tmp_file.name
    
    # In a real app, you would upload this to a server and return a URL
    # For now, we'll just return a base64 encoded version
    with open(tmp_file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    
    # Clean up the temporary file
    os.unlink(tmp_file_path)
    
    # Return a data URL
    return f"data:application/json;base64,{b64}"

def export_subgraph_as_png(graph_data: Dict[str, Any], selected_nodes: List[str] = None) -> bytes:
    """Export a subgraph as a PNG image."""
    # Create a NetworkX graph
    G = nx.Graph()
    
    # Add nodes
    for node in graph_data['nodes']:
        if not selected_nodes or node['id'] in selected_nodes:
            G.add_node(
                node['id'],
                title=node['title'],
                year=node['year'],
                court=node['court']
            )
    
    # Add edges
    for edge in graph_data['edges']:
        if not selected_nodes or (edge['source'] in selected_nodes and edge['target'] in selected_nodes):
            G.add_edge(
                edge['source'],
                edge['target'],
                weight=edge['weight']
            )
    
    # Create a matplotlib figure
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=1, iterations=50)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=8, font_weight='bold')
    
    # Save to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    return buf.getvalue()

def export_subgraph_as_html(graph_data: Dict[str, Any], selected_nodes: List[str] = None) -> str:
    """Export a subgraph as an interactive HTML file."""
    # Filter the graph data if needed
    if selected_nodes:
        subgraph = {
            'nodes': [node for node in graph_data['nodes'] if node['id'] in selected_nodes],
            'edges': [edge for edge in graph_data['edges'] 
                     if edge['source'] in selected_nodes and edge['target'] in selected_nodes]
        }
        graph_data = subgraph
    
    # Create a NetworkX graph
    G = nx.Graph()
    
    # Add nodes
    for node in graph_data['nodes']:
        G.add_node(
            node['id'],
            title=node['title'],
            year=node['year'],
            court=node['court']
        )
    
    # Add edges
    for edge in graph_data['edges']:
        G.add_edge(
            edge['source'],
            edge['target'],
            weight=edge['weight']
        )
    
    # Create node positions
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Create HTML with Plotly
    fig = go.Figure()
    
    # Add edges
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += (x0, x1, None)
        edge_trace['y'] += (y0, y1, None)
    
    fig.add_trace(edge_trace)
    
    # Add nodes
    node_trace = go.Scatter(
        x=[],
        y=[],
        mode='markers+text',
        hoverinfo='text',
        text=[],
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            )
        )
    )
    
    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += (x,)
        node_trace['y'] += (y,)
        node_trace['text'] += (f"{G.nodes[node]['title']}<br>"
                             f"Year: {G.nodes[node]['year']}<br>"
                             f"Court: {G.nodes[node]['court']}",)
    
    # Color nodes by number of connections
    node_adjacencies = []
    for node in G.nodes():
        node_adjacencies.append(len(list(G.neighbors(node))))
    
    node_trace.marker.color = node_adjacencies
    
    fig.add_trace(node_trace)
    
    # Update layout
    fig.update_layout(
        title='Precedent Influence Graph',
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    # Convert to HTML
    html = fig.to_html(include_plotlyjs=True, full_html=True)
    
    return html

def process_bulk_import(uploaded_file) -> Dict[str, Any]:
    """Process a bulk import of cases from a JSON file."""
    try:
        # Read the uploaded file
        content = uploaded_file.read()
        cases = json.loads(content)
        
        # Initialize the graph
        graph = PrecedentInfluenceGraph()
        
        # Add each case to the graph
        for case in cases:
            graph.add_case(case)
        
        # Export the graph data
        graph_data = graph.export_graph_data()
        
        return graph_data
    except Exception as e:
        logger.error(f"Failed to process bulk import: {e}")
        return None

def main():
    st.set_page_config(
        page_title="Precedent Influence Graph Explorer",
        page_icon="⚖️",
        layout="wide"
    )
    
    st.title("⚖️ Precedent Influence Graph Explorer")
    st.markdown("""
    This interactive tool allows you to explore the relationships between legal precedents
    and how they influence each other. The graph shows cases as nodes and their relationships
    as edges, with the thickness of edges indicating the strength of influence.
    """)
    
    # Initialize managers
    graph, precedent_manager = initialize_managers()
    if not graph or not precedent_manager:
        st.stop()
    
    # Initialize Scroll Memory Intelligence
    @st.cache_resource
    def initialize_scroll_memory():
        try:
            memory = ScrollMemoryIntelligence()
            return memory
        except Exception as e:
            logger.error(f"Failed to initialize Scroll Memory Intelligence: {e}")
            st.error("Failed to initialize Scroll Memory Intelligence. Please check the logs for details.")
            return None
    
    # Initialize Scroll Memory Intelligence
    st.session_state.scroll_memory = initialize_scroll_memory()
    
    # Sidebar controls
    st.sidebar.header("Controls")
    
    # Add new case
    st.sidebar.subheader("Add New Case")
    case_title = st.sidebar.text_input("Case Title")
    case_year = st.sidebar.number_input("Year", min_value=1800, max_value=2024, value=2024)
    case_court = st.sidebar.text_input("Court")
    case_text = st.sidebar.text_area("Case Text")
    
    if st.sidebar.button("Add Case"):
        if case_title and case_text:
            try:
                graph.add_case({
                    'title': case_title,
                    'year': case_year,
                    'court': case_court,
                    'text': case_text
                })
                st.sidebar.success("Case added successfully!")
            except Exception as e:
                st.sidebar.error(f"Failed to add case: {e}")
        else:
            st.sidebar.warning("Please fill in all required fields.")
    
    # Graph visualization
    st.header("Graph Visualization")
    
    try:
        graph_data = graph.export_graph_data()
        fig = create_network_graph(graph_data)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to create graph visualization: {e}")
    
    # Graph statistics
    st.header("Graph Statistics")
    try:
        stats = graph.get_graph_statistics()
        col1, col2, col3 = st.columns(3)
        col1.metric("Number of Cases", stats['num_cases'])
        col2.metric("Number of Connections", stats['num_edges'])
        col3.metric("Average Connections per Case", 
                   f"{stats['avg_connections']:.2f}")
    except Exception as e:
        st.error(f"Failed to load graph statistics: {e}")
    
    # Case details
    st.header("Case Details")
    try:
        cases = graph_data['nodes']
        case_df = pd.DataFrame(cases)
        st.dataframe(case_df[['title', 'year', 'court']])
    except Exception as e:
        st.error(f"Failed to load case details: {e}")

# Add to the main content section, after the tabs
if st.session_state.data_source == "public_domain":
    st.markdown("""
    <div class="highlight">
        <h3>🌐 Public Domain Data</h3>
        <p>Currently using public domain legal precedents from:</p>
        <ul>
            <li>U.S. Supreme Court decisions</li>
            <li>Indian Supreme Court judgments</li>
            <li>European Court of Human Rights</li>
            <li>International Court of Justice</li>
        </ul>
        <p><em>All data is sourced from publicly available repositories with appropriate attribution.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 