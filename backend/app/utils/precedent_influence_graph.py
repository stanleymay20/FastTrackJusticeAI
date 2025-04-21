import os
import json
import logging
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
from backend.app.utils.legal_principle_synthesizer import LegalPrincipleSynthesizer

logger = logging.getLogger(__name__)

class PrecedentInfluenceGraph:
    """
    Builds a graph representation of legal precedents based on principle similarity.
    Nodes represent cases, edges represent influence based on shared principles.
    Supports visualization and analysis of precedent relationships.
    """
    
    def __init__(
        self,
        synthesizer: Optional[LegalPrincipleSynthesizer] = None,
        cache_dir: str = "cache/graphs",
        similarity_threshold: float = 0.7,
        max_principles_per_case: int = 10,
        layout_algorithm: str = "spring",
        output_dir: str = "output/graphs"
    ):
        """
        Initialize the PrecedentInfluenceGraph.
        
        Args:
            synthesizer: LegalPrincipleSynthesizer instance for processing cases
            cache_dir: Directory to store graph data
            similarity_threshold: Minimum similarity score to create an edge
            max_principles_per_case: Maximum number of principles to consider per case
            layout_algorithm: Algorithm for graph layout (spring, kamada_kawai, etc.)
            output_dir: Directory to store generated visualizations
        """
        self.synthesizer = synthesizer or LegalPrincipleSynthesizer()
        self.cache_dir = cache_dir
        self.similarity_threshold = similarity_threshold
        self.max_principles_per_case = max_principles_per_case
        self.layout_algorithm = layout_algorithm
        self.output_dir = output_dir
        
        # Create directories if they don't exist
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize graph
        self.graph = nx.DiGraph()
        
        # Case data storage
        self.case_data = {}
        
        logger.info(f"Initialized PrecedentInfluenceGraph with {layout_algorithm} layout")
        
    def add_case(
        self, 
        case_id: str, 
        case_text: str, 
        holding: Optional[str] = None,
        reasoning: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        force_reprocess: bool = False
    ) -> Dict[str, Any]:
        """
        Add a case to the graph and process its principles.
        
        Args:
            case_id: Unique identifier for the case
            case_text: The full case text
            holding: Optional holding section of the case
            reasoning: Optional reasoning section of the case
            metadata: Optional metadata about the case (date, court, etc.)
            force_reprocess: Whether to reprocess even if cached
            
        Returns:
            Dictionary with case processing results
        """
        # Check if case is already in graph
        if case_id in self.graph.nodes and not force_reprocess:
            logger.info(f"Case {case_id} already in graph, using cached data")
            return self.case_data.get(case_id, {})
            
        # Process the case
        cache_key = f"case_{case_id}"
        result = self.synthesizer.process_case(
            case_text, 
            holding=holding, 
            reasoning=reasoning,
            cache_key=cache_key
        )
        
        # Store case data
        case_info = {
            "id": case_id,
            "principles": result["raw_principles"],
            "summarized_principles": result["summarized_principles"],
            "classified_principles": result["classified_principles"],
            "metadata": metadata or {},
            "categories": result["metadata"]["categories_found"]
        }
        
        self.case_data[case_id] = case_info
        
        # Add node to graph
        self.graph.add_node(
            case_id,
            **case_info
        )
        
        # Update edges with existing cases
        self._update_edges(case_id)
        
        logger.info(f"Added case {case_id} with {len(result['raw_principles'])} principles")
        return case_info
        
    def add_cases(
        self, 
        cases: List[Dict[str, Any]], 
        force_reprocess: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Add multiple cases to the graph.
        
        Args:
            cases: List of case dictionaries with required fields
            force_reprocess: Whether to reprocess even if cached
            
        Returns:
            List of case processing results
        """
        results = []
        for case in cases:
            case_id = case.get("id")
            if not case_id:
                logger.warning("Skipping case without ID")
                continue
                
            result = self.add_case(
                case_id=case_id,
                case_text=case.get("text", ""),
                holding=case.get("holding"),
                reasoning=case.get("reasoning"),
                metadata=case.get("metadata"),
                force_reprocess=force_reprocess
            )
            results.append(result)
            
        return results
        
    def _update_edges(self, new_case_id: str) -> None:
        """
        Update edges for a newly added case.
        
        Args:
            new_case_id: ID of the newly added case
        """
        new_case = self.case_data[new_case_id]
        new_principles = new_case["summarized_principles"]
        
        # Skip if no principles
        if not new_principles:
            return
            
        # Compare with all other cases
        for other_id, other_case in self.case_data.items():
            if other_id == new_case_id:
                continue
                
            other_principles = other_case["summarized_principles"]
            
            # Skip if no principles
            if not other_principles:
                continue
                
            # Calculate similarity
            similarity = self._calculate_case_similarity(
                new_principles, 
                other_principles
            )
            
            # Add edge if similarity exceeds threshold
            if similarity >= self.similarity_threshold:
                # Determine direction based on dates if available
                direction = self._determine_influence_direction(
                    new_case, 
                    other_case
                )
                
                if direction == "forward":
                    self.graph.add_edge(
                        other_id, 
                        new_case_id, 
                        weight=similarity,
                        principles=self._get_shared_principles(
                            new_principles, 
                            other_principles
                        )
                    )
                else:
                    self.graph.add_edge(
                        new_case_id, 
                        other_id, 
                        weight=similarity,
                        principles=self._get_shared_principles(
                            new_principles, 
                            other_principles
                        )
                    )
                    
    def _calculate_case_similarity(
        self, 
        principles1: List[str], 
        principles2: List[str]
    ) -> float:
        """
        Calculate similarity between two sets of principles.
        
        Args:
            principles1: First set of principles
            principles2: Second set of principles
            
        Returns:
            Similarity score between 0 and 1
        """
        if not principles1 or not principles2:
            return 0.0
            
        # Limit to max principles per case
        principles1 = principles1[:self.max_principles_per_case]
        principles2 = principles2[:self.max_principles_per_case]
        
        # Get embeddings for all principles
        embeddings1 = self.synthesizer.extraction_model.encode(
            principles1, 
            convert_to_numpy=True
        )
        embeddings2 = self.synthesizer.extraction_model.encode(
            principles2, 
            convert_to_numpy=True
        )
        
        # Calculate pairwise similarities
        similarities = []
        for emb1 in embeddings1:
            for emb2 in embeddings2:
                similarity = np.dot(emb1, emb2) / (
                    np.linalg.norm(emb1) * np.linalg.norm(emb2)
                )
                similarities.append(float(similarity))
                
        # Return max similarity as the case similarity
        return max(similarities) if similarities else 0.0
        
    def _get_shared_principles(
        self, 
        principles1: List[str], 
        principles2: List[str]
    ) -> List[str]:
        """
        Find principles that are semantically similar between two cases.
        
        Args:
            principles1: First set of principles
            principles2: Second set of principles
            
        Returns:
            List of shared principles
        """
        if not principles1 or not principles2:
            return []
            
        # Limit to max principles per case
        principles1 = principles1[:self.max_principles_per_case]
        principles2 = principles2[:self.max_principles_per_case]
        
        # Get embeddings for all principles
        embeddings1 = self.synthesizer.extraction_model.encode(
            principles1, 
            convert_to_numpy=True
        )
        embeddings2 = self.synthesizer.extraction_model.encode(
            principles2, 
            convert_to_numpy=True
        )
        
        # Find shared principles
        shared = []
        for i, emb1 in enumerate(embeddings1):
            for j, emb2 in enumerate(embeddings2):
                similarity = np.dot(emb1, emb2) / (
                    np.linalg.norm(emb1) * np.linalg.norm(emb2)
                )
                if similarity >= self.similarity_threshold:
                    # Add the shorter principle (usually the summary)
                    if len(principles1[i]) <= len(principles2[j]):
                        shared.append(principles1[i])
                    else:
                        shared.append(principles2[j])
                        
        return list(set(shared))  # Remove duplicates
        
    def _determine_influence_direction(
        self, 
        case1: Dict[str, Any], 
        case2: Dict[str, Any]
    ) -> str:
        """
        Determine the direction of influence between two cases.
        
        Args:
            case1: First case data
            case2: Second case data
            
        Returns:
            "forward" if case1 influenced case2, "backward" otherwise
        """
        # Try to get dates from metadata
        date1 = self._extract_date(case1.get("metadata", {}))
        date2 = self._extract_date(case2.get("metadata", {}))
        
        # If both dates are available, use them to determine direction
        if date1 and date2:
            return "forward" if date1 < date2 else "backward"
            
        # Default to treating the first case as influencing the second
        return "forward"
        
    def _extract_date(self, metadata: Dict[str, Any]) -> Optional[datetime]:
        """
        Extract date from case metadata.
        
        Args:
            metadata: Case metadata dictionary
            
        Returns:
            datetime object if date found, None otherwise
        """
        # Try different date fields
        date_fields = ["date", "decision_date", "issued_date", "filed_date"]
        
        for field in date_fields:
            if field in metadata:
                try:
                    # Try to parse the date
                    if isinstance(metadata[field], str):
                        # Try different formats
                        formats = [
                            "%Y-%m-%d", 
                            "%Y/%m/%d", 
                            "%d-%m-%Y", 
                            "%d/%m/%Y",
                            "%B %d, %Y",
                            "%d %B %Y"
                        ]
                        
                        for fmt in formats:
                            try:
                                return datetime.strptime(metadata[field], fmt)
                            except ValueError:
                                continue
                    elif isinstance(metadata[field], (int, float)):
                        # Assume it's a timestamp
                        return datetime.fromtimestamp(metadata[field])
                except Exception as e:
                    logger.warning(f"Error parsing date from {field}: {str(e)}")
                    
        return None
        
    def get_case_influences(self, case_id: str) -> Dict[str, Any]:
        """
        Get cases that influenced or were influenced by a given case.
        
        Args:
            case_id: ID of the case to analyze
            
        Returns:
            Dictionary with influencing and influenced cases
        """
        if case_id not in self.graph:
            logger.warning(f"Case {case_id} not found in graph")
            return {"influencing": [], "influenced": []}
            
        # Get predecessors (cases that influenced this case)
        predecessors = list(self.graph.predecessors(case_id))
        predecessor_data = []
        
        for pred_id in predecessors:
            edge_data = self.graph.get_edge_data(pred_id, case_id)
            predecessor_data.append({
                "id": pred_id,
                "similarity": edge_data.get("weight", 0.0),
                "shared_principles": edge_data.get("principles", []),
                "metadata": self.case_data.get(pred_id, {}).get("metadata", {})
            })
            
        # Get successors (cases influenced by this case)
        successors = list(self.graph.successors(case_id))
        successor_data = []
        
        for succ_id in successors:
            edge_data = self.graph.get_edge_data(case_id, succ_id)
            successor_data.append({
                "id": succ_id,
                "similarity": edge_data.get("weight", 0.0),
                "shared_principles": edge_data.get("principles", []),
                "metadata": self.case_data.get(succ_id, {}).get("metadata", {})
            })
            
        return {
            "influencing": sorted(predecessor_data, key=lambda x: x["similarity"], reverse=True),
            "influenced": sorted(successor_data, key=lambda x: x["similarity"], reverse=True)
        }
        
    def get_principle_evolution(
        self, 
        principle: str, 
        max_cases: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Track the evolution of a principle across cases.
        
        Args:
            principle: The principle to track
            max_cases: Maximum number of cases to return
            
        Returns:
            List of cases where the principle appears, sorted by date
        """
        # Get embedding for the principle
        principle_embedding = self.synthesizer.extraction_model.encode(
            principle, 
            convert_to_numpy=True
        )
        
        # Find cases with similar principles
        case_scores = []
        
        for case_id, case_data in self.case_data.items():
            principles = case_data.get("summarized_principles", [])
            
            if not principles:
                continue
                
            # Calculate similarity with each principle
            max_similarity = 0.0
            for p in principles:
                p_embedding = self.synthesizer.extraction_model.encode(
                    p, 
                    convert_to_numpy=True
                )
                similarity = float(np.dot(principle_embedding, p_embedding) / (
                    np.linalg.norm(principle_embedding) * np.linalg.norm(p_embedding)
                ))
                max_similarity = max(max_similarity, similarity)
                
            if max_similarity >= self.similarity_threshold:
                # Extract date from metadata
                date = self._extract_date(case_data.get("metadata", {}))
                
                case_scores.append({
                    "id": case_id,
                    "similarity": max_similarity,
                    "date": date,
                    "principle": principle,
                    "metadata": case_data.get("metadata", {})
                })
                
        # Sort by date if available
        case_scores.sort(key=lambda x: (x["date"] is None, x["date"]))
        
        # Limit to max cases
        return case_scores[:max_cases]
        
    def visualize(
        self, 
        output_file: Optional[str] = None,
        title: str = "Precedent Influence Graph",
        node_size: int = 1000,
        edge_width_scale: float = 2.0,
        show_labels: bool = True,
        highlight_cases: Optional[List[str]] = None,
        color_by_category: bool = False
    ) -> str:
        """
        Generate a visualization of the precedent influence graph.
        
        Args:
            output_file: Path to save the visualization (if None, auto-generated)
            title: Title for the visualization
            node_size: Size of nodes in the graph
            edge_width_scale: Scale factor for edge widths
            show_labels: Whether to show case labels
            highlight_cases: List of case IDs to highlight
            color_by_category: Whether to color nodes by legal category
            
        Returns:
            Path to the generated visualization
        """
        if not self.graph.nodes:
            logger.warning("Graph is empty, nothing to visualize")
            return ""
            
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Choose layout algorithm
        if self.layout_algorithm == "spring":
            pos = nx.spring_layout(self.graph, seed=42)
        elif self.layout_algorithm == "kamada_kawai":
            pos = nx.kamada_kawai_layout(self.graph)
        elif self.layout_algorithm == "circular":
            pos = nx.circular_layout(self.graph)
        elif self.layout_algorithm == "shell":
            pos = nx.shell_layout(self.graph)
        else:
            pos = nx.spring_layout(self.graph, seed=42)
            
        # Draw edges
        edge_weights = [self.graph[u][v]["weight"] * edge_width_scale 
                       for u, v in self.graph.edges()]
        nx.draw_networkx_edges(
            self.graph, 
            pos, 
            edge_color='gray',
            arrows=True,
            arrowsize=20,
            width=edge_weights,
            alpha=0.7
        )
        
        # Draw nodes
        if color_by_category:
            # Group nodes by primary category
            categories = {}
            for node in self.graph.nodes:
                node_categories = self.graph.nodes[node].get("categories", [])
                if node_categories:
                    primary_category = node_categories[0]
                    if primary_category not in categories:
                        categories[primary_category] = []
                    categories[primary_category].append(node)
                    
            # Draw nodes by category with different colors
            colors = plt.cm.tab20(np.linspace(0, 1, len(categories)))
            for (category, nodes), color in zip(categories.items(), colors):
                nx.draw_networkx_nodes(
                    self.graph, 
                    pos, 
                    nodelist=nodes,
                    node_color=[color],
                    node_size=node_size,
                    alpha=0.8
                )
                
            # Add legend
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor=color, label=category, markersize=10)
                              for category, color in zip(categories.keys(), colors)]
            plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        else:
            # Draw all nodes with the same color
            node_colors = 'lightblue'
            if highlight_cases:
                # Highlight specific cases
                node_colors = ['red' if node in highlight_cases else 'lightblue' 
                              for node in self.graph.nodes]
                
            nx.draw_networkx_nodes(
                self.graph, 
                pos, 
                node_color=node_colors,
                node_size=node_size,
                alpha=0.8
            )
            
        # Draw labels
        if show_labels:
            labels = {node: node for node in self.graph.nodes}
            nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)
            
        # Add title
        plt.title(title, fontsize=16)
        
        # Remove axes
        plt.axis('off')
        
        # Generate output filename if not provided
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(
                self.output_dir, 
                f"precedent_graph_{timestamp}.png"
            )
            
        # Save figure
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved graph visualization to {output_file}")
        return output_file
        
    def export_graph_data(self, output_file: Optional[str] = None) -> str:
        """
        Export the graph data to a JSON file.
        
        Args:
            output_file: Path to save the data (if None, auto-generated)
            
        Returns:
            Path to the exported file
        """
        # Convert graph to serializable format
        graph_data = {
            "nodes": [],
            "edges": []
        }
        
        # Add nodes
        for node in self.graph.nodes:
            node_data = self.graph.nodes[node]
            # Convert numpy arrays to lists for JSON serialization
            serializable_data = {}
            for key, value in node_data.items():
                if isinstance(value, np.ndarray):
                    serializable_data[key] = value.tolist()
                else:
                    serializable_data[key] = value
            graph_data["nodes"].append(serializable_data)
            
        # Add edges
        for u, v in self.graph.edges():
            edge_data = self.graph[u][v]
            # Convert numpy arrays to lists for JSON serialization
            serializable_data = {}
            for key, value in edge_data.items():
                if isinstance(value, np.ndarray):
                    serializable_data[key] = value.tolist()
                else:
                    serializable_data[key] = value
            graph_data["edges"].append({
                "source": u,
                "target": v,
                **serializable_data
            })
            
        # Generate output filename if not provided
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(
                self.output_dir, 
                f"precedent_graph_data_{timestamp}.json"
            )
            
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2)
            
        logger.info(f"Exported graph data to {output_file}")
        return output_file
        
    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the graph.
        
        Returns:
            Dictionary with graph statistics
        """
        if not self.graph.nodes:
            return {
                "num_cases": 0,
                "num_edges": 0,
                "density": 0.0,
                "avg_degree": 0.0,
                "categories": {}
            }
            
        # Basic statistics
        num_cases = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()
        density = nx.density(self.graph)
        avg_degree = sum(dict(self.graph.degree()).values()) / num_cases
        
        # Category distribution
        categories = {}
        for node in self.graph.nodes:
            node_categories = self.graph.nodes[node].get("categories", [])
            for category in node_categories:
                if category not in categories:
                    categories[category] = 0
                categories[category] += 1
                
        # Centrality measures
        try:
            betweenness = nx.betweenness_centrality(self.graph)
            pagerank = nx.pagerank(self.graph)
            
            # Get top influential cases
            top_influential = sorted(
                pagerank.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
            
            top_influential_data = []
            for case_id, score in top_influential:
                top_influential_data.append({
                    "id": case_id,
                    "score": score,
                    "metadata": self.case_data.get(case_id, {}).get("metadata", {})
                })
        except Exception as e:
            logger.warning(f"Error calculating centrality: {str(e)}")
            betweenness = {}
            pagerank = {}
            top_influential_data = []
            
        return {
            "num_cases": num_cases,
            "num_edges": num_edges,
            "density": density,
            "avg_degree": avg_degree,
            "categories": categories,
            "top_influential_cases": top_influential_data
        }
        
    def save(self, filename: Optional[str] = None) -> str:
        """
        Save the graph to a file.
        
        Args:
            filename: Path to save the graph (if None, auto-generated)
            
        Returns:
            Path to the saved file
        """
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(
                self.cache_dir, 
                f"precedent_graph_{timestamp}.json"
            )
            
        # Prepare data for saving
        data = {
            "case_data": self.case_data,
            "graph": {
                "nodes": list(self.graph.nodes(data=True)),
                "edges": list(self.graph.edges(data=True))
            },
            "metadata": {
                "similarity_threshold": self.similarity_threshold,
                "max_principles_per_case": self.max_principles_per_case,
                "layout_algorithm": self.layout_algorithm,
                "created_at": datetime.now().isoformat()
            }
        }
        
        # Save to file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
            
        logger.info(f"Saved graph to {filename}")
        return filename
        
    @classmethod
    def load(cls, filename: str, synthesizer: Optional[LegalPrincipleSynthesizer] = None) -> 'PrecedentInfluenceGraph':
        """
        Load a graph from a file.
        
        Args:
            filename: Path to the saved graph file
            synthesizer: LegalPrincipleSynthesizer instance
            
        Returns:
            Loaded PrecedentInfluenceGraph instance
        """
        # Load data from file
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Create instance
        graph = cls(
            synthesizer=synthesizer,
            cache_dir=os.path.dirname(filename),
            similarity_threshold=data["metadata"]["similarity_threshold"],
            max_principles_per_case=data["metadata"]["max_principles_per_case"],
            layout_algorithm=data["metadata"]["layout_algorithm"]
        )
        
        # Restore case data
        graph.case_data = data["case_data"]
        
        # Restore graph
        for node, node_data in data["graph"]["nodes"]:
            graph.graph.add_node(node, **node_data)
            
        for u, v, edge_data in data["graph"]["edges"]:
            graph.graph.add_edge(u, v, **edge_data)
            
        logger.info(f"Loaded graph from {filename} with {len(graph.graph.nodes)} nodes")
        return graph 