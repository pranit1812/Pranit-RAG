"""
Specialized drawing processing for chunking with spatial organization.
"""
from typing import List, Tuple, Dict, Any, Optional
import math

# Optional imports with fallbacks
try:
    import numpy as np
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Simple fallback for numpy-like operations
    class linalg:
        @staticmethod
        def norm(vector):
            return math.sqrt(sum(x*x for x in vector))
    
    class np:
        linalg = linalg
        
        @staticmethod
        def array(data):
            return data
        
        @staticmethod
        def median(data):
            sorted_data = sorted(data)
            n = len(sorted_data)
            if n % 2 == 0:
                return (sorted_data[n//2-1] + sorted_data[n//2]) / 2
            else:
                return sorted_data[n//2]
        
        @staticmethod
        def std(data):
            if not data:
                return 0
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            return math.sqrt(variance)
        
        @staticmethod
        def zeros(size, dtype=None):
            return [0] * size

from models.types import Block, PageParse


class DrawingProcessor:
    """Specialized processor for drawing blocks with spatial clustering."""
    
    def __init__(self):
        """Initialize drawing processor."""
        pass
    
    def process_drawing_page(
        self, 
        blocks: List[Block], 
        page: PageParse,
        cluster_text: bool = True,
        max_regions: int = 8
    ) -> Tuple[str, List[List[float]], List[Tuple[str, List[List[float]]]]]:
        """
        Process drawing blocks for a page.
        
        Args:
            blocks: Drawing and titleblock blocks
            page: Page information
            cluster_text: Whether to cluster text regions
            max_regions: Maximum number of regions to create
            
        Returns:
            Tuple of (combined_text, all_bboxes, clustered_regions)
            where clustered_regions is list of (region_text, region_bboxes)
        """
        if not blocks:
            return "", [], []
        
        # Extract text and bboxes
        texts = []
        bboxes = []
        
        for block in blocks:
            text = block["text"].strip()
            bbox = block.get("bbox", [])
            
            if text and bbox and len(bbox) == 4:
                texts.append(text)
                bboxes.append(bbox)
        
        if not texts:
            return "", [], []
        
        combined_text = '\n'.join(texts)
        
        # If clustering disabled or insufficient data, return single region
        if not cluster_text or len(texts) < 3:
            return combined_text, bboxes, [(combined_text, bboxes)]
        
        # Perform spatial clustering
        clustered_regions = self._cluster_drawing_regions(
            texts, bboxes, page, max_regions
        )
        
        return combined_text, bboxes, clustered_regions
    
    def _cluster_drawing_regions(
        self,
        texts: List[str],
        bboxes: List[List[float]],
        page: PageParse,
        max_regions: int
    ) -> List[Tuple[str, List[List[float]]]]:
        """
        Cluster drawing text regions using spatial analysis.
        
        Args:
            texts: Text content for each region
            bboxes: Bounding boxes for each text region
            page: Page information for normalization
            max_regions: Maximum number of regions
            
        Returns:
            List of (region_text, region_bboxes) tuples
        """
        if len(texts) < 2:
            return [(texts[0] if texts else "", bboxes)]
        
        # Calculate centers and normalize to page dimensions
        centers = self._calculate_normalized_centers(bboxes, page)
        
        if centers is None or len(centers) < 2:
            return [('\n'.join(texts), bboxes)]
        
        # Apply DBSCAN clustering
        clusters = self._apply_dbscan_clustering(centers)
        
        # Group texts and bboxes by cluster
        clustered_regions = self._group_by_clusters(
            texts, bboxes, clusters, max_regions
        )
        
        return clustered_regions
    
    def _calculate_normalized_centers(
        self, 
        bboxes: List[List[float]], 
        page: PageParse
    ) -> Optional[list]:
        """Calculate normalized center points from bounding boxes."""
        try:
            centers = []
            for bbox in bboxes:
                if len(bbox) != 4:
                    continue
                
                x0, y0, x1, y1 = bbox
                x_center = (x0 + x1) / 2
                y_center = (y0 + y1) / 2
                centers.append([x_center, y_center])
            
            if not centers:
                return None
            
            # Normalize to page dimensions
            page_width = page.get("width", 1)
            page_height = page.get("height", 1)
            
            if page_width > 0 and page_height > 0:
                normalized_centers = []
                for x, y in centers:
                    normalized_centers.append([x / page_width, y / page_height])
                centers = normalized_centers
            
            if SKLEARN_AVAILABLE:
                return np.array(centers)
            else:
                return centers
            
        except Exception:
            return None
    
    def _apply_dbscan_clustering(self, centers) -> list:
        """Apply DBSCAN clustering to center points."""
        if not SKLEARN_AVAILABLE:
            # Fallback: simple distance-based clustering
            return self._simple_clustering(centers)
        
        try:
            # Adaptive eps based on data distribution
            eps = self._calculate_adaptive_eps(centers)
            min_samples = max(2, min(3, len(centers) // 3))
            
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centers)
            return clustering.labels_
            
        except Exception:
            # Fallback: all points in one cluster
            return [0] * len(centers)
    
    def _simple_clustering(self, centers) -> list:
        """Simple distance-based clustering fallback."""
        if len(centers) <= 2:
            return [0] * len(centers)
        
        # Simple clustering based on distance threshold
        clusters = []
        cluster_id = 0
        threshold = 0.2  # 20% of normalized space
        
        for i, center in enumerate(centers):
            assigned = False
            
            # Check if point belongs to existing cluster
            for j in range(i):
                other_center = centers[j]
                distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(center, other_center)))
                
                if distance < threshold:
                    clusters.append(clusters[j])
                    assigned = True
                    break
            
            if not assigned:
                clusters.append(cluster_id)
                cluster_id += 1
        
        return clusters
    
    def _calculate_adaptive_eps(self, centers) -> float:
        """Calculate adaptive eps parameter for DBSCAN."""
        try:
            # Calculate pairwise distances
            n_points = len(centers)
            if n_points < 2:
                return 0.15
            
            distances = []
            for i in range(n_points):
                for j in range(i + 1, n_points):
                    if SKLEARN_AVAILABLE:
                        dist = np.linalg.norm(centers[i] - centers[j])
                    else:
                        # Manual distance calculation
                        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(centers[i], centers[j])))
                    distances.append(dist)
            
            if not distances:
                return 0.15
            
            # Use median distance as base, scaled by density
            median_dist = np.median(distances)
            
            # Adaptive scaling based on point density
            if n_points <= 5:
                scale = 1.5
            elif n_points <= 10:
                scale = 1.2
            else:
                scale = 1.0
            
            eps = median_dist * scale
            
            # Clamp to reasonable range
            return max(0.05, min(0.3, eps))
            
        except Exception:
            return 0.15
    
    def _group_by_clusters(
        self,
        texts: List[str],
        bboxes: List[List[float]],
        cluster_labels: list,
        max_regions: int
    ) -> List[Tuple[str, List[List[float]]]]:
        """Group texts and bboxes by cluster labels."""
        try:
            # Group by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append((texts[i], bboxes[i]))
            
            # Sort clusters by size (largest first)
            sorted_clusters = sorted(
                clusters.items(), 
                key=lambda x: len(x[1]), 
                reverse=True
            )
            
            # Limit to max_regions
            regions = []
            for cluster_id, cluster_items in sorted_clusters[:max_regions]:
                cluster_texts = [item[0] for item in cluster_items]
                cluster_bboxes = [item[1] for item in cluster_items]
                
                region_text = '\n'.join(cluster_texts)
                regions.append((region_text, cluster_bboxes))
            
            return regions if regions else [('\n'.join(texts), bboxes)]
            
        except Exception:
            # Fallback: single region
            return [('\n'.join(texts), bboxes)]
    
    def analyze_drawing_layout(
        self, 
        blocks: List[Block], 
        page: PageParse
    ) -> Dict[str, Any]:
        """
        Analyze drawing layout characteristics.
        
        Args:
            blocks: Drawing blocks
            page: Page information
            
        Returns:
            Dictionary with layout analysis
        """
        analysis = {
            "total_text_regions": 0,
            "text_density": 0.0,
            "spatial_distribution": "unknown",
            "dominant_regions": [],
            "title_block_detected": False,
            "drawing_type": "unknown"
        }
        
        if not blocks:
            return analysis
        
        # Count text regions
        text_blocks = [b for b in blocks if b["text"].strip()]
        analysis["total_text_regions"] = len(text_blocks)
        
        if not text_blocks:
            return analysis
        
        # Calculate text density (text area / page area)
        page_area = page.get("width", 1) * page.get("height", 1)
        total_text_area = 0
        
        for block in text_blocks:
            bbox = block.get("bbox", [])
            if len(bbox) == 4:
                x0, y0, x1, y1 = bbox
                area = (x1 - x0) * (y1 - y0)
                total_text_area += area
        
        analysis["text_density"] = total_text_area / page_area if page_area > 0 else 0
        
        # Detect title block
        title_blocks = [b for b in blocks if b["type"] == "titleblock"]
        analysis["title_block_detected"] = len(title_blocks) > 0
        
        # Analyze spatial distribution
        if len(text_blocks) >= 3:
            bboxes = [b["bbox"] for b in text_blocks if len(b.get("bbox", [])) == 4]
            if bboxes:
                analysis["spatial_distribution"] = self._analyze_spatial_distribution(bboxes, page)
        
        # Determine drawing type based on characteristics
        analysis["drawing_type"] = self._classify_drawing_type(blocks, analysis)
        
        return analysis
    
    def _analyze_spatial_distribution(
        self, 
        bboxes: List[List[float]], 
        page: PageParse
    ) -> str:
        """Analyze spatial distribution of text regions."""
        try:
            centers = self._calculate_normalized_centers(bboxes, page)
            if centers is None or len(centers) < 3:
                return "sparse"
            
            # Calculate spread in x and y directions
            x_spread = np.std(centers[:, 0])
            y_spread = np.std(centers[:, 1])
            
            # Classify distribution
            if x_spread > 0.3 and y_spread > 0.3:
                return "distributed"
            elif x_spread > 0.3:
                return "horizontal"
            elif y_spread > 0.3:
                return "vertical"
            else:
                return "clustered"
                
        except Exception:
            return "unknown"
    
    def _classify_drawing_type(
        self, 
        blocks: List[Block], 
        analysis: Dict[str, Any]
    ) -> str:
        """Classify drawing type based on content and layout."""
        # Simple heuristics for drawing type classification
        text_content = ' '.join(b["text"].lower() for b in blocks if b["text"].strip())
        
        # Check for common drawing type indicators
        if any(term in text_content for term in ["plan", "floor", "layout"]):
            return "floor_plan"
        elif any(term in text_content for term in ["elevation", "section", "detail"]):
            return "elevation_section"
        elif any(term in text_content for term in ["electrical", "power", "lighting"]):
            return "electrical"
        elif any(term in text_content for term in ["plumbing", "water", "sewer"]):
            return "plumbing"
        elif any(term in text_content for term in ["hvac", "mechanical", "duct"]):
            return "mechanical"
        elif any(term in text_content for term in ["structural", "beam", "column"]):
            return "structural"
        elif analysis["title_block_detected"]:
            return "technical_drawing"
        else:
            return "general"
    
    def optimize_region_boundaries(
        self, 
        regions: List[Tuple[str, List[List[float]]]]
    ) -> List[Tuple[str, List[List[float]]]]:
        """
        Optimize region boundaries to minimize overlap and improve coherence.
        
        Args:
            regions: List of (text, bboxes) tuples
            
        Returns:
            Optimized regions
        """
        if len(regions) <= 1:
            return regions
        
        # For now, return regions as-is
        # Future enhancement: implement boundary optimization
        return regions