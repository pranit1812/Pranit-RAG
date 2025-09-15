"""
Bounding box manipulation utilities for coordinate handling.
"""
from typing import List, Tuple, Optional, Union
import math


def normalize_bbox(bbox: List[float], page_width: float, page_height: float) -> List[float]:
    """
    Normalize bounding box coordinates to 0-1 range.
    
    Args:
        bbox: Bounding box [x0, y0, x1, y1] in absolute coordinates
        page_width: Page width in points/pixels
        page_height: Page height in points/pixels
        
    Returns:
        Normalized bounding box [x0, y0, x1, y1] in 0-1 range
    """
    x0, y0, x1, y1 = bbox
    return [
        x0 / page_width,
        y0 / page_height,
        x1 / page_width,
        y1 / page_height
    ]


def denormalize_bbox(bbox: List[float], page_width: float, page_height: float) -> List[float]:
    """
    Convert normalized bounding box back to absolute coordinates.
    
    Args:
        bbox: Normalized bounding box [x0, y0, x1, y1] in 0-1 range
        page_width: Page width in points/pixels
        page_height: Page height in points/pixels
        
    Returns:
        Absolute bounding box [x0, y0, x1, y1]
    """
    x0, y0, x1, y1 = bbox
    return [
        x0 * page_width,
        y0 * page_height,
        x1 * page_width,
        y1 * page_height
    ]


def bbox_area(bbox: List[float]) -> float:
    """
    Calculate the area of a bounding box.
    
    Args:
        bbox: Bounding box [x0, y0, x1, y1]
        
    Returns:
        Area of the bounding box
    """
    x0, y0, x1, y1 = bbox
    return max(0, x1 - x0) * max(0, y1 - y0)


def bbox_intersection(bbox1: List[float], bbox2: List[float]) -> Optional[List[float]]:
    """
    Calculate intersection of two bounding boxes.
    
    Args:
        bbox1: First bounding box [x0, y0, x1, y1]
        bbox2: Second bounding box [x0, y0, x1, y1]
        
    Returns:
        Intersection bounding box or None if no intersection
    """
    x0 = max(bbox1[0], bbox2[0])
    y0 = max(bbox1[1], bbox2[1])
    x1 = min(bbox1[2], bbox2[2])
    y1 = min(bbox1[3], bbox2[3])
    
    if x0 < x1 and y0 < y1:
        return [x0, y0, x1, y1]
    return None


def bbox_union(bbox1: List[float], bbox2: List[float]) -> List[float]:
    """
    Calculate union (bounding box) of two bounding boxes.
    
    Args:
        bbox1: First bounding box [x0, y0, x1, y1]
        bbox2: Second bounding box [x0, y0, x1, y1]
        
    Returns:
        Union bounding box
    """
    return [
        min(bbox1[0], bbox2[0]),
        min(bbox1[1], bbox2[1]),
        max(bbox1[2], bbox2[2]),
        max(bbox1[3], bbox2[3])
    ]


def bbox_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        bbox1: First bounding box [x0, y0, x1, y1]
        bbox2: Second bounding box [x0, y0, x1, y1]
        
    Returns:
        IoU value between 0 and 1
    """
    intersection = bbox_intersection(bbox1, bbox2)
    if intersection is None:
        return 0.0
    
    intersection_area = bbox_area(intersection)
    union_area = bbox_area(bbox1) + bbox_area(bbox2) - intersection_area
    
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area


def bbox_contains_point(bbox: List[float], point: Tuple[float, float]) -> bool:
    """
    Check if a point is contained within a bounding box.
    
    Args:
        bbox: Bounding box [x0, y0, x1, y1]
        point: Point coordinates (x, y)
        
    Returns:
        True if point is inside bbox, False otherwise
    """
    x, y = point
    x0, y0, x1, y1 = bbox
    return x0 <= x <= x1 and y0 <= y <= y1


def bbox_center(bbox: List[float]) -> Tuple[float, float]:
    """
    Calculate the center point of a bounding box.
    
    Args:
        bbox: Bounding box [x0, y0, x1, y1]
        
    Returns:
        Center point (x, y)
    """
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) / 2, (y0 + y1) / 2)


def bbox_distance(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate distance between centers of two bounding boxes.
    
    Args:
        bbox1: First bounding box [x0, y0, x1, y1]
        bbox2: Second bounding box [x0, y0, x1, y1]
        
    Returns:
        Euclidean distance between centers
    """
    center1 = bbox_center(bbox1)
    center2 = bbox_center(bbox2)
    
    dx = center1[0] - center2[0]
    dy = center1[1] - center2[1]
    
    return math.sqrt(dx * dx + dy * dy)


def merge_bboxes(bboxes: List[List[float]]) -> Optional[List[float]]:
    """
    Merge multiple bounding boxes into a single encompassing bbox.
    
    Args:
        bboxes: List of bounding boxes [x0, y0, x1, y1]
        
    Returns:
        Merged bounding box or None if input is empty
    """
    if not bboxes:
        return None
    
    if len(bboxes) == 1:
        return bboxes[0].copy()
    
    merged = bboxes[0].copy()
    for bbox in bboxes[1:]:
        merged = bbox_union(merged, bbox)
    
    return merged


def validate_bbox(bbox: List[float]) -> bool:
    """
    Validate that a bounding box has correct format and values.
    
    Args:
        bbox: Bounding box [x0, y0, x1, y1]
        
    Returns:
        True if bbox is valid, False otherwise
    """
    if len(bbox) != 4:
        return False
    
    x0, y0, x1, y1 = bbox
    
    # Check that coordinates are numbers
    if not all(isinstance(coord, (int, float)) for coord in bbox):
        return False
    
    # Check that x1 >= x0 and y1 >= y0
    return x1 >= x0 and y1 >= y0