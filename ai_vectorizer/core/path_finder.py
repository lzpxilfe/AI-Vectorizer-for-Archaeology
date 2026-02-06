# -*- coding: utf-8 -*-
"""
Path Finding Module for AI Vectorizer
Implements A* algorithm to find optimal paths along contour lines.
"""

import numpy as np
import heapq

class PathFinder:
    def __init__(self):
        pass

    def heuristic(self, a, b):
        """Euclidean distance heuristic."""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def find_path(self, start_pt, end_pt, cost_map: np.ndarray):
        """
        A* Pathfinding on the cost map.
        
        Args:
            start_pt (tuple): (x, y) starting pixel coordinates.
            end_pt (tuple): (x, y) ending pixel coordinates.
            cost_map (np.ndarray): 2D array of costs (height, width).
            
        Returns:
            list: List of (x, y) tuples representing the path.
        """
        # Convert inputs to int
        start = (int(start_pt[1]), int(start_pt[0])) # (row, col)
        end = (int(end_pt[1]), int(end_pt[0]))       # (row, col)
        
        rows, cols = cost_map.shape
        
        # Priority Queue: (cost, current_node)
        frontier = []
        heapq.heappush(frontier, (0, start))
        
        came_from = {}
        cost_so_far = {}
        
        came_from[start] = None
        cost_so_far[start] = 0
        
        # 8-connected neighbors
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), 
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]

        found = False
        
        # Safety break for very long searches
        max_steps = 50000
        steps = 0

        while frontier:
            steps += 1
            if steps > max_steps:
                print("Pathfinding timeout")
                break
                
            current_priority, current = heapq.heappop(frontier)
            
            if current == end:
                found = True
                break
            
            # Optimization: if we are close enough to end? 
            # For now, exact match.

            for dx, dy in neighbors:
                next_node = (current[0] + dx, current[1] + dy)
                
                # Check bounds
                if 0 <= next_node[0] < rows and 0 <= next_node[1] < cols:
                    # Calculate new cost
                    # Base move cost is 1 (or 1.414 for diagonal) * map cost
                    move_cost = 1.414 if dx != 0 and dy != 0 else 1.0
                    
                    # Add map cost (pixel value)
                    node_cost = cost_map[next_node]
                    
                    new_cost = cost_so_far[current] + (move_cost * node_cost)
                    
                    if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                        cost_so_far[next_node] = new_cost
                        came_from[next_node] = current  # Critical for path reconstruction
                        priority = new_cost + self.heuristic(end, next_node)
                        heapq.heappush(frontier, (priority, next_node))
        
        if not found:
            return []
            
        # Reconstruct path
        path = []
        curr = end
        while curr != start:
            path.append((curr[1], curr[0])) # Convert back to (x, y)
            curr = came_from[curr]
        path.append((start[1], start[0]))
        path.reverse()
        
        return path
