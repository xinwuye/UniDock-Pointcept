import torch
import numpy as np
import math
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import json
import os

# Atomic numbers for tie-breaking rule 2
ATOMIC_NUMBERS = {
    "H": 1, "LI": 3, "BE": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9,
    "NA": 11, "MG": 12, "AL": 13, "SI": 14, "P": 15, "S": 16, "CL": 17,
    "K": 19, "CA": 20, "MN": 25, "FE": 26, "CO": 27, "NI": 28, "CU": 29,
    "ZN": 30, "BR": 35, "SR": 38, "SE": 34, "I": 53, "CS": 55, "HG": 80
}

class BWMSCoder:
    def __init__(self, atom_types_json_path):
        with open(atom_types_json_path, "r") as f:
            self.atom_types = json.load(f)
        
        self.c_idx = self.atom_types.index("C") if "C" in self.atom_types else -1
        # Pre-calculate atomic weights for sorting
        self.atomic_weights = []
        for at in self.atom_types:
            # Clean type string (e.g., 'FE' -> 'FE')
            clean_at = at.upper()
            self.atomic_weights.append(ATOMIC_NUMBERS.get(clean_at, 0))
        self.atomic_weights = np.array(self.atomic_weights)

    def get_weight_multiplier(self, type_i, type_j):
        """
        Phase 2: Bio-chemical Weighting Matrix M
        """
        is_c_i = (type_i == self.c_idx)
        is_c_j = (type_j == self.c_idx)
        
        # Rule: C-C (0.1)
        if is_c_i and is_c_j:
            return 0.1
        # Rule: C-Others (1.0)
        if is_c_i or is_c_j:
            return 1.0
        
        # Rule: Others-Others (2.0)
        # Rule: Metal-Others (10.0) - Simulating by checking common metals in your list
        metals = {"FE", "CU", "ZN", "MG", "MN", "NI", "CO", "NA", "K", "LI", "HG", "SR", "CS"}
        at_i = self.atom_types[type_i]
        at_j = self.atom_types[type_j]
        if at_i in metals or at_j in metals:
            return 10.0
        
        return 2.0

    @torch.no_grad()
    def encode(self, coord, atom_type_onehot, cutoff=5.0):
        """
        Bio-Weighted MST Serialization (BWMS)
        coord: (N, 3) tensor or ndarray
        atom_type_onehot: (N, C) one-hot tensor or ndarray
        """
        if torch.is_tensor(coord): coord = coord.cpu().numpy()
        if torch.is_tensor(atom_type_onehot): atom_type_onehot = atom_type_onehot.cpu().numpy()
        
        n = coord.shape[0]
        if n == 0: return np.array([], dtype=np.int64)
        
        # Phase 1: Spatial Indexing & Graph Construction
        tree = KDTree(coord)
        pairs = tree.query_pairs(r=cutoff)
        
        # Get atom types (indices)
        types = np.argmax(atom_type_onehot, axis=1)
        
        # Phase 2: Weighting
        rows, cols, weights = [], [], []
        for i, j in pairs:
            dist = np.linalg.norm(coord[i] - coord[j])
            w = dist * self.get_weight_multiplier(types[i], types[j])
            rows.append(i); cols.append(j); weights.append(w)
            rows.append(j); cols.append(i); weights.append(w)
            
        if not weights: # Fallback for isolated atoms
            return np.arange(n).astype(np.int64)
            
        graph = csr_matrix((weights, (rows, cols)), shape=(n, n))
        
        # Phase 3: MST Computation (Kruskal)
        mst = minimum_spanning_tree(graph)
        
        # Phase 4: Deterministic Traversal
        # Build adjacency list from MST
        adj = [[] for _ in range(n)]
        cx = mst.tocoo()
        for i, j, v in zip(cx.row, cx.col, cx.data):
            adj[i].append(j)
            adj[j].append(i)
            
        # Rooting: Furthest from centroid
        centroid = coord.mean(axis=0)
        dists_to_centroid = np.linalg.norm(coord - centroid, axis=1)
        root = np.argmax(dists_to_centroid)
        
        # DFS with Tie-breaking
        order = []
        visited = np.zeros(n, dtype=bool)
        stack = [root]
        
        while stack:
            u = stack.pop()
            if not visited[u]:
                visited[u] = True
                order.append(u)
                
                # Sort neighbors for deterministic traversal
                neighbors = adj[u]
                if not neighbors: continue
                
                # Tie-breaking rules (reversed because of stack.pop())
                # Priority: Carbon (rule 1) > High Atomic Num (rule 2) > Closest (rule 3)
                
                def sort_key(v):
                    is_c = 1 if types[v] == self.c_idx else 0
                    atomic_z = self.atomic_weights[types[v]]
                    dist_to_u = np.linalg.norm(coord[v] - coord[u])
                    # Higher is better for stack pop order
                    return (is_c, atomic_z, -dist_to_u)
                
                sorted_neighbors = sorted(neighbors, key=sort_key)
                for v in sorted_neighbors:
                    if not visited[v]:
                        stack.append(v)
        
        # If graph was disconnected, handle remaining nodes
        if len(order) < n:
            remaining = np.where(~visited)[0]
            # Simple fallback for disconnected components
            order.extend(remaining.tolist())
            
        # The model expects a 'code' that when argsorted gives the sequence.
        # So the code is the rank/position in the DFS order.
        code = np.zeros(n, dtype=np.int64)
        for rank, idx in enumerate(order):
            code[idx] = rank
            
        return code

# Global cache to avoid repeated JSON loading for the same path
_coders = {}

def get_bwms_coder(atom_types_json_path):
    global _coders
    # If it's a list (from collate_fn), take the first element
    if isinstance(atom_types_json_path, (list, tuple)):
        atom_types_json_path = atom_types_json_path[0]
    if atom_types_json_path not in _coders:
        if not os.path.exists(atom_types_json_path):
            raise FileNotFoundError(f"BWMS requires atom_types.json at: {atom_types_json_path}")
        _coders[atom_types_json_path] = BWMSCoder(atom_types_json_path)
    return _coders[atom_types_json_path]

