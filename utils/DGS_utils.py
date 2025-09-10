import numpy as np
from scipy.spatial import cKDTree
from sknw import build_sknw
from collections import defaultdict
import networkx as nx
import torch
from skimage.morphology import skeletonize
import cv2 # Still needed for drawing masks in example

def aggregate_graph_nodes(graph: nx.Graph, tolerance: float = 2.0):
    """
    聚合 NetworkX 图中地理位置上非常接近的节点。

    Args:
        graph (nx.Graph): sknw.build_sknw 构建的 NetworkX 图。
        tolerance (float): 聚合的距离阈值。如果两个节点的距离小于此阈性，则将它们视为同一个节点。

    Returns:
        nx.Graph: 聚合后的新图。
        dict: 原始节点到新图节点ID的映射。
    """
    node_coords = np.array([graph.nodes[node_id]['o'] for node_id in graph.nodes()])
    original_node_ids = list(graph.nodes())

    if not original_node_ids:
        return graph.copy(), {}

    kdtree = cKDTree(node_coords)
    pairs_to_merge = kdtree.query_pairs(r=tolerance)

    parent = {node_id: node_id for node_id in original_node_ids}

    def find(node_id):
        if parent[node_id] == node_id:
            return node_id
        parent[node_id] = find(parent[node_id])
        return parent[node_id]

    def union(node1_id, node2_id):
        root1 = find(node1_id)
        root2 = find(node2_id)
        if root1 != root2:
            parent[root2] = root1
            return True
        return False

    for idx1, idx2 in pairs_to_merge:
        union(original_node_ids[idx1], original_node_ids[idx2])

    aggregated_nodes_map = defaultdict(list)
    aggregated_coords = {}

    for original_node_id in original_node_ids:
        root_id = find(original_node_id)
        aggregated_nodes_map[root_id].append(original_node_id)

    for root_id, component_nodes in aggregated_nodes_map.items():
        coords_in_component = np.array([graph.nodes[nid]['o'] for nid in component_nodes])
        aggregated_coords[root_id] = np.mean(coords_in_component, axis=0)

    new_graph = nx.Graph()
    original_to_new_node_id_map = {}

    new_node_counter = 0
    for root_id in sorted(aggregated_nodes_map.keys()):
        new_node_id = f"agg_node_{new_node_counter}"
        new_graph.add_node(new_node_id, o=aggregated_coords[root_id])
        for original_node_id in aggregated_nodes_map[root_id]:
            original_to_new_node_id_map[original_node_id] = new_node_id
        new_node_counter += 1

    for u, v, data in graph.edges(data=True):
        new_u = original_to_new_node_id_map[u]
        new_v = original_to_new_node_id_map[v]
        if new_u != new_v:
            if not new_graph.has_edge(new_u, new_v):
                new_graph.add_edge(new_u, new_v, pts=data['pts'].tolist())
            else:
                existing_pts = new_graph[new_u][new_v].get('pts', [])
                new_pts = data['pts'].tolist()

                if len(existing_pts) > 0 and np.allclose(np.array(existing_pts[-1]), np.array(new_pts[0]), atol=tolerance):
                    new_graph[new_u][new_v]['pts'] = existing_pts + new_pts[1:]
                elif len(existing_pts) > 0 and np.allclose(np.array(existing_pts[0]), np.array(new_pts[-1]), atol=tolerance):
                    new_graph[new_u][new_v]['pts'] = new_pts[:-1] + existing_pts
                else:
                    new_graph[new_u][new_v]['pts'] = existing_pts + new_pts

    for u, v, data in new_graph.edges(data=True):
        if 'pts' in data:
            new_graph[u][v]['pts'] = np.array(data['pts'])

    return new_graph, original_to_new_node_id_map


def create_adjacency_matrix_from_skeleton(node_tensor: torch.Tensor, score_tensor: torch.Tensor, mask_true: torch.Tensor, score_threshold: float = 0.5, dist_threshold: float = 5, node_aggregation_tolerance: float = 2.0, bidirectional_roads: bool = False):
    """
    根据骨架线分析结果和预测节点，构建预测节点间的邻接矩阵。

    Args:
        node_tensor (torch.Tensor): 预测的节点坐标，形状为 [B, N, 2]，其中 B 是批量大小，N 是节点数量。
        score_tensor (torch.Tensor): 预测节点的置信度，形状为 [B, N]。
        mask_true (torch.Tensor): 道路二值分割标签，形状为 [B, 1, H, W]。
        score_threshold (float): 用于过滤低置信度预测节点的阈值。
        dist_threshold (float): 预测节点与骨架线上点匹配的最大距离。
        node_aggregation_tolerance (float): 在骨架图中聚合节点时使用的距离阈值，
                                            用于处理浮点误差导致的交叉点不一致。
        bidirectional_roads (bool): 如果为True，则为每条道路片段添加双向连接。
                                    如果为False，则只添加骨架线片段方向的单向连接。

    Returns:
        torch.Tensor: 构建的邻接矩阵标签，形状为 [B, N, N]。
    """
    mask_true_squeezed = mask_true.squeeze(1)

    node_tensor_np = node_tensor.cpu().numpy()
    score_tensor_np = score_tensor.squeeze(2).cpu().numpy()
    mask_true_np = mask_true_squeezed.cpu().numpy()

    B, N, _ = node_tensor_np.shape
    adjacency_matrix_label_np = np.zeros((B, N, N), dtype=np.float32)

    for b in range(B):
        current_node_coords = node_tensor_np[b]
        current_node_scores = score_tensor_np[b]

        # 1. 骨架线分析并聚合节点
        skeleton_bool = skeletonize(mask_true_np[b].astype(bool))
        skeleton = (skeleton_bool * 255).astype(np.uint8) 

        graph = build_sknw(skeleton)
        aggregated_graph, _ = aggregate_graph_nodes(graph, tolerance=node_aggregation_tolerance)

        skeleton_paths = []
        for u, v, data in aggregated_graph.edges(data=True):
            if 'pts' in data and len(data['pts']) > 0:
                skeleton_paths.append(data['pts'])
        
        all_skeleton_points = np.vstack(skeleton_paths) if skeleton_paths else np.empty((0, 2))

        if all_skeleton_points.shape[0] == 0:
            for i in range(N):
                adjacency_matrix_label_np[b, i, i] = 1.0
            continue
            
        # 2. 从真值骨架点出发，查找最近的高置信度预测节点
        high_conf_predicted_coords = current_node_coords[current_node_scores >= score_threshold]
        high_conf_predicted_indices = np.where(current_node_scores >= score_threshold)[0]
        # Map from high_conf_predicted_coords index to original node index
        high_conf_coord_to_original_idx_map = {tuple(coord): original_idx for coord, original_idx in zip(high_conf_predicted_coords, high_conf_predicted_indices)}

        if high_conf_predicted_coords.shape[0] == 0:
            for i in range(N):
                adjacency_matrix_label_np[b, i, i] = 1.0
            continue

        predicted_kdtree = cKDTree(high_conf_predicted_coords)

        # map: skeleton_point_coord_tuple -> matched_predicted_node_idx (original index)
        skeleton_point_to_matched_predicted_node_idx = {}
        matched_predicted_nodes_set = set() # Set of original predicted node indices that got matched

        for skel_coord_raw in all_skeleton_points:
            skel_coord_tuple = tuple(skel_coord_raw)
            distances, nearest_pred_idx_in_high_conf_list = predicted_kdtree.query(skel_coord_raw, k=1)

            if distances < dist_threshold:
                matched_pred_coord = high_conf_predicted_coords[nearest_pred_idx_in_high_conf_list]
                p_node_idx = high_conf_coord_to_original_idx_map[tuple(matched_pred_coord)]
                
                # Check if this skeleton point has already been claimed by a better (closer) predicted node
                # Or if the current predicted node is a better match for this skeleton point.
                # For simplicity, assign directly. More complex strategy might involve a "best match" selection
                # if multiple predicted nodes are close to the same skeleton point.
                skeleton_point_to_matched_predicted_node_idx[skel_coord_tuple] = p_node_idx
                matched_predicted_nodes_set.add(p_node_idx)

        # 3. 构建邻接矩阵
        # Self-connect low-confidence nodes OR high-confidence nodes that did not match any skeleton point
        for i in range(N):
            if current_node_scores[i] < score_threshold or i not in matched_predicted_nodes_set:
                adjacency_matrix_label_np[b, i, i] = 1.0

        # Build directed connections based on skeleton paths and their inherent order
        for path_points in skeleton_paths: # Iterate through each individual path segment (e.g., from A to B)
            # Find all high-confidence predicted nodes that match a point on THIS path,
            # and store their original index in the path.
            matched_nodes_on_this_path = [] # List of (predicted_node_idx, skel_point_idx_in_path)

            for skel_point_idx, skel_coord_raw in enumerate(path_points):
                skel_coord_tuple = tuple(skel_coord_raw)
                if skel_coord_tuple in skeleton_point_to_matched_predicted_node_idx:
                    p_node_idx = skeleton_point_to_matched_predicted_node_idx[skel_coord_tuple]
                    # Ensure it's a high-confidence node
                    if current_node_scores[p_node_idx] >= score_threshold:
                        matched_nodes_on_this_path.append((p_node_idx, skel_point_idx))
            
            # Sort matched nodes by their position along the current skeleton path
            matched_nodes_on_this_path.sort(key=lambda x: x[1])

            # Establish directed connections along this segment
            # The direction is implicitly given by the order in 'path_points'
            for i in range(len(matched_nodes_on_this_path) - 1):
                p_node_idx1, _ = matched_nodes_on_this_path[i]
                p_node_idx2, _ = matched_nodes_on_this_path[i+1]

                if p_node_idx1 != p_node_idx2: # Ensure connecting distinct predicted nodes
                    # Directed connection from p_node_idx1 to p_node_idx2 (following skeleton path direction)
                    adjacency_matrix_label_np[b, p_node_idx1, p_node_idx2] = 1.0
                    
                    if bidirectional_roads:
                        # For bidirectional roads, also add the reverse connection
                        adjacency_matrix_label_np[b, p_node_idx2, p_node_idx1] = 1.0 
    
    return adjacency_matrix_label_np

