import cv2
import numpy as np
import pandas as pd
import insightface
from insightface.app import FaceAnalysis
import csv
from collections import defaultdict, deque
import sys
import os
import glob
import time
import argparse
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from scipy.spatial.distance import cosine
import faiss
from typing import List, Dict, Tuple, Optional

class Tracklet:
    def __init__(self, track_id: int, initial_face_data: dict):
        self.track_id = track_id
        self.faces = [initial_face_data]
        self.frames = [initial_face_data['frame_number']]
        self.bboxes = [initial_face_data['bbox']]
        self.embeddings = [initial_face_data['embedding']]
        self.person_id = None
        
    def add_face(self, face_data: dict):
        self.faces.append(face_data)
        self.frames.append(face_data['frame_number'])
        self.bboxes.append(face_data['bbox'])
        self.embeddings.append(face_data['embedding'])
    
    def get_quality_weighted_embedding(self) -> np.ndarray:
        if len(self.embeddings) == 1:
            return self.embeddings[0]
        
        weights = []
        for bbox in self.bboxes:
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            weights.append(area)
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        weighted_embedding = np.average(self.embeddings, axis=0, weights=weights)
        return weighted_embedding / np.linalg.norm(weighted_embedding)
    
    def get_medoid_embedding(self) -> np.ndarray:
        if len(self.embeddings) == 1:
            return self.embeddings[0]
        
        embeddings_array = np.array(self.embeddings)
        similarities = cosine_similarity(embeddings_array)
        medoid_idx = np.argmax(similarities.sum(axis=1))
        return embeddings_array[medoid_idx]

class FaceTracker:
    def __init__(self, iou_threshold: float = 0.3, max_disappeared: int = 30):
        self.tracklets = {}
        self.next_track_id = 0
        self.iou_threshold = iou_threshold
        self.max_disappeared = max_disappeared
        self.disappeared = {}
    
    def update(self, frame_faces: List[dict]) -> List[dict]:
        if not frame_faces:
            for track_id in list(self.tracklets.keys()):
                self.disappeared[track_id] = self.disappeared.get(track_id, 0) + 1
                if self.disappeared[track_id] > self.max_disappeared:
                    del self.tracklets[track_id]
                    del self.disappeared[track_id]
            return []
        
        if not self.tracklets:
            for face_data in frame_faces:
                self.tracklets[self.next_track_id] = Tracklet(self.next_track_id, face_data)
                self.disappeared[self.next_track_id] = 0
                self.next_track_id += 1
            return frame_faces
        
        track_ids = list(self.tracklets.keys())
        iou_matrix = np.zeros((len(track_ids), len(frame_faces)))
        
        for i, track_id in enumerate(track_ids):
            last_bbox = self.tracklets[track_id].bboxes[-1]
            for j, face_data in enumerate(frame_faces):
                iou_matrix[i, j] = self._calculate_iou(last_bbox, face_data['bbox'])
        
        from scipy.optimize import linear_sum_assignment
        track_indices, face_indices = linear_sum_assignment(-iou_matrix)
        
        matched_faces = set()
        for track_idx, face_idx in zip(track_indices, face_indices):
            if iou_matrix[track_idx, face_idx] >= self.iou_threshold:
                track_id = track_ids[track_idx]
                self.tracklets[track_id].add_face(frame_faces[face_idx])
                self.disappeared[track_id] = 0
                matched_faces.add(face_idx)
        
        for track_idx, track_id in enumerate(track_ids):
            if track_idx not in track_indices:
                self.disappeared[track_id] = self.disappeared.get(track_id, 0) + 1
                if self.disappeared[track_id] > self.max_disappeared:
                    del self.tracklets[track_id]
                    del self.disappeared[track_id]
        
        for face_idx, face_data in enumerate(frame_faces):
            if face_idx not in matched_faces:
                self.tracklets[self.next_track_id] = Tracklet(self.next_track_id, face_data)
                self.disappeared[self.next_track_id] = 0
                self.next_track_id += 1
        
        return frame_faces
    
    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def get_tracklets(self) -> List[Tracklet]:
        return list(self.tracklets.values())

class ConstrainedGraphClustering:
    def __init__(self, similarity_threshold: float = 0.6, k_neighbors: int = 10):
        self.similarity_threshold = similarity_threshold
        self.k_neighbors = k_neighbors
    
    def cluster_tracklets(self, tracklets: List[Tracklet], 
                         must_link_constraints: List[Tuple[int, int]] = None,
                         cannot_link_constraints: List[Tuple[int, int]] = None) -> List[int]:
        if len(tracklets) <= 1:
            return [0] * len(tracklets)
        
        embeddings = []
        for tracklet in tracklets:
            embedding = tracklet.get_quality_weighted_embedding()
            embeddings.append(embedding)
        
        embeddings = np.array(embeddings)
        
        graph = self._build_knn_graph(embeddings)
        
        if must_link_constraints:
            self._apply_must_link_constraints(graph, must_link_constraints)
        
        if cannot_link_constraints:
            self._apply_cannot_link_constraints(graph, cannot_link_constraints)
        
        clusters = self._detect_communities(graph)
        
        return clusters
    
    def _build_knn_graph(self, embeddings: np.ndarray) -> nx.Graph:
        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(embeddings.astype('float32'))
        
        k = min(self.k_neighbors + 1, len(embeddings))
        similarities, indices = index.search(embeddings.astype('float32'), k)
        
        graph = nx.Graph()
        for i in range(len(embeddings)):
            graph.add_node(i)
        
        for i in range(len(embeddings)):
            for j, sim in zip(indices[i][1:], similarities[i][1:]):
                if sim >= self.similarity_threshold:
                    graph.add_edge(i, j, weight=sim)
        
        return graph
    
    def _apply_must_link_constraints(self, graph: nx.Graph, constraints: List[Tuple[int, int]]):
        for i, j in constraints:
            if i < len(graph.nodes) and j < len(graph.nodes):
                graph.add_edge(i, j, weight=1.0)
    
    def _apply_cannot_link_constraints(self, graph: nx.Graph, constraints: List[Tuple[int, int]]):
        for i, j in constraints:
            if i < len(graph.nodes) and j < len(graph.nodes):
                if graph.has_edge(i, j):
                    graph.remove_edge(i, j)
    
    def _detect_communities(self, graph: nx.Graph) -> List[int]:
        try:
            import leidenalg as la
            import igraph as ig
            
            edges = list(graph.edges(data=True))
            g_ig = ig.Graph()
            g_ig.add_vertices(len(graph.nodes))
            g_ig.add_edges([(e[0], e[1]) for e in edges])
            g_ig.es['weight'] = [e[2]['weight'] for e in edges]
            
            partition = la.find_partition(g_ig, la.ModularityVertexPartition, weights='weight')
            clusters = [0] * len(graph.nodes)
            for i, cluster_id in enumerate(partition.membership):
                clusters[i] = cluster_id
            
            return clusters
            
        except ImportError:
            return self._louvain_clustering(graph)
    
    def _louvain_clustering(self, graph: nx.Graph) -> List[int]:
        try:
            import community
            
            partition = community.best_partition(graph)
            clusters = [partition[node] for node in range(len(graph.nodes))]
            return clusters
            
        except ImportError:
            return self._connected_components_clustering(graph)
    
    def _connected_components_clustering(self, graph: nx.Graph) -> List[int]:
        components = list(nx.connected_components(graph))
        clusters = [0] * len(graph.nodes)
        for cluster_id, component in enumerate(components):
            for node in component:
                clusters[node] = cluster_id
        return clusters

def load_ground_truth(ground_truth_path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(ground_truth_path):
        print(f" Warning: Ground truth file not found: {ground_truth_path}")
        return None

    try:
        df = pd.read_csv(ground_truth_path, header=0)
        df = df.rename(columns={
            'frame_number': 'frame_number',
            'person_id': 'person_id',
            'x1': 'x1', 'x2': 'x2', 'y1': 'y1', 'y2': 'y2'
        })
        df = df[pd.to_numeric(df['person_id'], errors='coerce').notnull()]
        df['person_id'] = df['person_id'].astype(int)
        print(f" Loaded ground truth: {len(df)} records")
        print(f" Ground truth unique person IDs: {sorted(df['person_id'].unique())}")
        return df

    except Exception as e:
        print(f" Error loading ground truth: {e}")
        return None

def extract_tracklets(video_path: str, model: FaceAnalysis, 
                     max_frames: Optional[int] = None,
                     ground_truth_df: Optional[pd.DataFrame] = None) -> List[Tracklet]:
    print(f"  Extracting tracklets from: {os.path.basename(video_path)}...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    Error: Could not open {video_path}")
        return []
    
    tracker = FaceTracker(iou_threshold=0.3, max_disappeared=30)
    frame_number = 0
    
    while cap.isOpened() and (max_frames is None or frame_number < max_frames):
        ret, frame = cap.read()
        if not ret: 
            break
            
        faces = model.get(frame)
        frame_faces = []
        
        for face in faces:
            face_data = {
                'video_path': video_path,
                'frame_number': frame_number,
                'bbox': face.bbox.astype(int),
                'embedding': face.normed_embedding
            }
            
            if ground_truth_df is not None:
                face_data = match_with_ground_truth(face_data, ground_truth_df)
            
            frame_faces.append(face_data)
        
        tracker.update(frame_faces)
        frame_number += 1
    
    cap.release()
    return tracker.get_tracklets()

def match_with_ground_truth(face_data: dict, ground_truth_df: pd.DataFrame, 
                           iou_threshold: float = 0.3) -> dict:
    frame_num = face_data['frame_number']
    detected_bbox = face_data['bbox']
    
    frame_gt = ground_truth_df[ground_truth_df['frame_number'] == frame_num]
    
    best_iou = 0
    best_person_id = None
    
    for _, gt_row in frame_gt.iterrows():
        gt_bbox = [gt_row['x1'], gt_row['y1'], gt_row['x2'], gt_row['y2']]
        iou = calculate_iou(detected_bbox, gt_bbox)
        
        if iou > best_iou and iou >= iou_threshold:
            best_iou = iou
            best_person_id = gt_row['person_id']
    
    if best_person_id is not None:
        face_data['person_id'] = str(int(best_person_id))
    
    return face_data

def calculate_iou(box1: List[int], box2: List[int]) -> float:
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def create_constraints(tracklets: List[Tracklet]) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    must_link = []
    cannot_link = []
    
    frame_to_tracklets = defaultdict(list)
    for i, tracklet in enumerate(tracklets):
        for frame_num in tracklet.frames:
            frame_to_tracklets[frame_num].append(i)
    
    for frame_num, tracklet_indices in frame_to_tracklets.items():
        if len(tracklet_indices) > 1:
            for i in range(len(tracklet_indices)):
                for j in range(i + 1, len(tracklet_indices)):
                    cannot_link.append((tracklet_indices[i], tracklet_indices[j]))
    
    return must_link, cannot_link

def save_outputs(video_path: str, tracklets: List[Tracklet], output_dir: str):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_csv_path = os.path.join(output_dir, f"{base_name}_tracklet_constrained.csv")
    output_video_path = os.path.join(output_dir, f"{base_name}_tracklet_constrained.mp4")

    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_number', 'person_id', 'x1', 'y1', 'x2', 'y2'])
        
        for tracklet in tracklets:
            if tracklet.person_id is not None:
                for i, frame_num in enumerate(tracklet.frames):
                    bbox = tracklet.bboxes[i]
                    writer.writerow([frame_num, tracklet.person_id, bbox[0], bbox[1], bbox[2], bbox[3]])
    
    print(f"    - Saved annotations to: {output_csv_path}")

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_to_faces = defaultdict(list)
    for tracklet in tracklets:
        if tracklet.person_id is not None:
            for i, frame_num in enumerate(tracklet.frames):
                frame_to_faces[frame_num].append({
                    'bbox': tracklet.bboxes[i],
                    'person_id': tracklet.person_id
                })

    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break

        if frame_number in frame_to_faces:
            for face_data in frame_to_faces[frame_number]:
                bbox = face_data['bbox']
                person_id = face_data['person_id']
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(frame, person_id, (bbox[0], bbox[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        out.write(frame)
        frame_number += 1

    cap.release()
    out.release()
    print(f"    - Saved labeled video to: {output_video_path}")

def main(args):
    print("Initializing InsightFace model...")
    app = FaceAnalysis(name='buffalo_l', providers=[args.execution_provider])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file '{args.video_path}' does not exist!")
        return
    
    print(f"\nProcessing video file: {os.path.basename(args.video_path)}")
    
    ground_truth_df = None
    if args.ground_truth_dir:
        video_basename = os.path.splitext(os.path.basename(args.video_path))[0]
        ground_truth_path = os.path.join(args.ground_truth_dir, f"{video_basename}.csv")
        ground_truth_df = load_ground_truth(ground_truth_path)
    
    print("\n--- Step 1: Extracting Tracklets ---")
    start_time = time.time()
    
    tracklets = extract_tracklets(args.video_path, app, 
                                 max_frames=args.max_frames, 
                                 ground_truth_df=ground_truth_df)
    
    if not tracklets:
        print("Error: No tracklets found in the video.")
        return
    
    print(f"Extracted {len(tracklets)} tracklets")
    
    print("\n--- Step 2: Creating Constraints ---")
    must_link, cannot_link = create_constraints(tracklets)
    print(f"Created {len(must_link)} must-link and {len(cannot_link)} cannot-link constraints")
    
    print("\n--- Step 3: Constrained Graph Clustering ---")
    clusterer = ConstrainedGraphClustering(
        similarity_threshold=args.similarity_threshold,
        k_neighbors=args.k_neighbors
    )
    
    cluster_labels = clusterer.cluster_tracklets(tracklets, must_link, cannot_link)
    
    unique_clusters = sorted(set(cluster_labels))
    cluster_to_person = {cluster_id: str(i + 1) for i, cluster_id in enumerate(unique_clusters)}
    
    for tracklet, cluster_id in zip(tracklets, cluster_labels):
        tracklet.person_id = cluster_to_person[cluster_id]
    
    print(f"Clustered into {len(unique_clusters)} unique people")
    
    print("\n--- Step 4: Processing Full Video ---")
    full_tracklets = extract_tracklets(args.video_path, app, 
                                     max_frames=None, 
                                     ground_truth_df=ground_truth_df)
    
    gallery_embeddings = []
    gallery_ids = []
    
    for cluster_id in unique_clusters:
        cluster_tracklets = [t for i, t in enumerate(tracklets) if cluster_labels[i] == cluster_id]
        if cluster_tracklets:
            cluster_embeddings = [t.get_quality_weighted_embedding() for t in cluster_tracklets]
            gallery_embedding = np.mean(cluster_embeddings, axis=0)
            gallery_embedding = gallery_embedding / np.linalg.norm(gallery_embedding)
            gallery_embeddings.append(gallery_embedding)
            gallery_ids.append(cluster_to_person[cluster_id])
    
    for tracklet in full_tracklets:
        if not gallery_embeddings:
            tracklet.person_id = 'unknown'
            continue
            
        tracklet_embedding = tracklet.get_quality_weighted_embedding()
        similarities = cosine_similarity([tracklet_embedding], gallery_embeddings)[0]
        best_match_idx = np.argmax(similarities)
        
        if similarities[best_match_idx] >= args.recognition_threshold:
            tracklet.person_id = gallery_ids[best_match_idx]
        else:
            tracklet.person_id = 'unknown'
    
    print("\n--- Step 5: Saving Results ---")
    save_outputs(args.video_path, full_tracklets, args.output_dir)
    
    end_time = time.time()
    print(f"\nFinished processing '{os.path.basename(args.video_path)}' in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tracklet-based constrained face recognition for video files.")
    parser.add_argument('--video_path', type=str, required=True, help="Path to the video file to process.")
    parser.add_argument('--output_dir', type=str, default="processed_videos", help="Directory to save output files.")
    parser.add_argument('--similarity_threshold', type=float, default=0.6, help="Cosine similarity threshold for clustering.")
    parser.add_argument('--k_neighbors', type=int, default=10, help="Number of neighbors for kNN graph.")
    parser.add_argument('--recognition_threshold', type=float, default=0.7, help="Cosine similarity threshold for recognition.")
    parser.add_argument('--execution_provider', type=str, default='CUDAExecutionProvider', 
                       help="Execution provider for ONNX Runtime.")
    parser.add_argument('--ground_truth_dir', type=str, help="Directory containing ground truth CSV files (optional).")
    parser.add_argument('--max_frames', type=int, default=36000, 
                       help="Maximum frames to process for gallery creation (default: 36000 = 20 minutes at 30fps).")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)
