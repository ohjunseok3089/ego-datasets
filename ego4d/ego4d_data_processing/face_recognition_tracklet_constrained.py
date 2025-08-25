#!/usr/bin/env python3
"""
Graph-based Face Recognition System for Ego4D Videos
Using tracklet extraction, constrained clustering, and multi-prototype galleries
"""

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
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Optional imports with fallbacks
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available, using sklearn for similarity search")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("Warning: NetworkX not available, using basic clustering")

class Tracklet:
    """Represents a face tracklet across multiple frames"""
    
    def __init__(self, track_id: int, initial_face_data: dict):
        self.track_id = track_id
        self.faces = [initial_face_data]
        self.frames = [initial_face_data['frame_number']]
        self.bboxes = [initial_face_data['bbox']]
        self.embeddings = [initial_face_data['embedding']]
        self.qualities = [self._compute_quality(initial_face_data['bbox'])]
        self.person_id = None
        
    def add_face(self, face_data: dict):
        self.faces.append(face_data)
        self.frames.append(face_data['frame_number'])
        self.bboxes.append(face_data['bbox'])
        self.embeddings.append(face_data['embedding'])
        self.qualities.append(self._compute_quality(face_data['bbox']))
    
    def _compute_quality(self, bbox):
        """Compute face quality based on bbox area"""
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        return np.sqrt(area)  # Use square root to reduce extreme weights
    
    def get_quality_weighted_embedding(self) -> np.ndarray:
        """Get quality-weighted average embedding"""
        if len(self.embeddings) == 1:
            return self.embeddings[0]
        
        weights = np.array(self.qualities)
        weights = weights / weights.sum()
        
        weighted_embedding = np.average(self.embeddings, axis=0, weights=weights)
        return weighted_embedding / np.linalg.norm(weighted_embedding)
    
    def get_medoid_embedding(self) -> np.ndarray:
        """Get medoid embedding (most representative)"""
        if len(self.embeddings) == 1:
            return self.embeddings[0]
        
        embeddings_array = np.array(self.embeddings)
        similarities = cosine_similarity(embeddings_array)
        medoid_idx = np.argmax(similarities.sum(axis=1))
        return embeddings_array[medoid_idx]
    
    def get_multi_prototype_embeddings(self, n_prototypes=3):
        """Get multiple prototype embeddings for better representation"""
        if len(self.embeddings) <= n_prototypes:
            return self.embeddings
        
        # Use k-means to find representative prototypes
        from sklearn.cluster import KMeans
        embeddings_array = np.array(self.embeddings)
        kmeans = KMeans(n_clusters=n_prototypes, n_init=10, random_state=42)
        kmeans.fit(embeddings_array)
        
        prototypes = []
        for center in kmeans.cluster_centers_:
            center_norm = center / np.linalg.norm(center)
            prototypes.append(center_norm)
        
        return prototypes

class FaceTracker:
    """Simple IoU-based face tracker"""
    
    def __init__(self, iou_threshold: float = 0.3, max_disappeared: int = 30):
        self.tracklets = {}
        self.next_track_id = 0
        self.iou_threshold = iou_threshold
        self.max_disappeared = max_disappeared
        self.disappeared = {}
    
    def update(self, frame_faces: list) -> list:
        """Update tracklets with new frame faces"""
        if not frame_faces:
            # Mark all existing tracklets as disappeared
            for track_id in list(self.tracklets.keys()):
                self.disappeared[track_id] = self.disappeared.get(track_id, 0) + 1
                if self.disappeared[track_id] > self.max_disappeared:
                    del self.tracklets[track_id]
                    del self.disappeared[track_id]
            return []
        
        if not self.tracklets:
            # Create new tracklets for all faces
            for face_data in frame_faces:
                self.tracklets[self.next_track_id] = Tracklet(self.next_track_id, face_data)
                self.disappeared[self.next_track_id] = 0
                self.next_track_id += 1
            return frame_faces
        
        # Compute IoU matrix
        track_ids = list(self.tracklets.keys())
        iou_matrix = np.zeros((len(track_ids), len(frame_faces)))
        
        for i, track_id in enumerate(track_ids):
            last_bbox = self.tracklets[track_id].bboxes[-1]
            for j, face_data in enumerate(frame_faces):
                iou_matrix[i, j] = self._calculate_iou(last_bbox, face_data['bbox'])
        
        # Hungarian matching
        from scipy.optimize import linear_sum_assignment
        track_indices, face_indices = linear_sum_assignment(-iou_matrix)
        
        matched_faces = set()
        for track_idx, face_idx in zip(track_indices, face_indices):
            if iou_matrix[track_idx, face_idx] >= self.iou_threshold:
                track_id = track_ids[track_idx]
                self.tracklets[track_id].add_face(frame_faces[face_idx])
                self.disappeared[track_id] = 0
                matched_faces.add(face_idx)
        
        # Handle unmatched tracklets
        for track_idx, track_id in enumerate(track_ids):
            if track_idx not in track_indices or \
               iou_matrix[track_idx, face_indices[list(track_indices).index(track_idx)]] < self.iou_threshold:
                self.disappeared[track_id] = self.disappeared.get(track_id, 0) + 1
                if self.disappeared[track_id] > self.max_disappeared:
                    del self.tracklets[track_id]
                    del self.disappeared[track_id]
        
        # Create new tracklets for unmatched faces
        for face_idx, face_data in enumerate(frame_faces):
            if face_idx not in matched_faces:
                self.tracklets[self.next_track_id] = Tracklet(self.next_track_id, face_data)
                self.disappeared[self.next_track_id] = 0
                self.next_track_id += 1
        
        return frame_faces
    
    def _calculate_iou(self, box1, box2) -> float:
        """Calculate IoU between two bboxes"""
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
    
    def get_tracklets(self) -> list:
        """Get all current tracklets"""
        return list(self.tracklets.values())

class GraphClusterer:
    """Graph-based clustering with constraints"""
    
    def __init__(self, similarity_threshold: float = 0.6, k_neighbors: int = 10):
        self.similarity_threshold = similarity_threshold
        self.k_neighbors = k_neighbors
    
    def cluster_tracklets(self, tracklets: list, 
                         must_link_constraints: list = None,
                         cannot_link_constraints: list = None) -> list:
        """Cluster tracklets using graph-based methods"""
        if len(tracklets) <= 1:
            return [0] * len(tracklets)
        
        # Extract embeddings
        embeddings = []
        for tracklet in tracklets:
            embedding = tracklet.get_quality_weighted_embedding()
            embeddings.append(embedding)
        embeddings = np.array(embeddings).astype('float32')
        
        # Build similarity graph
        if FAISS_AVAILABLE:
            graph = self._build_faiss_graph(embeddings)
        elif NETWORKX_AVAILABLE:
            graph = self._build_sklearn_graph(embeddings)
        else:
            return self._simple_clustering(embeddings)
        
        # Apply constraints
        if NETWORKX_AVAILABLE and graph is not None:
            if must_link_constraints:
                self._apply_must_link_constraints(graph, must_link_constraints)
            if cannot_link_constraints:
                self._apply_cannot_link_constraints(graph, cannot_link_constraints)
            
            # Detect communities
            clusters = self._detect_communities(graph)
        else:
            clusters = self._simple_clustering(embeddings)
        
        return clusters
    
    def _build_faiss_graph(self, embeddings: np.ndarray):
        """Build kNN graph using FAISS"""
        if not NETWORKX_AVAILABLE:
            return None
            
        import networkx as nx
        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)  # Inner product (cosine similarity for normalized vectors)
        index.add(embeddings)
        
        k = min(self.k_neighbors + 1, len(embeddings))
        similarities, indices = index.search(embeddings, k)
        
        graph = nx.Graph()
        for i in range(len(embeddings)):
            graph.add_node(i)
        
        # Add edges for k-NN connections
        for i in range(len(embeddings)):
            for j, sim in zip(indices[i][1:], similarities[i][1:]):
                if sim >= self.similarity_threshold:
                    graph.add_edge(i, j, weight=float(sim))
        
        return graph
    
    def _build_sklearn_graph(self, embeddings: np.ndarray):
        """Build kNN graph using sklearn"""
        if not NETWORKX_AVAILABLE:
            return None
            
        import networkx as nx
        from sklearn.neighbors import NearestNeighbors
        
        k = min(self.k_neighbors, len(embeddings) - 1)
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine')
        nbrs.fit(embeddings)
        distances, indices = nbrs.kneighbors(embeddings)
        
        graph = nx.Graph()
        for i in range(len(embeddings)):
            graph.add_node(i)
        
        # Convert distances to similarities
        for i in range(len(embeddings)):
            for j, dist in zip(indices[i][1:], distances[i][1:]):
                sim = 1 - dist  # Convert cosine distance to similarity
                if sim >= self.similarity_threshold:
                    graph.add_edge(i, j, weight=float(sim))
        
        return graph
    
    def _apply_must_link_constraints(self, graph, constraints):
        """Apply must-link constraints to graph"""
        for i, j in constraints:
            if i < len(graph.nodes) and j < len(graph.nodes):
                graph.add_edge(i, j, weight=1.0)
    
    def _apply_cannot_link_constraints(self, graph, constraints):
        """Apply cannot-link constraints to graph"""
        for i, j in constraints:
            if i < len(graph.nodes) and j < len(graph.nodes):
                if graph.has_edge(i, j):
                    graph.remove_edge(i, j)
    
    def _detect_communities(self, graph):
        """Detect communities in graph"""
        if not NETWORKX_AVAILABLE or graph is None:
            return list(range(len(graph.nodes)))
        
        # Try different community detection methods
        try:
            # Try Leiden algorithm (best quality)
            import leidenalg as la
            import igraph as ig
            
            edges = list(graph.edges(data=True))
            if not edges:
                return list(range(len(graph.nodes)))
                
            g_ig = ig.Graph()
            g_ig.add_vertices(len(graph.nodes))
            g_ig.add_edges([(e[0], e[1]) for e in edges])
            g_ig.es['weight'] = [e[2].get('weight', 1.0) for e in edges]
            
            partition = la.find_partition(g_ig, la.ModularityVertexPartition, weights='weight')
            return partition.membership
            
        except ImportError:
            pass
        
        try:
            # Try Louvain algorithm (good quality, faster)
            import community as community_louvain
            partition = community_louvain.best_partition(graph, weight='weight')
            return [partition[node] for node in range(len(graph.nodes))]
            
        except ImportError:
            pass
        
        # Fallback to connected components
        import networkx as nx
        components = list(nx.connected_components(graph))
        clusters = [0] * len(graph.nodes)
        for cluster_id, component in enumerate(components):
            for node in component:
                clusters[node] = cluster_id
        return clusters
    
    def _simple_clustering(self, embeddings):
        """Simple clustering fallback using cosine similarity threshold"""
        n = len(embeddings)
        clusters = list(range(n))  # Start with each in own cluster
        
        # Compute pairwise similarities
        similarities = cosine_similarity(embeddings)
        
        # Greedy merging
        for i in range(n):
            for j in range(i + 1, n):
                if similarities[i, j] >= self.similarity_threshold:
                    # Merge clusters
                    old_cluster = clusters[j]
                    new_cluster = clusters[i]
                    for k in range(n):
                        if clusters[k] == old_cluster:
                            clusters[k] = new_cluster
        
        # Renumber clusters
        unique_clusters = sorted(set(clusters))
        cluster_map = {old: new for new, old in enumerate(unique_clusters)}
        return [cluster_map[c] for c in clusters]

def get_execution_provider():
    """Automatically select the best available execution provider"""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        
        # Priority order
        provider_priority = [
            'TensorrtExecutionProvider',
            'CUDAExecutionProvider', 
            'ROCMExecutionProvider',
            'CPUExecutionProvider'
        ]
        
        for provider in provider_priority:
            if provider in providers:
                print(f"Using execution provider: {provider}")
                return provider
                
    except Exception as e:
        print(f"Error detecting providers: {e}")
    
    print("Using default CPUExecutionProvider")
    return 'CPUExecutionProvider'

def create_constraints(tracklets: list):
    """Create must-link and cannot-link constraints from tracklets"""
    must_link = []
    cannot_link = []
    
    # Build frame-to-tracklets mapping
    frame_to_tracklets = defaultdict(list)
    for i, tracklet in enumerate(tracklets):
        for frame_num in tracklet.frames:
            frame_to_tracklets[frame_num].append(i)
    
    # Cannot-link: tracklets appearing in same frame
    for frame_num, tracklet_indices in frame_to_tracklets.items():
        if len(tracklet_indices) > 1:
            for i in range(len(tracklet_indices)):
                for j in range(i + 1, len(tracklet_indices)):
                    cannot_link.append((tracklet_indices[i], tracklet_indices[j]))
    
    return must_link, cannot_link

def extract_tracklets(video_path: str, model: FaceAnalysis, 
                     max_frames: int = None, skip_frames: int = 1) -> list:
    """Extract face tracklets from video"""
    print(f"  Extracting tracklets from: {os.path.basename(video_path)}...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    Error: Could not open {video_path}")
        return []
    
    tracker = FaceTracker(iou_threshold=0.3, max_disappeared=30)
    frame_number = 0
    processed_frames = 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total_frames = min(total_frames, max_frames)
    
    while cap.isOpened() and (max_frames is None or frame_number < max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames for faster processing
        if frame_number % skip_frames != 0:
            frame_number += 1
            continue
            
        faces = model.get(frame)
        frame_faces = []
        
        for face in faces:
            # Filter low-quality faces
            bbox_area = (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
            if bbox_area < 900:  # Skip very small faces (30x30)
                continue
                
            face_data = {
                'video_path': video_path,
                'frame_number': frame_number,
                'bbox': face.bbox.astype(int),
                'embedding': face.normed_embedding
            }
            frame_faces.append(face_data)
        
        tracker.update(frame_faces)
        frame_number += 1
        processed_frames += 1
        
        # Progress update
        if processed_frames % 300 == 0:
            progress = (frame_number / total_frames) * 100
            print(f"    Progress: {progress:.1f}% ({frame_number}/{total_frames} frames)")
    
    cap.release()
    
    tracklets = tracker.get_tracklets()
    print(f"    Extracted {len(tracklets)} tracklets from {processed_frames} frames")
    
    # Filter out very short tracklets
    tracklets = [t for t in tracklets if len(t.frames) >= 3]
    print(f"    After filtering: {len(tracklets)} tracklets")
    
    return tracklets

def save_outputs(video_path: str, tracklets: list, output_dir: str, suffix: str = "graph_based"):
    """Save processing results"""
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_csv_path = os.path.join(output_dir, f"{base_name}_{suffix}.csv")
    output_video_path = os.path.join(output_dir, f"{base_name}_{suffix}.mp4")

    # Save CSV annotations
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_number', 'person_id', 'x1', 'y1', 'x2', 'y2', 'confidence'])
        
        for tracklet in tracklets:
            if tracklet.person_id is not None and tracklet.person_id != 'unknown':
                for i, frame_num in enumerate(tracklet.frames):
                    bbox = tracklet.bboxes[i]
                    quality = tracklet.qualities[i] if hasattr(tracklet, 'qualities') else 1.0
                    writer.writerow([
                        frame_num, tracklet.person_id, 
                        bbox[0], bbox[1], bbox[2], bbox[3],
                        quality
                    ])
    
    print(f"    Saved annotations to: {output_csv_path}")

    # Generate labeled video
    print(f"    Generating labeled video...")
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Build frame-to-faces mapping
    frame_to_faces = defaultdict(list)
    for tracklet in tracklets:
        if tracklet.person_id is not None and tracklet.person_id != 'unknown':
            for i, frame_num in enumerate(tracklet.frames):
                frame_to_faces[frame_num].append({
                    'bbox': tracklet.bboxes[i],
                    'person_id': tracklet.person_id
                })

    # Color mapping for person IDs
    colors = {}
    np.random.seed(42)
    for tracklet in tracklets:
        if tracklet.person_id and tracklet.person_id not in colors:
            colors[tracklet.person_id] = tuple(map(int, np.random.randint(50, 255, 3)))

    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number in frame_to_faces:
            for face_data in frame_to_faces[frame_number]:
                bbox = face_data['bbox']
                person_id = face_data['person_id']
                color = colors.get(person_id, (0, 255, 0))
                
                # Draw bbox
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                
                # Draw label with background
                label = f"ID: {person_id}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, 
                            (bbox[0], bbox[1] - label_size[1] - 10),
                            (bbox[0] + label_size[0], bbox[1]),
                            color, -1)
                cv2.putText(frame, label, (bbox[0], bbox[1] - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        out.write(frame)
        frame_number += 1

    cap.release()
    out.release()
    print(f"    Saved labeled video to: {output_video_path}")

def main(args):
    """Main processing pipeline"""
    print("="*60)
    print("Graph-based Face Recognition System")
    print("="*60)
    
    # Initialize model
    print("\nInitializing InsightFace model...")
    provider = args.execution_provider if args.execution_provider != 'auto' else get_execution_provider()
    
    app = FaceAnalysis(name='buffalo_l', providers=[provider])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    # Validate input
    if not os.path.exists(args.video_path):
        print(f"Error: Video file '{args.video_path}' does not exist!")
        return
    
    print(f"\nProcessing video: {os.path.basename(args.video_path)}")
    print(f"Output directory: {args.output_dir}")
    
    # Step 1: Extract tracklets from gallery portion
    print("\n" + "="*60)
    print("STEP 1: Extracting Tracklets for Gallery Creation")
    print("="*60)
    start_time = time.time()
    
    gallery_tracklets = extract_tracklets(
        args.video_path, app, 
        max_frames=args.max_frames,
        skip_frames=args.skip_frames
    )
    
    if not gallery_tracklets:
        print("Error: No tracklets found in the video.")
        return
    
    # Step 2: Create constraints
    print("\n" + "="*60)
    print("STEP 2: Creating Constraints")
    print("="*60)
    must_link, cannot_link = create_constraints(gallery_tracklets)
    print(f"  Created {len(must_link)} must-link constraints")
    print(f"  Created {len(cannot_link)} cannot-link constraints")
    
    # Step 3: Graph-based clustering
    print("\n" + "="*60)
    print("STEP 3: Graph-based Clustering")
    print("="*60)
    clusterer = GraphClusterer(
        similarity_threshold=args.similarity_threshold,
        k_neighbors=args.k_neighbors
    )
    
    cluster_labels = clusterer.cluster_tracklets(
        gallery_tracklets, must_link, cannot_link
    )
    
    # Assign person IDs
    unique_clusters = sorted(set(cluster_labels))
    cluster_to_person = {cluster_id: str(i + 1) for i, cluster_id in enumerate(unique_clusters)}
    
    for tracklet, cluster_id in zip(gallery_tracklets, cluster_labels):
        tracklet.person_id = cluster_to_person[cluster_id]
    
    print(f"  Identified {len(unique_clusters)} unique individuals")
    
    # Step 4: Build gallery with multi-prototypes
    print("\n" + "="*60)
    print("STEP 4: Building Multi-Prototype Gallery")
    print("="*60)
    
    gallery = defaultdict(list)
    for cluster_id in unique_clusters:
        cluster_tracklets = [t for i, t in enumerate(gallery_tracklets) 
                            if cluster_labels[i] == cluster_id]
        
        # Collect all embeddings for this person
        all_embeddings = []
        for tracklet in cluster_tracklets:
            if args.use_multi_prototype:
                prototypes = tracklet.get_multi_prototype_embeddings(n_prototypes=3)
                all_embeddings.extend(prototypes)
            else:
                all_embeddings.append(tracklet.get_quality_weighted_embedding())
        
        person_id = cluster_to_person[cluster_id]
        gallery[person_id] = all_embeddings
        print(f"  Person {person_id}: {len(all_embeddings)} prototypes")
    
    # Step 5: Process full video with gallery matching
    print("\n" + "="*60)
    print("STEP 5: Processing Full Video with Gallery Matching")
    print("="*60)
    
    full_tracklets = extract_tracklets(
        args.video_path, app,
        max_frames=None,
        skip_frames=1  # Process all frames for final output
    )
    
    # Match tracklets to gallery
    for tracklet in full_tracklets:
        tracklet_embedding = tracklet.get_quality_weighted_embedding()
        
        best_person_id = 'unknown'
        best_similarity = 0
        
        for person_id, prototypes in gallery.items():
            # Compute maximum similarity across all prototypes
            similarities = [
                np.dot(tracklet_embedding, proto) 
                for proto in prototypes
            ]
            max_sim = max(similarities)
            
            if max_sim > best_similarity:
                best_similarity = max_sim
                best_person_id = person_id
        
        if best_similarity >= args.recognition_threshold:
            tracklet.person_id = best_person_id
        else:
            tracklet.person_id = 'unknown'
    
    # Statistics
    recognized_tracklets = [t for t in full_tracklets if t.person_id != 'unknown']
    print(f"  Recognized {len(recognized_tracklets)}/{len(full_tracklets)} tracklets")
    
    # Step 6: Save results
    print("\n" + "="*60)
    print("STEP 6: Saving Results")
    print("="*60)
    save_outputs(args.video_path, full_tracklets, args.output_dir)
    
    # Final statistics
    end_time = time.time()
    processing_time = end_time - start_time
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Total processing time: {processing_time:.2f} seconds")
    print(f"Video: {os.path.basename(args.video_path)}")
    print(f"Unique individuals identified: {len(unique_clusters)}")
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Graph-based face recognition for video files using tracklets and constrained clustering."
    )
    
    # Required arguments
    parser.add_argument('--video_path', type=str, required=True,
                       help="Path to the video file to process.")
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default="processed_videos",
                       help="Directory to save output files.")
    
    # Clustering parameters
    parser.add_argument('--similarity_threshold', type=float, default=0.6,
                       help="Cosine similarity threshold for clustering (0.0-1.0).")
    parser.add_argument('--k_neighbors', type=int, default=10,
                       help="Number of neighbors for kNN graph construction.")
    parser.add_argument('--recognition_threshold', type=float, default=0.65,
                       help="Cosine similarity threshold for recognition.")
    
    # Model configuration
    parser.add_argument('--execution_provider', type=str, default='auto',
                       choices=['auto', 'CUDAExecutionProvider', 'CPUExecutionProvider', 
                               'TensorrtExecutionProvider', 'ROCMExecutionProvider'],
                       help="Execution provider for ONNX Runtime (auto for automatic selection).")
    
    # Processing configuration
    parser.add_argument('--max_frames', type=int, default=36000,
                       help="Maximum frames to process for gallery creation (default: 36000 = 20 min @ 30fps).")
    parser.add_argument('--skip_frames', type=int, default=2,
                       help="Skip frames for faster gallery creation (1 = no skip, 2 = every other frame).")
    parser.add_argument('--use_multi_prototype', action='store_true',
                       help="Use multiple prototypes per person for better representation.")
    
    # Ground truth (optional)
    parser.add_argument('--ground_truth_dir', type=str,
                       help="Directory containing ground truth CSV files (optional).")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run main processing
    try:
        main(args)
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)