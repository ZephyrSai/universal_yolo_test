import cv2
import numpy as np
import time
import argparse
from pathlib import Path
from ultralytics import YOLO
import threading
from collections import defaultdict

class YOLOTracker:
    def __init__(self, model_name="yolo11n.pt", tracker_type="bytetrack", inference_fps=30, device="auto"):
        """
        Initialize YOLO tracker with specified model and tracker type.
        
        Args:
            model_name (str): YOLO model name
            tracker_type (str): Tracker type ('bytetrack', 'botsort', or 'deepsort')
            inference_fps (int): FPS for inference processing
            device (str): Device to use ('auto', 'cpu', 'cuda', 'mps')
        """
        # Available YOLO models
        self.available_models = {
            # YOLOv5 models
            'yolov5n': 'yolov5n.pt',
            'yolov5s': 'yolov5s.pt', 
            'yolov5m': 'yolov5m.pt',
            'yolov5l': 'yolov5l.pt',
            'yolov5x': 'yolov5x.pt',
            
            # YOLOv8 models
            'yolov8n': 'yolov8n.pt',
            'yolov8s': 'yolov8s.pt',
            'yolov8m': 'yolov8m.pt',
            'yolov8l': 'yolov8l.pt',
            'yolov8x': 'yolov8x.pt',
            
            # YOLO11 models (latest)
            'yolo11n': 'yolo11n.pt',
            'yolo11s': 'yolo11s.pt',
            'yolo11m': 'yolo11m.pt',
            'yolo11l': 'yolo11l.pt',
            'yolo11x': 'yolo11x.pt',
            
            # YOLO12 models (newest)
            'yolo12n': 'yolo12n.pt',
            'yolo12s': 'yolo12s.pt',
            'yolo12m': 'yolo12m.pt',
            'yolo12l': 'yolo12l.pt',
            'yolo12x': 'yolo12x.pt',
        }
        
        # Load YOLO model
        self.device = self._setup_device(device)
        self.model = YOLO(model_name)
        self.model.to(self.device)
        self.tracker_type = tracker_type
        self.inference_fps = inference_fps
        self.inference_interval = 1.0 / inference_fps
        
        # Tracking variables
        self.track_history = defaultdict(lambda: [])
        self.last_inference_time = 0
        self.current_frame = None
        self.results = None
        
        # Colors for different object classes
        self.colors = self._generate_colors()
        
        print(f"Loaded model: {model_name}")
        print(f"Device: {self.device}")
        print(f"Tracker type: {tracker_type}")
        print(f"Inference FPS: {inference_fps}")
        
    def _setup_device(self, device):
        """
        Setup and validate the computing device
        
        Args:
            device (str): Device preference ('auto', 'cpu', 'cuda', 'mps')
            
        Returns:
            str: Validated device string
        """
        import torch
        
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                print("CUDA GPU detected - using GPU acceleration")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
                print("Apple Metal Performance Shaders detected - using MPS acceleration")
            else:
                device = "cpu"
                print("No GPU acceleration detected - using CPU")
        elif device == "cuda":
            if not torch.cuda.is_available():
                print("Warning: CUDA not available, falling back to CPU")
                device = "cpu"
            else:
                print(f"Using CUDA GPU: {torch.cuda.get_device_name()}")
        elif device == "mps":
            if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                print("Warning: MPS not available, falling back to CPU")
                device = "cpu"
            else:
                print("Using Apple Metal Performance Shaders (MPS)")
        elif device == "cpu":
            print("Using CPU")
        else:
            print(f"Unknown device '{device}', falling back to CPU")
            device = "cpu"
            
        return device
    
    def _generate_colors(self):
        """Generate random colors for different object classes"""
        np.random.seed(42)
        return [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) 
                for _ in range(100)]
    
    def list_available_models(self):
        """Print available YOLO models"""
        print("\n=== Available YOLO Models ===")
        for category in ['YOLOv5', 'YOLOv8', 'YOLO11', 'YOLO12']:
            print(f"\n{category} Models:")
            for key, value in self.available_models.items():
                if category.lower().replace('v', '') in key.lower():
                    print(f"  - {key}: {value}")
    
    def setup_video_source(self, source):
        """
        Setup video source (camera, video file, or RTSP stream)
        
        Args:
            source: Video source (0 for camera, file path, or RTSP URL)
        """
        if isinstance(source, str):
            if source.startswith('rtsp://') or source.startswith('http://'):
                print(f"Setting up RTSP/HTTP stream: {source}")
            elif Path(source).exists():
                print(f"Setting up video file: {source}")
            else:
                print(f"Warning: Source path may not exist: {source}")
        else:
            print(f"Setting up camera: {source}")
            
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video properties: {width}x{height} @ {fps} FPS")
        return cap, fps, width, height
    
    def should_process_frame(self):
        """Check if enough time has passed to process the next frame"""
        current_time = time.time()
        if current_time - self.last_inference_time >= self.inference_interval:
            self.last_inference_time = current_time
            return True
        return False
    
    def draw_tracks(self, frame, results):
        """
        Draw bounding boxes, labels, and tracking lines on the frame
        
        Args:
            frame: Current frame
            results: YOLO results with tracking information
        """
        if results[0].boxes is not None and results[0].boxes.id is not None:
            # Get boxes, track IDs, and confidence scores
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confidences = results[0].boxes.conf.float().cpu().tolist()
            classes = results[0].boxes.cls.int().cpu().tolist()
            
            # Draw bounding boxes and labels
            for box, track_id, conf, cls in zip(boxes, track_ids, confidences, classes):
                x, y, w, h = box
                x1, y1 = int(x - w/2), int(y - h/2)
                x2, y2 = int(x + w/2), int(y + h/2)
                
                # Get class name and color
                class_name = self.model.names[cls]
                color = self.colors[cls % len(self.colors)]
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with ID and confidence
                label = f"{class_name} ID:{track_id} {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Store track history
                track = self.track_history[track_id]
                track.append((int(x), int(y)))
                if len(track) > 30:  # Keep only last 30 points
                    track.pop(0)
                
                # Draw tracking trail
                points = np.array(track).reshape((-1, 1, 2))
                if len(points) > 1:
                    cv2.polylines(frame, [points], isClosed=False, 
                                 color=color, thickness=2)
        
        return frame
    
    def run_detection_only(self, source, save_output=False, output_path="output.mp4"):
        """
        Run object detection without tracking
        
        Args:
            source: Video source
            save_output (bool): Whether to save output video
            output_path (str): Path to save output video
        """
        cap, fps, width, height = self.setup_video_source(source)
        
        # Setup video writer if saving output
        writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print("Starting object detection... Press 'q' to quit")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame only at specified FPS
                if self.should_process_frame():
                    # Run detection
                    results = self.model(frame, verbose=False)
                    
                    # Draw detections
                    annotated_frame = results[0].plot()
                    self.current_frame = annotated_frame
                else:
                    # Use last processed frame
                    annotated_frame = self.current_frame if self.current_frame is not None else frame
                
                # Display frame
                cv2.imshow('YOLO Detection', annotated_frame)
                
                # Save frame if required
                if save_output and writer is not None:
                    writer.write(annotated_frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\nDetection stopped by user")
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()
    
    def run_tracking(self, source, save_output=False, output_path="output_tracking.mp4"):
        """
        Run object detection with tracking
        
        Args:
            source: Video source
            save_output (bool): Whether to save output video
            output_path (str): Path to save output video
        """
        cap, fps, width, height = self.setup_video_source(source)
        
        # Setup video writer if saving output
        writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print("Starting object tracking... Press 'q' to quit")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame only at specified FPS
                if self.should_process_frame():
                    # Run tracking
                    if self.tracker_type == 'deepsort':
                        # For DeepSORT, use detection + manual tracking
                        results = self.model(frame, verbose=False)
                        self.results = results
                        # Note: DeepSORT integration would require additional implementation
                        # For now, falling back to detection only
                        annotated_frame = results[0].plot()
                    else:
                        # Use built-in tracking (ByteTrack or BoT-SORT)
                        results = self.model.track(frame, persist=True, 
                                                 tracker=f"{self.tracker_type}.yaml", 
                                                 verbose=False)
                        self.results = results
                        annotated_frame = self.draw_tracks(frame, results)
                else:
                    # Use last processed frame
                    if self.results is not None:
                        annotated_frame = self.draw_tracks(frame, self.results)
                    else:
                        annotated_frame = frame
                
                # Add info text
                info_text = f"Model: {self.model.model_name} | Device: {self.device} | Tracker: {self.tracker_type} | FPS: {self.inference_fps}"
                cv2.putText(annotated_frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow('YOLO Tracking', annotated_frame)
                
                # Save frame if required
                if save_output and writer is not None:
                    writer.write(annotated_frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\nTracking stopped by user")
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='YOLO Object Detection and Tracking')
    parser.add_argument('--source', type=str, default='0', 
                       help='Video source (0 for camera, file path, or RTSP URL)')
    parser.add_argument('--model', type=str, default='yolo11n.pt',
                       help='YOLO model name (e.g., yolo11n.pt, yolov8s.pt)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to use (auto, cpu, cuda for NVIDIA GPU, mps for Apple GPU)')
    parser.add_argument('--tracker', type=str, default='bytetrack',
                       choices=['bytetrack', 'botsort', 'deepsort'],
                       help='Tracker type')
    parser.add_argument('--inference-fps', type=int, default=30,
                       help='FPS for inference processing')
    parser.add_argument('--mode', type=str, default='track',
                       choices=['detect', 'track'],
                       help='Mode: detect (detection only) or track (with tracking)')
    parser.add_argument('--save', action='store_true',
                       help='Save output video')
    parser.add_argument('--output', type=str, default='output.mp4',
                       help='Output video path')
    parser.add_argument('--list-models', action='store_true',
                       help='List available YOLO models')
    
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = YOLOTracker(
        model_name=args.model,
        tracker_type=args.tracker,
        inference_fps=args.inference_fps,
        device=args.device
    )
    
    # List models if requested
    if args.list_models:
        tracker.list_available_models()
        return
    
    # Convert source to appropriate type
    source = args.source
    if source.isdigit():
        source = int(source)
    
    # Run detection or tracking
    if args.mode == 'detect':
        tracker.run_detection_only(source, args.save, args.output)
    else:
        tracker.run_tracking(source, args.save, args.output)

if __name__ == "__main__":
    main()