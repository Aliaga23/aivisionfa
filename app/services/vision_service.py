from __future__ import annotations

import os
# Optimize for GPU performance
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'
os.environ['OPENCV_IO_ENABLE_JASPER'] = '0'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'

import cv2
if hasattr(cv2, 'setNumThreads'):
    cv2.setNumThreads(0)  # Use all available threads

import numpy as np
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any, Generator
import tempfile

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO


class CarOrientationPredictor:
    def __init__(self, model_path: str = '../models/car_orientation_model.pth'):
        # Debug CUDA availability
        print(f"[CarOrientationPredictor] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[CarOrientationPredictor] CUDA device count: {torch.cuda.device_count()}")
            print(f"[CarOrientationPredictor] CUDA device name: {torch.cuda.get_device_name(0)}")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[CarOrientationPredictor] Using device: {self.device}")
        
        self.model = models.resnet50(pretrained=False)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 2)
        self.enabled = False
        
        try:
            if model_path and Path(model_path).exists():
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.enabled = True
                print(f"[CarOrientationPredictor] Model loaded from: {model_path}")
            else:
                print(f"[CarOrientationPredictor] Model not found at: {model_path}")
                print(f"[CarOrientationPredictor] Orientation detection disabled")
        except Exception as e:
            print(f"[CarOrientationPredictor] Error loading model: {e}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.classes = ['Frontal', 'Reversa']

    def predict(self, image):
        if not self.enabled:
            return None, 0.0
            
        if isinstance(image, np.ndarray):
            if image.size == 0:
                return None, 0.0
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        return self.classes[predicted.item()], confidence.item()


class ParkingSystemComplete:
    def __init__(self, zone_model_path: str, vehicle_model_path: str, orientation_model_path: str):
        # Debug CUDA availability
        print(f"[ParkingSystemComplete] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[ParkingSystemComplete] CUDA device count: {torch.cuda.device_count()}")
            print(f"[ParkingSystemComplete] CUDA device name: {torch.cuda.get_device_name(0)}")
            print(f"[ParkingSystemComplete] PyTorch version: {torch.__version__}")
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[ParkingSystemComplete] Using device: {self.device}")
        
        # Load models
        self.zone_model = YOLO(zone_model_path)
        self.zone_model.to(self.device)
        
        self.vehicle_model = YOLO(vehicle_model_path)
        self.vehicle_model.to(self.device)
        
        self.orientation_predictor = CarOrientationPredictor(orientation_model_path)
        
        # Vehicle configuration
        self.vehicle_classes = [2, 3, 5, 7]
        self.vehicle_names = {2: 'auto', 3: 'moto', 5: 'bus', 7: 'camion'}
        self.zone_class_names = self.zone_model.names
        
        # Colors for visualization
        self.zone_color = (0, 0, 255)        # Red for empty zones
        self.occupied_color = (0, 165, 255)   # Orange for occupied zones

    def box_in_zone(self, box_center, box_coords, zone_mask) -> bool:
        """Check if vehicle center or bottom center is inside zone mask"""
        x, y = box_center
        if 0 <= y < zone_mask.shape[0] and 0 <= x < zone_mask.shape[1]:
            if zone_mask[y, x]:
                return True
        
        # Check bottom center of bounding box
        x1, y1, x2, y2 = box_coords
        bottom_center = (int((x1 + x2) / 2), int(y2))
        if 0 <= bottom_center[1] < zone_mask.shape[0] and 0 <= bottom_center[0] < zone_mask.shape[1]:
            if zone_mask[bottom_center[1], bottom_center[0]]:
                return True
        
        return False

    def process_frame(self, frame, frame_number: int, zone_conf=0.5, vehicle_conf=0.4):
        """Process single frame and return annotated frame with detections"""
        frame_result = frame.copy()
        h, w = frame.shape[:2]
        zones = []
        
        # Detect parking zones
        zone_results = self.zone_model(frame, conf=zone_conf, verbose=False, device=self.device, half=True)
        for result in zone_results:
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                boxes = result.boxes
                
                for mask, box in zip(masks, boxes):
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = self.zone_class_names[cls_id]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Resize mask to frame size
                    mask_resized = cv2.resize(mask, (w, h))
                    mask_bool = mask_resized > 0.5
                    
                    zones.append({
                        'class': class_name,
                        'confidence': conf,
                        'bbox': (x1, y1, x2, y2),
                        'mask': mask_bool,
                        'vehicles': []
                    })

        # Detect vehicles
        vehicle_results = self.vehicle_model(frame, conf=vehicle_conf, verbose=False, device=self.device, half=True)
        vehicles = []
        vehicle_counter = 0
        
        for result in vehicle_results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id in self.vehicle_classes:
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    
                    vehicles.append({
                        'id': vehicle_counter,
                        'type': self.vehicle_names[cls_id],
                        'confidence': conf,
                        'bbox': (x1, y1, x2, y2),
                        'center': center,
                        'in_zone': False,
                        'zone_id': None,
                        'orientation': None,
                        'orientation_confidence': None
                    })
                    vehicle_counter += 1

        # Match vehicles to zones
        for v_idx, vehicle in enumerate(vehicles):
            for z_idx, zone in enumerate(zones):
                if self.box_in_zone(vehicle['center'], vehicle['bbox'], zone['mask']):
                    vehicle['in_zone'] = True
                    vehicle['zone_id'] = z_idx
                    zone['vehicles'].append(v_idx)
                    break

        # Predict orientation for cars in zones
        for vehicle in vehicles:
            if vehicle['in_zone'] and vehicle['type'] == 'auto':
                x1, y1, x2, y2 = vehicle['bbox']
                car_roi = frame[y1:y2, x1:x2]
                
                if car_roi.size > 0:
                    orientation, orientation_conf = self.orientation_predictor.predict(car_roi)
                    vehicle['orientation'] = orientation
                    vehicle['orientation_confidence'] = orientation_conf

        # Draw zones with masks
        mask_overlay = frame_result.copy()
        for zone in zones:
            is_occupied = len(zone['vehicles']) > 0
            color = self.occupied_color if is_occupied else self.zone_color
            
            # Fill mask area
            mask_overlay[zone['mask']] = color
            
            # Draw contours
            contours, _ = cv2.findContours(zone['mask'].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame_result, contours, -1, color, 2)
            
            # Draw zone label
            x1, y1, x2, y2 = zone['bbox']
            status = f"OCUPADA ({len(zone['vehicles'])})" if is_occupied else "LIBRE"
            label = f"{zone['class']} - {status}"
            
            (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame_result, (x1, y1 - h_text - 6), (x1 + w_text, y1), color, -1)
            cv2.putText(frame_result, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Blend mask overlay
        frame_result = cv2.addWeighted(frame_result, 1, mask_overlay, 0.25, 0)

        # Collect statistics
        stats = {
            'total_in_zone': 0,
            'moving': 0,
            'parked': 0,
            'frontal': 0,
            'reversa': 0,
            'speeds': []
        }
        
        # Draw vehicles with orientation
        for vehicle in vehicles:
            if vehicle['in_zone'] and vehicle['orientation'] is not None:
                x1, y1, x2, y2 = vehicle['bbox']
                orientation = vehicle['orientation']
                
                stats['total_in_zone'] += 1
                stats['frontal' if orientation == 'Frontal' else 'reversa'] += 1
                stats['parked'] += 1
                
                # Color based on orientation
                color = (0, 255, 0) if orientation == 'Frontal' else (0, 0, 255)
                
                # Draw bounding box
                cv2.rectangle(frame_result, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{orientation} | ESTACIONADO"
                (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame_result, (x1, y2), (x1 + w_text, y2 + h_text + 10), color, -1)
                cv2.putText(frame_result, label, (x1, y2 + h_text + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Draw center point
                cv2.circle(frame_result, vehicle['center'], 5, color, -1)

        return frame_result, zones, stats


class VisionService:
    """Main service for video processing and WebSocket streaming"""
    
    def __init__(self, uploads_dir: Optional[Path] = None, image_format: str = "jpg"):
        self.detector: Optional[ParkingSystemComplete] = None
        self.camera: Optional[cv2.VideoCapture] = None
        self.output_frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.processing_active = False
        self.video_source: Optional[str] = None
        self.image_format = "jpg"  # Always use JPEG for RTX 3060 compatibility
        self._temp_video_path: Optional[Path] = None
        
        # Pre-encoded frame for WebSocket
        self._encoded_frame: Optional[bytes] = None
        self._encoder_active: bool = False
        self._encoder_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.stats: Dict[str, Any] = {
            'zones_total': 0,
            'zones_occupied': 0,
            'vehicles_in_zone': 0,
            'moving': 0,
            'parked': 0,
            'frontal': 0,
            'reversa': 0,
            'avg_speed': 0,
            'max_speed': 0
        }
        
        print("[VisionService] Service initialized")

    def configure(self, uploads_dir: Optional[Path] = None, image_format: Optional[str] = None):
        """Configure service parameters"""
        # Always use JPEG for RTX 3060 ultra-smooth streaming
        self.image_format = "jpg"
        print(f"[VisionService] Image format: JPEG (RTX 3060 optimized)")

    def init_models(self, zone_model_path: str, vehicle_model_path: str, orientation_model_path: str):
        """Initialize detection models"""
        print(f"[VisionService] Initializing models...")
        print(f"  - Zone model: {zone_model_path}")
        print(f"  - Vehicle model: {vehicle_model_path}")
        print(f"  - Orientation model: {orientation_model_path}")
        
        self.detector = ParkingSystemComplete(zone_model_path, vehicle_model_path, orientation_model_path)
        print("[VisionService] Models initialized successfully")

    def process_video_stream(self):
        """Main video processing loop - runs in separate thread"""
        frame_count = 0
        print(f"[VisionService] Video processing started. Detector: {self.detector is not None}")
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        while self.processing_active and self.camera is not None:
            ret, frame = self.camera.read()
            if not ret:
                print("[VisionService] Failed to read frame, stopping...")
                self.processing_active = False
                break

            frame_count += 1

            if self.detector is not None:
                try:
                    # Process frame with detection
                    result_frame, zones, stats = self.detector.process_frame(frame, frame_count)

                    # Update stats and output frame
                    with self.lock:
                        self.stats['zones_total'] = len(zones)
                        self.stats['zones_occupied'] = sum(1 for z in zones if z['vehicles'])
                        self.stats['vehicles_in_zone'] = stats['total_in_zone']
                        self.stats['moving'] = 0
                        self.stats['parked'] = stats['parked']
                        self.stats['frontal'] = stats['frontal']
                        self.stats['reversa'] = stats['reversa']
                        self.stats['avg_speed'] = 0
                        self.stats['max_speed'] = 0
                        self.output_frame = result_frame
                        
                except Exception as e:
                    print(f"[VisionService] Processing error: {e}")
                    with self.lock:
                        self.output_frame = frame
            else:
                # No detector, show raw feed
                with self.lock:
                    self.output_frame = frame
                    
            time.sleep(0.01)  # Small delay
        
        print("[VisionService] Video processing stopped")

    def _encode_current_frame(self) -> bool:
        """RTX 3060 Ultra-High Quality encoding for 60 FPS"""
        with self.lock:
            if self.output_frame is None:
                return False
            frame_local = self.output_frame.copy()

        # RTX 3060 can handle full resolution - no downsizing!
        height, width = frame_local.shape[:2]
        frame_resized = frame_local  # Keep original quality
        
        # Premium JPEG encoding for RTX 3060 ultra-smooth 60 FPS streaming
        # Use maximum quality JPEG (95%) - WebP removed for compatibility
        ok, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])

        if not ok:
            return False

        with self.lock:
            self._encoded_frame = buffer.tobytes()
        return True

    def _encoder_loop(self, target_fps: float = 60.0):
        """RTX 3060 Ultra-High Performance encoding loop at 60 FPS"""
        self._encoder_active = True
        interval = 1.0 / target_fps  # 0.0166 seconds for 60 FPS
        print(f"[VisionService] RTX 3060 Encoder: {target_fps} FPS (interval: {interval:.4f}s)")
        
        while self._encoder_active:
            start = time.perf_counter()  # More precise timing
            success = self._encode_current_frame()
            
            if success:
                elapsed = time.perf_counter() - start
                sleep_time = interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            else:
                time.sleep(0.001)  # Tiny sleep if no frame
        
        print("[VisionService] Encoder loop stopped")

    def start(self, source_type: str = 'webcam', video_path: Optional[str] = None) -> Dict[str, Any]:
        """Start video processing"""
        if self.processing_active:
            return {"status": "already_running", "message": "Processing already active"}
        
        if self.detector is None:
            return {"status": "error", "message": "Detector not initialized"}
        
        # Open video source
        if source_type == 'webcam' or source_type == 'camera':
            print("[VisionService] Opening webcam...")
            self.camera = cv2.VideoCapture(0)
            
            if not self.camera.isOpened():
                print("[VisionService] Failed to open camera")
                return {"status": "error", "message": "Camera not available"}
            
            # Test frame read
            ret, test_frame = self.camera.read()
            if ret:
                print(f"[VisionService] Camera test successful: {test_frame.shape}")
            else:
                print("[VisionService] Camera test failed")
                self.camera.release()
                self.camera = None
                return {"status": "error", "message": "Camera test failed"}
            
            # RTX 3060 Optimized Settings for 60 FPS Ultra Smooth
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.camera.set(cv2.CAP_PROP_FPS, 60)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
            
            # Get actual settings
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[VisionService] RTX 3060 Camera: {actual_width}x{actual_height} @ {actual_fps} FPS")
            
            self.video_source = 'Camara Web'
            
        else:
            # Video file
            if not video_path:
                return {"status": "error", "message": "Video path required"}
            
            # Check uploads folder
            if not video_path.startswith('/') and not video_path.startswith('C:'):
                vp = Path("uploads") / video_path
            else:
                vp = Path(video_path)
            
            if not vp.exists():
                return {"status": "error", "message": f"Video not found: {vp}"}
            
            self.camera = cv2.VideoCapture(str(vp))
            if not self.camera.isOpened():
                return {"status": "error", "message": "Failed to open video file"}
            
            self.video_source = video_path
        
        # Start processing
        self.processing_active = True
        
        # Start video processing thread
        thread = threading.Thread(target=self.process_video_stream, daemon=True)
        thread.start()
        
        # Start RTX 3060 optimized encoder thread at 60 FPS
        if not self._encoder_active:
            self._encoder_thread = threading.Thread(
                target=self._encoder_loop, 
                kwargs={"target_fps": 60.0}, 
                daemon=True
            )
            self._encoder_thread.start()
            print("[VisionService] RTX 3060 Encoder started at 60 FPS")
        
        print(f"[VisionService] Started: {self.video_source}")
        return {"status": "started", "source": self.video_source}

    def start_from_bytes(self, content: bytes, suffix: Optional[str] = None) -> Dict[str, Any]:
        """Start processing from uploaded video bytes"""
        if self.processing_active:
            return {"status": "already_running"}
        
        if self.detector is None:
            return {"status": "error", "message": "Detector not initialized"}
        
        try:
            # Create temporary file
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".mp4")
            tmp.write(content)
            tmp.flush()
            tmp.close()
            self._temp_video_path = Path(tmp.name)
            
            # Open video
            self.camera = cv2.VideoCapture(tmp.name)
            
            if not self.camera.isOpened():
                if self._temp_video_path.exists():
                    self._temp_video_path.unlink(missing_ok=True)
                self._temp_video_path = None
                return {"status": "error", "message": "Failed to open uploaded video"}
            
            self.video_source = "Uploaded Video"
            self.processing_active = True
            
            # Start processing
            thread = threading.Thread(target=self.process_video_stream, daemon=True)
            thread.start()
            
            # Start encoder if needed
            if not self._encoder_active:
                self._encoder_thread = threading.Thread(
                    target=self._encoder_loop,
                    kwargs={"target_fps": 30.0},
                    daemon=True
                )
                self._encoder_thread.start()
            
            return {"status": "started", "source": self.video_source}
            
        except Exception as e:
            print(f"[VisionService] Error in start_from_bytes: {e}")
            return {"status": "error", "message": str(e)}

    def stop(self) -> Dict[str, Any]:
        """Stop video processing"""
        print("[VisionService] Stopping...")
        
        self.processing_active = False
        self._encoder_active = False
        
        time.sleep(0.3)  # Allow threads to finish
        
        # Release camera
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        
        # Wait for encoder thread
        if self._encoder_thread is not None and self._encoder_thread.is_alive():
            self._encoder_thread.join(timeout=0.5)
        
        # Clean up temp file
        if self._temp_video_path is not None:
            try:
                if self._temp_video_path.exists():
                    self._temp_video_path.unlink(missing_ok=True)
            except Exception as e:
                print(f"[VisionService] Error cleaning temp file: {e}")
            self._temp_video_path = None
        
        # Reset stats
        with self.lock:
            self.output_frame = None
            self.stats = {
                'zones_total': 0,
                'zones_occupied': 0,
                'vehicles_in_zone': 0,
                'moving': 0,
                'parked': 0,
                'frontal': 0,
                'reversa': 0,
                'avg_speed': 0,
                'max_speed': 0
            }
        
        print("[VisionService] Stopped")
        return {"status": "stopped"}

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        with self.lock:
            return dict(self.stats)

    def get_encoded_frame(self) -> Optional[bytes]:
        """Get pre-encoded frame for WebSocket"""
        with self.lock:
            if self._encoded_frame is not None:
                return self._encoded_frame
        
        # Fallback: encode on demand
        ok = self._encode_current_frame()
        if not ok:
            # Return placeholder if no frame available
            return self._generate_placeholder_frame()
        
        with self.lock:
            return self._encoded_frame

    def _generate_placeholder_frame(self) -> bytes:
        """Generate placeholder frame when no video active"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        cv2.putText(frame, "Sistema de Vision Artificial", (100, 200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Esperando inicio...", (180, 250), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "GPU Optimizado", (210, 300), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        
        ok, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buffer.tobytes() if ok else b''

    def mjpeg_generator(self) -> Generator[bytes, None, None]:
        """RTX 3060 Optimized MJPEG stream at 60 FPS Ultra Smooth"""
        while True:
            with self.lock:
                frame_local = None if self.output_frame is None else self.output_frame.copy()
            
            if frame_local is None:
                # RTX 3060 GPU Ready placeholder
                placeholder = np.zeros((1080, 1920, 3), dtype=np.uint8)
                placeholder.fill(15)  # Dark background
                cv2.putText(placeholder, "RTX 3060 GPU READY", (600, 480), 
                          cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 4)
                cv2.putText(placeholder, "Ultra Smooth 60 FPS", (680, 540), 
                          cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
                cv2.putText(placeholder, "Presiona 'Iniciar Camara' para comenzar", (520, 600), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
                frame_local = placeholder
            
            # Premium quality encoding for RTX 3060
            ok, buffer = cv2.imencode('.jpg', frame_local, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                time.sleep(0.001)
                continue
            
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.0166)  # Exactly 60.24 FPS for ultra smoothness


# Global singleton instance
vision_service = VisionService()