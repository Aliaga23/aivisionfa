from __future__ import annotations

import cv2
import numpy as np
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any, Generator
import json
import tempfile

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO


class CarOrientationPredictor:
    def __init__(self, model_path: str = 'car_orientation_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = models.resnet50(pretrained=False)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 2)
        self.enabled = False
        try:
            if model_path and Path(model_path).exists():
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.enabled = True
            else:
                print("[VisionService] Orientation model not found; orientation disabled")
        except Exception as e:
            print(f"[VisionService] Warning loading orientation model: {e}")
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
        self.zone_model = YOLO(zone_model_path)
        self.vehicle_model = YOLO(vehicle_model_path)
        self.orientation_predictor = CarOrientationPredictor(orientation_model_path)
        self.vehicle_classes = [2, 3, 5, 7]
        self.vehicle_names = {2: 'auto', 3: 'moto', 5: 'bus', 7: 'camion'}
        self.zone_class_names = self.zone_model.names
        self.zone_color = (0, 0, 255)
        self.occupied_color = (0, 165, 255)

    def box_in_zone(self, box_center, box_coords, zone_mask) -> bool:
        x, y = box_center
        if 0 <= y < zone_mask.shape[0] and 0 <= x < zone_mask.shape[1]:
            if zone_mask[y, x]:
                return True
        x1, y1, x2, y2 = box_coords
        bottom_center = (int((x1 + x2) / 2), int(y2))
        if 0 <= bottom_center[1] < zone_mask.shape[0] and 0 <= bottom_center[0] < zone_mask.shape[1]:
            if zone_mask[bottom_center[1], bottom_center[0]]:
                return True
        return False

    def process_frame(self, frame, frame_number: int, zone_conf=0.5, vehicle_conf=0.4):
        frame_result = frame.copy()
        h, w = frame.shape[:2]
        zones = []
        zone_results = self.zone_model(frame, conf=zone_conf, verbose=False)
        for result in zone_results:
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                boxes = result.boxes
                for mask, box in zip(masks, boxes):
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = self.zone_class_names[cls_id]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    mask_resized = cv2.resize(mask, (w, h))
                    mask_bool = mask_resized > 0.5
                    zones.append({
                        'class': class_name,
                        'confidence': conf,
                        'bbox': (x1, y1, x2, y2),
                        'mask': mask_bool,
                        'vehicles': []
                    })

        vehicle_results = self.vehicle_model(frame, conf=vehicle_conf, verbose=False)
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

        for v_idx, vehicle in enumerate(vehicles):
            for z_idx, zone in enumerate(zones):
                if self.box_in_zone(vehicle['center'], vehicle['bbox'], zone['mask']):
                    vehicle['in_zone'] = True
                    vehicle['zone_id'] = z_idx
                    zone['vehicles'].append(v_idx)
                    break

        for vehicle in vehicles:
            if vehicle['in_zone'] and vehicle['type'] == 'auto':
                x1, y1, x2, y2 = vehicle['bbox']
                car_roi = frame[y1:y2, x1:x2]
                if car_roi.size > 0:
                    orientation, orientation_conf = self.orientation_predictor.predict(car_roi)
                    vehicle['orientation'] = orientation
                    vehicle['orientation_confidence'] = orientation_conf

        mask_overlay = frame_result.copy()
        for zone in zones:
            is_occupied = len(zone['vehicles']) > 0
            color = self.occupied_color if is_occupied else self.zone_color
            mask_overlay[zone['mask']] = color
            contours, _ = cv2.findContours(zone['mask'].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame_result, contours, -1, color, 2)
            x1, y1, x2, y2 = zone['bbox']
            status = f"OCUPADA ({len(zone['vehicles'])})" if is_occupied else "LIBRE"
            label = f"{zone['class']} - {status}"
            (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame_result, (x1, y1 - h_text - 6), (x1 + w_text, y1), color, -1)
            cv2.putText(frame_result, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        frame_result = cv2.addWeighted(frame_result, 1, mask_overlay, 0.25, 0)

        stats = {
            'total_in_zone': 0,
            'moving': 0,
            'parked': 0,
            'frontal': 0,
            'reversa': 0,
            'speeds': []
        }
        for vehicle in vehicles:
            if vehicle['in_zone'] and vehicle['orientation'] is not None:
                x1, y1, x2, y2 = vehicle['bbox']
                orientation = vehicle['orientation']
                stats['total_in_zone'] += 1
                stats['frontal' if orientation == 'Frontal' else 'reversa'] += 1
                stats['parked'] += 1
                color = (0, 255, 0) if orientation == 'Frontal' else (0, 0, 255)
                cv2.rectangle(frame_result, (x1, y1), (x2, y2), color, 2)
                label = f"{orientation} | ESTACIONADO"
                (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame_result, (x1, y2), (x1 + w_text, y2 + h_text + 10), color, -1)
                cv2.putText(frame_result, label, (x1, y2 + h_text + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.circle(frame_result, vehicle['center'], 5, color, -1)

        return frame_result, zones, stats


class VisionService:
    def __init__(self, uploads_dir: Optional[Path] = None, image_format: str = "webp"):
        self.detector: Optional[ParkingSystemComplete] = None
        self.camera: Optional[cv2.VideoCapture] = None
        self.output_frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.processing_active = False
        self.video_source: Optional[str] = None
        self.image_format = image_format.lower() if image_format in ("webp", "jpg") else "webp"
        self._temp_video_path: Optional[Path] = None
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

    def configure(self, uploads_dir: Optional[Path] = None, image_format: Optional[str] = None):
        if image_format is not None:
            fmt = image_format.lower()
            if fmt in ("webp", "jpg"):
                self.image_format = fmt

    def init_models(self,
                    zone_model_path: str,
                    vehicle_model_path: str,
                    orientation_model_path: str):
        self.detector = ParkingSystemComplete(zone_model_path, vehicle_model_path, orientation_model_path)

    def process_video_stream(self):
        frame_count = 0
        while self.processing_active and self.camera is not None:
            ret, frame = self.camera.read()
            if not ret:
                self.processing_active = False
                break
            frame_count += 1
            result_frame, zones, stats = self.detector.process_frame(frame, frame_count)
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
                self.output_frame = result_frame.copy()

    def start(self, source_type: str = 'webcam', video_path: Optional[str] = None) -> Dict[str, Any]:
        if self.processing_active:
            return {"status": "already_running"}
        if self.detector is None:
            return {"status": "error", "message": "Detector no inicializado"}
        if source_type == 'webcam':
            self.camera = cv2.VideoCapture(0)
            self.video_source = 'Webcam'
        else:
            if not video_path:
                return {"status": "error", "message": "Falta nombre de archivo"}
            vp = Path(video_path)
            if not vp.exists():
                return {"status": "error", "message": f"Video no encontrado: {vp}"}
            self.camera = cv2.VideoCapture(str(vp))
            self.video_source = str(vp)
        if not self.camera.isOpened():
            self.camera = None
            return {"status": "error", "message": "No se pudo abrir la fuente de video"}
        self.processing_active = True
        thread = threading.Thread(target=self.process_video_stream, daemon=True)
        thread.start()
        return {"status": "started", "source": self.video_source}

    def start_from_bytes(self, content: bytes, suffix: Optional[str] = None) -> Dict[str, Any]:
        if self.processing_active:
            return {"status": "already_running"}
        if self.detector is None:
            return {"status": "error", "message": "Detector no inicializado"}
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".mp4")
            tmp.write(content)
            tmp.flush()
            tmp.close()
            self._temp_video_path = Path(tmp.name)
            self.camera = cv2.VideoCapture(tmp.name)
        except Exception as e:
            return {"status": "error", "message": f"No se pudo crear archivo temporal: {e}"}
        if not self.camera.isOpened():
            try:
                if self._temp_video_path and self._temp_video_path.exists():
                    self._temp_video_path.unlink(missing_ok=True)
            except Exception:
                pass
            self._temp_video_path = None
            self.camera = None
            return {"status": "error", "message": "No se pudo abrir el video subido"}
        self.video_source = "upload-temp"
        self.processing_active = True
        thread = threading.Thread(target=self.process_video_stream, daemon=True)
        thread.start()
        return {"status": "started", "source": self.video_source}

    def stop(self) -> Dict[str, Any]:
        self.processing_active = False
        time.sleep(0.3)
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        # Clean ephemeral temp file if created
        if self._temp_video_path is not None:
            try:
                if self._temp_video_path.exists():
                    self._temp_video_path.unlink(missing_ok=True)
            except Exception:
                pass
            self._temp_video_path = None
        return {"status": "stopped"}

    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.stats)

    # Backwards compatibility: write to temp and return its path (not persisted)
    def save_upload(self, filename: str, content: bytes) -> Path:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix or ".mp4")
        tmp.write(content)
        tmp.flush()
        tmp.close()
        return Path(tmp.name)

    def mjpeg_generator(self) -> Generator[bytes, None, None]:
        while True:
            with self.lock:
                frame_local = None if self.output_frame is None else self.output_frame.copy()
            if frame_local is None:
                time.sleep(0.05)
                continue
            ok, buffer = cv2.imencode('.jpg', frame_local, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                time.sleep(0.02)
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    def get_encoded_frame(self) -> Optional[bytes]:
        with self.lock:
            if self.output_frame is None:
                return None
            frame_local = self.output_frame.copy()
        # Encode according to configured format
        if self.image_format == 'webp':
            ok, buffer = cv2.imencode('.webp', frame_local, [cv2.IMWRITE_WEBP_QUALITY, 80])
            if not ok:
                # Fallback to JPEG if OpenCV lacks WebP support
                ok, buffer = cv2.imencode('.jpg', frame_local, [cv2.IMWRITE_JPEG_QUALITY, 85])
        else:
            ok, buffer = cv2.imencode('.jpg', frame_local, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return None
        return buffer.tobytes()


# Global singleton for DI in routers
vision_service = VisionService()
