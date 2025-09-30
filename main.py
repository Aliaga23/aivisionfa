from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
from ultralytics import YOLO
import numpy as np
import os
from datetime import datetime
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import threading
import asyncio
import time
from pathlib import Path

app = FastAPI()

# Configuración
UPLOAD_FOLDER = 'uploads'
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)

templates = Jinja2Templates(directory="templates")

# Variables globales
detector = None
camera = None
output_frame = None
lock = threading.Lock()
processing_active = False
video_source = None
current_stats = {
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


class CarOrientationPredictor:
    def __init__(self, model_path='car_orientation_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = models.resnet50(pretrained=False)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 2)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        self.classes = ['Frontal', 'Reversa']
    
    def predict(self, image):
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
    def __init__(self, zone_model_path, vehicle_model_path, orientation_model_path, fps=30):
        self.zone_model = YOLO(zone_model_path)
        self.vehicle_model = YOLO(vehicle_model_path)
        self.orientation_predictor = CarOrientationPredictor(orientation_model_path)
        
        self.vehicle_classes = [2, 3, 5, 7]
        self.vehicle_names = {2: 'auto', 3: 'moto', 5: 'bus', 7: 'camion'}
        self.zone_class_names = self.zone_model.names
        self.zone_color = (0, 0, 255)
        self.occupied_color = (0, 165, 255)
    
    def box_in_zone(self, box_center, box_coords, zone_mask):
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
    
    def process_frame(self, frame, frame_number, zone_conf=0.5, vehicle_conf=0.4):
        frame_result = frame.copy()
        h, w = frame.shape[:2]
        
        zone_results = self.zone_model(frame, conf=zone_conf, verbose=False, stream=True)
        zones = []
        
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
        
        vehicle_results = self.vehicle_model(frame, conf=vehicle_conf, verbose=False, stream=True)
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
            
            contours, _ = cv2.findContours(
                zone['mask'].astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(frame_result, contours, -1, color, 3)
            
            x1, y1, x2, y2 = zone['bbox']
            status = f"OCUPADA ({len(zone['vehicles'])})" if is_occupied else "LIBRE"
            label = f"{zone['class']} - {status}"
            
            (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame_result, (x1, y1 - h_text - 10), (x1 + w_text, y1), color, -1)
            cv2.putText(frame_result, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        frame_result = cv2.addWeighted(frame_result, 1, mask_overlay, 0.3, 0)
        
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
                thickness = 2
                cv2.rectangle(frame_result, (x1, y1), (x2, y2), color, thickness)
                
                label = f"{orientation} | ESTACIONADO"
                
                (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame_result, (x1, y2), (x1 + w_text, y2 + h_text + 10), color, -1)
                cv2.putText(frame_result, label, (x1, y2 + h_text + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.circle(frame_result, vehicle['center'], 5, color, -1)
        
        return frame_result, zones, stats


def process_video_stream():
    global output_frame, lock, processing_active, current_stats, detector, camera
    
    frame_count = 0
    
    while processing_active:
        ret, frame = camera.read()
        if not ret:
            processing_active = False
            break
        
        frame_count += 1
        result_frame, zones, stats = detector.process_frame(frame, frame_count)
        
        with lock:
            current_stats['zones_total'] = len(zones)
            current_stats['zones_occupied'] = sum(1 for z in zones if z['vehicles'])
            current_stats['vehicles_in_zone'] = stats['total_in_zone']
            current_stats['moving'] = 0
            current_stats['parked'] = stats['parked']
            current_stats['frontal'] = stats['frontal']
            current_stats['reversa'] = stats['reversa']
            current_stats['avg_speed'] = 0
            current_stats['max_speed'] = 0
            output_frame = result_frame.copy()


def generate_frames():
    global output_frame, lock
    
    while True:
        # Tomar una copia SIN bloquear el resto de la lógica
        with lock:
            frame_local = None if output_frame is None else output_frame.copy()
        
        # Si no hay frame, dormir FUERA del lock
        if frame_local is None:
            time.sleep(0.05)
            continue
        
        # Codificar el frame
        ret, buffer = cv2.imencode('.jpg', frame_local, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.03)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(),
                            media_type='multipart/x-mixed-replace; boundary=frame')


@app.get("/stats")
async def stats():
    with lock:
        return current_stats


@app.post("/upload")
async def upload_file(video: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_FOLDER, video.filename)
        with open(file_path, "wb") as buffer:
            content = await video.read()
            buffer.write(content)
        return {"status": "success", "filename": video.filename}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/start")
async def start_processing(request: Request):
    global processing_active, camera, detector, video_source
    
    if processing_active:
        return {"status": "already_running"}
    
    data = await request.json()
    source_type = data.get('source_type', 'webcam')
    video_filename = data.get('video_filename', '')
    
    try:
        if source_type == 'webcam':
            camera = cv2.VideoCapture(0)
            video_source = 'Webcam'
        else:
            video_path = os.path.join(UPLOAD_FOLDER, video_filename)
            if not os.path.exists(video_path):
                return {"status": "error", "message": "Video no encontrado"}
            camera = cv2.VideoCapture(video_path)
            video_source = video_filename
        
        if not camera.isOpened():
            return {"status": "error", "message": "No se pudo abrir la fuente de video"}
        
        processing_active = True
        
        thread = threading.Thread(target=process_video_stream)
        thread.daemon = True
        thread.start()
        
        return {"status": "started", "source": video_source}
    
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/stop")
async def stop_processing():
    global processing_active, camera
    
    processing_active = False
    time.sleep(0.5)
    
    if camera is not None:
        camera.release()
        camera = None
    
    return {"status": "stopped"}


@app.on_event("startup")
async def startup_event():
    global detector
    
    zone_model_path = r"C:\Users\PC\Desktop\sw2p12\parking_zone_training\parking_zones4\weights\best.pt"
    vehicle_model_path = 'yolov8n.pt'
    orientation_model_path = 'car_orientation_model.pth'
    
    detector = ParkingSystemComplete(
        zone_model_path,
        vehicle_model_path,
        orientation_model_path,
        fps=30
    )
    
    print("\n" + "=" * 60)
    print("SERVIDOR FASTAPI DE ESTACIONAMIENTO INICIADO")
    print("=" * 60)
    print("Abre tu navegador en: http://localhost:8000")
    print("=" * 60)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)