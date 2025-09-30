import cv2
from ultralytics import YOLO
import numpy as np
from tkinter import Tk, filedialog
import os
from datetime import datetime
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


class CarOrientationPredictor:
    """
    Predice orientación de autos usando ResNet50
    """
    def __init__(self, model_path='car_orientation_model.pth'):
        print("Cargando modelo de orientación...")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = models.resnet50(pretrained=False)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 2)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontró el modelo: {model_path}")
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Modelo de orientación cargado")
        
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
        
        orientation = self.classes[predicted.item()]
        conf = confidence.item()
        
        return orientation, conf


class SpeedTracker:
    """
    Rastrea vehículos y calcula velocidad
    """
    def __init__(self, fps=30, pixel_to_meter_ratio=0.05):
        self.fps = fps
        self.pixel_to_meter_ratio = pixel_to_meter_ratio
        
        self.tracked_cars = {}
        self.next_car_id = 0
        self.max_distance_match = 150
        
    def get_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def calculate_distance(self, pos1, pos2):
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def match_detection_to_track(self, center):
        min_distance = float('inf')
        matched_id = None
        
        for car_id, track_data in self.tracked_cars.items():
            if len(track_data['positions']) == 0:
                continue
            
            last_pos = track_data['positions'][-1]
            distance = self.calculate_distance(center, last_pos)
            
            if distance < self.max_distance_match and distance < min_distance:
                min_distance = distance
                matched_id = car_id
        
        return matched_id
    
    def update(self, detections, frame_number):
        """
        Actualiza tracking y calcula velocidades
        
        Args:
            detections: Lista con {'bbox': tuple, 'orientation': str, 'vehicle_id': int}
        
        Returns:
            dict: {vehicle_id: {'speed_kmh': float, 'is_moving': bool}}
        """
        current_detections = {}
        
        for det in detections:
            center = self.get_center(det['bbox'])
            vehicle_id = det['vehicle_id']
            
            car_id = self.match_detection_to_track(center)
            
            if car_id is None:
                car_id = self.next_car_id
                self.next_car_id += 1
                self.tracked_cars[car_id] = {
                    'positions': [],
                    'frame_numbers': [],
                    'vehicle_id': vehicle_id
                }
            
            self.tracked_cars[car_id]['positions'].append(center)
            self.tracked_cars[car_id]['frame_numbers'].append(frame_number)
            self.tracked_cars[car_id]['vehicle_id'] = vehicle_id
            
            current_detections[car_id] = vehicle_id
        
        speeds = {}
        for car_id, vehicle_id in current_detections.items():
            track = self.tracked_cars[car_id]
            
            if len(track['positions']) < 5:
                speeds[vehicle_id] = {'speed_kmh': 0, 'is_moving': False}
                continue
            
            window_size = min(15, len(track['positions']))
            recent_positions = track['positions'][-window_size:]
            recent_frames = track['frame_numbers'][-window_size:]
            
            total_pixel_distance = 0
            for i in range(1, len(recent_positions)):
                total_pixel_distance += self.calculate_distance(
                    recent_positions[i-1], 
                    recent_positions[i]
                )
            
            frames_elapsed = recent_frames[-1] - recent_frames[0]
            
            if frames_elapsed > 0 and total_pixel_distance > 5:
                time_elapsed = frames_elapsed / self.fps
                distance_meters = total_pixel_distance * self.pixel_to_meter_ratio
                speed_ms = distance_meters / time_elapsed
                speed_kmh = speed_ms * 3.6
                speed_kmh = min(speed_kmh, 150)
                speed_kmh = max(speed_kmh, 0)
            else:
                speed_kmh = 0
            
            is_moving = speed_kmh > 3.0
            
            speeds[vehicle_id] = {
                'speed_kmh': speed_kmh,
                'is_moving': is_moving
            }
        
        cars_to_remove = []
        for car_id, track in self.tracked_cars.items():
            if car_id not in current_detections:
                if len(track['frame_numbers']) > 0:
                    last_frame = track['frame_numbers'][-1]
                    if frame_number - last_frame > 30:
                        cars_to_remove.append(car_id)
        
        for car_id in cars_to_remove:
            del self.tracked_cars[car_id]
        
        return speeds


class ParkingSystemComplete:
    """
    Sistema completo: Zonas + Vehículos + Orientación + Velocidad
    """
    def __init__(self, zone_model_path, vehicle_model_path, orientation_model_path, fps=30):
        print("=" * 60)
        print("SISTEMA COMPLETO DE ESTACIONAMIENTO")
        print("=" * 60)
        
        print(f"\n1. Cargando modelo de zonas...")
        if not os.path.exists(zone_model_path):
            raise FileNotFoundError(f"No se encontró: {zone_model_path}")
        
        self.zone_model = YOLO(zone_model_path)
        print("Modelo de zonas cargado")
        
        print(f"\n2. Cargando modelo de vehículos...")
        self.vehicle_model = YOLO(vehicle_model_path)
        print("Modelo de vehículos cargado")
        
        print(f"\n3. Cargando clasificador de orientación...")
        self.orientation_predictor = CarOrientationPredictor(orientation_model_path)
        
        print(f"\n4. Inicializando tracker de velocidad...")
        self.speed_tracker = SpeedTracker(fps=fps, pixel_to_meter_ratio=0.05)
        print("Tracker inicializado")
        
        self.vehicle_classes = [2, 3, 5, 7]
        self.vehicle_names = {2: 'auto', 3: 'moto', 5: 'bus', 7: 'camion'}
        
        self.zone_class_names = self.zone_model.names
        
        print("\n" + "=" * 60)
        print("Sistema completo listo!")
        print("=" * 60)
        
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
        
        # 1. DETECTAR ZONAS
        zone_results = self.zone_model(frame, conf=zone_conf, verbose=False)
        
        zones = []
        
        for result in zone_results:
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                boxes = result.boxes
                
                for i, (mask, box) in enumerate(zip(masks, boxes)):
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = self.zone_class_names[cls_id]
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    mask_resized = cv2.resize(mask, (w, h))
                    mask_bool = mask_resized > 0.5
                    
                    zone_info = {
                        'class': class_name,
                        'confidence': conf,
                        'bbox': (x1, y1, x2, y2),
                        'mask': mask_bool,
                        'vehicles': []
                    }
                    zones.append(zone_info)
        
        # 2. DETECTAR VEHÍCULOS
        vehicle_results = self.vehicle_model(frame, conf=vehicle_conf, verbose=False)
        
        vehicles = []
        vehicle_counter = 0
        
        for result in vehicle_results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                
                if cls_id in self.vehicle_classes:
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    
                    vehicle_info = {
                        'id': vehicle_counter,
                        'type': self.vehicle_names[cls_id],
                        'confidence': conf,
                        'bbox': (x1, y1, x2, y2),
                        'center': center,
                        'in_zone': False,
                        'zone_id': None,
                        'orientation': None,
                        'orientation_confidence': None
                    }
                    vehicles.append(vehicle_info)
                    vehicle_counter += 1
        
        # 3. VERIFICAR VEHÍCULOS EN ZONAS
        for v_idx, vehicle in enumerate(vehicles):
            for z_idx, zone in enumerate(zones):
                if self.box_in_zone(vehicle['center'], vehicle['bbox'], zone['mask']):
                    vehicle['in_zone'] = True
                    vehicle['zone_id'] = z_idx
                    zone['vehicles'].append(v_idx)
                    break
        
        # 4. CLASIFICAR ORIENTACIÓN PARA TODOS LOS AUTOS EN ZONA
        vehicles_for_tracking = []
        
        for vehicle in vehicles:
            if vehicle['in_zone'] and vehicle['type'] == 'auto':
                x1, y1, x2, y2 = vehicle['bbox']
                car_roi = frame[y1:y2, x1:x2]
                
                if car_roi.size > 0:
                    orientation, orientation_conf = self.orientation_predictor.predict(car_roi)
                    vehicle['orientation'] = orientation
                    vehicle['orientation_confidence'] = orientation_conf
                    
                    vehicles_for_tracking.append({
                        'bbox': vehicle['bbox'],
                        'orientation': orientation,
                        'vehicle_id': vehicle['id']
                    })
        
        # 5. CALCULAR VELOCIDADES
        speeds = self.speed_tracker.update(vehicles_for_tracking, frame_number)
        
        # 6. DIBUJAR ZONAS
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
            label = f"{zone['class']} {zone['confidence']*100:.1f}% - {status}"
            
            (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame_result, (x1, y1 - h_text - 10), (x1 + w_text, y1), color, -1)
            cv2.putText(frame_result, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        alpha = 0.3
        frame_result = cv2.addWeighted(frame_result, 1, mask_overlay, alpha, 0)
        
        # 7. DIBUJAR VEHÍCULOS EN ZONA CON ORIENTACIÓN
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
                
                # Obtener velocidad si existe
                speed_data = speeds.get(vehicle['id'], {'speed_kmh': 0, 'is_moving': False})
                speed_kmh = speed_data['speed_kmh']
                is_moving = speed_data['is_moving']
                
                stats['total_in_zone'] += 1
                
                if orientation == 'Frontal':
                    stats['frontal'] += 1
                else:
                    stats['reversa'] += 1
                
                if is_moving:
                    stats['moving'] += 1
                    stats['speeds'].append(speed_kmh)
                else:
                    stats['parked'] += 1
                
                # Color según orientación
                color = (0, 255, 0) if orientation == 'Frontal' else (0, 0, 255)
                
                # Bounding box más grueso si está en movimiento
                thickness = 4 if is_moving else 2
                cv2.rectangle(frame_result, (x1, y1), (x2, y2), color, thickness)
                
                # Etiqueta: mostrar velocidad solo si está en movimiento
                if is_moving:
                    label = f"{orientation} | {speed_kmh:.1f} km/h"
                else:
                    label = f"{orientation} | ESTACIONADO"
                
                (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame_result, (x1, y2), (x1 + w_text, y2 + h_text + 10), color, -1)
                cv2.putText(frame_result, label, (x1, y2 + h_text + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Círculo en centro
                cv2.circle(frame_result, vehicle['center'], 5, color, -1)
        
        return frame_result, zones, stats


def select_file(file_type):
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    if file_type == 'video':
        file_path = filedialog.askopenfilename(
            title="Selecciona un video",
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv"), ("Todos", "*.*")]
        )
    else:
        file_path = None
    
    root.destroy()
    return file_path if file_path else None


def process_video(detector, video_path, zone_conf=0.5, vehicle_conf=0.4, save_output=False):
    if video_path.lower() == 'webcam':
        cap = cv2.VideoCapture(0)
        print("\nUsando webcam...")
        is_webcam = True
    else:
        cap = cv2.VideoCapture(video_path)
        print(f"\nProcesando: {os.path.basename(video_path)}")
        is_webcam = False
    
    if not cap.isOpened():
        print("Error al abrir el video")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    detector.speed_tracker.fps = fps
    
    print(f"Resolución: {width}x{height} @ {fps} FPS")
    
    video_writer = None
    if save_output and not is_webcam:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"resultado_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Guardando video en: {output_path}")
    
    print("\nCONTROLES:")
    print("   Q = Salir | P = Pausar | S = Guardar frame")
    print("   +/- = Ajustar umbral zonas | [/] = Ajustar umbral vehiculos")
    
    frame_count = 0
    paused = False
    
    total_stats = {
        'total_in_zone': 0,
        'moving': 0,
        'parked': 0,
        'frontal': 0,
        'reversa': 0,
        'speeds': []
    }
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\nFin del video")
                break
            
            frame_count += 1
            
            result_frame, zones, frame_stats = detector.process_frame(
                frame, frame_count, zone_conf, vehicle_conf
            )
            
            # Acumular estadísticas
            total_stats['total_in_zone'] += frame_stats['total_in_zone']
            total_stats['moving'] += frame_stats['moving']
            total_stats['parked'] += frame_stats['parked']
            total_stats['frontal'] += frame_stats['frontal']
            total_stats['reversa'] += frame_stats['reversa']
            total_stats['speeds'].extend(frame_stats['speeds'])
            
            # Calcular promedios actuales
            if frame_stats['speeds']:
                avg_speed = np.mean(frame_stats['speeds'])
                max_speed = np.max(frame_stats['speeds'])
            else:
                avg_speed = 0
                max_speed = 0
            
            # Información en pantalla
            occupied = sum(1 for z in zones if z['vehicles'])
            
            info_lines = [
                f"Frame: {frame_count} | FPS: {fps}",
                f"Zonas: {len(zones)} ({occupied} ocupadas)",
                f"En zona: {frame_stats['total_in_zone']} | Mov: {frame_stats['moving']} | Est: {frame_stats['parked']}",
                f"F: {frame_stats['frontal']} | R: {frame_stats['reversa']}",
                f"Vel Max: {max_speed:.1f} | Prom: {avg_speed:.1f} km/h",
                f"Conf Z: {zone_conf*100:.0f}% | V: {vehicle_conf*100:.0f}%",
                "Q=Salir | P=Pausa | S=Guardar"
            ]
            
            y_offset = 30
            for line in info_lines:
                (w_text, h_text), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(result_frame, (5, y_offset - h_text - 5), (15 + w_text, y_offset + 5), (0, 0, 0), -1)
                cv2.putText(result_frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 30
            
            if video_writer is not None:
                video_writer.write(result_frame)
        
        cv2.imshow('Sistema de Estacionamiento', result_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('p') or key == ord('P'):
            paused = not paused
            print("\nPAUSADO" if paused else "\nREPRODUCIENDO")
        elif key == ord('s') or key == ord('S'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"captura_{frame_count}_{timestamp}.jpg"
            cv2.imwrite(filename, result_frame)
            print(f"\nFrame guardado: {filename}")
        elif key == ord('+') or key == ord('='):
            zone_conf = min(0.95, zone_conf + 0.05)
            print(f"\nUmbral zonas: {zone_conf*100:.0f}%")
        elif key == ord('-') or key == ord('_'):
            zone_conf = max(0.05, zone_conf - 0.05)
            print(f"\nUmbral zonas: {zone_conf*100:.0f}%")
        elif key == ord('['):
            vehicle_conf = max(0.1, vehicle_conf - 0.05)
            print(f"\nUmbral vehiculos: {vehicle_conf*100:.0f}%")
        elif key == ord(']'):
            vehicle_conf = min(0.95, vehicle_conf + 0.05)
            print(f"\nUmbral vehiculos: {vehicle_conf*100:.0f}%")
    
    cap.release()
    if video_writer is not None:
        video_writer.release()
        print(f"\nVideo guardado")
    cv2.destroyAllWindows()
    
    # Resumen
    if frame_count > 0 and total_stats['total_in_zone'] > 0:
        print("\n" + "=" * 60)
        print("RESUMEN DEL PROCESAMIENTO")
        print("=" * 60)
        print(f"Frames procesados: {frame_count}")
        print(f"Total detecciones en zona: {total_stats['total_in_zone']}")
        print(f"  En movimiento: {total_stats['moving']} ({100*total_stats['moving']/total_stats['total_in_zone']:.1f}%)")
        print(f"  Estacionados: {total_stats['parked']} ({100*total_stats['parked']/total_stats['total_in_zone']:.1f}%)")
        print(f"\nORIENTACIONES:")
        print(f"  Frontal: {total_stats['frontal']} ({100*total_stats['frontal']/total_stats['total_in_zone']:.1f}%)")
        print(f"  Reversa: {total_stats['reversa']} ({100*total_stats['reversa']/total_stats['total_in_zone']:.1f}%)")
        
        if total_stats['speeds']:
            print(f"\nVELOCIDADES (solo en movimiento):")
            print(f"  Promedio: {np.mean(total_stats['speeds']):.1f} km/h")
            print(f"  Maxima: {np.max(total_stats['speeds']):.1f} km/h")
            print(f"  Minima: {np.min(total_stats['speeds']):.1f} km/h")
        print("=" * 60)


def main():
    print("\n" + "=" * 60)
    print("SISTEMA COMPLETO DE ESTACIONAMIENTO")
    print("Zonas + Vehiculos + Orientacion + Velocidad")
    print("=" * 60)
    
    zone_model_path = r"C:\Users\PC\Desktop\sw2p12\parking_zone_training\parking_zones4\weights\best.pt"
    vehicle_model_path = 'yolov8n.pt'
    orientation_model_path = 'car_orientation_model.pth'
    
    try:
        detector = ParkingSystemComplete(
            zone_model_path, 
            vehicle_model_path,
            orientation_model_path,
            fps=30
        )
    except Exception as e:
        print(f"\nERROR: {e}")
        return
    
    while True:
        print("\n" + "=" * 60)
        print("MENU PRINCIPAL")
        print("=" * 60)
        print("1. Procesar VIDEO")
        print("2. Procesar VIDEO y GUARDAR")
        print("3. Usar WEBCAM")
        print("4. SALIR")
        print("=" * 60)
        
        opcion = input("\nSelecciona (1-4): ").strip()
        
        if opcion == '1':
            ruta = select_file('video')
            if ruta:
                try:
                    zone_conf = float(input("Umbral zonas (default 0.5): ") or 0.5)
                    vehicle_conf = float(input("Umbral vehiculos (default 0.4): ") or 0.4)
                except:
                    zone_conf, vehicle_conf = 0.5, 0.4
                process_video(detector, ruta, zone_conf, vehicle_conf, False)
            else:
                print("No se selecciono archivo")
        
        elif opcion == '2':
            ruta = select_file('video')
            if ruta:
                try:
                    zone_conf = float(input("Umbral zonas (default 0.5): ") or 0.5)
                    vehicle_conf = float(input("Umbral vehiculos (default 0.4): ") or 0.4)
                except:
                    zone_conf, vehicle_conf = 0.5, 0.4
                process_video(detector, ruta, zone_conf, vehicle_conf, True)
            else:
                print("No se selecciono archivo")
        
        elif opcion == '3':
            try:
                zone_conf = float(input("Umbral zonas (default 0.5): ") or 0.5)
                vehicle_conf = float(input("Umbral vehiculos (default 0.4): ") or 0.4)
            except:
                zone_conf, vehicle_conf = 0.5, 0.4
            process_video(detector, 'webcam', zone_conf, vehicle_conf, False)
        
        elif opcion == '4':
            print("\nHasta luego!")
            break
        
        else:
            print("Opcion invalida")


if __name__ == "__main__":
    main()