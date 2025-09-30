import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
from ultralytics import YOLO
import numpy as np
from tkinter import Tk, filedialog
import os
from collections import defaultdict
import time


class CarOrientationPredictor:
    """
    Usa el modelo entrenado para predecir orientación de autos
    """
    def __init__(self, model_path='car_orientation_model.pth'):
        print("🤖 Cargando modelo entrenado...")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"💻 Usando dispositivo: {self.device}")
        
        # Crear arquitectura del modelo
        self.model = models.resnet50(pretrained=False)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 2)
        
        # Cargar pesos entrenados
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ No se encontró el modelo: {model_path}")
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Modelo cargado desde: {model_path}")
        
        # Transformaciones (deben ser las mismas que en entrenamiento)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.classes = ['Frontal', 'Reversa']
    
    def predict(self, image):
        """
        Predice la orientación de un auto
        
        Args:
            image: numpy array (BGR) o PIL Image
            
        Returns:
            tuple: (orientacion, confianza)
        """
        # Convertir a PIL si es numpy array
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # Aplicar transformaciones
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predecir
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        orientation = self.classes[predicted.item()]
        conf = confidence.item()
        
        return orientation, conf


class SpeedTracker:
    """
    Rastrea autos y calcula velocidad basándose en distancia conocida
    """
    def __init__(self, distance_meters=39, fps=30, movement_threshold=20):
        """
        Args:
            distance_meters: Distancia conocida que cruza el auto (metros)
            fps: Frames por segundo del video
            movement_threshold: Píxeles mínimos de movimiento para considerar que está en movimiento
        """
        self.distance_meters = distance_meters
        self.fps = fps
        self.movement_threshold = movement_threshold
        
        # Tracking de autos: {car_id: {'positions': [], 'timestamps': [], 'orientation': str}}
        self.tracked_cars = {}
        self.next_car_id = 0
        
        # Para matching entre frames
        self.max_distance_match = 100  # Distancia máxima en píxeles para considerar el mismo auto
        
    def get_center(self, bbox):
        """Calcula el centro de un bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def calculate_distance(self, pos1, pos2):
        """Distancia euclidiana entre dos posiciones"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def match_detection_to_track(self, center, current_frame):
        """
        Encuentra el track más cercano a una nueva detección
        
        Returns:
            car_id o None si no hay match
        """
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
        Actualiza tracking con nuevas detecciones
        
        Args:
            detections: Lista de detecciones del frame actual
            frame_number: Número de frame actual
            
        Returns:
            dict: {car_id: {'speed_kmh': float, 'is_moving': bool, 'orientation': str}}
        """
        current_detections = {}
        
        # Procesar cada detección
        for det in detections:
            center = self.get_center(det['bbox'])
            orientation = det['orientation']
            
            # Intentar hacer match con tracks existentes
            car_id = self.match_detection_to_track(center, frame_number)
            
            if car_id is None:
                # Nuevo auto detectado
                car_id = self.next_car_id
                self.next_car_id += 1
                self.tracked_cars[car_id] = {
                    'positions': [],
                    'frame_numbers': [],
                    'orientation': orientation,
                    'bbox': det['bbox']
                }
            
            # Actualizar track
            self.tracked_cars[car_id]['positions'].append(center)
            self.tracked_cars[car_id]['frame_numbers'].append(frame_number)
            self.tracked_cars[car_id]['orientation'] = orientation
            self.tracked_cars[car_id]['bbox'] = det['bbox']
            
            current_detections[car_id] = det
        
        # Calcular velocidades
        speeds = {}
        for car_id in current_detections.keys():
            track = self.tracked_cars[car_id]
            
            if len(track['positions']) < 2:
                speeds[car_id] = {
                    'speed_kmh': 0,
                    'is_moving': False,
                    'orientation': track['orientation'],
                    'bbox': track['bbox']
                }
                continue
            
            # Calcular movimiento total (en píxeles)
            positions = track['positions']
            
            # Usar las últimas N posiciones para cálculo más estable
            window_size = min(10, len(positions))
            recent_positions = positions[-window_size:]
            
            # Calcular distancia total en píxeles
            total_pixel_distance = 0
            for i in range(1, len(recent_positions)):
                total_pixel_distance += self.calculate_distance(
                    recent_positions[i-1], 
                    recent_positions[i]
                )
            
            # Determinar si está en movimiento
            is_moving = total_pixel_distance > self.movement_threshold
            
            if is_moving and len(recent_positions) > 1:
                # Calcular velocidad
                # Frames transcurridos
                frames_elapsed = window_size - 1
                
                # Tiempo transcurrido (segundos)
                time_elapsed = frames_elapsed / self.fps
                
                # Velocidad en píxeles por segundo
                pixel_speed = total_pixel_distance / time_elapsed if time_elapsed > 0 else 0
                
                # Convertir a km/h (asumiendo que cruza la pantalla = distance_meters)
                # Estimamos el tamaño de la pantalla en píxeles para hacer la conversión
                # Usamos la distancia horizontal máxima entre el primer y último punto
                horizontal_distance = abs(recent_positions[-1][0] - recent_positions[0][0])
                
                if horizontal_distance > 10:  # Si hay movimiento horizontal significativo
                    # Proporción: píxeles recorridos / tamaño pantalla = metros recorridos / distance_meters
                    # Asumimos que el ancho de la pantalla corresponde a los 39 metros
                    meters_per_pixel = self.distance_meters / 1920  # Asumiendo 1920px de ancho típico
                    
                    # Velocidad en metros por segundo
                    speed_ms = pixel_speed * meters_per_pixel
                    
                    # Convertir a km/h
                    speed_kmh = speed_ms * 3.6
                    
                    # Limitar valores extremos
                    speed_kmh = min(speed_kmh, 200)  # Máximo 200 km/h
                    speed_kmh = max(speed_kmh, 0)    # Mínimo 0 km/h
                else:
                    speed_kmh = 0
            else:
                speed_kmh = 0
            
            speeds[car_id] = {
                'speed_kmh': speed_kmh,
                'is_moving': is_moving,
                'orientation': track['orientation'],
                'bbox': track['bbox']
            }
        
        # Limpiar tracks antiguos (no vistos en los últimos 30 frames)
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


class YOLOCarDetectorWithOrientation:
    """
    Sistema completo: YOLO detecta autos + Modelo clasifica orientación + Tracking de velocidad
    """
    def __init__(self, yolo_model='yolov8l.pt', classifier_model='car_orientation_model.pth', 
                 distance_meters=39, fps=30):
        print("=" * 60)
        print("🚗 SISTEMA DE DETECCIÓN DE ORIENTACIÓN Y VELOCIDAD")
        print("=" * 60)
        
        # Cargar YOLO
        print("\n1️⃣ Cargando YOLO para detección de autos...")
        self.yolo = YOLO(yolo_model)
        self.car_classes = [2]  # Clase 'car' en COCO
        print("✅ YOLO cargado")
        
        # Cargar clasificador
        print("\n2️⃣ Cargando clasificador de orientación...")
        self.predictor = CarOrientationPredictor(classifier_model)
        
        # Inicializar tracker de velocidad
        print("\n3️⃣ Inicializando sistema de tracking...")
        self.speed_tracker = SpeedTracker(distance_meters=distance_meters, fps=fps)
        print(f"✅ Distancia configurada: {distance_meters} metros")
        
        print("\n" + "=" * 60)
        print("✨ Sistema listo para usar!")
        print("=" * 60)
    
    def detect_and_classify(self, frame, frame_number=0):
        """
        Detecta autos, clasifica orientación y calcula velocidad
        
        Args:
            frame: Imagen/frame (numpy array BGR)
            frame_number: Número de frame actual
            
        Returns:
            frame_result: Frame con anotaciones
            detections: Lista de detecciones con velocidades
        """
        results = self.yolo(frame, conf=0.5, verbose=False)
        detections = []
        frame_result = frame.copy()
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                cls = int(box.cls[0])
                
                # Solo procesar autos
                if cls in self.car_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    yolo_conf = float(box.conf[0])
                    
                    # Extraer región del auto
                    car_roi = frame[y1:y2, x1:x2]
                    
                    if car_roi.size == 0:
                        continue
                    
                    # Clasificar orientación
                    orientation, orientation_conf = self.predictor.predict(car_roi)
                    
                    # Guardar detección
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'yolo_confidence': yolo_conf,
                        'orientation': orientation,
                        'orientation_confidence': orientation_conf
                    })
        
        # Actualizar tracking y calcular velocidades
        speeds = self.speed_tracker.update(detections, frame_number)
        
        # Dibujar resultados
        for car_id, speed_data in speeds.items():
            x1, y1, x2, y2 = speed_data['bbox']
            orientation = speed_data['orientation']
            speed_kmh = speed_data['speed_kmh']
            is_moving = speed_data['is_moving']
            
            # Color según orientación
            if not is_moving:
                color = (128, 128, 128)  # Gris para estacionados
            elif orientation == 'Frontal':
                color = (0, 255, 0)  # Verde para frontal
            else:
                color = (0, 0, 255)  # Rojo para reversa
            
            # Dibujar bounding box
            thickness = 4 if is_moving else 2
            cv2.rectangle(frame_result, (x1, y1), (x2, y2), color, thickness)
            
            # Preparar etiqueta
            if is_moving:
                label = f"{orientation} | {speed_kmh:.1f} km/h"
                status = "EN MOVIMIENTO"
            else:
                label = f"{orientation} | ESTACIONADO"
                status = "ESTACIONADO"
            
            # Dibujar etiqueta principal
            (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame_result, (x1, y1 - h_text - 10), 
                        (x1 + w_text, y1), color, -1)
            cv2.putText(frame_result, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # ID del auto (pequeño)
            cv2.putText(frame_result, f"ID:{car_id}", (x1, y2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Actualizar detecciones con información de velocidad
        detection_results = []
        for car_id, speed_data in speeds.items():
            detection_results.append({
                'car_id': car_id,
                'bbox': speed_data['bbox'],
                'orientation': speed_data['orientation'],
                'speed_kmh': speed_data['speed_kmh'],
                'is_moving': speed_data['is_moving']
            })
        
        return frame_result, detection_results


def select_file(file_type):
    """Abre explorador de archivos"""
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    if file_type == 'image':
        file_path = filedialog.askopenfilename(
            title="Selecciona una imagen",
            filetypes=[
                ("Imágenes", "*.jpg *.jpeg *.png *.bmp"),
                ("Todos", "*.*")
            ]
        )
    elif file_type == 'video':
        file_path = filedialog.askopenfilename(
            title="Selecciona un video",
            filetypes=[
                ("Videos", "*.mp4 *.avi *.mov *.mkv"),
                ("Todos", "*.*")
            ]
        )
    else:
        file_path = None
    
    root.destroy()
    return file_path if file_path else None


def process_video(detector, video_path, fps=30):
    """Procesa un video o webcam con detección de velocidad"""
    if video_path.lower() == 'webcam':
        cap = cv2.VideoCapture(0)
        print("\n📹 Usando webcam...")
    else:
        cap = cv2.VideoCapture(video_path)
        # Obtener FPS real del video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        print(f"\n🎬 Procesando: {os.path.basename(video_path)}")
        print(f"📊 FPS del video: {fps}")
    
    # Actualizar FPS en el tracker
    detector.speed_tracker.fps = fps
    
    if not cap.isOpened():
        print("❌ Error al abrir el video/webcam")
        return
    
    print("▶️  Presiona 'Q' para salir, 'P' para pausar, 'S' para guardar frame")
    
    frame_count = 0
    paused = False
    
    # Estadísticas del video completo
    stats = {
        'total_detections': 0,
        'moving_cars': 0,
        'parked_cars': 0,
        'frontal_count': 0,
        'reversa_count': 0,
        'speeds': []
    }
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\n🏁 Fin del video")
                break
            
            frame_count += 1
            
            # Detectar y clasificar con velocidad
            result_frame, detections = detector.detect_and_classify(frame, frame_count)
            
            # Actualizar estadísticas
            for det in detections:
                if det['is_moving']:
                    stats['moving_cars'] += 1
                    stats['speeds'].append(det['speed_kmh'])
                else:
                    stats['parked_cars'] += 1
                
                if det['orientation'] == 'Frontal':
                    stats['frontal_count'] += 1
                else:
                    stats['reversa_count'] += 1
            
            # Calcular estadísticas de velocidad
            if stats['speeds']:
                avg_speed = np.mean(stats['speeds'])
                max_speed = np.max(stats['speeds'])
            else:
                avg_speed = 0
                max_speed = 0
            
            # Información en pantalla
            moving_now = sum(1 for d in detections if d['is_moving'])
            parked_now = len(detections) - moving_now
            
            info_lines = [
                f"Frame: {frame_count} | FPS: {fps}",
                f"Autos: {len(detections)} (Mov: {moving_now}, Est: {parked_now})",
                f"Vel Max: {max_speed:.1f} km/h | Vel Prom: {avg_speed:.1f} km/h",
                f"Total: F:{stats['frontal_count']} R:{stats['reversa_count']}",
                "Q=Salir | P=Pausa | S=Guardar"
            ]
            
            y_offset = 30
            for line in info_lines:
                (w_text, h_text), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(result_frame, (5, y_offset - h_text - 5), 
                            (15 + w_text, y_offset + 5), (0, 0, 0), -1)
                
                cv2.putText(result_frame, line, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 30
            
            # Mostrar detecciones cada 30 frames
            if frame_count % 30 == 0 and detections:
                print(f"\n📊 Frame {frame_count}:")
                for det in detections:
                    status = "🚗" if det['is_moving'] else "🅿️"
                    if det['is_moving']:
                        print(f"  {status} Auto {det['car_id']}: {det['orientation']} - {det['speed_kmh']:.1f} km/h")
                    else:
                        print(f"  {status} Auto {det['car_id']}: {det['orientation']} - ESTACIONADO")
        
        cv2.imshow('Detección de Orientación y Velocidad', result_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            print("\n⏹️  Detenido por el usuario")
            break
        elif key == ord('p') or key == ord('P'):
            paused = not paused
            if paused:
                print("\n⏸️  PAUSADO - Presiona 'P' para continuar")
            else:
                print("\n▶️  REPRODUCIENDO")
        elif key == ord('s') or key == ord('S'):
            filename = f"captura_speed_{frame_count}.jpg"
            cv2.imwrite(filename, result_frame)
            print(f"\n💾 Frame guardado: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Mostrar resumen final
    if frame_count > 0 and (stats['moving_cars'] + stats['parked_cars']) > 0:
        total_cars = stats['moving_cars'] + stats['parked_cars']
        print("\n" + "=" * 60)
        print("📊 RESUMEN DEL VIDEO")
        print("=" * 60)
        print(f"Total de frames procesados: {frame_count}")
        print(f"Total de detecciones: {total_cars}")
        print(f"\n🚗 ESTADO DE MOVIMIENTO:")
        print(f"  • En movimiento: {stats['moving_cars']} ({100*stats['moving_cars']/total_cars:.1f}%)")
        print(f"  • Estacionados: {stats['parked_cars']} ({100*stats['parked_cars']/total_cars:.1f}%)")
        print(f"\n🧭 ORIENTACIÓN:")
        print(f"  • Frontales: {stats['frontal_count']} ({100*stats['frontal_count']/total_cars:.1f}%)")
        print(f"  • Reversa: {stats['reversa_count']} ({100*stats['reversa_count']/total_cars:.1f}%)")
        
        if stats['speeds']:
            print(f"\n⚡ VELOCIDADES (solo autos en movimiento):")
            print(f"  • Velocidad promedio: {np.mean(stats['speeds']):.1f} km/h")
            print(f"  • Velocidad máxima: {np.max(stats['speeds']):.1f} km/h")
            print(f"  • Velocidad mínima: {np.min(stats['speeds']):.1f} km/h")
        
        print("=" * 60)


def main():
    print("\n" + "=" * 60)
    print("🚗 DETECTOR DE ORIENTACIÓN Y VELOCIDAD DE AUTOS")
    print("   Detecta: Orientación + Velocidad + Movimiento")
    print("=" * 60)
    
    # Configuración de distancia
    print("\n⚙️  CONFIGURACIÓN:")
    print(f"   Distancia de cruce: 39 metros")
    print(f"   Los autos estacionados NO se contarán en velocidad")
    
    # Inicializar sistema
    try:
        detector = YOLOCarDetectorWithOrientation(
            yolo_model='yolov8l.pt',
            classifier_model='car_orientation_model.pth',
            distance_meters=39,
            fps=30  # Se actualizará automáticamente con el FPS real del video
        )
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\n💡 Asegúrate de que el archivo 'car_orientation_model.pth' existe")
        return
    
    # Menú principal
    while True:
        print("\n" + "=" * 60)
        print("MENÚ PRINCIPAL")
        print("=" * 60)
        print("1. 🎬 Procesar VIDEO (con velocidad)")
        print("2. 📹 Usar WEBCAM (con velocidad)")
        print("3. 🚪 SALIR")
        print("=" * 60)
        
        opcion = input("\nSelecciona (1-3): ").strip()
        
        if opcion == '1':
            print("\n📁 Selecciona un video...")
            ruta = select_file('video')
            if ruta:
                process_video(detector, ruta)
            else:
                print("❌ No se seleccionó archivo")
        
        elif opcion == '2':
            process_video(detector, 'webcam')
        
        elif opcion == '3':
            print("\n👋 ¡Hasta luego!")
            break
        
        else:
            print("\n❌ Opción inválida")


if __name__ == "__main__":
    main()