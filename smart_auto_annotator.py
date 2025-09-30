import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
from ultralytics import YOLO
import numpy as np
from tkinter import Tk, filedialog
import os


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


class YOLOCarDetectorWithOrientation:
    """
    Sistema completo: YOLO detecta autos + Modelo clasifica orientación
    """
    def __init__(self, yolo_model='yolov8l.pt', classifier_model='car_orientation_model.pth'):
        print("=" * 60)
        print("🚗 SISTEMA DE DETECCIÓN DE ORIENTACIÓN DE AUTOS")
        print("=" * 60)
        
        # Cargar YOLO
        print("\n1️⃣ Cargando YOLO para detección de autos...")
        self.yolo = YOLO(yolo_model)
        self.car_classes = [2]  # Clase 'car' en COCO
        print("✅ YOLO cargado")
        
        # Cargar clasificador
        print("\n2️⃣ Cargando clasificador de orientación...")
        self.predictor = CarOrientationPredictor(classifier_model)
        
        print("\n" + "=" * 60)
        print("✨ Sistema listo para usar!")
        print("=" * 60)
    
    def detect_and_classify(self, frame):
        """
        Detecta autos y clasifica su orientación
        
        Args:
            frame: Imagen/frame (numpy array BGR)
            
        Returns:
            frame_result: Frame con anotaciones
            detections: Lista de detecciones
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
                    
                    # Dibujar en el frame
                    color = (0, 255, 0) if orientation == 'Frontal' else (0, 0, 255)
                    cv2.rectangle(frame_result, (x1, y1), (x2, y2), color, 3)
                    
                    # Etiqueta con orientación y confianza
                    label = f"{orientation} {orientation_conf*100:.1f}%"
                    
                    # Fondo para el texto
                    (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(frame_result, (x1, y1 - h_text - 10), 
                                (x1 + w_text, y1), color, -1)
                    
                    # Texto
                    cv2.putText(frame_result, label, (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame_result, detections


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


def process_image(detector, image_path):
    """Procesa una imagen"""
    print(f"\n📸 Procesando: {os.path.basename(image_path)}")
    
    frame = cv2.imread(image_path)
    if frame is None:
        print("❌ Error al cargar la imagen")
        return
    
    # Detectar y clasificar
    result_frame, detections = detector.detect_and_classify(frame)
    
    # Mostrar resultados en consola
    print(f"\n🚗 Autos detectados: {len(detections)}")
    for i, det in enumerate(detections, 1):
        print(f"  {i}. {det['orientation']} - "
              f"Confianza: {det['orientation_confidence']*100:.1f}% - "
              f"BBox: {det['bbox']}")
    
    # Mostrar imagen
    # Redimensionar si es muy grande
    height, width = result_frame.shape[:2]
    max_height = 900
    if height > max_height:
        scale = max_height / height
        new_width = int(width * scale)
        result_frame = cv2.resize(result_frame, (new_width, max_height))
    
    cv2.imshow('Detección de Orientación - Presiona cualquier tecla', result_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video(detector, video_path):
    """Procesa un video o webcam"""
    if video_path.lower() == 'webcam':
        cap = cv2.VideoCapture(0)
        print("\n📹 Usando webcam...")
    else:
        cap = cv2.VideoCapture(video_path)
        print(f"\n🎬 Procesando: {os.path.basename(video_path)}")
    
    if not cap.isOpened():
        print("❌ Error al abrir el video/webcam")
        return
    
    print("▶️  Presiona 'Q' para salir, 'P' para pausar, 'S' para guardar frame")
    
    frame_count = 0
    paused = False
    
    # Estadísticas del video completo
    stats = {
        'total_detections': 0,
        'frontal_count': 0,
        'reversa_count': 0,
        'low_confidence_count': 0
    }
    
    # Para tracking de autos (evitar contar el mismo auto múltiples veces)
    detected_cars = {}  # {car_id: {'orientation': str, 'confidences': []}}
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\n🏁 Fin del video")
                break
            
            frame_count += 1
            
            # Detectar y clasificar
            result_frame, detections = detector.detect_and_classify(frame)
            
            # Actualizar estadísticas
            for det in detections:
                stats['total_detections'] += 1
                
                if det['orientation'] == 'Frontal':
                    stats['frontal_count'] += 1
                else:
                    stats['reversa_count'] += 1
                
                if det['orientation_confidence'] < 0.7:
                    stats['low_confidence_count'] += 1
            
            # Información en pantalla
            info_lines = [
                f"Frame: {frame_count}",
                f"Autos: {len(detections)}",
                f"Total F: {stats['frontal_count']} | R: {stats['reversa_count']}",
                "Q=Salir | P=Pausa | S=Guardar"
            ]
            
            y_offset = 30
            for line in info_lines:
                # Fondo negro para legibilidad
                (w_text, h_text), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(result_frame, (5, y_offset - h_text - 5), 
                            (15 + w_text, y_offset + 5), (0, 0, 0), -1)
                
                cv2.putText(result_frame, line, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 30
            
            # Mostrar detecciones cada 30 frames
            if frame_count % 30 == 0 and detections:
                print(f"\n📊 Frame {frame_count}:")
                for i, det in enumerate(detections, 1):
                    conf_emoji = "✅" if det['orientation_confidence'] > 0.9 else "⚠️" if det['orientation_confidence'] > 0.7 else "❌"
                    print(f"  Auto {i}: {det['orientation']} ({det['orientation_confidence']*100:.1f}%) {conf_emoji}")
        
        cv2.imshow('Detección de Orientación', result_frame)
        
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
            filename = f"captura_frame_{frame_count}.jpg"
            cv2.imwrite(filename, result_frame)
            print(f"\n💾 Frame guardado: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Mostrar resumen final
    if frame_count > 0:
        print("\n" + "=" * 60)
        print("📊 RESUMEN DEL VIDEO")
        print("=" * 60)
        print(f"Total de frames procesados: {frame_count}")
        print(f"Total de detecciones: {stats['total_detections']}")
        print(f"  • Frontales: {stats['frontal_count']} ({100*stats['frontal_count']/stats['total_detections']:.1f}%)")
        print(f"  • Reversa: {stats['reversa_count']} ({100*stats['reversa_count']/stats['total_detections']:.1f}%)")
        
        if stats['low_confidence_count'] > 0:
            print(f"\n⚠️  Detecciones con baja confianza (<70%): {stats['low_confidence_count']}")
            print(f"   ({100*stats['low_confidence_count']/stats['total_detections']:.1f}% del total)")
        
        print("=" * 60)


def main():
    print("\n" + "=" * 60)
    print("🚗 DETECTOR DE ORIENTACIÓN DE AUTOS")
    print("   Con tu modelo entrenado")
    print("=" * 60)
    
    # Inicializar sistema
    try:
        detector = YOLOCarDetectorWithOrientation(
            yolo_model='yolov8l.pt',
            classifier_model='car_orientation_model.pth'
        )
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\n💡 Asegúrate de que el archivo 'car_orientation_model.pth' existe")
        print("   Debes entrenar el modelo primero usando el script de etiquetado")
        return
    
    # Menú principal
    while True:
        print("\n" + "=" * 60)
        print("MENÚ PRINCIPAL")
        print("=" * 60)
        print("1. 📸 Procesar IMAGEN")
        print("2. 🎬 Procesar VIDEO")
        print("3. 📹 Usar WEBCAM")
        print("4. 🚪 SALIR")
        print("=" * 60)
        
        opcion = input("\nSelecciona (1-4): ").strip()
        
        if opcion == '1':
            print("\n📁 Selecciona una imagen...")
            ruta = select_file('image')
            if ruta:
                process_image(detector, ruta)
            else:
                print("❌ No se seleccionó archivo")
        
        elif opcion == '2':
            print("\n📁 Selecciona un video...")
            ruta = select_file('video')
            if ruta:
                process_video(detector, ruta)
            else:
                print("❌ No se seleccionó archivo")
        
        elif opcion == '3':
            process_video(detector, 'webcam')
        
        elif opcion == '4':
            print("\n👋 ¡Hasta luego!")
            break
        
        else:
            print("\n❌ Opción inválida")


if __name__ == "__main__":
    main()