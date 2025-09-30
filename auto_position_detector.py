import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import cv2
from ultralytics import YOLO
import numpy as np
from tkinter import Tk, filedialog
import os
import json
from pathlib import Path
import shutil


class CarOrientationLabeler:
    """
    Herramienta para etiquetar autos detectados como Frontal o Reversa
    """
    def __init__(self, yolo_model='yolov8l.pt', output_dir='dataset_etiquetado'):
        print("🚗 Inicializando sistema de etiquetado...")
        self.yolo = YOLO(yolo_model)
        self.car_classes = [2]  # Clase 'car' en COCO
        
        # Crear estructura de carpetas
        self.output_dir = Path(output_dir)
        self.frontal_dir = self.output_dir / 'frontal'
        self.reversa_dir = self.output_dir / 'reversa'
        self.metadata_file = self.output_dir / 'metadata.json'
        
        self.frontal_dir.mkdir(parents=True, exist_ok=True)
        self.reversa_dir.mkdir(parents=True, exist_ok=True)
        
        # Cargar o crear metadata
        self.metadata = self.load_metadata()
        
        print(f"✓ Sistema listo!")
        print(f"📁 Datos guardados en: {self.output_dir}")
        print(f"   - Frontales: {len(list(self.frontal_dir.glob('*.jpg')))} imágenes")
        print(f"   - Reversa: {len(list(self.reversa_dir.glob('*.jpg')))} imágenes")
    
    def load_metadata(self):
        """Cargar metadata existente"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {'labeled_count': 0, 'frontal_count': 0, 'reversa_count': 0}
    
    def save_metadata(self):
        """Guardar metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, indent=2, fp=f)
    
    def detect_cars(self, frame):
        """
        Detecta autos en un frame
        
        Returns:
            list: Lista de bounding boxes [(x1, y1, x2, y2, confidence), ...]
        """
        results = self.yolo(frame, conf=0.5)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if cls in self.car_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    detections.append((x1, y1, x2, y2, conf))
        
        return detections
    
    def label_cars_interactive(self, image_path):
        """
        Etiqueta autos de forma interactiva en una imagen
        
        Args:
            image_path: Ruta de la imagen
        """
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"❌ Error al cargar: {image_path}")
            return
        
        # Detectar autos
        detections = self.detect_cars(frame)
        
        if not detections:
            print(f"⚠ No se detectaron autos en: {os.path.basename(image_path)}")
            return
        
        print(f"\n📸 Procesando: {os.path.basename(image_path)}")
        print(f"🚗 Autos detectados: {len(detections)}")
        
        # Etiquetar cada auto
        for idx, (x1, y1, x2, y2, conf) in enumerate(detections):
            # Extraer ROI del auto
            car_roi = frame[y1:y2, x1:x2].copy()
            if car_roi.size == 0:
                continue
            
            # Crear imagen de visualización
            display_frame = frame.copy()
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.putText(display_frame, f"Auto {idx+1}/{len(detections)}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Mostrar instrucciones
            instructions = [
                "Presiona:",
                "F = Frontal",
                "R = Reversa", 
                "S = Saltar",
                "Q = Salir"
            ]
            
            y_offset = 30
            for instruction in instructions:
                cv2.putText(display_frame, instruction, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 25
            
            # Mostrar imagen
            cv2.imshow('Etiquetado de Autos', display_frame)
            
            # Esperar tecla
            while True:
                key = cv2.waitKey(0) & 0xFF
                
                if key == ord('f') or key == ord('F'):
                    # Guardar como frontal
                    self.save_car_image(car_roi, 'frontal')
                    print(f"  ✓ Auto {idx+1}: FRONTAL")
                    break
                
                elif key == ord('r') or key == ord('R'):
                    # Guardar como reversa
                    self.save_car_image(car_roi, 'reversa')
                    print(f"  ✓ Auto {idx+1}: REVERSA")
                    break
                
                elif key == ord('s') or key == ord('S'):
                    # Saltar este auto
                    print(f"  ⊘ Auto {idx+1}: SALTADO")
                    break
                
                elif key == ord('q') or key == ord('Q'):
                    # Salir completamente
                    cv2.destroyAllWindows()
                    print("\n🛑 Etiquetado cancelado por el usuario")
                    return False
        
        cv2.destroyAllWindows()
        return True
    
    def save_car_image(self, car_roi, label):
        """
        Guarda imagen de auto etiquetado
        
        Args:
            car_roi: Región del auto
            label: 'frontal' o 'reversa'
        """
        # Determinar carpeta destino
        if label == 'frontal':
            save_dir = self.frontal_dir
            self.metadata['frontal_count'] += 1
        else:
            save_dir = self.reversa_dir
            self.metadata['reversa_count'] += 1
        
        # Generar nombre único
        self.metadata['labeled_count'] += 1
        filename = f"car_{self.metadata['labeled_count']:05d}.jpg"
        filepath = save_dir / filename
        
        # Guardar imagen
        cv2.imwrite(str(filepath), car_roi)
        self.save_metadata()
    
    def process_images_from_folder(self, folder_path):
        """
        Procesa todas las imágenes de una carpeta
        
        Args:
            folder_path: Ruta de la carpeta con imágenes
        """
        folder = Path(folder_path)
        
        # Buscar todas las imágenes
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        images = []
        for ext in image_extensions:
            images.extend(folder.glob(f'*{ext}'))
            images.extend(folder.glob(f'*{ext.upper()}'))
        
        if not images:
            print("❌ No se encontraron imágenes en la carpeta")
            return
        
        print(f"\n📂 Encontradas {len(images)} imágenes")
        print("=" * 60)
        
        for idx, img_path in enumerate(images, 1):
            print(f"\n[{idx}/{len(images)}] ", end="")
            result = self.label_cars_interactive(img_path)
            if result == False:  # Usuario presionó Q
                break
        
        print("\n" + "=" * 60)
        print("✅ RESUMEN DEL ETIQUETADO:")
        print(f"   Total etiquetados: {self.metadata['labeled_count']}")
        print(f"   Frontales: {self.metadata['frontal_count']}")
        print(f"   Reversa: {self.metadata['reversa_count']}")
        print("=" * 60)
    
    def extract_frames_from_video(self, video_path, frame_interval=30):
        """
        Extrae frames de un video y los guarda como imágenes
        
        Args:
            video_path: Ruta del video
            frame_interval: Guardar 1 frame cada N frames (default: 30)
        
        Returns:
            str: Ruta de la carpeta con frames extraídos
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ No se pudo abrir el video: {video_path}")
            return None
        
        # Crear carpeta para frames
        video_name = Path(video_path).stem
        frames_dir = self.output_dir / f'frames_{video_name}'
        frames_dir.mkdir(exist_ok=True)
        
        frame_count = 0
        saved_count = 0
        
        print(f"\n🎬 Extrayendo frames de: {os.path.basename(video_path)}")
        print(f"📁 Guardando en: {frames_dir}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Guardar cada N frames
            if frame_count % frame_interval == 0:
                frame_filename = frames_dir / f'frame_{saved_count:05d}.jpg'
                cv2.imwrite(str(frame_filename), frame)
                saved_count += 1
            
            frame_count += 1
        
        cap.release()
        
        print(f"✅ Extraídos {saved_count} frames de {frame_count} totales")
        return str(frames_dir)


class CarOrientationTrainer:
    """
    Entrena un modelo para clasificar orientación de autos
    """
    def __init__(self, dataset_dir='dataset_etiquetado', model_path='car_orientation_model.pth'):
        self.dataset_dir = Path(dataset_dir)
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"💻 Usando dispositivo: {self.device}")
        
        # Transformaciones
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Crear modelo
        self.model = models.resnet50(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 2)
        self.model = self.model.to(self.device)
        
        self.classes = ['Frontal', 'Reversa']
    
    def prepare_datasets(self, train_split=0.8):
        """Prepara datasets de entrenamiento y validación"""
        from torchvision.datasets import ImageFolder
        
        # Cargar dataset completo
        full_dataset = ImageFolder(str(self.dataset_dir), transform=self.transform)
        
        # Dividir en train/val
        dataset_size = len(full_dataset)
        train_size = int(train_split * dataset_size)
        val_size = dataset_size - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
        
        return train_loader, val_loader, dataset_size
    
    def train(self, epochs=20, learning_rate=0.001):
        """Entrena el modelo"""
        print("\n🚀 INICIANDO ENTRENAMIENTO")
        print("=" * 60)
        
        # Preparar datos
        train_loader, val_loader, total_samples = self.prepare_datasets()
        print(f"📊 Total de muestras: {total_samples}")
        print(f"   Entrenamiento: {len(train_loader.dataset)}")
        print(f"   Validación: {len(val_loader.dataset)}")
        print("=" * 60)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # ENTRENAMIENTO
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_acc = 100 * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            # VALIDACIÓN
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_acc = 100 * val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)
            
            # Mostrar progreso
            print(f'Época {epoch+1}/{epochs}:')
            print(f'  Train - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}%')
            print(f'  Val   - Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%')
            
            # Guardar mejor modelo
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.model_path)
                print(f'  ✓ Mejor modelo guardado! (Acc: {val_acc:.2f}%)')
            print()
        
        print("=" * 60)
        print(f"🎉 ENTRENAMIENTO COMPLETADO!")
        print(f"🏆 Mejor precisión en validación: {best_val_acc:.2f}%")
        print(f"💾 Modelo guardado en: {self.model_path}")
        print("=" * 60)


def select_folder():
    """Abre diálogo para seleccionar carpeta"""
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    folder_path = filedialog.askdirectory(title="Selecciona una carpeta")
    root.destroy()
    return folder_path if folder_path else None


def select_file(title, filetypes):
    """Abre diálogo para seleccionar archivo"""
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return file_path if file_path else None


def main():
    print("=" * 60)
    print("🚗 SISTEMA DE ETIQUETADO Y ENTRENAMIENTO")
    print("   Orientación de Autos (Frontal/Reversa)")
    print("=" * 60)
    
    # Inicializar etiquetador
    labeler = CarOrientationLabeler()
    
    while True:
        print("\n" + "=" * 60)
        print("MENÚ PRINCIPAL")
        print("=" * 60)
        print("1. 📂 Etiquetar imágenes de una CARPETA")
        print("2. 🎬 Extraer frames de un VIDEO")
        print("3. 🧠 ENTRENAR modelo con datos etiquetados")
        print("4. 📊 Ver estadísticas del dataset")
        print("5. 🚪 SALIR")
        print("=" * 60)
        
        opcion = input("\nSelecciona (1-5): ").strip()
        
        if opcion == '1':
            print("\n📁 Selecciona la carpeta con imágenes...")
            folder = select_folder()
            if folder:
                labeler.process_images_from_folder(folder)
            else:
                print("❌ No se seleccionó carpeta")
        
        elif opcion == '2':
            print("\n🎬 Selecciona un video...")
            video = select_file(
                "Selecciona un video",
                [("Videos", "*.mp4 *.avi *.mov *.mkv"), ("Todos", "*.*")]
            )
            if video:
                interval = input("¿Cada cuántos frames extraer? (default=30): ").strip()
                interval = int(interval) if interval.isdigit() else 30
                
                frames_folder = labeler.extract_frames_from_video(video, interval)
                
                if frames_folder:
                    respuesta = input("\n¿Quieres etiquetar estos frames ahora? (s/n): ").strip().lower()
                    if respuesta == 's':
                        labeler.process_images_from_folder(frames_folder)
            else:
                print("❌ No se seleccionó video")
        
        elif opcion == '3':
            # Verificar que hay datos suficientes
            frontal_count = len(list(labeler.frontal_dir.glob('*.jpg')))
            reversa_count = len(list(labeler.reversa_dir.glob('*.jpg')))
            
            if frontal_count < 50 or reversa_count < 50:
                print(f"\n⚠ ADVERTENCIA: Datos insuficientes")
                print(f"   Se recomienda al menos 50 imágenes de cada clase")
                print(f"   Actual: Frontal={frontal_count}, Reversa={reversa_count}")
                respuesta = input("¿Continuar de todos modos? (s/n): ").strip().lower()
                if respuesta != 's':
                    continue
            
            epochs = input("\n¿Cuántas épocas de entrenamiento? (default=20): ").strip()
            epochs = int(epochs) if epochs.isdigit() else 20
            
            trainer = CarOrientationTrainer()
            trainer.train(epochs=epochs)
        
        elif opcion == '4':
            frontal_count = len(list(labeler.frontal_dir.glob('*.jpg')))
            reversa_count = len(list(labeler.reversa_dir.glob('*.jpg')))
            total = frontal_count + reversa_count
            
            print("\n" + "=" * 60)
            print("📊 ESTADÍSTICAS DEL DATASET")
            print("=" * 60)
            print(f"Total de imágenes etiquetadas: {total}")
            print(f"  • Frontales: {frontal_count} ({100*frontal_count/total if total > 0 else 0:.1f}%)")
            print(f"  • Reversa: {reversa_count} ({100*reversa_count/total if total > 0 else 0:.1f}%)")
            print(f"\n📁 Ubicación: {labeler.output_dir}")
            
            if total < 100:
                print(f"\n💡 TIP: Se recomienda al menos 100 imágenes totales")
                print(f"   Te faltan aproximadamente {100-total} imágenes más")
            elif frontal_count < 50 or reversa_count < 50:
                print(f"\n⚠ Las clases están desbalanceadas")
                print(f"   Intenta tener cantidades similares de cada clase")
            else:
                print(f"\n✅ ¡Dataset listo para entrenar!")
            print("=" * 60)
        
        elif opcion == '5':
            print("\n👋 ¡Hasta luego!")
            break
        
        else:
            print("\n❌ Opción inválida")


if __name__ == "__main__":
    main()