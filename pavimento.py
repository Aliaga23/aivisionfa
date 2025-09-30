from ultralytics import YOLO
import torch
from pathlib import Path
import yaml

def setup_roboflow_dataset():
    """
    Configurar dataset de Roboflow
    Tu estructura debe ser:
    condo2/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    ├── test/
    │   ├── images/
    │   └── labels/
    └── data.yaml
    """
    
    # Verificar estructura
    base_path = Path('condo5')
    
    required_dirs = [
        base_path / 'train' / 'images',
        base_path / 'train' / 'labels',
        base_path / 'valid' / 'images',
        base_path / 'valid' / 'labels'
    ]
    
    for dir_path in required_dirs:
        if not dir_path.exists():
            print(f"ERROR: No existe {dir_path}")
            return False
    
    # Verificar data.yaml
    yaml_path = base_path / 'data.yaml'
    if not yaml_path.exists():
        print("ERROR: No existe data.yaml")
        return False
    
    # Leer y mostrar configuración
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\nConfiguración del dataset:")
    print(f"  Clases: {config.get('names', [])}")
    print(f"  Número de clases: {config.get('nc', 0)}")
    
    # Contar imágenes
    train_images = list((base_path / 'train' / 'images').glob('*.*'))
    valid_images = list((base_path / 'valid' / 'images').glob('*.*'))
    
    print(f"\nImágenes de entrenamiento: {len(train_images)}")
    print(f"Imágenes de validación: {len(valid_images)}")
    
    if len(train_images) == 0:
        print("\nERROR: No hay imágenes de entrenamiento")
        return False
    
    return True


def train_parking_zone_detector(
    data_yaml='condo5/data.yaml',
    model_size='yolov8n-seg.pt',  # Cambiar a segmentación
    epochs=100,
    imgsz=640,
    batch=16,
    project_name='parking_zone_training',
    use_segmentation=True  # Nueva opción
):
    """
    Entrenar YOLOv8 para detectar zonas de parqueo
    
    Args:
        data_yaml: Ruta al archivo data.yaml de Roboflow
        model_size: Tamaño del modelo 
        epochs: Número de épocas de entrenamiento
        imgsz: Tamaño de imagen
        batch: Tamaño del batch
        project_name: Nombre del proyecto
        use_segmentation: True para segmentación, False para detection
    """
    
    print("\n" + "="*60)
    print("ENTRENAMIENTO DE YOLOV8 PARA ZONAS DE PARQUEO")
    print("="*60)
    
    # Verificar CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDispositivo: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Ajustar nombre del modelo según tipo
    if use_segmentation and not model_size.endswith('-seg.pt'):
        model_size = model_size.replace('.pt', '-seg.pt')
    
    # Cargar modelo pre-entrenado
    print(f"\nCargando modelo: {model_size}")
    print(f"Tipo: {'Segmentación' if use_segmentation else 'Detection'}")
    model = YOLO(model_size)
    
    # Iniciar entrenamiento
    print(f"\nIniciando entrenamiento...")
    print(f"  Épocas: {epochs}")
    print(f"  Batch size: {batch}")
    print(f"  Tamaño de imagen: {imgsz}")
    print("="*60)
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project_name,
        name='parking_zones',
        patience=50,
        save=True,
        plots=True,
        
        # Augmentaciones
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        
        # Optimización
        optimizer='auto',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # Otros
        amp=True,
        verbose=True
    )
    
    print("\n" + "="*60)
    print("ENTRENAMIENTO COMPLETADO")
    print("="*60)
    print(f"\nModelo guardado en: {project_name}/parking_zones/weights/")
    print(f"  - best.pt (mejor modelo)")
    print(f"  - last.pt (último epoch)")
    
    return results


def test_trained_model(model_path, test_images_dir='condo5/valid/images'):
    """
    Probar el modelo entrenado
    
    Args:
        model_path: Ruta al modelo entrenado (best.pt)
        test_images_dir: Carpeta con imágenes de prueba
    """
    print("\n" + "="*60)
    print("PROBANDO MODELO ENTRENADO")
    print("="*60)
    
    # Cargar modelo
    model = YOLO(model_path)
    
    # Buscar imágenes
    test_dir = Path(test_images_dir)
    images = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png'))
    
    if not images:
        print(f"\nNo se encontraron imágenes en {test_dir}")
        return
    
    print(f"\nProbando con {len(images)} imágenes...")
    
    # Predecir
    results = model.predict(
        source=str(test_dir),
        save=True,
        project='parking_zone_test',
        name='predictions',
        conf=0.25,
        iou=0.45,
        show_labels=True,
        show_conf=True
    )
    
    print(f"\nResultados guardados en: parking_zone_test/predictions/")
    
    # Mostrar estadísticas
    total_detections = sum(len(r.boxes) for r in results)
    print(f"\nTotal de zonas detectadas: {total_detections}")
    print(f"Promedio por imagen: {total_detections/len(results):.1f}")


def export_model(model_path, export_format='onnx'):
    """
    Exportar modelo a otros formatos
    
    Args:
        model_path: Ruta al modelo (best.pt)
        export_format: Formato (onnx, torchscript, coreml, tflite, etc)
    """
    print(f"\nExportando modelo a formato {export_format}...")
    
    model = YOLO(model_path)
    model.export(format=export_format)
    
    print(f"Modelo exportado exitosamente")


def main():
    print("\nSISTEMA DE ENTRENAMIENTO - ZONAS DE PARQUEO")
    
    while True:
        print("\n" + "="*60)
        print("MENU")
        print("="*60)
        print("1. Verificar dataset")
        print("2. Entrenar modelo (rápido - yolov8n)")
        print("3. Entrenar modelo (preciso - yolov8l)")
        print("4. Entrenar modelo (personalizado)")
        print("5. Probar modelo entrenado")
        print("6. Exportar modelo")
        print("7. Salir")
        print("="*60)
        
        op = input("\nOpcion (1-7): ").strip()
        
        if op == '1':
            print("\nVerificando dataset...")
            if setup_roboflow_dataset():
                print("\n✓ Dataset configurado correctamente")
            else:
                print("\n✗ Hay problemas con el dataset")
        
        elif op == '2':
            print("\nEntrenamiento RÁPIDO con SEGMENTACIÓN (yolov8n-seg)")
            if setup_roboflow_dataset():
                train_parking_zone_detector(
                    model_size='yolov8n-seg.pt',
                    epochs=100,
                    batch=16,
                    use_segmentation=True
                )
        
        elif op == '3':
            print("\nEntrenamiento PRECISO con SEGMENTACIÓN (yolov8l-seg)")
            if setup_roboflow_dataset():
                train_parking_zone_detector(
                    model_size='yolov8l-seg.pt',
                    epochs=150,
                    batch=8,
                    use_segmentation=True
                )
        
        elif op == '4':
            print("\nEntrenamiento PERSONALIZADO")
            
            model_size = input("Tamaño (n/s/m/l/x) [default: n]: ").strip().lower() or 'n'
            epochs = input("Épocas [default: 100]: ").strip()
            batch = input("Batch size [default: 16]: ").strip()
            
            epochs = int(epochs) if epochs.isdigit() else 100
            batch = int(batch) if batch.isdigit() else 16
            
            if setup_roboflow_dataset():
                train_parking_zone_detector(
                    model_size=f'yolov8{model_size}.pt',
                    epochs=epochs,
                    batch=batch
                )
        
        elif op == '5':
            model_path = input("\nRuta al modelo (ej: parking_zone_training/parking_zones/weights/best.pt): ").strip()
            if Path(model_path).exists():
                test_trained_model(model_path)
            else:
                print(f"ERROR: No existe {model_path}")
        
        elif op == '6':
            model_path = input("\nRuta al modelo: ").strip()
            if Path(model_path).exists():
                formato = input("Formato (onnx/torchscript/tflite) [default: onnx]: ").strip() or 'onnx'
                export_model(model_path, formato)
            else:
                print(f"ERROR: No existe {model_path}")
        
        elif op == '7':
            print("\nSaliendo...")
            break


if __name__ == "__main__":
    main()