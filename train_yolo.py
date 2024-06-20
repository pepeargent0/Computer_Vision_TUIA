from ultralytics import YOLO

# Cargar el modelo YOLOv8 preentrenado
model = YOLO('yolov8s.pt')

# Definir el tamaño de imagen esperado por el modelo YOLOv8
EXPECTED_SIZE = (640, 640)

# Configuración del entrenamiento
data_yaml = 'data.yaml'  # Asegúrate de que este archivo esté en tu directorio de trabajo

# Entrenar el modelo
model.train(
    data=data_yaml,
    imgsz=EXPECTED_SIZE[0],
    epochs=100,
    batch=16,
    name='yolo_cartas'
)

# Guardar el modelo entrenado
model.save('yolov8s_cartas.pt')
