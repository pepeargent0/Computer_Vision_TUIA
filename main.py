from ultralytics import YOLO
from PIL import Image

# Definir el tamaño de imagen esperado por el modelo YOLOv8
EXPECTED_SIZE = (640, 640)

# Cargar el modelo YOLOv8 preentrenado
model = YOLO('yolov8s.pt')

# Cargar una imagen
img_path = '1.jpeg'  # Cambia esto a la ruta de tu imagen
img = Image.open(img_path)

# Verificar el tamaño de la imagen y redimensionar si es necesario
if img.size != EXPECTED_SIZE:
    print(f"Redimensionando la imagen de {img.size} a {EXPECTED_SIZE}")
    img = img.resize(EXPECTED_SIZE)

# Realizar la detección
results = model.predict(img, save=True, save_txt=True)

# Mostrar los resultados
for result in results:
    result.show()

# Guardar los resultados en el directorio 'runs/detect'
# results.save()  # Este método no es necesario porque 'save=True' en predict ya guarda los resultados
