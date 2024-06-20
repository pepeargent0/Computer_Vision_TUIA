import json
import os
import glob

# Directorios de entrada y salida
labelme_dir = 'datasets/cartas/labels/train'
yolo_dir = 'datasets/cartas/labels/train'
images_dir = 'datasets/cartas/images/train'

os.makedirs(yolo_dir, exist_ok=True)


# Función para convertir las anotaciones de Labelme a formato YOLO
def convert_labelme_to_yolo(json_file, img_dir, output_dir):
    print(f"Procesando archivo: {json_file}")

    with open(json_file) as f:
        data = json.load(f)

    img_height = data.get('imageHeight')
    img_width = data.get('imageWidth')

    if img_height is None or img_width is None:
        print(f"Error: no se encontraron 'imageHeight' o 'imageWidth' en {json_file}")
        return

    yolo_labels = []
    for shape in data['shapes']:
        label = shape['label']
        points = shape['points']

        # Calcular las coordenadas del cuadro delimitador en formato YOLO
        x_min = min(points[0][0], points[1][0])
        x_max = max(points[0][0], points[1][0])
        y_min = min(points[0][1], points[1][1])
        y_max = max(points[0][1], points[1][1])

        x_center = (x_min + x_max) / 2 / img_width
        y_center = (y_min + y_max) / 2 / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height

        yolo_labels.append(f"{label} {x_center} {y_center} {width} {height}")

    # Guardar las etiquetas YOLO en el directorio de salida
    base_name = os.path.basename(json_file).replace('.json', '.txt')
    yolo_file = os.path.join(output_dir, base_name)

    print(f"Guardando etiquetas YOLO en: {yolo_file}")
    with open(yolo_file, 'w') as f:
        for label in yolo_labels:
            f.write(f"{label}\n")


# Convertir todos los archivos JSON de Labelme en el directorio
json_files = glob.glob(os.path.join(labelme_dir, '*.json'))
print(f"Archivos JSON encontrados: {json_files}")

for json_file in json_files:
    convert_labelme_to_yolo(json_file, images_dir, yolo_dir)

print("Conversión completa.")
