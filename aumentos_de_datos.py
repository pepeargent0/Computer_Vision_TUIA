import os
import cv2
import ssl
import urllib3

# Deshabilitar advertencias de SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

import albumentations as A
from glob import glob

# Mapeo de etiquetas de texto a índices numéricos
label_map = {
    "1B": 0, "2B": 1, "3B": 2, "4B": 3, "5B": 4, "6B": 5, "7B": 6, "8B": 7, "9B": 8, "10B": 9,
    "1C": 10, "2C": 11, "3C": 12, "4C": 13, "5C": 14, "6C": 15, "7C": 16, "8C": 17, "9C": 18, "10C": 19,
    "1O": 20, "2O": 21, "3O": 22, "4O": 23, "5O": 24, "6O": 25, "7O": 26, "8O": 27, "9O": 28, "10O": 29,
    "1E": 30, "2E": 31, "3E": 32, "4E": 33, "5E": 34, "6E": 35, "7E": 36, "8E": 37, "9E": 38, "10E": 39
}

# Directorios de entrada y salida
images_dir = 'datasets/cartas/images/train'
labels_dir = 'datasets/cartas/labels/train'

# Directorio para las imágenes y etiquetas aumentadas
aug_images_dir = 'datasets/cartas_augmented/images/train'
aug_labels_dir = 'datasets/cartas_augmented/labels/train'
os.makedirs(aug_images_dir, exist_ok=True)
os.makedirs(aug_labels_dir, exist_ok=True)

# Definir las transformaciones de aumento
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=45, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.RandomGamma(p=0.5),
    A.GaussianBlur(p=0.5),
    A.HueSaturationValue(p=0.5),
    A.RGBShift(p=0.5),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Obtener todas las imágenes y etiquetas
image_paths = glob(os.path.join(images_dir, '*.jpeg'))
label_paths = glob(os.path.join(labels_dir, '*.txt'))

# Función para leer etiquetas YOLO
def read_yolo_label(label_path):
    labels = []
    with open(label_path, 'r') as file:
        for line in file.readlines():
            parts = line.strip().split()
            class_label = parts[0]
            if class_label in label_map:
                class_id = label_map[class_label]
                x_center, y_center, width, height = map(float, parts[1:])
                labels.append([class_id, x_center, y_center, width, height])
            else:
                print(f"Etiqueta no reconocida: {class_label}")
    return labels

# Función para escribir etiquetas YOLO
def write_yolo_label(label_path, labels):
    with open(label_path, 'w') as file:
        for label in labels:
            file.write(" ".join(map(str, label)) + '\n')

# Aumentar las imágenes y etiquetas
for img_path, lbl_path in zip(image_paths, label_paths):
    image = cv2.imread(img_path)
    labels = read_yolo_label(lbl_path)

    for i in range(25):  # Generar 15 imágenes aumentadas por cada imagen original
        bboxes = [label[1:] for label in labels]
        class_labels = [int(label[0]) for label in labels]

        augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        aug_image = augmented['image']
        aug_bboxes = augmented['bboxes']

        # Guardar la imagen aumentada
        base_name = os.path.basename(img_path).split('.')[0]
        aug_img_path = os.path.join(aug_images_dir, f"{base_name}_aug_{i}.jpeg")
        cv2.imwrite(aug_img_path, aug_image)

        # Guardar las etiquetas aumentadas
        aug_labels = [[class_labels[j]] + list(aug_bboxes[j]) for j in range(len(aug_bboxes))]
        aug_lbl_path = os.path.join(aug_labels_dir, f"{base_name}_aug_{i}.txt")
        write_yolo_label(aug_lbl_path, aug_labels)

print("Aumento de datos completado.")
