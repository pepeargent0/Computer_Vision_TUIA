import os

def verify_paths(paths):
    for path in paths:
        if not os.path.exists(path):
            print(f"Path does not exist: {path}")
        else:
            print(f"Path exists: {path}")

train_images = 'datasets/cartas/images/train'
val_images = 'datasets/cartas/images/val'
train_labels = 'datasets/cartas/labels/train'
val_labels = 'datasets/cartas/labels/val'

verify_paths([train_images, val_images, train_labels, val_labels])
