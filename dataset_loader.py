from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
def load_data(dataset_path):
    train_path = os.path.join(dataset_path, "Training")
    val_path = os.path.join(dataset_path, "Testing")
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        zoom_range=0.1,
        horizontal_flip=True
    )
    val_datagen = ImageDataGenerator(
        rescale=1./255
    )
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    val_generator = val_datagen.flow_from_directory(
        val_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    return train_generator, val_generator
