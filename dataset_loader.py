from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (160,160)
BATCH_SIZE = 32

def load_data(dataset_path):

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.1
    )

    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.1
    )

    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )

    val_generator = val_datagen.flow_from_directory(
        dataset_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )

    return train_generator, val_generator