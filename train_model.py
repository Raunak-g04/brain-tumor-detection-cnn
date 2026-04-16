from dataset_loader import load_data
from model import build_model
DATASET_PATH = "dataset/"
train_gen, val_gen = load_data(DATASET_PATH)
model = build_model()
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)
model.save("brain_tumor_model.h5")
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train','Validation'])
plt.savefig('outputs/accuracy.png')
    return model
