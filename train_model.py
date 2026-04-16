from dataset_loader import load_data
from model import build_model
import matplotlib.pyplot as plt
import os
DATASET_PATH = "../dataset/" 
train_gen, val_gen = load_data(DATASET_PATH)
model = build_model()
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)
os.makedirs("../models", exist_ok=True)
os.makedirs("../outputs", exist_ok=True)
model.save("../models/brain_tumor_model.h5")
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])
plt.savefig('../outputs/accuracy.png')
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])
plt.savefig('../outputs/loss.png')
print("Training complete. Model and graphs saved.")
