from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model("cats_vs_dogs.keras")

# source images
imgSrc = "dog6.jpg" 

# input configuration
gambar = cv2.imread(imgSrc)
gambar = cv2.resize(gambar, (224,224))
gambar = np.expand_dims(gambar, axis=0)

# output configuration
hasil = model.predict(gambar)


if hasil[0][0] > 0.5:
    print("Dog 🐶")
else:
    print("Cat 🐱")

print(hasil)
