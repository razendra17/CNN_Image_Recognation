from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.callbacks import EarlyStopping


train_dataset = image_dataset_from_directory(
    "dataset/train",
    image_size=(224,224),
    batch_size=32,
    shuffle=True
)
print(train_dataset.class_names)
validation_dataset = image_dataset_from_directory(
    "dataset/validation",
    image_size=(224,224),
    batch_size=32,
    shuffle=False
)

# test_dataset = image_dataset_from_directory(
#     "dataset/test",
#     image_size=(224,224),
#     batch_size=32,
#     shuffle=False
# )

print(train_dataset.class_names)

model = Sequential()

model.add(
    Rescaling(1./255)
)

model.add(
    Conv2D(
        filters= 32,
        kernel_size=(3,3),
        padding="same",
        activation="relu",
        input_shape = (224,224,3)
    )
)

model.add(
    MaxPooling2D(
        pool_size=(2,2)
    )
)

model.add(
    Conv2D(
        filters=64,
        kernel_size=(3,3),
        padding="same",
        activation="relu",
    )
)

model.add(
    MaxPooling2D((2,2))
)

model.add(
    Flatten()
)

model.add(
    Dense(
        128,
        activation="relu"
    )
)

model.add(
    Dense(
        1,
        activation="sigmoid"
    )
)

model.compile(
    optimizer = "adam",
    loss = "binary_crossentropy",
    metrics = ["accuracy"]
)
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10
)

history.history

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.show()

model.save("cats_vs_dogs.keras")
model.summary()

