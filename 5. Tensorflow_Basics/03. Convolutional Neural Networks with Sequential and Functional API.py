import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10

# physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape)

x_train = x_train.astype("float32") / 255.0  # normalization
x_test = x_test.astype("float32") / 255.0

model = keras.Sequential(
    [
        keras.Input(shape=(32, 32, 3)), # Height / Width / Channel
        layers.Conv2D(filters=32, kernel_size=3, padding="valid", activation="relu"),
        # padding -> "same"은 입력 가로/세로 그대로 유지 & "valid"는 kernel size에 맞춰서
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"),
        layers.Flatten(),
        layers.Dense(units=64, activation="relu"),
        layers.Dense(10),
    ]
)

print(model.summary())

def my_model_origin():
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, 3)(inputs)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)
    x = layers.Conv2D(64, 3)(x)
    x = layers.BatchNormalization()(x)  # after Conv, before Activation
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3)(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Flatten()(x)  # Dense Layer에 넘기기 위해서 Flatten 과정이 필요
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(10)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def my_model():

    # TODO : Modeling from simple to complex
    # inputs = tf.keras.Input(shape=self._config.img_shape)

    inputs = keras.Input(shape=(128, 128, 1))
    x = layers.Flatten()(inputs)
    x = layers.Dense(1024, activation="relu", name="0_layer")(x)
    x = layers.Dense(512, activation="relu", name="first_layer")(x)
    x = layers.Dense(256, activation="relu", name="second_layer")(x)
    x = layers.Dense(128, activation="relu", name="third_layer")(x)
    output_c = layers.Dense(19)(x)
    output_j = layers.Dense(21)(x)
    output_z = layers.Dense(28)(x)
    outputs = [output_c, output_j, output_z]

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


model = my_model()
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # output에 softmax 연산이 없어도 괜찮음
    optimizer=keras.optimizers.Adam(lr=3e-4),
    metrics=["accuracy"],
)

model.fit(x_train, y_train, batch_size=1024, epochs=10, verbose=2)
model.evaluate(x_test, y_test, batch_size=1024, verbose=2)