import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignore information for Tensorflow

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)  # (60000, 28, 28) -> 따로 채널수에 대한 명시는 없네? (어차피 Dense로 넘기긴 할거지만,,)

x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0  # flatten + Normalize 하기 위한 과정
x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0
print(x_train.shape)  # (60000, 784) -> Dense Layer에 넘겨주기 위해서 flatten을 진행한 결과

# 전체적인 process: model 정의 -> Compile -> fit

# 1. Sequential API (Very convenient, not very flexible)
model = keras.Sequential(
    [
        keras.Input(shape=(28 * 28)),  # input size를 정해줘야한다.
        layers.Dense(512, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(10),
    ]
)
print(model.summary())  # model의 전체구조 한번 보여주는 역할 -> Common Debugging Tool로 활용 가능

model = keras.Sequential()
model.add(keras.Input(shape=(784)))
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dense(256, activation="relu", name="my_layer"))
model.add(layers.Dense(10))

# 1-1. Extracting output in middle layers -> Debugging 할때도 사용할 수 있겠네
model = keras.Model(inputs=model.inputs,
                    # outputs=[model.layers[-2].output]) # layers[-1]은 마지막 layer의 출력결과물
                    outputs=[layer.output for layer in model.layers])

# feature = model.predict(x_train)
# print(feature.shape)

features = model.predict(x_train)
for feature in features:
    print(feature.shape)

# 2. Functional API (A bit more flexible) -> if you can't use Sequential API or 간단하고 직관적인 연산인 경우
inputs = keras.Input(shape=(784))
x = layers.Dense(512, activation="relu", name="first_layer")(inputs)
x = layers.Dense(256, activation="relu", name="second_layer")(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(  # model training configuration을 정의한다고 생각
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # y label이 just integer로 구성된 경우 / sparse가 없는 경우에는 one-hot encoding 필요
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"],
)

model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)  # epoch별로 정보 print
model.evaluate(x_test, y_test, batch_size=32, verbose=2)