import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # warning 안뜨게 만들어주는 역할

import tensorflow as tf

# Initialization of Tensors
x = tf.constant(4)  # tf.Tensor(4, shape=(), dtype=int32)
x = tf.constant(4, shape=(1,1), dtype=tf.float32)  # tf.Tensor([[4.]], shape=(1, 1), dtype=float32)
x = tf.constant([[1,2,3], [4,5,6]])  # tf.Tensor([[1 2 3]
                                     #           [4 5 6]], shape=(2, 3), dtype=int32)
x = tf.ones((3,3))
x = tf.zeros((3,3))
x = tf.eye(3)  # Identity Matrix
x = tf.random.normal((3,3,4), mean=0, stddev=1)  # 평균과 표준편차를 적용한 normal distribution으로 생성
x = tf.random.uniform((3,3,4), minval=0, maxval=1)  # uniform distribution으로 생성
x = tf.range(start=1, limit=10, delta=2)  # delta는 step을 의미한다.

x = tf.cast(x, dtype=tf.float64)  # 주어진 data의 type casting을 이용하는 방법


# Mathematical Operations
x = tf.constant([1,2,3])
y = tf.constant([9,8,7])

z = tf.add(x, y)  # tf.Tensor([10 10 10], shape=(3,), dtype=int32)
z = x + y

z = tf.subtract(x, y)  # tf.Tensor([-8 -6 -4], shape=(3,), dtype=int32)
z = x - y

z = tf.divide(x, y)
z = x / y

z = tf.multiply(x, y)
z = x * y

z = x ** 5

z = tf.tensordot(x, y, axes=1)  # dot production -> tf.Tensor(46, shape=(), dtype=int32)
z = tf.reduce_sum(x * y, axis=0)

x = tf.random.normal((2,3))
y = tf.random.normal((3,4))
z = tf.matmul(x, y)
z = x @ y


# Indexing
x = tf.constant([0, 1, 1, 2, 3, 1, 2, 3])
print(x[:])  # tf.Tensor([0 1 1 2 3 1 2 3], shape=(8,), dtype=int32)
print(x[1:])  # tf.Tensor([1 1 2 3 1 2 3], shape=(7,), dtype=int32)
print(x[1:3])  # tf.Tensor([1 1], shape=(2,), dtype=int32)
print(x[::2])  # tf.Tensor([0 1 3 2], shape=(4,), dtype=int32)
print(x[::-1])  # tf.Tensor([3 2 1 3 2 1 1 0], shape=(8,), dtype=int32)

indices = tf.constant([0,3])
x_ind = tf.gather(x, indices)
print(x_ind)  # tf.Tensor([0 2], shape=(2,), dtype=int32)


# Reshaping
x = tf.range(9)

x = tf.reshape(x, (3,3))
x = tf.transpose(x, perm=[1, 0])  # swap axes