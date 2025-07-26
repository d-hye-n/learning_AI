# import tensorflow as tf
#
# 키 = [170, 180, 175, 160]
# 신발 = [260, 270, 265, 255]
#
# a = tf.Variable(0.1)
# b = tf.Variable(0.2)
#
# def 손실함수():
#     return tf.square(신발 - (키 * a + b))
#
# opt = tf.keras.optimizers.Adam(learning_rate=0.1)
#
# for i in range(1000):
#     opt.minimize(손실함수, var_list=[a,b])
#     print(a.numpy(),b.numpy())

import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("Build info:", tf.sysconfig.get_build_info())  # 빌드에 포함된 CUDA/cuDNN 버전 확인
print("Physical GPUs:", tf.config.list_physical_devices("GPU"))

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
