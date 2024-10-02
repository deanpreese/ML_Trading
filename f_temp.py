import keras
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt


print(f"Keras   {keras.__version__}")
print("TensorFlow version:", tf.__version__)
print("TensorRT version:", trt.trt_utils._pywrap_py_utils.get_linked_tensorrt_version())