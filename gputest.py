import tensorflow as tf

# Check if CUDA is working
print("Built with CUDA:", tf.test.is_built_with_cuda())

# List all available physical GPUs detected by the system
gpus = tf.config.list_physical_devices('GPU')
print("Available GPUs:", gpus)