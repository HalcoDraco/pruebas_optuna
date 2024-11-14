import tensorflow as tf

# Check number of threads available
print("Threads inter: ", tf.config.threading.get_inter_op_parallelism_threads())
print("Threads intra: ", tf.config.threading.get_intra_op_parallelism_threads())
