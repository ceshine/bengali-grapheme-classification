import tensorflow as tf


def prepare_tpu(tpu=None, zone=None, project=None):
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu=tpu, zone=zone, project=project
        )  # TPU detection
        print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
    except ValueError:
        tpu = None
    strategy = tf.distribute.get_strategy()
    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    return strategy, tpu
