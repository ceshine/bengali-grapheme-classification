import tensorflow as tf
import efficientnet.tfkeras as efn

from .dataset import CLASS_COUNTS

# Avoids cudnn initialization problem
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class SELayer(tf.keras.layers.Layer):
    def __init__(self, channels, reduction):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(
            channels // reduction,
            kernel_initializer=tf.keras.initializers.he_normal(seed=None),
            name="fc1",
            activation="relu"
        )
        self.fc2 = tf.keras.layers.Dense(
            channels,
            kernel_initializer=tf.keras.initializers.he_normal(seed=None),
            name="fc2",
            activation="sigmoid"
        )

    def call(self, x):
        tmp = self.fc1(x)
        tmp = self.fc2(tmp)
        return tmp * x


def get_model(arch="b3", pretrained="imagenet", image_size=(128, 128, 3)):
    image_input = tf.keras.layers.Input(
        shape=image_size, dtype='float32', name='image_input'
    )
    if arch.startswith("b2"):
        base_model = efn.EfficientNetB2(
            weights=pretrained, input_shape=image_size, include_top=False)
    elif arch.startswith("b3"):
        base_model = efn.EfficientNetB3(
            weights=pretrained, input_shape=image_size, include_top=False)
    elif arch.startswith("b4"):
        base_model = efn.EfficientNetB4(
            weights=pretrained, input_shape=image_size, include_top=False)
    elif arch.startswith("b5"):
        base_model = efn.EfficientNetB5(
            weights=pretrained, input_shape=image_size, include_top=False)
    elif arch.startswith("b6"):
        base_model = efn.EfficientNetB6(
            weights=pretrained, input_shape=image_size, include_top=False)
    elif arch.startswith("b7"):
        base_model = efn.EfficientNetB7(
            weights=pretrained, input_shape=image_size, include_top=False)
    else:
        raise ValueError("Unknown arch!")
    base_model.trainable = True
    tmp = base_model(image_input)
    hidden_dim = base_model.output_shape[-1]
    tmp = tf.keras.layers.GlobalAveragePooling2D()(tmp)
    tmp = tf.keras.layers.Dropout(0.5)(tmp)
    if arch.endswith("g"):
        prediction_0 = tf.keras.layers.Dense(
            CLASS_COUNTS[0], activation='softmax', name="root", dtype='float32'
        )(SELayer(hidden_dim, 8)(tmp))
        prediction_1 = tf.keras.layers.Dense(
            CLASS_COUNTS[1], activation='softmax', name="vowel", dtype='float32'
        )(SELayer(hidden_dim, 8)(tmp))
        prediction_2 = tf.keras.layers.Dense(
            CLASS_COUNTS[2], activation='softmax', name="consonant", dtype='float32'
        )(SELayer(hidden_dim, 8)(tmp))
    else:
        prediction_0 = tf.keras.layers.Dense(
            CLASS_COUNTS[0], activation='softmax', name="root", dtype='float32')(tmp)
        prediction_1 = tf.keras.layers.Dense(
            CLASS_COUNTS[1], activation='softmax', name="vowel", dtype='float32')(tmp)
        prediction_2 = tf.keras.layers.Dense(
            CLASS_COUNTS[2], activation='softmax', name="consonant", dtype='float32')(tmp)
    prediction = tf.keras.layers.Concatenate(axis=-1)([
        prediction_0, prediction_1, prediction_2])
    return tf.keras.Model(image_input, prediction)
