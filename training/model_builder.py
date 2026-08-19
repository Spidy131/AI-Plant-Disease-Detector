import tensorflow as tf
from tensorflow.keras import layers, models
from config import IMAGE_SIZE, LEARNING_RATE


def build_model(num_classes):
    # Load EfficientNetB0 without the top classification layer
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=IMAGE_SIZE + (3,),
        include_top=False,
        weights="imagenet"
    )

    # Freeze pretrained layers for initial training
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model