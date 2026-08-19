import tensorflow as tf

from config import (
    DATASET_PATH,
    IMAGE_SIZE,
    BATCH_SIZE,
    VALIDATION_SPLIT,
    SEED
)


def load_datasets():

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )

    class_names = train_ds.class_names

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names