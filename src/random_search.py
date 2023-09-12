import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from kerastuner import RandomSearch

def load_dataset(base_path, image_size=(320, 240), batch_size=32, validation_split=0.2, seed=1):
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        base_path,
        image_size=image_size,
        color_mode='grayscale',
        batch_size=batch_size,
        label_mode='categorical',
        subset='training',
        validation_split=validation_split,
        shuffle=True,
        seed=seed
    )

    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        base_path,
        image_size=image_size,
        color_mode='grayscale',
        batch_size=batch_size,
        label_mode='categorical',
        subset='validation',
        validation_split=validation_split,
        shuffle=True,
        seed=seed
    )
    
    return train_dataset, validation_dataset


def build_model(hp, num_classes):
    inputs = keras.Input((320, 240, 1))
    x = inputs

    for i in range(2):
        x = keras.layers.Conv2D(
            filters=hp.Int(f"conv_{i+1}_filter", min_value=16, max_value=96, step=16),
            kernel_size=hp.Choice(f"conv_{i+1}_kernel", values=[3, 5]),
            padding="same",
            kernel_regularizer=tf.keras.regularizers.l1(hp.Choice(f"conv_{i+1}_l1", values=[0.01, 0.001])),
        )(x)
        x = keras.layers.Activation('relu')(x)
        if i == 0:
            x = keras.layers.MaxPooling2D(3, strides=3, padding="same")(x)

    x = keras.layers.GlobalMaxPooling2D()(x)
    outputs = keras.layers.Dense(num_classes, activation='sigmoid')(x)

    net = keras.Model(inputs, outputs)
    net.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.RMSprop(learning_rate=hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])),
        metrics=['accuracy']
    )

    return net


if __name__ == "__main__":
    INPUT_PATH = 'input_directory_path_here'
    train_dataset, validation_dataset = load_dataset(INPUT_PATH)

    num_classes = len(train_dataset.class_names)

    tuner = RandomSearch(
        lambda hp: build_model(hp, num_classes=num_classes),
        objective='val_accuracy',
        max_trials=5,
        executions_per_trial=3,
        directory='output',
        project_name='CNN_RandomSearch'
    )

    tuner.search_space_summary()

    tuner.search(train_dataset, epochs=10, validation_data=validation_dataset)
    
    tuner.results_summary()

    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.fit(train_dataset.concatenate(validation_dataset), epochs=10)