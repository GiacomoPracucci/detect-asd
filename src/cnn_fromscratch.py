from random_search import load_dataset
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

def build_cnn_from_scratch(num_classes):
    inputs = keras.Input((224,224,3))
    x = inputs

    x = keras.layers.Conv2D(32, 3, padding="same", kernel_regularizer=tf.keras.regularizers.l1(0.001))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling2D(3, strides=3, padding="same")(x)

    x = keras.layers.Conv2D(48, 5, padding='same', kernel_regularizer=tf.keras.regularizers.l1(0.001))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.GlobalMaxPooling2D()(x)
    outputs = keras.layers.Dense(num_classes, activation='sigmoid')(x)

    net = keras.Model(inputs, outputs)
    net.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    
    return net

def plot_training_history(history):
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylim([0.0, 2.5])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['train', 'valid'])

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.ylim([0.3, 1.0])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['train', 'valid'])

    # Enlarge the figure
    plt.gcf().set_size_inches(10, 5)
    plt.show()

if __name__ == "__main__":
    INPUT_PATH = 'input_directory_path_here'
    train_dataset, validation_dataset = load_dataset(INPUT_PATH, image_size=(224, 224))

    num_classes = len(train_dataset.class_names)
    net = build_cnn_from_scratch(num_classes)
    
    history = net.fit(train_dataset, epochs=20, validation_data=validation_dataset)
    plot_training_history(history)