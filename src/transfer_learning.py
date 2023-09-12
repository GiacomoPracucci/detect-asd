import os
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Flatten, Dense, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class ImageLoader:
    def __init__(self, train_dir):
        self.train_dir = train_dir
        self.train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    def get_generators(self, target_size, batch_size):
        train_generator = self.train_datagen.flow_from_directory(
            self.train_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='binary'
        )

        validation_generator = self.train_datagen.flow_from_directory(
            self.train_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='binary',
            subset='validation',
            shuffle=False
        )
        return train_generator, validation_generator


class TransferLearningVGG:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = self.create_model()

    def create_model(self):
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape)
        for layer in base_model.layers:
            layer.trainable = False

        x = Flatten()(base_model.output)
        x = Dense(16, activation=LeakyReLU(alpha=0.2))(x)
        x = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=base_model.input, outputs=x)
        optimizer = Adam(learning_rate=0.0001)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model

    def train_model(self, train_generator, validation_generator, epochs):
        history = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.n // train_generator.batch_size,
            validation_data=validation_generator,
            epochs=epochs
        )
        return history
    
if __name__ == "__main__":
    # Definizione degli iperparametri
    EPOCHS = 50
    BATCH_SIZE = 50
    TARGET_SIZE = (640, 480) # Dimensions of input images
    TRAIN_DIR = 'drive/MyDrive/Images2/' # Path to the folder containing the images

    loader = ImageLoader(TRAIN_DIR)
    train_generator, validation_generator = loader.get_generators(TARGET_SIZE, BATCH_SIZE)

    vgg_model = TransferLearningVGG(input_shape=TARGET_SIZE + (3,))
    history = vgg_model.train_model(train_generator, validation_generator, EPOCHS)

    # Visualizzazione risultati
    score = vgg_model.model.evaluate(train_generator, verbose=0)
    print('Test accuracy:', score[1])
    print("Numero totale di immagini utilizzate:", train_generator.n)

    # Plot della loss
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot dell'accuracy
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()