import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt


base_dir = 'dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')


train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# Dense model
model_dense = models.Sequential([
    layers.Flatten(input_shape=(150, 150, 3)),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model_dense.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history_dense = model_dense.fit(train_generator, epochs=20, validation_data=val_generator, callbacks=[early_stopping])
model_dense.save('dense_model_2.h5')

# CNN model
model_cnn = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_cnn = model_cnn.fit(train_generator, epochs=20, validation_data=val_generator, callbacks=[early_stopping])
model_cnn.save('cnn_model_2.h5')


def plot_training_history(history, title):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'o-', label='Training accuracy', color='green')
    plt.plot(epochs, val_acc, 'o-', label='Validation accuracy', color='purple')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'o-', label='Training loss', color='green')
    plt.plot(epochs, val_loss, 'o-', label='Validation loss', color='purple')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


plot_training_history(history_dense, 'Dense model')
plot_training_history(history_cnn, 'CNN model')


loss_dense, acc_dense = model_dense.evaluate(test_generator)
print(f'Dense model accuracy on test set: {acc_dense}')
loss_cnn, acc_cnn = model_cnn.evaluate(test_generator)
print(f'CNN model accuracy on test set: {acc_cnn}')


def visualize_predictions(model, generator):
    class_labels = list(generator.class_indices.keys())
    x_batch, y_batch = next(generator)

    predictions = model.predict(x_batch)
    predicted_labels = (predictions > 0.5).astype("int32")

    plt.figure(figsize=(12, 12))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(x_batch[i])

        true_label = int(y_batch[i])
        predicted_label = int(predicted_labels[i])
        color = "green" if true_label == predicted_label else "red"

        plt.title(f'True: {class_labels[true_label]}, Pred: {class_labels[predicted_label]}', color=color)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# model_dense=tf.keras.models.load_model('dense_model_2.h5')
# model_cnn=tf.keras.models.load_model('cnn_model_2.h5')

print("Predictions for Dense Model:")
visualize_predictions(model_dense, test_generator)
print("Predictions for CNN Model:")
visualize_predictions(model_cnn, test_generator)
