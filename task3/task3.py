import tensorflow as tf
from tensorflow.keras.applications import VGG19, ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.python.keras.callbacks import EarlyStopping

# Specifying paths to the dataset
base_dir = 'dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Preparation of image generators for training, validation and testing
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
# def build_vgg19_model():
#     base_model = VGG19(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
#
#     # Freezing the base layers
#     for layer in base_model.layers:
#         layer.trainable = False
#
#     # Adding custom layers
#     model = models.Sequential([
#         base_model,
#         layers.Flatten(),
#         layers.Dense(512, activation='relu'),
#         layers.Dropout(0.5),
#         layers.Dense(1, activation='sigmoid')  # Для бінарної класифікації
#     ])
#
#     model.compile(optimizer='adam',
#                   loss='binary_crossentropy',
#                   metrics=['accuracy'])
#     return model
#
#
# vgg19_model = build_vgg19_model()
#
# early_stopping = EarlyStopping(
#     monitor='val_accuracy',
#     patience=3,
#     restore_best_weights=True # We restore the scales of the best model
# )
#
# # Model training
# vgg19_model.fit(
#     train_generator,
#     epochs=20,
#     validation_data=val_generator,
#     callbacks=[early_stopping]
# )
#
# # Saving
# vgg19_model.save('vgg19_transfer_model.h5')







# def build_resnet50_model():
#     base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
#
#     # Freezing the base layers
#     for layer in base_model.layers:
#         layer.trainable = False
#
#     # Adding custom layers
#     model = models.Sequential([
#         base_model,
#         layers.Flatten(),
#         layers.Dense(512, activation='relu'),
#         layers.Dropout(0.5),
#         layers.Dense(1, activation='sigmoid')  # For binary classification
#     ])
#
#     model.compile(optimizer='adam',
#                   loss='binary_crossentropy',
#                   metrics=['accuracy'])
#     return model
#
#
# resnet50_model = build_resnet50_model()
#
#
# early_stopping = EarlyStopping(
#     monitor='val_accuracy',
#     patience=3,
#     restore_best_weights=True 
# )
# 
# resnet50_model.fit(
#     train_generator,
#     epochs=20,
#     validation_data=val_generator,
#     callbacks=[early_stopping]
# )
#
#
# resnet50_model.save('resnet50_transfer_model.h5')


# Loading models
vgg19_model = tf.keras.models.load_model('vgg19_transfer_model.h5')
resnet50_model = tf.keras.models.load_model('resnet50_transfer_model.h5')

# Evaluation of models on test data
vgg19_loss, vgg19_accuracy = vgg19_model.evaluate(test_generator)
resnet50_loss, resnet50_accuracy = resnet50_model.evaluate(test_generator)

print(f"VGG19 Test Accuracy: {vgg19_accuracy:.2f}")
print(f"ResNet50 Test Accuracy: {resnet50_accuracy:.2f}")

# Visualization of predictions
sample_images, sample_labels = next(test_generator)
predictions_vgg19 = vgg19_model.predict(sample_images)
predictions_resnet50 = resnet50_model.predict(sample_images)

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for i, ax in enumerate(axes[0]):
    ax.imshow(sample_images[i])
    predicted_label = 'Dog' if predictions_vgg19[i] > 0.5 else 'Cat'
    true_label = 'Dog' if sample_labels[i] == 1 else 'Cat'
    ax.set_title(f"VGG19 - Pred: {predicted_label}, True: {true_label}")
    ax.axis('off')

for i, ax in enumerate(axes[1]):
    ax.imshow(sample_images[i])
    predicted_label = 'Dog' if predictions_resnet50[i] > 0.5 else 'Cat'
    true_label = 'Dog' if sample_labels[i] == 1 else 'Cat'
    ax.set_title(f"ResNet50 - Pred: {predicted_label}, True: {true_label}")
    ax.axis('off')

plt.show()

