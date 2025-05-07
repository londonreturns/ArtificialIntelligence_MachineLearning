#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image Classification with Convolutional Neural Network

This script implements a comprehensive end-to-end deep learning project for image classification.
The goal is to deepen understanding of the entire deep learning pipeline, from data
preprocessing to model building and evaluation.

The script has two parts:
- Part A: Train and experiment with baseline CNN model from scratch.
- Part B: Fine-tune a pre-trained model for real-world application.
"""

# Importing Libraries
import matplotlib.pyplot as plt
import os
import random
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import Precision, Recall, F1Score
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Input
from sklearn.metrics import classification_report, confusion_matrix
import time


def count_images(directory):
    """Count the number of images in each class directory"""
    total_images = 0
    class_counts = {}
    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            num_images = len([f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
            class_counts[class_name] = num_images
            total_images += num_images
    return total_images, class_counts


def visualize_samples(train_path):
    """Visualize sample images from each class"""
    class_dirs = os.listdir(train_path)
    images = []
    labels = []

    for class_dir in class_dirs:
        class_path = os.path.join(train_path, class_dir)
        if os.path.isdir(class_path):
            image_files = os.listdir(class_path)
            if image_files:  # Check if there are any images
                random_image_file = random.choice(image_files)
                image_path = os.path.join(class_path, random_image_file)
                try:
                    images.append(plt.imread(image_path))
                    labels.append(class_dir)
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")

    cols = 4
    rows = (len(images) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
    axes = axes.flatten()

    for i, (image, label) in enumerate(zip(images, labels)):
        axes[i].imshow(image)
        axes[i].set_title(label)
        axes[i].axis('off')

    # Turn off axes for empty subplots
    for j in range(len(images), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def check_corrupted_images(directory):
    """Check for corrupted images in the directory"""
    corrupted_count = 0
    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            image_files = os.listdir(class_path)
            for image_file in image_files:
                image_path = os.path.join(class_path, image_file)
                try:
                    with Image.open(image_path) as img:
                        img.verify()
                except (IOError, SyntaxError) as e:
                    print(f"Corrupted image found: {image_path}")
                    corrupted_count += 1
                    # Optionally remove corrupted images
                    # os.remove(image_path)

    if corrupted_count == 0:
        print(f"No corrupted images found in {directory}")
    else:
        print(f"Found {corrupted_count} corrupted images in {directory}")


def generate_augmented_images(input_base_dir, output_base_dir):
    """Generate and save augmented images to disk if not already generated"""
    # Check if augmented images already exist
    if os.path.exists(output_base_dir) and any(os.scandir(output_base_dir)):
        print("Augmented images already exist. Skipping augmentation.")
        return

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rescale=1. / 255
    )

    # Counter for processed images
    total_processed = 0
    total_augmented = 0

    # Process each dataset directory separately to respect the validation split
    for dataset_dir in ['train', 'val', 'test']:
        dataset_path = os.path.join(input_base_dir, dataset_dir)

        # Skip if the directory doesn't exist
        if not os.path.exists(dataset_path):
            print(f"Skipping {dataset_dir} directory as it doesn't exist")
            continue

        print(f"Processing {dataset_dir} dataset")

        for class_dir in os.listdir(dataset_path):
            class_path = os.path.join(dataset_path, class_dir)

            # Skip if not a directory
            if not os.path.isdir(class_path):
                continue

            # Create output directory for this class
            output_dir = os.path.join(output_base_dir, dataset_dir, class_dir)
            os.makedirs(output_dir, exist_ok=True)

            print(f"Processing directory: {dataset_dir}/{class_dir}")

            # Get all image files in this class directory
            image_files = [f for f in os.listdir(class_path) if
                           f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]

            for idx, filename in enumerate(image_files):
                if idx % 10 == 0:
                    print(f"  Processing image {idx + 1}/{len(image_files)} in {dataset_dir}/{class_dir}")

                image_path = os.path.join(class_path, filename)

                # Get filename without extension
                file_basename = os.path.splitext(filename)[0]

                try:
                    # Load and resize image
                    img = tf.keras.preprocessing.image.load_img(image_path)
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    img_resized = tf.image.resize(img_array, (256, 256))
                    img_resized_np = img_resized.numpy().astype('uint8')

                    # Save the resized original image
                    original_output_path = os.path.join(output_dir, f"{file_basename}_original.jpg")
                    Image.fromarray(img_resized_np).save(original_output_path, quality=95)
                    total_processed += 1

                    # Generate and save augmented images
                    img_resized = tf.expand_dims(img_resized, axis=0)
                    augmented_images = datagen.flow(img_resized, batch_size=1)

                    for i in range(2, 10):
                        # Get next augmented image
                        augmented_image = next(augmented_images)[0]

                        # Convert from float [0,1] to uint8 [0,255]
                        augmented_image_uint8 = (augmented_image * 255).astype('uint8')

                        # Save the augmented image
                        aug_output_path = os.path.join(output_dir, f"{file_basename}_aug_{i - 1}.jpg")
                        Image.fromarray(augmented_image_uint8).save(aug_output_path, quality=95)
                        total_augmented += 1

                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

    print(f"Processing complete! Saved {total_processed} original images and {total_augmented} augmented images.")
    print(f"Total images saved: {total_processed + total_augmented}")


def create_baseline_model(input_shape, num_classes):
    """Create a baseline CNN model"""
    # Create the model using Sequential
    model = Sequential()

    # Add Input layer first
    model.add(Input(shape=input_shape))

    # First convolutional layer
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    # Second convolutional layer
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    # Third convolutional layer
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    # Flatten the output for the fully connected layers
    model.add(Flatten())

    # First fully connected layer
    model.add(Dense(512, activation='relu'))

    # Second fully connected layer
    model.add(Dense(256, activation='relu'))

    # Third fully connected layer
    model.add(Dense(128, activation='relu'))

    # Output layer
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', Precision(), Recall(), F1Score()]
    )

    return model


def create_deeper_model(input_shape, num_classes):
    """Create a deeper CNN model with specified architecture"""
    model = Sequential([
        Input(shape=input_shape),
        Rescaling(1. / 255),

        # Repeat the block 3 times
        Conv2D(128, (3, 3), padding='same', activation=None),
        BatchNormalization(),
        Dropout(0.25),
        tf.keras.layers.Activation('relu'),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), padding='same', activation=None),
        BatchNormalization(),
        Dropout(0.25),
        tf.keras.layers.Activation('relu'),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), padding='same', activation=None),
        BatchNormalization(),
        Dropout(0.25),
        tf.keras.layers.Activation('relu'),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy', Precision(), Recall(), F1Score()]
    )

    return model


def create_vgg16_model(input_shape, num_classes, trainable=False):
    """Create a model using VGG16 pre-trained base model"""

    # Define input tensor
    inputs = Input(shape=input_shape)

    # Create VGG16 base model
    base_model = VGG16(include_top=False, weights='imagenet', input_tensor=inputs)

    # Freeze the base model if not trainable
    base_model.trainable = trainable

    # Add custom layers on top
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy', Precision(), Recall(), F1Score()]
    )

    return model


def plot_training_history(history, title):
    """Plot the training history"""
    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 5))

    # Plot training and validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{title} - Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{title} - Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()

    # Save the figure to the output directory
    filename = f"{output_dir}\\{title.replace(' ', '_')}_history.png"
    plt.savefig(filename)
    print(f"Saved training history plot to {filename}")

    plt.show()


def plot_loss_accuracy(history, title="Model"):
    """Plot training and validation loss and accuracy"""
    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss', color='blue')
    plt.plot(range(1, len(val_loss) + 1), val_loss, label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{title} - Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_acc) + 1), train_acc, label='Training Accuracy', color='blue')
    plt.plot(range(1, len(val_acc) + 1), val_acc, label='Validation Accuracy', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'{title} - Training and Validation Accuracy')
    plt.legend()

    # Save the figure to the output directory
    filename = f"{output_dir}\\{title.replace(' ', '_')}_loss_accuracy.png"
    plt.savefig(filename)
    print(f"Saved {title} loss and accuracy plot to {filename}")

    plt.show()


def main():
    # Dataset path
    path = os.path.join('data', 'pest')

    train_path = os.path.join(path, 'train')
    test_path = os.path.join(path, 'test')
    val_path = os.path.join(path, 'val')

    # Getting the number of classes and their names
    classes = os.listdir(train_path)
    num_classes = len(classes)

    print(f'Total number of classes: {num_classes}')
    print(f'Classes: {classes}')

    # Counting images in each dataset
    train_total, train_counts = count_images(train_path)
    test_total, test_counts = count_images(test_path)
    val_total, val_counts = count_images(val_path) if os.path.exists(val_path) else (0, {})

    print(f'Number of training images: {train_total}')
    print(f'Number of validation images: {val_total}')
    print(f'Number of test images: {test_total}')
    print(f'Total number of images: {train_total + val_total + test_total}')

    # Generate augmented images
    input_base_dir = os.path.join('data', 'pest')
    output_base_dir = os.path.join('data', 'augmented_images', 'pest')
    generate_augmented_images(input_base_dir, output_base_dir)

    # Define image size for the model
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    BATCH_SIZE = 32

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        subset='validation'
    )

    val_test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = val_test_datagen.flow_from_directory(
        test_path,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    # Create the baseline model
    input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)  # RGB images
    baseline_model = create_baseline_model(input_shape, num_classes)

    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(
        'baseline_model_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )

    # Train the baseline model
    EPOCHS = 20

    # Record start time
    start_time = time.time()

    baseline_history = baseline_model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        callbacks=[early_stopping, checkpoint]
    )

    # Record end time
    baseline_training_time = time.time() - start_time
    print(f"Baseline model training time: {baseline_training_time:.2f} seconds")

    # Plot the training history for the baseline model
    plot_training_history(baseline_history, 'Baseline Model')
    plot_loss_accuracy(baseline_history, 'Baseline Model')

    # Create the deeper model
    deeper_model = create_deeper_model(input_shape, num_classes)

    # Define callbacks for the deeper model
    early_stopping_deeper = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    checkpoint_deeper = ModelCheckpoint(
        'deeper_model_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )

    # Train the deeper model
    EPOCHS_DEEPER = 30

    # Record start time
    start_time_deeper = time.time()

    deeper_history = deeper_model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS_DEEPER,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        callbacks=[early_stopping_deeper, checkpoint_deeper]
    )

    # Record end time
    deeper_training_time = time.time() - start_time_deeper
    print(f"Deeper model training time: {deeper_training_time:.2f} seconds")

    # Plot the training history for the deeper model
    plot_training_history(deeper_history, 'Deeper Model')
    plot_loss_accuracy(deeper_history, 'Deeper Model')

    # Create data generators with appropriate preprocessing for the pre-trained model

    train_datagen_pretrained = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_test_datagen_pretrained = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    # Create data generators
    train_generator_pretrained = train_datagen_pretrained.flow_from_directory(
        train_path,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    validation_generator_pretrained = val_test_datagen_pretrained.flow_from_directory(
        val_path,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    test_generator_pretrained = val_test_datagen_pretrained.flow_from_directory(
        test_path,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    # Create the VGG16 model with frozen base layers (feature extraction)
    pretrained_model = create_vgg16_model(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        num_classes=num_classes,
        trainable=False  # Start with frozen base layers
    )

    # Define callbacks for the pre-trained model
    early_stopping_pretrained = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    checkpoint_pretrained = ModelCheckpoint(
        'vgg16_feature_extraction_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )

    # Train the model with frozen base layers (feature extraction)
    EPOCHS_PRETRAINED = 20

    # Record start time
    start_time_pretrained = time.time()

    pretrained_history = pretrained_model.fit(
        train_generator_pretrained,
        steps_per_epoch=train_generator_pretrained.samples // BATCH_SIZE,
        epochs=EPOCHS_PRETRAINED,
        validation_data=validation_generator_pretrained,
        validation_steps=validation_generator_pretrained.samples // BATCH_SIZE,
        callbacks=[early_stopping_pretrained, checkpoint_pretrained]
    )

    # Record end time
    feature_extraction_time = time.time() - start_time_pretrained
    print(f"Feature extraction training time: {feature_extraction_time:.2f} seconds")

    # Plot the training history
    plot_training_history(pretrained_history, 'VGG16 Feature Extraction')
    plot_loss_accuracy(pretrained_history, 'VGG16 Feature Extraction')

    # Now, let's fine-tune the model by unfreezing some of the top layers of the VGG16 base model
    # Unfreeze the top 2 blocks of VGG16
    for layer in pretrained_model.layers[15:]:
        layer.trainable = True

    # Recompile the model with a lower learning rate
    pretrained_model.compile(
        optimizer=Adam(learning_rate=1e-5),  # Much lower learning rate for fine-tuning
        loss='categorical_crossentropy',
        metrics=['accuracy', Precision(), Recall(), F1Score()]
    )

    # Define callbacks for fine-tuning
    early_stopping_finetuning = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    checkpoint_finetuning = ModelCheckpoint(
        'vgg16_finetuned_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )

    # Fine-tune the model
    EPOCHS_FINETUNING = 20

    # Record start time
    start_time_finetuning = time.time()

    finetuning_history = pretrained_model.fit(
        train_generator_pretrained,
        steps_per_epoch=train_generator_pretrained.samples // BATCH_SIZE,
        epochs=EPOCHS_FINETUNING,
        validation_data=validation_generator_pretrained,
        validation_steps=validation_generator_pretrained.samples // BATCH_SIZE,
        callbacks=[early_stopping_finetuning, checkpoint_finetuning]
    )

    # Record end time
    finetuning_time = time.time() - start_time_finetuning
    print(f"Fine-tuning time: {finetuning_time:.2f} seconds")
    print(f"Total transfer learning time: {feature_extraction_time + finetuning_time:.2f} seconds")

    # Plot the fine-tuning history
    plot_training_history(finetuning_history, 'VGG16 Fine-Tuning')
    plot_loss_accuracy(finetuning_history, 'VGG16 Fine-Tuning')

    # Evaluate all models on the test set
    print("\nEvaluating Baseline Model:")
    baseline_evaluation = baseline_model.evaluate(test_generator)
    print(f"Test Loss: {baseline_evaluation[0]:.4f}")
    print(f"Test Accuracy: {baseline_evaluation[1]:.4f}")
    print(f"Test F1 Score: {baseline_evaluation[3]:.4f}")

    print("\nEvaluating Deeper Model:")
    deeper_evaluation = deeper_model.evaluate(test_generator)
    print(f"Test Loss: {deeper_evaluation[0]:.4f}")
    print(f"Test Accuracy: {deeper_evaluation[1]:.4f}")
    print(f"Test F1 Score: {deeper_evaluation[3]:.4f}")

    print("\nEvaluating Fine-tuned VGG16 Model:")
    pretrained_evaluation = pretrained_model.evaluate(test_generator_pretrained)
    print(f"Test Loss: {pretrained_evaluation[0]:.4f}")
    print(f"Test Accuracy: {pretrained_evaluation[1]:.4f}")
    print(f"Test F1 Score: {pretrained_evaluation[3]:.4f}")

    print("\nTraining Time Comparison:")
    print(f"Baseline Model: {baseline_training_time:.2f} seconds")
    print(f"Deeper Model: {deeper_training_time:.2f} seconds")
    print(f"VGG16 (Feature Extraction + Fine-Tuning): {feature_extraction_time + finetuning_time:.2f} seconds")


if __name__ == "__main__":
    main()
