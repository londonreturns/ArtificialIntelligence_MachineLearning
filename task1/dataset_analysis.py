import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

def analyze_dataset_balance(base_path):
    """Analyze the balance of the dataset"""
    # Paths
    train_path = os.path.join(base_path, 'train')
    test_path = os.path.join(base_path, 'test')
    
    # Count images
    train_total, train_counts = count_images(train_path)
    test_total, test_counts = count_images(test_path)
    
    print(f"Total training images: {train_total}")
    print(f"Total test images: {test_total}")
    print(f"Total images: {train_total + test_total}")
    
    print("\nTraining images per class:")
    for class_name, count in train_counts.items():
        print(f"  {class_name}: {count}")
    
    print("\nTest images per class:")
    for class_name, count in test_counts.items():
        print(f"  {class_name}: {count}")
    
    # Plot class distribution
    plt.figure(figsize=(12, 6))
    
    # Training data
    plt.subplot(1, 2, 1)
    plt.bar(train_counts.keys(), train_counts.values())
    plt.title('Training Images per Class')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Test data
    plt.subplot(1, 2, 2)
    plt.bar(test_counts.keys(), test_counts.values())
    plt.title('Test Images per Class')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the figure
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}\\class_distribution.png")
    print(f"Saved class distribution plot to {output_dir}\\class_distribution.png")
    
    plt.show()
    
    return train_counts, test_counts

def analyze_validation_split(train_path, validation_split=0.2, img_height=224, img_width=224, batch_size=32):
    """Analyze the validation split"""
    # Create data generator with validation split
    datagen = ImageDataGenerator(validation_split=validation_split)
    
    # Create training generator
    train_generator = datagen.flow_from_directory(
        train_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    # Create validation generator
    validation_generator = datagen.flow_from_directory(
        train_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    print(f"\nTraining generator:")
    print(f"  Number of samples: {train_generator.samples}")
    print(f"  Number of classes: {len(train_generator.class_indices)}")
    print(f"  Class indices: {train_generator.class_indices}")
    print(f"  Steps per epoch (batch_size={batch_size}): {train_generator.samples // batch_size}")
    
    print(f"\nValidation generator:")
    print(f"  Number of samples: {validation_generator.samples}")
    print(f"  Number of classes: {len(validation_generator.class_indices)}")
    print(f"  Class indices: {validation_generator.class_indices}")
    print(f"  Steps per epoch (batch_size={batch_size}): {validation_generator.samples // batch_size}")
    
    # Count samples per class in training and validation sets
    train_class_counts = {}
    validation_class_counts = {}
    
    # Reset generators
    train_generator.reset()
    validation_generator.reset()
    
    # Count training samples per class
    for i in range(len(train_generator.filenames)):
        class_name = train_generator.filenames[i].split(os.sep)[0]
        if class_name in train_class_counts:
            train_class_counts[class_name] += 1
        else:
            train_class_counts[class_name] = 1
    
    # Count validation samples per class
    for i in range(len(validation_generator.filenames)):
        class_name = validation_generator.filenames[i].split(os.sep)[0]
        if class_name in validation_class_counts:
            validation_class_counts[class_name] += 1
        else:
            validation_class_counts[class_name] = 1
    
    print("\nTraining samples per class:")
    for class_name, count in train_class_counts.items():
        print(f"  {class_name}: {count}")
    
    print("\nValidation samples per class:")
    for class_name, count in validation_class_counts.items():
        print(f"  {class_name}: {count}")
    
    # Plot validation split distribution
    plt.figure(figsize=(12, 6))
    
    # Get all class names
    all_classes = sorted(list(set(list(train_class_counts.keys()) + list(validation_class_counts.keys()))))
    
    # Prepare data for plotting
    train_counts = [train_class_counts.get(cls, 0) for cls in all_classes]
    val_counts = [validation_class_counts.get(cls, 0) for cls in all_classes]
    
    # Create positions for bars
    x = np.arange(len(all_classes))
    width = 0.35
    
    # Plot bars
    plt.bar(x - width/2, train_counts, width, label='Training')
    plt.bar(x + width/2, val_counts, width, label='Validation')
    
    # Add labels and title
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.title('Training vs Validation Split')
    plt.xticks(x, all_classes, rotation=45, ha='right')
    plt.legend()
    
    plt.tight_layout()
    
    # Save the figure
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}\\validation_split.png")
    print(f"Saved validation split plot to {output_dir}\\validation_split.png")
    
    plt.show()
    
    return train_generator.samples, validation_generator.samples

def analyze_augmented_images(base_path, augmented_path):
    """Analyze the augmented images"""
    # Original dataset
    train_path = os.path.join(base_path, 'train')
    test_path = os.path.join(base_path, 'test')
    
    # Augmented dataset
    aug_train_path = os.path.join(augmented_path, 'train')
    aug_test_path = os.path.join(augmented_path, 'test')
    
    # Count original images
    train_total, train_counts = count_images(train_path)
    test_total, test_counts = count_images(test_path)
    
    # Count augmented images
    aug_train_total, aug_train_counts = count_images(aug_train_path) if os.path.exists(aug_train_path) else (0, {})
    aug_test_total, aug_test_counts = count_images(aug_test_path) if os.path.exists(aug_test_path) else (0, {})
    
    print(f"\nOriginal dataset:")
    print(f"  Total training images: {train_total}")
    print(f"  Total test images: {test_total}")
    
    print(f"\nAugmented dataset:")
    print(f"  Total augmented training images: {aug_train_total}")
    print(f"  Total augmented test images: {aug_test_total}")
    
    # Plot comparison
    plt.figure(figsize=(15, 10))
    
    # Get all class names
    all_train_classes = sorted(list(set(list(train_counts.keys()) + list(aug_train_counts.keys()))))
    all_test_classes = sorted(list(set(list(test_counts.keys()) + list(aug_test_counts.keys()))))
    
    # Training data comparison
    plt.subplot(2, 1, 1)
    
    # Prepare data for plotting
    orig_train_counts = [train_counts.get(cls, 0) for cls in all_train_classes]
    aug_train_counts_list = [aug_train_counts.get(cls, 0) for cls in all_train_classes]
    
    # Create positions for bars
    x = np.arange(len(all_train_classes))
    width = 0.35
    
    # Plot bars
    plt.bar(x - width/2, orig_train_counts, width, label='Original Training')
    plt.bar(x + width/2, aug_train_counts_list, width, label='Augmented Training')
    
    # Add labels and title
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.title('Original vs Augmented Training Images')
    plt.xticks(x, all_train_classes, rotation=45, ha='right')
    plt.legend()
    
    # Test data comparison
    plt.subplot(2, 1, 2)
    
    # Prepare data for plotting
    orig_test_counts = [test_counts.get(cls, 0) for cls in all_test_classes]
    aug_test_counts_list = [aug_test_counts.get(cls, 0) for cls in all_test_classes]
    
    # Create positions for bars
    x = np.arange(len(all_test_classes))
    width = 0.35
    
    # Plot bars
    plt.bar(x - width/2, orig_test_counts, width, label='Original Test')
    plt.bar(x + width/2, aug_test_counts_list, width, label='Augmented Test')
    
    # Add labels and title
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.title('Original vs Augmented Test Images')
    plt.xticks(x, all_test_classes, rotation=45, ha='right')
    plt.legend()
    
    plt.tight_layout()
    
    # Save the figure
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}\\augmentation_comparison.png")
    print(f"Saved augmentation comparison plot to {output_dir}\\augmentation_comparison.png")
    
    plt.show()

def calculate_steps_per_epoch(num_samples, batch_size):
    """Calculate the correct steps_per_epoch value"""
    # Integer division
    steps = num_samples // batch_size
    # If there's a remainder, add one more step
    if num_samples % batch_size > 0:
        steps += 1
    return steps

def main():
    # Paths
    base_path = os.path.join('data', 'pest')
    augmented_path = os.path.join('data', 'augmented_images', 'pest')
    
    # Analyze dataset balance
    print("Analyzing dataset balance...")
    train_counts, test_counts = analyze_dataset_balance(base_path)
    
    # Analyze validation split
    print("\nAnalyzing validation split...")
    train_samples, val_samples = analyze_validation_split(os.path.join(base_path, 'train'))
    
    # Analyze augmented images
    print("\nAnalyzing augmented images...")
    analyze_augmented_images(base_path, augmented_path)
    
    # Calculate correct steps_per_epoch
    batch_size = 32
    print("\nCalculating correct steps_per_epoch values:")
    
    # For training
    train_steps = calculate_steps_per_epoch(train_samples, batch_size)
    print(f"Training steps_per_epoch: {train_steps} (samples: {train_samples}, batch_size: {batch_size})")
    
    # For validation
    val_steps = calculate_steps_per_epoch(val_samples, batch_size)
    print(f"Validation steps_per_epoch: {val_steps} (samples: {val_samples}, batch_size: {batch_size})")
    
    # Recommendation
    print("\nRecommendation to fix 'Training Data Exhaustion' warning:")
    print("1. Use the correct steps_per_epoch values calculated above")
    print("2. Or use .repeat() if working with a tf.data.Dataset")
    print("3. Make sure the validation_split is properly set and there are enough samples in each class")

if __name__ == "__main__":
    main()