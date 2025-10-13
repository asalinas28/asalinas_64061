import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

os.environ["KERAS_HOME"] = "C:/austi/.keras"


# ======================================================
# Download and prepare dataset
# ======================================================
def get_dataset_dirs():
    zip_path = tf.keras.utils.get_file(
        'cats_and_dogs_filtered.zip',
        'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip',
        extract=True)
    
    base_dir = os.path.join(os.path.dirname(zip_path), 'cats_and_dogs_filtered')
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')

    if not (os.path.isdir(train_dir) and os.path.isdir(validation_dir)):
        print("Dataset not found — redownloading manually...")
        import zipfile, requests

        url = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
        local_zip = os.path.expanduser('~/cats_and_dogs_filtered.zip')

        if not os.path.exists(local_zip):
            r = requests.get(url)
            with open(local_zip, 'wb') as f:
                f.write(r.content)

        with zipfile.ZipFile(local_zip, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(local_zip))

        base_dir = os.path.join(os.path.dirname(local_zip), 'cats_and_dogs_filtered')
        train_dir = os.path.join(base_dir, 'train')
        validation_dir = os.path.join(base_dir, 'validation')

    return base_dir, train_dir, validation_dir


# ======================================================
# Prepare datasets (with augmentation)
# ======================================================
def prepare_datasets(train_dir, validation_dir, image_size=(150,150), batch_size=32, sample_limit=None):
    """Loads, augments, and batches image data. Optionally limit training samples."""
    if not os.path.isdir(train_dir) or not os.path.isdir(validation_dir):
        raise SystemExit(f"Missing dataset directories. Expected train:{train_dir} and validation:{validation_dir}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=image_size,
        batch_size=batch_size
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        validation_dir,
        image_size=image_size,
        batch_size=batch_size
    )

    if sample_limit:
        train_ds = train_ds.take(sample_limit)

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.1),
    ])

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds


# ======================================================
# CNN built from scratch
# ======================================================
def build_model_scratch(input_shape=(150,150,3)):
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=input_shape),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# ======================================================
# Pretrained model (VGG16)
# ======================================================
def build_model_pretrained(input_shape=(150,150,3), trainable=False):
    base_model = tf.keras.applications.VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = trainable  

    model = models.Sequential([
        layers.Rescaling(1./255),
        base_model,
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# ======================================================
# Plot training history
# ======================================================
def plot_history(history, title='Training Results', out_path='training_plot.png'):
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, acc, 'bo-', label='Training acc')
    plt.plot(epochs, val_acc, 'ro-', label='Validation acc')
    plt.title(f'{title} - Accuracy')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
    plt.title(f'{title} - Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path)
    print(f'Saved training plot to {out_path}')


# ======================================================
# Run experiment (scratch or pretrained)
# ======================================================
def run_experiment(model_type, sample_limit, args):
    base_dir, train_dir, validation_dir = get_dataset_dirs()
    print(f'\nRunning {model_type.upper()} model | Training sample limit: {sample_limit or "Full"}')

    train_ds, val_ds = prepare_datasets(train_dir, validation_dir,
                                        image_size=(args.img_size, args.img_size),
                                        batch_size=args.batch_size,
                                        sample_limit=sample_limit)

    if model_type == 'scratch':
        model = build_model_scratch(input_shape=(args.img_size, args.img_size, 3))
    else:
        model = build_model_pretrained(input_shape=(args.img_size, args.img_size, 3), trainable=False)

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]

    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=args.epochs,
                        callbacks=callbacks,
                        verbose=1)

    loss, acc = model.evaluate(val_ds, verbose=0)
    print(f'{model_type.title()} model final val_accuracy: {acc:.4f}')

    plot_history(history, title=f'{model_type.title()} (limit={sample_limit})',
                 out_path=f'{model_type}_plot_{sample_limit or "full"}.png')

    return acc


# ======================================================
# Main execution
# ======================================================
def main(args):
    results = []
    sample_sizes = [30, 100, 500]  

    for s in sample_sizes:
        acc = run_experiment('scratch', s, args)
        results.append(('Scratch', s, acc))

    for s in sample_sizes:
        acc = run_experiment('pretrained', s, args)
        results.append(('Pretrained', s, acc))

    import pandas as pd
    df = pd.DataFrame(results, columns=['Model Type', 'Training Sample Size', 'Validation Accuracy'])
    print("\n=== Summary of Results ===")
    print(df)
    df.to_csv('results_summary.csv', index=False)
    print("Saved results_summary.csv")

    plt.figure(figsize=(6,4))
    for model_type in df['Model Type'].unique():
        subset = df[df['Model Type'] == model_type]
        plt.plot(subset['Training Sample Size'], subset['Validation Accuracy'],
                 marker='o', label=model_type)
    plt.title('Validation Accuracy vs Training Sample Size')
    plt.xlabel('Training Sample Size')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('comparison_summary.png')
    print('Saved comparison_summary.png')


# ======================================================
# Run script
# ======================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Assignment 2 Convolution (Scratch vs Pretrained)')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--img-size', type=int, default=150, help='Image width/height')
    args = parser.parse_args()
    main(args)


