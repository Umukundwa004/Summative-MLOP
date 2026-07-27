import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class BrainTumorModel:
    def __init__(self, img_size=(224, 224), batch_size=32, num_classes=4):
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.model = None

    def build_custom_cnn(self):
        """Builds a light, efficient Convolutional Neural Network from scratch."""
        model = models.Sequential([
            layers.Input(shape=(*self.img_size, 3)),
            
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            # Block 4
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.4),
            
            # Dense Classifier Head
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return self.model

    def compile_model(self, learning_rate=1e-3):
        """Compiles the model with Adam optimizer and Categorical Crossentropy loss."""
        if self.model is None:
            raise ValueError("Model has not been built yet. Call build_custom_cnn() first.")
            
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
        )
        print(" Model compiled successfully.")

    def get_data_generators(self, train_dir, test_dir):
        """Prepares train and test ImageDataGenerators with rescaling and light augmentation."""
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

        train_gen = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )

        test_gen = test_datagen.flow_from_directory(
            test_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )

        return train_gen, test_gen

    def train(self, train_gen, test_gen, epochs=25, save_path='best_brain_tumor_model.keras'):
        """Trains the neural network using early stopping and learning rate reduction callbacks."""
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1),
            ModelCheckpoint(save_path, monitor='val_accuracy', save_best_only=True, verbose=1)
        ]

        print(f"🚀 Starting model training for {epochs} epochs...")
        history = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=test_gen,
            callbacks=callbacks
        )
        return history


def resolve_data_paths():
    """Detects whether dataset exists in local 'data/' directory or Kagglehub cache."""
    project_root = Path.cwd().parent if Path.cwd().name == "notebook" else Path.cwd()
    local_train = project_root / "data" / "train"
    local_test = project_root / "data" / "test"

    if local_train.exists() and local_test.exists():
        return str(local_train), str(local_test)
    
    # Fallback to local user cache folder if data/ is not populated yet
    cache_base = Path(os.path.expanduser("~")) / ".cache" / "kagglehub" / "datasets" / "masoudnickparvar" / "brain-tumor-mri-dataset" / "versions" / "2"
    cache_train = cache_base / "Training"
    cache_test = cache_base / "Testing"

    if cache_train.exists() and cache_test.exists():
        return str(cache_train), str(cache_test)
    
    raise FileNotFoundError("Could not locate training/testing dataset folders in project or Kaggle cache.")


# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # 1. Resolve dataset paths dynamically
    train_path, test_path = resolve_data_paths()
    print(f" Training Data Path: {train_path}")
    print(f" Testing Data Path:  {test_path}\n")

    # 2. Instantiate and build model pipeline
    classifier = BrainTumorModel(img_size=(224, 224), batch_size=32, num_classes=4)
    model = classifier.build_custom_cnn()
    model.summary()
    
    classifier.compile_model(learning_rate=1e-3)

    # 3. Create data streams
    train_generator, test_generator = classifier.get_data_generators(train_path, test_path)

    # 4. Run model training
    history = classifier.train(
        train_gen=train_generator,
        test_gen=test_generator,
        epochs=20,
        save_path='best_brain_tumor_model.keras'
    )