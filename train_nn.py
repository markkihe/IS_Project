import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import json
import os

IMG_SIZE = 224
BATCH_SIZE = 32

# ======================
# 1️⃣ Data Generator
# ======================

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

valid_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_data = train_datagen.flow_from_directory(
    "train",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

valid_data = valid_datagen.flow_from_directory(
    "valid",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    "test",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

NUM_CLASSES = train_data.num_classes
print("Number of classes:", NUM_CLASSES)

# ======================
# 🔥 Save class mapping (สำคัญมาก)
# ======================

with open("class_indices.json", "w") as f:
    json.dump(train_data.class_indices, f)

print("class_indices.json saved!")

# ======================
# 2️⃣ Build Model
# ======================

base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=2,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "best_model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1,
    save_format="h5"
)
# ======================
# 3️⃣ Train (Freeze base)
# ======================

model.fit(
    train_data,
    validation_data=valid_data,
    epochs=10,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

# ======================
# 4️⃣ Fine-tuning
# ======================

base_model.trainable = True

for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_data,
    validation_data=valid_data,
    epochs=5,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

# ======================
# 5️⃣ Evaluate Best Model
# ======================

best_model = tf.keras.models.load_model("best_model.h5", compile=False)

loss, acc = best_model.evaluate(test_data)
print("Final Test Accuracy:", acc)

# ✅ Save เป็น .h5 (compatible กับ TF 2.13)
best_model.save("nn_model.h5")
print("Final model saved as nn_model.h5")