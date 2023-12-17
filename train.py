
import scipy
import numpy as np

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.image import load_img

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# To train using MobileNetV2

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

checkpoint = keras.callbacks.ModelCheckpoint(
    'models/mobilenetv2_v7_{epoch:02d}_{val_accuracy:.3f}.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    mode="max",
)

callback = [early_stop, checkpoint]

# Define the data generators
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input, 
    rotation_range=15,
    height_shift_range=0.2    
)

valid_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input, 
)

# Create the training and validation data generators
train_ds = train_gen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=32
)

validation_ds = valid_gen.flow_from_directory(
    'data/valid',
    target_size=(224, 224),
    batch_size=32,
)

# Hyperparameters
learning_rate = 0.01
droprate = 0.2

# Define the model preparation logic
def make_model(learning_rate=0.01, droprate=0.2):
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False

    #########################################

    inputs = keras.Input(shape=(224, 224, 3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    drop = keras.layers.Dropout(droprate)(vectors)
    outputs = keras.layers.Dense(100)(drop)
    model = keras.Model(inputs, outputs)
    
    #########################################

    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    
    return model

# Create the model
model = make_model(learning_rate=learning_rate, 
                   droprate=droprate)

# Monitor the scores achieved
scores = []
    
# Train the model
history = model.fit(
    train_ds, 
    epochs=50, 
    validation_data=validation_ds, 
    callbacks=callback
)
    
# Append to scores
scores.append(history.history)