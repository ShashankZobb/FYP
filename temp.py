import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Embedding, Flatten, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import ModelCheckpoint
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model


# Define classes and image URLs as shown in the previous example
# ...

# Download and organize images as shown in the previous example
# ...
# classes

file = open('result.txt', 'a')
classes = ["Normal", "Turberculosis", "COVID19", "Normal", "Pneumonia", "adenocarcinoma", "squamous.cell.carcinoma","large.cell.carcinoma", "adenocarcinoma", "COVID19", "large.cell.carcinoma", "Normal", "Pneumonia", "squamous.cell.carcinoma"]
path = [r"C:\Users\shash\Desktop\projects\TB_Chest_Radiography_Database\Normal", r"C:\Users\shash\Desktop\projects\TB_Chest_Radiography_Database\Tuberculosis", r"C:\Users\shash\Desktop\projects\Data\test\COVID19", r"C:\Users\shash\Desktop\projects\Data\test\NORMAL", r"C:\Users\shash\Desktop\projects\Data\test\PNEUMONIA" , r"C:\Users\shash\Desktop\projects\Data1\test\adenocarcinoma", r"C:\Users\shash\Desktop\projects\Data1\test\squamous.cell.carcinoma", r"C:\Users\shash\Desktop\projects\Data1\test\large.cell.carcinoma", r"C:\Users\shash\Desktop\projects\Data1\train\adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib", r"C:\Users\shash\Desktop\projects\Data\train\COVID19", r"C:\Users\shash\Desktop\projects\Data1\train\large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa", r"C:\Users\shash\Desktop\projects\Data1\train\normal", r"C:\Users\shash\Desktop\projects\Data\train\PNEUMONIA", r"C:\Users\shash\Desktop\projects\Data1\train\squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa"]

# Load and process images using OpenCV
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return img

# Load images and labels into arrays
def load_images_and_labels(class_names, dataset_dir):
    images = []
    labels = []
    freq = {}
    for class_index, class_name in enumerate(class_names):
        print(class_index)
        for filename in os.listdir(dataset_dir[class_index]):
            if class_name not in freq:
                freq[class_name] = 0
            if freq[class_name] >= 1000:
                break
            freq[class_name] += 1
            img = image.load_img(os.path.join(dataset_dir[class_index], filename), target_size = (224, 224))
            img_array = image.img_to_array(img)

            #img_expanded = np.expand_dims(img_array, axis = 0)
            img_ready = preprocess_input(img_array)
            images.append(img_ready)
            labels.append(class_name)
    return np.array(images), np.array(labels)
# Load and preprocess images
images, labels = load_images_and_labels(classes, path)


print(Counter(labels))

#Encode the label
lb = LabelEncoder()
labels = lb.fit_transform(labels)
# Balancing
images = images.reshape(len(images), -1)
smote = SMOTE(sampling_strategy='auto', random_state=42)
images, labels = smote.fit_resample(images, labels)
images = images.reshape(-1, 224, 224, 3)
print(Counter(labels))
labels = to_categorical(labels)
w = "imagenet"
batch_Size = 16
split = 0.6
rate = 0.0001
input_tensor=Input(shape=(224,224,3))
resnet_model = VGG19(weights=w, include_top=False, input_tensor=input_tensor)
# Split data into train and test sets
image_train_filenames, image_test_filenames, labels_train, labels_test = train_test_split(
    images, labels, train_size=split, random_state=42, stratify=labels)
image_test_filenames, image_valid_filenames, labels_test, labels_valid = train_test_split(
    image_test_filenames, labels_test, test_size=0.5, random_state=42, stratify=labels_test)
# Define a custom data generator
def data_generator(image_filenames, labels, batch_size):
    num_samples = len(labels)
    while True:
        for i in range(0, num_samples, batch_size):
            batch_image = [
                img_filename for img_filename in image_filenames[i:i + batch_size]
            ]

            # Remove None values (if any) from batch_image
            batch_image = [img for img in batch_image if img is not None]

            if len(batch_image) == 0:
                continue  # Skip this batch if there are no valid images
            batch_labels = labels[i:i + len(batch_image)]  # Match the length of labels with valid images

            # Convert batch_image list to numpy array
            batch_image = np.array(batch_image)

            yield (batch_image, batch_labels)

valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
image_train_size = len(image_train_filenames)
image_valid_size = len(image_valid_filenames)
train_datagen.fit(image_train_filenames)
valid_datagen.fit(image_valid_filenames)
# Create data generators for train and validation sets
batch_size = batch_Size  # Adjust the batch size if needed
train_generator = data_generator(image_train_filenames, labels_train, batch_Size)
validation_generator = data_generator(image_valid_filenames, labels_valid, batch_Size)
# Define the image branch (ResNet-50)

# image_input = Input(shape=(224, 224, 3))
# image_features = resnet_model(image_input)
# image_features_flat = Flatten()(image_features)

# Add fully connected layers for classification
x = resnet_model.output
x = tf.keras.layers.GlobalMaxPooling2D()(x)
x = Flatten()(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = tf.keras.layers.Dense(len(set(classes)), activation='softmax')(x)
#predictions = (Dropout(0.25))(predictions)

# Create the multi-modal model
model = Model(inputs=resnet_model.input, outputs=predictions)
# Compile the model
model.compile(optimizer=Adam(learning_rate=rate), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model using the custom data generators
steps_per_epoch = image_train_size // batch_Size  # Use image_train_filenames for steps_per_epoch
validation_steps = image_valid_size // batch_Size  # Use image_test_filenames for validation_steps
#model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=10, validation_data=validation_generator, validation_steps=validation_steps)

# Train the model using the custom data generators
steps_per_epoch = image_train_size // batch_Size
validation_steps = image_valid_size // (batch_Size)

# Define a ModelCheckpoint callback with patience
model_checkpoint = ModelCheckpoint(
    filepath=r"vgg19 best_model.h5",
    monitor='val_accuracy',  # Monitor validation accuracy
    save_best_only=True,
    save_weights_only=False,
    mode='max',  # Save the best model based on validation accuracy
    save_freq='epoch',  # Save after every epoch
    period=2,  # Save after every 2 epochs
    patience=3  # Stop training if no improvement for 3 consecutive epochs
)  # <-- Add the missing closing parenthesis here

# Train the model and store the training history
history = model.fit(
    train_datagen.flow(image_train_filenames, labels_train, batch_Size),
    steps_per_epoch=steps_per_epoch,
    epochs=20,
    validation_data=valid_datagen.flow(image_valid_filenames, labels_valid, batch_Size),
    validation_steps=validation_steps,
    callbacks=[model_checkpoint]  # Add the ModelCheckpoint callback
)
#test_eval = model.evaluate(image_test_filenames, labels_test, verbose=0)
#file.write("Resnet50 model balanced dataset learning rate:"+str(rate)+" batch_size:"+str(batch_Size)+" spilt"+str(split)+" weights"+str(w)+"\n")
#file.write('Test loss:'+str(test_eval[0])+"\n")
#file.write('Test accuracy:'+str(test_eval[1])+"\n")
# Plot training history: Loss
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('Model Loss vgg19')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Model Loss vgg19.jpg')
plt.clf()
# Plot training history: Accuracy
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title('Model Accuracy vgg19')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('Model Accuracy vgg19.jpg')
plt.clf()