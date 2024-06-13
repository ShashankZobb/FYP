import tensorflow 
from datetime import datetime
from tensorflow.keras.applications.vgg16 import preprocess_input
import pandas as pd
from sklearn.utils import class_weight
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#Importing Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from tensorflow.keras.models import load_model
from keras.models import Model
from imblearn.over_sampling import SMOTE
from collections import Counter
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
# Input Layer ----> Transfer Learning Algorithm ---> Output Layer
classes = ["Normal", "Turberculosis", "COVID19", "Normal", "Pneumonia", "adenocarcinoma", "squamous.cell.carcinoma","large.cell.carcinoma", "adenocarcinoma", "COVID19", "large.cell.carcinoma", "Normal", "Pneumonia", "squamous.cell.carcinoma"]
path = [r"C:\Users\shash\Desktop\projects\TB_Chest_Radiography_Database\Normal", r"C:\Users\shash\Desktop\projects\TB_Chest_Radiography_Database\Tuberculosis", r"C:\Users\shash\Desktop\projects\Data\test\COVID19", r"C:\Users\shash\Desktop\projects\Data\test\NORMAL", r"C:\Users\shash\Desktop\projects\Data\test\PNEUMONIA" , r"C:\Users\shash\Desktop\projects\Data1\test\adenocarcinoma", r"C:\Users\shash\Desktop\projects\Data1\test\squamous.cell.carcinoma", r"C:\Users\shash\Desktop\projects\Data1\test\large.cell.carcinoma", r"C:\Users\shash\Desktop\projects\Data1\train\adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib", r"C:\Users\shash\Desktop\projects\Data\train\COVID19", r"C:\Users\shash\Desktop\projects\Data1\train\large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa", r"C:\Users\shash\Desktop\projects\Data1\train\normal", r"C:\Users\shash\Desktop\projects\Data\train\PNEUMONIA", r"C:\Users\shash\Desktop\projects\Data1\train\squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa"]


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
            if freq[class_name] >= 500:
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
le_name_mapping = dict(zip(lb.classes_, lb.transform(lb.classes_)))
print(le_name_mapping)
labels = to_categorical(labels)
# Balancing
images = images.reshape(len(images), -1)
smote = SMOTE(sampling_strategy='auto', random_state=42)
images, labels = smote.fit_resample(images, labels)
images = images.reshape(-1, 224, 224, 3)
# print(Counter(labels))
w = "imagenet"
batch_Size = 8
split = 0.6
rate = 0.0001

#resnet_model = ResNet50(weights=w, include_top=False, input_shape=(224, 224, 3))
# Split data into train and test sets
image_train_filenames, image_test_filenames, labels_train, labels_test = train_test_split(
    images, labels, train_size=split, random_state=42, stratify=labels)
image_test_filenames, image_valid_filenames, labels_test, labels_valid = train_test_split(
    image_test_filenames, labels_test, test_size=0.5, random_state=42, stratify=labels_test)

valid_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator()
train_datagen = ImageDataGenerator(
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
test_datagen.fit(image_test_filenames)
batch_size=8
# file2 = open('labels_test', 'ab')
# pickle.dump(labels_test, file2)
# file2.close()
# file2 = open('labels_train', 'ab')
# pickle.dump(labels_train, file2)
# file2.close()
# file2 = open('labels_valid', 'ab')
# pickle.dump(labels_valid, file2)
# file2.close()
# exit()
#gen=ImageDataGenerator(preprocessing_function=preprocess_input, horizontal_flip=True, vertical_flip=False, rotation_range=10, width_shift_range=[-.1,.1], height_shift_range=[-.1,.1],)
#train_gen=gen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels', target_size=img_size,class_mode='categorical',shuffle=True, batch_size=batch_size)
#length= len(valid_df) # determine test batch size and test steps such that test_batch_size X test_steps = number of test samples
#test_batch_size=sorted([int(length/n) for n in range(1,length+1) if length % n ==0 and length/n<=batch_size],reverse=True)[0]
#test_steps=int(length/test_batch_size)
model1 = load_model('xception unabalanced.tf')
# layer = model1.get_layer('dd11')
# model2 = Model(inputs=model1.input, outputs=layer.output)
# for i in image_test_filenames:
#     temp = model1.predict(i)
#     print(temp)
y_test = np.argmax(labels_test, axis=1)
pred = np.argmax(model1.predict(image_test_filenames, batch_size=1), axis=1)
# pred = model2.predict(image_valid_filenames, batch_size=8)
# file1 = open("vgg16 output valid", 'ab')
# pickle.dump(pred, file1)
# file1.close()
# for i in y_test:
#     print(i)
#print(accuracy_score(y_test, pred))
cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(8, 6))
xlabels = []
for i in range(7):
    for j in le_name_mapping:
        if le_name_mapping[j] == i:
            xlabels.append(j)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=xlabels, yticklabels=xlabels)
plt.xlabel('Predicted')
plt.ylabel('True')
#plt.title('Confusion Matrix')
plt.show()
print("accuracy=", accuracy_score(y_test, pred))
print("f1 score= ",f1_score(y_test, pred, average="macro"))
print("precision_score= ", precision_score(y_test, pred, average="macro"))
print("recall_score= ", recall_score(y_test, pred, average="macro"))

