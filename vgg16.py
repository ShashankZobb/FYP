import tensorflow 
from datetime import datetime
from tensorflow.keras.applications.xception import preprocess_input
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
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential,Model, load_model
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow.keras.layers import MaxPooling2D,BatchNormalization,Dropout,Flatten,Dense,Conv2D,Input,GlobalAveragePooling2D
from tensorflow.keras import layers
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
# from keras import optimizers
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.callbacks import ModelCheckpoint
from imblearn.over_sampling import SMOTE
from collections import Counter
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.utils.vis_utils import plot_model
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



#Encode the label
lb = LabelEncoder()
labels = lb.fit_transform(labels)
le_name_mapping = dict(zip(lb.classes_, lb.transform(lb.classes_)))
# print("Before Smote:")
# print(Counter(labels))

# # Balancing
# images = images.reshape(len(images), -1)
# smote = SMOTE(sampling_strategy='auto', random_state=42)
# images, labels = smote.fit_resample(images, labels)
# images = images.reshape(-1, 224, 224, 3)
# print("After Smote:")
# print(Counter(labels))
labels = to_categorical(labels)
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

#gen=ImageDataGenerator(preprocessing_function=preprocess_input, horizontal_flip=True, vertical_flip=False, rotation_range=10, width_shift_range=[-.1,.1], height_shift_range=[-.1,.1],)
#train_gen=gen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels', target_size=img_size,class_mode='categorical',shuffle=True, batch_size=batch_size)
#length= len(valid_df) # determine test batch size and test steps such that test_batch_size X test_steps = number of test samples
#test_batch_size=sorted([int(length/n) for n in range(1,length+1) if length % n ==0 and length/n<=batch_size],reverse=True)[0]
#test_steps=int(length/test_batch_size)

#test_gen=gen2.flow_from_dataframe( valid_df,  x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
                                   # color_mode='rgb', shuffle=False, batch_size=test_batch_size)
#valid_gen=gen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical', shuffle=True,batch_size=batch_size)
#train_steps=int(len(train_gen.labels)/batch_size)
# def build_model():
full_name='concatenate'
classes_number=7 #Number of classes
input_tensor=Input(shape=(224,224,3))
######################################################################################################
base_model1 = Xception(weights=w, include_top=False, input_tensor=input_tensor)
for layer in base_model1.layers:
    layer._name = layer._name + str('_A')
#print(base_model1.summary())
# tf.keras.utils.plot_model(
#     base_model1
# )
count = 0
for layer in base_model1.layers:
    count += 1
count = count//2
# for layer in base_model1.layers:
#     layer.trainable = False
features1 = base_model1.output
# featuresConv2 = Conv2D(1024, (1, 1),padding='same')(features2)
# featuresPool2 = MaxPooling2D()(featuresConv2)
x = tf.keras.layers.GlobalMaxPooling2D()(features1)
x = Flatten()(x)
# x = BatchNormalization()(x)
# featuresDense2 = Dense(1024, activation="relu")(featuresFlatten2)
# featuresDense2 = Dense(512, activation="relu")(featuresDense2)
#x = Dense(512, activation="relu")(x)
# x = BatchNormalization()(x)
featuresDense1 = Dense(256, activation="relu")(x)
# featuresDense1 = BatchNormalization()(featuresDense1)
print(featuresDense1.shape)
# print(featuresFlatten1.shape)
######################################################################################################
base_model2 = VGG16(weights=w, include_top=False, input_tensor=input_tensor)
for layer in base_model2.layers:
    layer._name = layer._name + str('_C')
# tf.keras.utils.plot_model(
#     base_model2
# )

count = 0
for layer in base_model2.layers:
    count += 1
count = count//2
# for layer in base_model2.layers:
#     layer.trainable = False
features2 = base_model2.output
# featuresConv2 = Conv2D(1024, (1, 1),padding='same')(features2)
# x = MaxPooling2D()(featuresConv2)
x = tf.keras.layers.GlobalMaxPooling2D()(features2)
# featuresFlatten2 = Flatten()(features2)
x = Flatten()(x)
# x = BatchNormalization()(x)
# featuresDense2 = Dense(1024, activation="relu")(featuresFlatten2)
# featuresDense2 = Dense(512, activation="relu")(featuresDense2)
#x = Dense(512, activation="relu", name="dd11")(x)
# x = BatchNormalization()(x)
featuresDense2 = Dense(256, activation="relu", name="dd12")(x)
# featuresDense2 = BatchNormalization()(featuresDense2)
print(featuresDense2.shape)
#print(base_model2.summary())
# print(featuresFlatten2.shape)
########################################################################################################
#base_model3 = InceptionV3(weights=w, include_top=False, input_tensor=input_tensor)
#for layer in base_model3.layers:
#    layer._name = layer._name + str('_B')

# print(base_model1.input)
# tf.keras.utils.plot_model(
#     base_model3
# )
#count = 0
#for layer in base_model3.layers:
#    count += 1
#count = count//2
# for layer in base_model3.layers:
#     layer.trainable = False
#features3 = base_model3.output
# featuresConv2 = Conv2D(1024, (1, 1),padding='same')(features2)
# x = MaxPooling2D()(features3)
#x = tf.keras.layers.GlobalMaxPooling2D()(features3)
#x = Flatten()(x)
# featuresFlatten2 = Flatten()(features2)
# featuresDense2 = Dense(1024, activation="relu")(featuresFlatten2)
# featuresDense2 = Dense(512, activation="relu")(featuresDense2)
#x = Dense(512, activation="relu")(x)
#featuresDense3 = Dense(256, activation="relu")(x)
#print(featuresDense3.shape)
#print(base_model3.summary())
# print(featuresFlatten3.shape)
########################################################################################################
# # df = pd.DataFrame(features3)
# # df.drop_duplicates()
#featuresDense2 = featuresDense3
base_model4 = VGG19(weights=w, include_top=False, input_tensor=input_tensor)
count = 0
for layer in base_model4.layers:
    layer._name = layer._name + str('_D')
for layer in base_model4.layers:
    count += 1
count = count//2
# for layer in base_model4.layers:
#     layer.trainable = False
#print(base_model4.summary())
features3 = base_model4.output
# featuresConv2 = Conv2D(1024, (1, 1),padding='same')(features2)
# featuresPool2 = MaxPooling2D()(featuresConv2)
x = tf.keras.layers.GlobalMaxPooling2D()(features3)
x = Flatten()(x)
# featuresFlatten2 = Flatten()(features2)
# featuresDense2 = Dense(1024, activation="relu")(featuresFlatten2)
# featuresDense2 = Dense(512, activation="relu")(featuresDense2)
# x = BatchNormalization()(x)
#x = Dense(512, activation="relu")(x)
# x = BatchNormalization()(x)
featuresDense4 = Dense(256, activation="relu")(x)

# print(featuresDense4.shape)
#print(base_model4.summary())
concatenated=tensorflow.keras.layers.concatenate([featuresDense1,  featuresDense2, featuresDense4])
print(concatenated.shape)
#print(input_tensor)
# # # concatenated=tensorflow.keras.layers.Concatenate()([features1,features2,features3]) #Concatenate the extracted features
# # ####################################################################################################
# conv=tensorflow.keras.layers.Conv2D(1024, (1, 1),padding='same')(concatenated) #add the concatenated features to a convolutional layer
# feature = Flatten(name='flatten')(conv)
# print(feature.shape)
#concatenated = BatchNormalization()(concatenated)
dp = Dropout(0.5)(concatenated) #add dropout
preds = Dense(classes_number, activation='softmax', kernel_initializer=RandomNormal(mean=0.0, stddev=0.001), name="dense_7")(dp)
Concatenated_model = Model(inputs=input_tensor, outputs=preds)
plot_model(Concatenated_model, to_file='simple_model.png', show_shapes=True, show_layer_names=False)
# #######################################################
# for layer in Concatenated_model.layers:
#     layer.trainable = True
# compute class weight 
# based on appearance of each class in y_trian
# cls_weights = class_weight.compute_sample_weight(class_weight="balanced", y=(labels_train))
# d_class_weights = dict(enumerate(cls_weights))
# print(d_class_weights)
# dict mapping

# cls_wgts = {}
# tot = 0
# for i, label in enumerate(np.unique(labels_train)):
#     weight = len(labels_train)/(classes_number*np.sum(np.array(labels_train)==label))
#     cls_wgts[label] = weight
#     tot += weight
# for i in cls_wgts:
#     cls_wgts[i] = cls_wgts[i]/tot
#cls_wgts = {i : cls_wgts[i] for i, label in enumerate((np.unique(labels_train)))}
#optimizers = RMSprop(learning_rate=0.0001, rho=0.9, epsilon=1e-08)
opt = Adam(learning_rate=0.0001)
Concatenated_model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])
filepath="ttttt best1_model2.tf"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', save_best_only=True, mode='max') #creating checkpoint to save the best validation accuracy
callbacks_list = [checkpoint]
#Concatenated_model.summary()

start = datetime.now()
step_per_epoch = image_train_size // batch_Size
# cnnModel=Concatenated_model.fit_generator(train_gen, epochs=10,validation_data=valid_gen,shuffle=True,callbacks=callbacks_list)
#
history=Concatenated_model.fit(
  train_datagen.flow(image_train_filenames, labels_train, batch_Size),
  validation_data=valid_datagen.flow(image_valid_filenames, labels_valid, batch_Size),
  steps_per_epoch=step_per_epoch,
  epochs=20,shuffle=True,
  )
#callbacks=callbacks_list
# opt = Adam(learning_rate=0.00001)
# history=Concatenated_model.fit(
#   train_datagen.flow(image_train_filenames, labels_train, batch_Size),
#   validation_data=valid_datagen.flow(image_valid_filenames, labels_valid, batch_Size),
#   steps_per_epoch=step_per_epoch,
#   epochs=20,shuffle=True,
#   callbacks=callbacks_list)
  #class_weight=d_class_weights,
# title = "ttt12 cmobined model"
# # Plot training history: Loss
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='val')
# plt.title('Model Loss '+title)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# ax = plt.gca()
# ax.set_ylim([0 ,1])
# plt.legend()
# plt.savefig('Model Loss '+title+".jpg")
# plt.clf()
# # Plot training history: Accuracy
# plt.plot(history.history['accuracy'], label='train')
# plt.plot(history.history['val_accuracy'], label='val')
# plt.title('Model Accuracy '+title)
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# ax = plt.gca()
# ax.set_ylim([0 ,1])
# plt.legend()
# plt.savefig('Model Accuracy '+title+'.jpg')
# plt.clf()

duration = datetime.now() - start
print("Training completed in time: ", duration)

y_test = np.argmax(labels_test, axis=1)
pred = np.argmax(Concatenated_model.predict(image_test_filenames, batch_size=8), axis=1)
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