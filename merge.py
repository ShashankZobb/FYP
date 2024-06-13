import pickle
import numpy as np
from keras.layers import Dense, BatchNormalization, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
file1 = open('labels test unb', 'rb')
labels_test = pickle.load(file1)
file1.close()
file1 = open('labels train unb', 'rb')
labels_train = pickle.load(file1)
file1.close()
file1 = open('labels valid unb', 'rb')
labels_valid = pickle.load(file1)
file1.close()
file1 = open('vgg16 output test', 'rb')
vgg16_test = pickle.load(file1)
file1.close() 
file1 = open('vgg19 output test', 'rb')
vgg19_test = pickle.load(file1)
file1.close() 
file1 = open('xception output test', 'rb')
xception_test = pickle.load(file1)
file1.close() 
file1 = open('vgg16 unb output train', 'rb')
vgg16_train = pickle.load(file1)
file1.close() 
file1 = open('vgg19 unb output train', 'rb')
vgg19_train = pickle.load(file1)
file1.close() 
file1 = open('xception unb output train', 'rb')
xception_train = pickle.load(file1)
file1.close() 
file1 = open('vgg16 unb output valid', 'rb')
vgg16_valid = pickle.load(file1)
file1.close() 
file1 = open('vgg19 unb output valid', 'rb')
vgg19_valid = pickle.load(file1)
file1.close() 
file1 = open('xception unb output valid', 'rb')
xception_valid = pickle.load(file1)
file1.close() 
image_train = np.concatenate([vgg16_train, vgg19_train, xception_train], axis=1)
image_test = np.concatenate([vgg16_test, vgg19_test, xception_test], axis=1)
image_valid = np.concatenate([vgg16_valid, vgg19_valid, xception_valid], axis=1)

np.random.seed(668)

opt = Adam(learning_rate=1e-3)
model = Sequential()
model.add(Dropout(0.5, input_shape=(768,)))
model.add(BatchNormalization())
model.add(Dense(7, activation="softmax"))
model1=model
model1.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])


epochs = 100
model_checkpoint = ModelCheckpoint(
    filepath="./merge.tf",
    monitor='val_acc',  # Monitor validation accuracy
    save_best_only=True,
    save_weights_only=False,
    mode='max',  # Save the best model based on validation accuracy
    save_freq='epoch',  # Save after every epoch
    period=2,  # Save after every 2 epochs
    patience=3  # Stop training if no improvement for 3 consecutive epochs
)
history=model1.fit(image_train, labels_train, batch_size=16, epochs=epochs, verbose=2, shuffle=True, validation_data=(image_valid, labels_valid))
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('Model Loss method2 unbalanced')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Model Loss method2 unb.jpg')
plt.clf()
# Plot training history: Accuracy
plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='val')
plt.title('Model Accuracy method2 unbalanced')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('Model Accuracy method2 unb.jpg')
plt.clf()
"""EVALUATE MODEL ON TEST DATA"""
le_name_mapping = {'COVID19': 0, 'Normal': 1, 'Pneumonia': 2, 'Turberculosis': 3, 'adenocarcinoma': 4, 'large.cell.carcinoma': 5, 'squamous.cell.carcinoma': 6}
y_test = np.argmax(labels_test, axis=1)
pred = np.argmax(model1.predict(image_test), axis=1)
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
plt.savefig('confusion merge unb.png', bbox_inches='tight')
plt.show()
print("accuracy=", accuracy_score(y_test, pred))
print("f1 score= ",f1_score(y_test, pred, average="macro"))
print("precision_score= ", precision_score(y_test, pred, average="macro"))
print("recall_score= ", recall_score(y_test, pred, average="macro"))