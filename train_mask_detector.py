from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adamax, Nadam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os
from time import sleep

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# hyper parameter GG
# learning rate, jmlh epoch, n batch size
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

# buat preprocessing
imagePaths = list(paths.list_images('dataset/'))
data = []
labels = []
# loop smpe smua yg ada di imagePaths
for imagePath in imagePaths:
    # ekstrak label kelas dari nama file
	# label berisi 'with_mask' atau 'without_mask' sesuai nama folder image-nya
	label = imagePath.split(os.path.sep)[-2]
	# load inputan gambar (224x224)
	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image)
    # men-skala intensitas piksel ke range[-1,1]
	image = preprocess_input(image)
	# mengupdate list data dan labels
	data.append(image)
	labels.append(label)
	# print('label : ',label)

# mengconvert data dan label ke arr
data = np.array(data, dtype="float32")
labels = np.array(labels)

# menggunakan one hot encoder
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# tts 
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# data augmen
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# base model tanpa kepala (inculde_top = False)
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# buat kepala model yg menggantikan include_top
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

# supaya weight pada base layers tidak ikut terupdate selama proses backpropagation
# weights head layernya saja yg dituned
for layer in baseModel.layers:
	layer.trainable = False

opt = Adamax(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

predIdxs = model.predict(testX, batch_size=BS)

# mencari index label dengan probabilitas prediksi tertinggi
predIdxs = np.argmax(predIdxs, axis=1)

# report
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# save model
model.save('model/model.model', save_format="h5")

# plot los mbe akurasi
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig('plot.png')
