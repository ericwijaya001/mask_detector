from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# load detektor wajah 
# caffe model tuh framework deeplearning GG
prototxtPath = r'face_detector\deploy.prototxt'
weightsPath = r'face_detector\res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# load masker classifier
model = load_model('model\mask_detector.model')	

# load inputan gambar
image = cv2.imread('examples\example_01.png')
orig = image.copy()
# height and width maybe
(h, w) = image.shape[:2]

# resize ke 300x300 px dan melakukan mean subraction
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
	(104.0, 177.0, 123.0))


net.setInput(blob)
detections = net.forward()

# for selama deteksi dan mengestrak nilai confirdence  

for i in range(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with
	# the detection
	confidence = detections[0, 0, i, 2]
    # bandingkan nilai confidencde dengan nilai minimal confidence
	if confidence > 0.5:
        # ngitung koordinat x dan y-nya bounding box
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
        # mastikan bounding box dalam bingkai gambar
		(startX, startY) = (max(0, startX), max(0, startY))
		(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
        # ekstrak 'face' ROI pake numpy slicing
		face = image[startY:endY, startX:endX]
		
        # preprocess ROI kek pas training
		# ekstrak ROI wajah dan convert dr BGR ke RGB, 

		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		face = cv2.resize(face, (224, 224))
		face = img_to_array(face)
		face = preprocess_input(face)
		face = np.expand_dims(face, axis=0)
        
        # prediksi pake masker ato ga pke masker
		(mask, withoutMask) = model.predict(face)[0]
		
        # init warna bounding box mera jika tanpa masker dan warna ijo jika menggunakan masker
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        
        # probabilitas label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        
        # display label dan bounding box pada output frame
		cv2.putText(image, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
# output
cv2.imshow("Output", image)
cv2.waitKey(0)