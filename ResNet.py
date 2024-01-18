import numpy as np
from keras.applications import ResNet50
from keras.applications.resnet import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

resnet_model = ResNet50(weights='imagenet', include_top=True)

img_path = '1.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

predictions = resnet_model.predict(img_array)
# print(predictions)
decoded_predictions = decode_predictions(predictions, top=5)[0]
print(decoded_predictions)
# for _, label, confidence in decoded_predictions:
#     print(f'{label}: {confidence:.2f}')
for number, label, confidence in decoded_predictions:
    print(f'{label} : {confidence:.2f}')
