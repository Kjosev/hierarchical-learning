from keras.applications import vgg16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import os
import tarfile

model = vgg16.VGG16(weights='imagenet', include_top=True)

data_path = 'C:\\Users\\kjosev\\Documents\\Master Thesis\\datasets\\Awa2\\AwA2-data\\Animals_with_Attributes2\\JPEGImages'
# data_path = '/cluster/scratch/tkjosev/datasets/awa2//'

for class_dir in os.listdir(data_path):
    class_dir_path = os.path.join(data_path,class_dir)
    image_paths = os.listdir(class_dir_path)
    image_paths = sorted(image_paths)
    image_paths = [os.path.join(class_dir_path, x) for x in image_paths]

    model_extractfeatures = Model(input=model.input, output=model.get_layer('fc2').output)

    print('Preprocessing images')
    images = []
    for image_paths in image_paths:
        img = image.load_img(image_paths, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        images.append(x)
    images = np.array(images)
    images = np.squeeze(images)

    print('Extracting features')

    fc2_features = model_extractfeatures.predict(images)
    fc2_features = fc2_features.reshape((-1,4096))

    np.save('data/awa2/features/%s.npy' % class_dir,fc2_features)