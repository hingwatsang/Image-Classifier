# Make all imports
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import json

# input arguments
from argument import image_path, model_file, top_k, category_names

# preprocessing of image
from preprocessing_image import im

# load the model
model = tf.keras.models.load_model(model_file,custom_objects={'KerasLayer': hub.KerasLayer})

# Prediction
result = model(im).numpy().squeeze()
index = np.argsort(result)
num_classes = len(result)
label = index[-1]+1      #index: 0 -101, label: 1-102
prob = result[label-1]

# Generate lists of top K label and probablity
if top_k:
    class_list = index[num_classes-top_k:]
    class_list = np.flip(class_list)
    prob_list = []
    for i in class_list:
        prob_list.append(result[i])
    class_list += 1

# Load the names of flowers from a json file
if category_names:
    with open(category_names, 'r') as f:
        class_names = json.load(f)
        name = class_names[str(label)]

print('--------------------------------------------')
# Print the predictions
if category_names:   #with name
    print('The name of flower is {}.'.format(name))

# The predicted label and probability
print('The label of the flower is {}.'.format(label))
print('The probablity is {:.4f}.'.format(prob))
print('--------------------------------------------')

# print probability of top K classes
if top_k:
    if category_names:     #with names
        print('\nThe most probably {} flowers are listed below.'.format(top_k))
        print('Label      Probability    Name of flower')
        for i in range(top_k):
            print(class_list[i],'       ',prob_list[i], '      ',class_names[str(class_list[i])] )
    else:      # without name
        print('\nThe most probably {} flower labels are listed below.'.format(top_k))
        print('Label      Probability')
        for i in range(top_k):
            print(class_list[i], '       ',prob_list[i] )
    print('----------------------------------------------------')
