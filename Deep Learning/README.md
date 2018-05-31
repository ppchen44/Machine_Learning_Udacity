# Artificial Intelligence Nanodegree
## Convolutional Neural Networks
## Project: Write an Algorithm for a Dog Identification App

### Why We're Here
In this notebook, you will make the first steps towards developing an algorithm that could be used as part of a mobile or web app. At the end of this project, your code will accept any user-supplied image as input. If a dog is detected in the image, it will provide an estimate of the dog's breed. If a human is detected, it will provide an estimate of the dog breed that is most resembling. The image below displays potential sample output of your finished project (... but we expect that each student's algorithm will behave differently!).

In this real-world setting, you will need to piece together a series of models to perform different tasks; for instance, the algorithm that detects humans in an image will be different from the CNN that infers dog breed. There are many points of possible failure, and no perfect algorithm exists. Your imperfect solution will nonetheless create a fun user experience!

### The Road Ahead
We break the notebook into separate steps. Feel free to use the links below to navigate the notebook.

- Step 0: Import Datasets
- Step 1: Detect Humans
- Step 2: Detect Dogs
- Step 3: Create a CNN to Classify Dog Breeds (from Scratch)
- Step 4: Use a CNN to Classify Dog Breeds (using Transfer Learning)
- Step 5: Create a CNN to Classify Dog Breeds (using Transfer Learning)
- Step 6: Write your Algorithm
- Step 7: Test Your Algorithm

### Step 0: Import Datasets
#### Import Dog Dataset
In the code cell below, we import a dataset of dog images. We populate a few variables through the use of the load_files function from the scikit-learn library:

- train_files, valid_files, test_files - numpy arrays containing file paths to images
- train_targets, valid_targets, test_targets - numpy arrays containing onehot-encoded classification labels
- dog_names - list of string-valued dog breed names for translating labels

```
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset('dogImages/train')
valid_files, valid_targets = load_dataset('dogImages/valid')
test_files, test_targets = load_dataset('dogImages/test')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))
```
```
There are 133 total dog categories.
There are 8351 total dog images.

There are 6680 training dog images.
There are 835 validation dog images.
There are 836 test dog images.
```
```
print np.array(load_files('dogImages/train')['target'])[0]
```
```
94
```
```
printprint  train_targetstrain_ta [0]
```
```
[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
  0.  0.  0.  0.  0.  0.  0.]
  ```
#### Import Human Dataset
In the code cell below, we import a dataset of human images, where the file paths are stored in the numpy array human_files.

```

importimport  randomrandom
 randomrandom..seedseed((86753098675309)

# load filenames in shuffled human dataset
human_files = np.array(glob("lfw/*/*"))
random.shuffle(human_files)

# print statistics about the dataset
print('There are %d total human images.' % len(human_files))
```
```
There are 13233 total human images.
```
### Step 1: Detect Humans
We use OpenCV's implementation of Haar feature-based cascade classifiers to detect human faces in images. OpenCV provides many pre-trained face detectors, stored as XML files on github. We have downloaded one of these detectors and stored it in the haarcascades directory.

In the next code cell, we demonstrate how to use this detector to find human faces in a sample image.

```
import cv2                
import matplotlib.pyplot as plt                        
%matplotlib inline                               

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# load color (BGR) image
img = cv2.imread(human_files[3])
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = face_cascade.detectMultiScale(gray)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x,y,w,h) in faces:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()
```
Before using any of the face detectors, it is standard procedure to convert the images to grayscale. The detectMultiScale function executes the classifier stored in face_cascade and takes the grayscale image as a parameter.

In the above code, faces is a numpy array of detected faces, where each row corresponds to a detected face. Each detected face is a 1D array with four entries that specifies the bounding box of the detected face. The first two entries in the array (extracted in the above code as x and y) specify the horizontal and vertical positions of the top left corner of the bounding box. The last two entries in the array (extracted here as w and h) specify the width and height of the box.

#### Write a Human Face Detector
We can use this procedure to write a function that returns True if a human face is detected in an image and False otherwise. This function, aptly named face_detector, takes a string-valued file path to an image as input and appears in the code block below.

```
# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0
 ```
 
#### (IMPLEMENTATION) Assess the Human Face Detector
#### Question 1: Use the code cell below to test the performance of the face_detector function.

- What percentage of the first 100 images in human_files have a detected human face?
- What percentage of the first 100 images in dog_files have a detected human face?

Ideally, we would like 100% of human images with a detected face and 0% of dog images with a detected face. You will see that our algorithm falls short of this goal, but still gives acceptable performance. We extract the file paths for the first 100 images from each of the datasets and store them in the numpy arrays human_files_short and dog_files_short.

**Answer:**

- There are 99 percentage of the first 100 images in human_files have a detected human face.
- There are 11 percentage of the first 100 images in dog_files have a detected human face.

```
human_files_short = human_files[:100]
dog_files_short = train_files[:100]
# Do NOT modify the code above this line.

## TODO: Test the performance of the face_detector algorithm 
humanface_detected_in_human=[]
humanface_detected_in_dog=[]
for i in range(100):
    humanface_detected_in_human.append(face_detector(human_files_short[i]))
    humanface_detected.append(face_detector(dog_files_short[i]))
print ('There are %f percent human images with a detected face.' % ((sum(humanface_detected_in_human)/100.0)*100))
print ('There are %f percent dog images with a detected face.' % ((sum(humanface_detected)/100.0)*100))
## on the images in human_files_short and dog_files_short.
```
```
There are 99.000000 percent human images with a detected face.
There are 11.000000 percent dog images with a detected face.
```
**Question 2:** This algorithmic choice necessitates that we communicate to the user that we accept human images only when they provide a clear view of a face (otherwise, we risk having unneccessarily frustrated users!). In your opinion, is this a reasonable expectation to pose on the user? If not, can you think of a way to detect humans in images that does not necessitate an image with a clearly presented face?

**Answer:**

- In my opinion, I think this is a reasonable expectation to pose on the user.

We suggest the face detector from OpenCV as a potential way to detect human images in your algorithm, but you are free to explore other approaches, especially approaches that make use of deep learning :). Please use the code cell below to design and test your own face detection algorithm. If you decide to pursue this optional task, report performance on each of the datasets.

### Step 2: Detect Dogs

In this section, we use a pre-trained ResNet-50 model to detect dogs in images. Our first line of code downloads the ResNet-50 model, along with weights that have been trained on ImageNet, a very large, very popular dataset used for image classification and other vision tasks. ImageNet contains over 10 million URLs, each linking to an image containing an object from one of 1000 categories. Given an image, this pre-trained ResNet-50 model returns a prediction (derived from the available categories in ImageNet) for the object that is contained in the image.

```
from keras.applications.resnet50 import ResNet50

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')
```
#### Pre-process the Data
When using TensorFlow as backend, Keras CNNs require a 4D array (which we'll also refer to as a 4D tensor) as input, with shape
                                   (nb_samples, rows, columns, channels),
where `nb_samples` corresponds to the total number of images (or samples), and rows, columns, and channels correspond to the number of rows, columns, and channels for each image, respectively.

The `path_to_tensor` function below takes a string-valued file path to a color image as input and returns a 4D tensor suitable for supplying to a Keras CNN. The function first loads the image and resizes it to a square image that is $224 \times 224$ pixels. Next, the image is converted to an array, which is then resized to a 4D tensor. In this case, since we are working with color images, each image has three channels. Likewise, since we are processing a single image (or sample), the returned tensor will always have shape(1, 224, 224, 3).
The `paths_to_tensor` function takes a numpy array of string-valued image paths as input and returns a 4D tensor with shape

(nb_samples, 224, 224, 3).
Here, `nb_samples` is the number of samples, or number of images, in the supplied array of image paths. It is best to think of `nb_samples` as the number of 3D tensors (where each 3D tensor corresponds to a different image) in your dataset!

```
from keras.preprocessing import image                  
from tqdm import tqdm

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)
```

#### Making Predictions with ResNet-50

Getting the 4D tensor ready for ResNet-50, and for any other pre-trained model in Keras, requires some additional processing. First, the RGB image is converted to BGR by reordering the channels. All pre-trained models have the additional normalization step that the mean pixel (expressed in RGB as [103.939, 116.779, 123.68] and calculated from all pixels in all images in ImageNet) must be subtracted from every pixel in each image. This is implemented in the imported function `preprocess_input`. If you're curious, you can check the code for `preprocess_input` here.

Now that we have a way to format our image for supplying to ResNet-50, we are now ready to use the model to extract the predictions. This is accomplished with the predict method, which returns an array whose $i$-th entry is the model's predicted probability that the image belongs to the $i$-th ImageNet category. This is implemented in the `ResNet50_predict_labels` function below.

By taking the argmax of the predicted probability vector, we obtain an integer corresponding to the model's predicted object class, which we can identify with an object category through the use of this dictionary.

```
from keras.applications.resnet50 import preprocess_input, decode_predictions

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))
```
#### Write a Dog Detector
While looking at the dictionary, you will notice that the categories corresponding to dogs appear in an uninterrupted sequence and correspond to dictionary keys 151-268, inclusive, to include all categories from 'Chihuahua' to 'Mexican hairless'. Thus, in order to check to see if an image is predicted to contain a dog by the pre-trained ResNet-50 model, we need only check if the `ResNet50_predict_labels` function above returns a value between 151 and 268 (inclusive).

We use these ideas to complete the `dog_detector` function below, which returns True if a dog is detected in an image (and False if not).

```
### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))
 ```

**(IMPLEMENTATION) Assess the Dog Detector***
**Question 3:** Use the code cell below to test the performance of your dog_detector function.

- What percentage of the images in human_files_short have a detected dog?
- What percentage of the images in dog_files_short have a detected dog?

**Answer:**

- There are 2 percentage of the images in human_files_short have a detected dog.
- There are 100 percentage of the images in dog_files_short have a detected dog.


```
### TODO: Test the performance of the dog_detector function
dogface_detected_in_human=[]
dogface_detected_in_dog=[]
for i in range(100):
    dogface_detected_in_human.append(dog_detector(human_files_short[i]))
    dogface_detected_in_dog.append(dog_detector(dog_files_short[i]))
    
print('There are %.2f%% percent human images with a detected dog face' % ((sum(dogface_detected_in_human)/100.0)*100))
print('There are %.2f%% percent dog images with a detected dog face' % ((sum(dogface_detected_in_dog)/100.0)*100))

### on the images in human_files_short and dog_files_short.
```
```
There are 2.00% percent human images with a detected dog face
There are 100.00% percent dog images with a detected dog face
```
```
print('There are %.2f%% percent human images with a detected dog face' % ((sum(dogface_detected_in_human)/100.0)*100))
print('There are %.2f%% percent dog images with a detected dog face' % ((sum(dogface_detected_in_dog)/100.0)*100))
```
```
There are 2.00% percent human images with a detected dog face
There are 100.00% percent dog images with a detected dog face
```

### Step 3: Create a CNN to Classify Dog Breeds (from Scratch)
Now that we have functions for detecting humans and dogs in images, we need a way to predict breed from images. In this step, you will create a CNN that classifies dog breeds. You must create your CNN from scratch (so, you can't use transfer learning yet!), and you must attain a test accuracy of at least 1%. In Step 5 of this notebook, you will have the opportunity to use transfer learning to create a CNN that attains greatly improved accuracy.

Be careful with adding too many trainable layers! More parameters means longer training, which means you are more likely to need a GPU to accelerate the training process. Thankfully, Keras provides a handy estimate of the time that each epoch is likely to take; you can extrapolate this estimate to figure out how long it will take for your algorithm to train.

We mention that the task of assigning breed to dogs from images is considered exceptionally challenging. To see why, consider that even a human would have great difficulty in distinguishing between a Brittany and a Welsh Springer Spaniel.

It is not difficult to find other dog breed pairs with minimal inter-class variation (for instance, Curly-Coated Retrievers and American Water Spaniels).

Likewise, recall that labradors come in yellow, chocolate, and black. Your vision-based algorithm will have to conquer this high intra-class variation to determine how to classify all of these different shades as the same breed.

We also mention that random chance presents an exceptionally low bar: setting aside the fact that the classes are slightly imabalanced, a random guess will provide a correct answer roughly 1 in 133 times, which corresponds to an accuracy of less than 1%.

Remember that the practice is far ahead of the theory in deep learning. Experiment with many different architectures, and trust your intuition. And, of course, have fun!

#### Pre-process the Data

We rescale the images by dividing every pixel in every image by 255.
```
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255
```


**(IMPLEMENTATION) Model Architecture***

Create a CNN to classify dog breed. At the end of your code cell block, summarize the layers of your model by executing the line:

    `model.summary()`
    
 We have imported some Python modules to get you started, but feel free to import as many modules as you need. If you end up getting stuck, here's a hint that specifies a model that trains relatively fast on CPU and attains >1% test accuracy in 5 epochs:


**Question 4:** Outline the steps you took to get to your final CNN architecture and your reasoning at each step. If you chose to use the hinted architecture above, describe why you think that CNN architecture should work well for the image classification task.

**Answer:***
```
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, strides=1, activation='relu',input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(Conv2D(filters=32, kernel_size=2, strides=1, activation='relu'))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(Conv2D(filters=64,kernel_size=2,strides=1,activation='relu'))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(GlobalAveragePooling2D())
#model.add(Flatten())
model.add(Dense(133, activation='softmax'))
### TODO: Define your architecture.

model.summary()
```
```
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_36 (Conv2D)           (None, 223, 223, 16)      208       
_________________________________________________________________
max_pooling2d_31 (MaxPooling (None, 111, 111, 16)      0         
_________________________________________________________________
conv2d_37 (Conv2D)           (None, 110, 110, 32)      2080      
_________________________________________________________________
max_pooling2d_32 (MaxPooling (None, 55, 55, 32)        0         
_________________________________________________________________
conv2d_38 (Conv2D)           (None, 54, 54, 64)        8256      
_________________________________________________________________
max_pooling2d_33 (MaxPooling (None, 27, 27, 64)        0         
_________________________________________________________________
global_average_pooling2d_5 ( (None, 64)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 133)               8645      
=================================================================
Total params: 19,189
Trainable params: 19,189
Non-trainable params: 0
_________________________________________________________________
```
#### Compile the Model
```
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
```

**(IMPLEMENTATION) Train the Model**

Train your model in the code cell below. Use model checkpointing to save the model that attains the best validation loss.

You are welcome to augment the training data, but this is not a requirement.

```
from keras.callbacks import ModelCheckpoint  

### TODO: specify the number of epochs that you would like to use to train the model.

epochs = 10

### Do NOT modify the code below this line.

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', 
                               verbose=1, save_best_only=True)

model.fit(train_tensors, train_targets, 
          validation_data=(valid_tensors, valid_targets),
          epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)
```
```

Train on 6680 samples, validate on 835 samples
Epoch 1/10
6660/6680 [============================>.] - ETA: 1s - loss: 4.8831 - acc: 0.0093Epoch 00000: val_loss improved from inf to 4.87134, saving model to saved_models/weights.best.from_scratch.hdf5
6680/6680 [==============================] - 373s - loss: 4.8831 - acc: 0.0094 - val_loss: 4.8713 - val_acc: 0.0108
Epoch 2/10
6660/6680 [============================>.] - ETA: 0s - loss: 4.8666 - acc: 0.0131Epoch 00001: val_loss improved from 4.87134 to 4.85588, saving model to saved_models/weights.best.from_scratch.hdf5
6680/6680 [==============================] - 304s - loss: 4.8667 - acc: 0.0130 - val_loss: 4.8559 - val_acc: 0.0180
Epoch 3/10
6660/6680 [============================>.] - ETA: 0s - loss: 4.8322 - acc: 0.0171Epoch 00002: val_loss improved from 4.85588 to 4.82620, saving model to saved_models/weights.best.from_scratch.hdf5
6680/6680 [==============================] - 279s - loss: 4.8323 - acc: 0.0172 - val_loss: 4.8262 - val_acc: 0.0180
Epoch 4/10
6660/6680 [============================>.] - ETA: 0s - loss: 4.7847 - acc: 0.0180Epoch 00003: val_loss improved from 4.82620 to 4.79278, saving model to saved_models/weights.best.from_scratch.hdf5
6680/6680 [==============================] - 272s - loss: 4.7850 - acc: 0.0181 - val_loss: 4.7928 - val_acc: 0.0287
Epoch 5/10
6660/6680 [============================>.] - ETA: 0s - loss: 4.7509 - acc: 0.0192Epoch 00004: val_loss improved from 4.79278 to 4.76127, saving model to saved_models/weights.best.from_scratch.hdf5
6680/6680 [==============================] - 265s - loss: 4.7508 - acc: 0.0192 - val_loss: 4.7613 - val_acc: 0.0216
Epoch 6/10
6660/6680 [============================>.] - ETA: 0s - loss: 4.7213 - acc: 0.0260Epoch 00005: val_loss improved from 4.76127 to 4.75532, saving model to saved_models/weights.best.from_scratch.hdf5
6680/6680 [==============================] - 262s - loss: 4.7219 - acc: 0.0259 - val_loss: 4.7553 - val_acc: 0.0228
Epoch 7/10
6660/6680 [============================>.] - ETA: 0s - loss: 4.6942 - acc: 0.0269Epoch 00006: val_loss improved from 4.75532 to 4.73605, saving model to saved_models/weights.best.from_scratch.hdf5
6680/6680 [==============================] - 264s - loss: 4.6940 - acc: 0.0268 - val_loss: 4.7361 - val_acc: 0.0251
Epoch 8/10
6660/6680 [============================>.] - ETA: 0s - loss: 4.6723 - acc: 0.0336Epoch 00007: val_loss improved from 4.73605 to 4.73030, saving model to saved_models/weights.best.from_scratch.hdf5
6680/6680 [==============================] - 293s - loss: 4.6720 - acc: 0.0338 - val_loss: 4.7303 - val_acc: 0.0228
Epoch 9/10
6660/6680 [============================>.] - ETA: 0s - loss: 4.6520 - acc: 0.0288Epoch 00008: val_loss improved from 4.73030 to 4.71218, saving model to saved_models/weights.best.from_scratch.hdf5
6680/6680 [==============================] - 265s - loss: 4.6523 - acc: 0.0287 - val_loss: 4.7122 - val_acc: 0.0263
Epoch 10/10
6660/6680 [============================>.] - ETA: 0s - loss: 4.6307 - acc: 0.0348Epoch 00009: val_loss improved from 4.71218 to 4.70778, saving model to saved_models/weights.best.from_scratch.hdf5
6680/6680 [==============================] - 265s - loss: 4.6298 - acc: 0.0352 - val_loss: 4.7078 - val_acc: 0.0251
```

#### Load the Model with the Best Validation Loss
```
model.load_weights('saved_models/weights.best.from_scratch.hdf5')
```
#### Test the Model

Try out your model on the test dataset of dog images. Ensure that your test accuracy is greater than 1%.
```
# get index of predicted dog breed for each image in test set
dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# report test accuracy
test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
```
```
Test accuracy: 3.0000%
```

### Step 4: Use a CNN to Classify Dog Breeds

To reduce training time without sacrificing accuracy, we show you how to train a CNN using transfer learning. In the following step, you will get a chance to use transfer learning to train your own CNN.

#### Obtain Bottleneck Features

```
bottleneck_features = np.load('bottleneck_features/DogVGG16Data.npz')
train_VGG16 = bottleneck_features['train']
valid_VGG16 = bottleneck_features['valid']
test_VGG16 = bottleneck_features['test']
```

#### Model Architecture

The model uses the the pre-trained VGG-16 model as a fixed feature extractor, where the last convolutional output of VGG-16 is fed as input to our model. We only add a global average pooling layer and a fully connected layer, where the latter contains one node for each dog category and is equipped with a softmax.

```
VGG16_model = Sequential()
VGG16_model.add(GlobalAveragePooling2D(input_shape=train_VGG16.shape[1:]))
VGG16_model.add(Dense(133, activation='softmax'))

VGG16_model.summary()
```

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
global_average_pooling2d_6 ( (None, 512)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 133)               68229     
=================================================================
Total params: 68,229
Trainable params: 68,229
Non-trainable params: 0
_________________________________________________________________
```

#### Compile the Model

```
VGG16_modelVGG16_mo .compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
```

#### Train the Model

```
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.VGG16.hdf5', 
                               verbose=1, save_best_only=True)

VGG16_model.fit(train_VGG16, train_targets, 
          validation_data=(valid_VGG16, valid_targets),
          epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)
```
```
Train on 6680 samples, validate on 835 samples
Epoch 1/20
6660/6680 [============================>.] - ETA: 0s - loss: 12.4798 - acc: 0.1120Epoch 00000: val_loss improved from inf to 11.15187, saving model to saved_models/weights.best.VGG16.hdf5
6680/6680 [==============================] - 4s - loss: 12.4745 - acc: 0.1120 - val_loss: 11.1519 - val_acc: 0.1880
Epoch 2/20
6540/6680 [============================>.] - ETA: 0s - loss: 10.5277 - acc: 0.2563Epoch 00001: val_loss improved from 11.15187 to 10.53154, saving model to saved_models/weights.best.VGG16.hdf5
6680/6680 [==============================] - 1s - loss: 10.5163 - acc: 0.2564 - val_loss: 10.5315 - val_acc: 0.2539
Epoch 3/20
6640/6680 [============================>.] - ETA: 0s - loss: 9.9128 - acc: 0.3268Epoch 00002: val_loss improved from 10.53154 to 10.25525, saving model to saved_models/weights.best.VGG16.hdf5
6680/6680 [==============================] - 2s - loss: 9.8970 - acc: 0.3274 - val_loss: 10.2553 - val_acc: 0.2862
Epoch 4/20
6540/6680 [============================>.] - ETA: 0s - loss: 9.6575 - acc: 0.3532Epoch 00003: val_loss improved from 10.25525 to 10.08651, saving model to saved_models/weights.best.VGG16.hdf5
6680/6680 [==============================] - 2s - loss: 9.6397 - acc: 0.3543 - val_loss: 10.0865 - val_acc: 0.3042
Epoch 5/20
6660/6680 [============================>.] - ETA: 0s - loss: 9.5342 - acc: 0.3767Epoch 00004: val_loss improved from 10.08651 to 10.06010, saving model to saved_models/weights.best.VGG16.hdf5
6680/6680 [==============================] - 1s - loss: 9.5359 - acc: 0.3765 - val_loss: 10.0601 - val_acc: 0.3066
Epoch 6/20
6640/6680 [============================>.] - ETA: 0s - loss: 9.3746 - acc: 0.3953Epoch 00005: val_loss improved from 10.06010 to 9.82192, saving model to saved_models/weights.best.VGG16.hdf5
6680/6680 [==============================] - 2s - loss: 9.3747 - acc: 0.3954 - val_loss: 9.8219 - val_acc: 0.3174
Epoch 7/20
6500/6680 [============================>.] - ETA: 0s - loss: 9.1269 - acc: 0.4105Epoch 00006: val_loss improved from 9.82192 to 9.67062, saving model to saved_models/weights.best.VGG16.hdf5
6680/6680 [==============================] - 1s - loss: 9.1396 - acc: 0.4102 - val_loss: 9.6706 - val_acc: 0.3449
Epoch 8/20
6540/6680 [============================>.] - ETA: 0s - loss: 9.0125 - acc: 0.4251Epoch 00007: val_loss improved from 9.67062 to 9.57740, saving model to saved_models/weights.best.VGG16.hdf5
6680/6680 [==============================] - 1s - loss: 9.0021 - acc: 0.4257 - val_loss: 9.5774 - val_acc: 0.3509
Epoch 9/20
6620/6680 [============================>.] - ETA: 0s - loss: 8.9269 - acc: 0.4329Epoch 00008: val_loss improved from 9.57740 to 9.42000, saving model to saved_models/weights.best.VGG16.hdf5
6680/6680 [==============================] - 2s - loss: 8.9166 - acc: 0.4334 - val_loss: 9.4200 - val_acc: 0.3473
Epoch 10/20
6540/6680 [============================>.] - ETA: 0s - loss: 8.7640 - acc: 0.4419Epoch 00009: val_loss did not improve
6680/6680 [==============================] - 1s - loss: 8.7505 - acc: 0.4427 - val_loss: 9.4855 - val_acc: 0.3629
Epoch 11/20
6520/6680 [============================>.] - ETA: 0s - loss: 8.6937 - acc: 0.4505Epoch 00010: val_loss improved from 9.42000 to 9.30081, saving model to saved_models/weights.best.VGG16.hdf5
6680/6680 [==============================] - 1s - loss: 8.6950 - acc: 0.4499 - val_loss: 9.3008 - val_acc: 0.3605
Epoch 12/20
6660/6680 [============================>.] - ETA: 0s - loss: 8.4943 - acc: 0.4572Epoch 00011: val_loss improved from 9.30081 to 9.16110, saving model to saved_models/weights.best.VGG16.hdf5
6680/6680 [==============================] - 1s - loss: 8.4956 - acc: 0.4570 - val_loss: 9.1611 - val_acc: 0.3725
Epoch 13/20
6520/6680 [============================>.] - ETA: 0s - loss: 8.4091 - acc: 0.4681Epoch 00012: val_loss improved from 9.16110 to 9.12366, saving model to saved_models/weights.best.VGG16.hdf5
6680/6680 [==============================] - 1s - loss: 8.3841 - acc: 0.4698 - val_loss: 9.1237 - val_acc: 0.3713
Epoch 14/20
6500/6680 [============================>.] - ETA: 0s - loss: 8.2387 - acc: 0.4755Epoch 00013: val_loss improved from 9.12366 to 8.98274, saving model to saved_models/weights.best.VGG16.hdf5
6680/6680 [==============================] - 1s - loss: 8.2571 - acc: 0.4746 - val_loss: 8.9827 - val_acc: 0.3784
Epoch 15/20
6580/6680 [============================>.] - ETA: 0s - loss: 8.1866 - acc: 0.4860Epoch 00014: val_loss improved from 8.98274 to 8.98247, saving model to saved_models/weights.best.VGG16.hdf5
6680/6680 [==============================] - 1s - loss: 8.1968 - acc: 0.4855 - val_loss: 8.9825 - val_acc: 0.3808
Epoch 16/20
6660/6680 [============================>.] - ETA: 0s - loss: 8.1772 - acc: 0.4868Epoch 00015: val_loss did not improve
6680/6680 [==============================] - 2s - loss: 8.1841 - acc: 0.4864 - val_loss: 9.0132 - val_acc: 0.3772
Epoch 17/20
6520/6680 [============================>.] - ETA: 0s - loss: 8.1335 - acc: 0.4851Epoch 00016: val_loss improved from 8.98247 to 8.94504, saving model to saved_models/weights.best.VGG16.hdf5
6680/6680 [==============================] - 1s - loss: 8.1436 - acc: 0.4846 - val_loss: 8.9450 - val_acc: 0.3772
Epoch 18/20
6620/6680 [============================>.] - ETA: 0s - loss: 7.9939 - acc: 0.4935Epoch 00017: val_loss improved from 8.94504 to 8.80880, saving model to saved_models/weights.best.VGG16.hdf5
6680/6680 [==============================] - 2s - loss: 8.0027 - acc: 0.4928 - val_loss: 8.8088 - val_acc: 0.3988
Epoch 19/20
6640/6680 [============================>.] - ETA: 0s - loss: 7.9015 - acc: 0.5002Epoch 00018: val_loss did not improve
6680/6680 [==============================] - 1s - loss: 7.9194 - acc: 0.4991 - val_loss: 8.9242 - val_acc: 0.3772
Epoch 20/20
6580/6680 [============================>.] - ETA: 0s - loss: 7.9003 - acc: 0.5023Epoch 00019: val_loss did not improve
6680/6680 [==============================] - 1s - loss: 7.8984 - acc: 0.5024 - val_loss: 8.8751 - val_acc: 0.3904
```

#### Load the Model with the Best Validation Loss

```
VGG16_model.load_weights('saved_models/weights.best.VGG16.hdf5')
```

#### Test the Model
Now, we can use the CNN to test how well it identifies breed within our test dataset of dog images. We print the test accuracy below.

```
# get index of predicted dog breed for each image in test set# get in 
VGG16_predictions = [np.argmax(VGG16_model.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG16]

# report test accuracy
test_accuracy = 100*np.sum(np.array(VGG16_predictions)==np.argmax(test_targets, axis=1))/len(VGG16_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
```
```
Test accuracy: 41.0000%
```

#### Predict Dog Breed with the Model

```
from extract_bottleneck_features import *

def VGG16_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_VGG16(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = VGG16_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]
```
```
print VGG16_predict_breed(train_files[0])
```
```
Downloading data from https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
58892288/58889256 [==============================] - 43s    
Bedlington_terrier
[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
  0.  0.  0.  0.  0.  0.  0.]
```
```
print train_targets[0]
```
```
[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
  0.  0.  0.  0.  0.  0.  0.]
```

### Step 5: Create a CNN to Classify Dog Breeds (using Transfer Learning)

You will now use transfer learning to create a CNN that can identify dog breed from images. Your CNN must attain at least 60% accuracy on the test set.

In Step 4, we used transfer learning to create a CNN using VGG-16 bottleneck features. In this section, you must use the bottleneck features from a different pre-trained model. To make things easier for you, we have pre-computed the features for all of the networks that are currently available in Keras:
- VGG-19 bottleneck features
- ResNet-50 bottleneck features
- Inception bottleneck features
- Xception bottleneck features
The files are encoded as such:

Dog{network}Data.npz

where {network}, in the above filename, can be one of VGG19, Resnet50, InceptionV3, or Xception. Pick one of the above architectures, download the corresponding bottleneck features, and store the downloaded file in the bottleneck_features/ folder in the repository.

(IMPLEMENTATION) Obtain Bottleneck Features
In the code block below, extract the bottleneck features corresponding to the train, test, and validation sets by running the following:

`bottleneck_features = np.load('bottleneck_features/Dog{network}Data.npz')`

`train_{network} = bottleneck_features['train']`

`valid_{network} = bottleneck_features['valid']`

`test_{network} = bottleneck_features['test']`

```
### TODO: Obtain bottleneck features from another pre-trained CNN.
bottleneck_features = np.load('bottleneck_features/DogResnet50Data.npz')
train_Resnet50 = bottleneck_features['train']
valid_Resnet50 = bottleneck_features['valid']
test_Resnet50 = bottleneck_features['test']
```
#### (IMPLEMENTATION) Model Architecture
Create a CNN to classify dog breed. At the end of your code cell block, summarize the layers of your model by executing the line:

    `<your model's name>.summary()`
    
**Question 5:** Outline the steps you took to get to your final CNN architecture and your reasoning at each step. Describe why you think the architecture is suitable for the current problem.    

**Answer:**

- `Resnet50_model = Sequential()` to create a neural network model.
- `Resnet50_model.add(GlobalAveragePooling2D(input_shape=train_Resnet50.shape[1:]))` to add a global average pooling layer, using pre-trained `Resnet50` model as input.
- `Resnet50_model.add(Dense(133, activation='softmax')`) to add a final dense layer with node 133, and it will give the probabilities for all nodes.
```
### TODO: Define your architecture.
Resnet50_model = Sequential()
Resnet50_model.add(GlobalAveragePooling2D(input_shape=train_Resnet50.shape[1:]))
Resnet50_model.add(Dense(133, activation='softmax'))

Resnet50_model.summary()
```
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
global_average_pooling2d_7 ( (None, 2048)              0         
_________________________________________________________________
dense_3 (Dense)              (None, 133)               272517    
=================================================================
Total params: 272,517
Trainable params: 272,517
Non-trainable params: 0
_________________________________________________________________
```
#### (IMPLEMENTATION) Compile the Model
```
### TODO: Compile the model.
Resnet50_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
```
#### (IMPLEMENTATION) Train the Model
Train your model in the code cell below. Use model checkpointing to save the model that attains the best validation loss.

You are welcome to augment the training data, but this is not a requirement.
```
### TODO: Train the model.### TODO 
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.Resnet50.hdf5',
                               verbose=1, save_best_only=True)

Resnet50_model.fit(train_Resnet50, train_targets,
                   validation_data=(valid_Resnet50, valid_targets),
                   epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)
```
#### (IMPLEMENTATION) Load the Model with the Best Validation Loss
```
### TODO: Load the model weights with the best validation loss.
Resnet50_model.load_weights('saved_models/weights.best.Resnet50.hdf5')
```

#### (IMPLEMENTATION) Test the Model

Try out your model on the test dataset of dog images. Ensure that your test accuracy is greater than 60%.

```
### TODO: Calculate classification accuracy on the test dataset.
Resnet50_predictions = [np.argmax(Resnet50_model.predict(np.expand_dims(feature, axis=0))) for feature in test_Resnet50]

test_accuracy = 100*np.sum(np.array(Resnet50_predictions)==np.argmax(test_targets, axis=1))/len(Resnet50_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
```
```
Test accuracy: 80.0000%
```

#### (IMPLEMENTATION) Predict Dog Breed with the Model

Write a function that takes an image path as input and returns the dog breed (Affenpinscher, Afghan_hound, etc) that is predicted by your model.

Similar to the analogous function in Step 5, your function should have three steps:

- Extract the bottleneck features corresponding to the chosen CNN model.
- Supply the bottleneck features as input to the model to return the predicted vector. Note that the argmax of this prediction vector gives the index of the predicted dog breed.
- Use the dog_names array defined in Step 0 of this notebook to return the corresponding breed.

The functions to extract the bottleneck features can be found in `extract_bottleneck_features.py`, and they have been imported in an earlier code cell. To obtain the bottleneck features corresponding to your chosen CNN architecture, you need to use the function

extract_{network}

where {network}, in the above filename, should be one of VGG19, Resnet50, InceptionV3, or Xception.

```
### TODO: Write a function that takes a path to an image as input
### and returns the dog breed that is predicted by the model.
def Resnet50_predict_breed(img_path):
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    predicted_vector = Resnet50_model.predict(bottleneck_feature)
    return dog_names[np.argmax(predicted_vector)]
```

### Step 6: Write your Algorithm

Write an algorithm that accepts a file path to an image and first determines whether the image contains a human, dog, or neither. Then,

- if a **dog** is detected in the image, return the predicted breed.
- if a **human** is detected in the image, return the resembling dog breed.
- if **neither** is detected in the image, provide output that indicates an error.

You are welcome to write your own functions for detecting humans and dogs in images, but feel free to use the `face_detector` and `dog_detector` functions developed above. You are **required** to use your CNN from Step 5 to predict dog breed.

Some sample output for our algorithm is provided below, but feel free to design your own user experience!

Sample Human Output

#### (IMPLEMENTATION) Write your Algorithm¶
