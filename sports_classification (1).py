#!/usr/bin/env python
# coding: utf-8

# In[47]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import random
import tensorflow as tf
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import load_img
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPool2D, BatchNormalization, GaussianNoise
from google.colab import files
from tqdm.notebook import tqdm
from tensorflow.keras import layers
from PIL import Image
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
from google.colab import drive
from google.colab import files


# In[2]:


drive.mount('/content/drive/')
Trainingset = 'drive/MyDrive/sports/Train'
Testingset = 'drive/MyDrive/sports/Test'


# In[3]:



def data(directory):
  path = []
  L = []

  for i in os.listdir(directory):
      PathOfImg = os.path.join(directory, i)
      path.append(PathOfImg)
      if i.startswith("Basketball"):
        L.append(0)
      elif i.startswith("Football"):
        L.append(1)
      elif i.startswith("Rowing"):
        L.append(2)
      elif i.startswith("Swimming"):
        L.append(3)
      elif i.startswith("Tennis"):
        L.append(4)
      else:
        L.append(5)
  return path, L


# In[4]:


training = pd.DataFrame() 
training['image'], training['label'] = data(Trainingset)

testing = pd.DataFrame()   
testing['image'], testing['label'] = data(Testingset)

training = training.sample(frac=1).reset_index(drop=True) 


# In[ ]:


plt.figure(figsize=(10,10))
j = training[training['label']==0]['image']
begin = random.randint(0, len(j))
z = j[begin:begin+9]

for i, file in enumerate(z):
  plt.subplot(3,3, i+1)
  image = load_img(file)
  image = np.array(image)
  plt.imshow(image)
  plt.title('Basketball')
  plt.axis('off')


# In[ ]:



plt.figure(figsize=(10,10))

j = training[training['label']==1]['image']

begin = random.randint(0, len(j))

z = j[begin:begin+9]

for i, file in enumerate(z):
  plt.subplot(3,3, i+1)
  image = load_img(file)
  image = np.array(image)
  plt.imshow(image)
  plt.title('Football')
  plt.axis('off')


# In[ ]:



plt.figure(figsize=(10,10))

j = training[training['label']==2]['image']

begin = random.randint(0, len(j))

z = j[begin:begin+9]

for i, file in enumerate(z):
  plt.subplot(3,3, i+1)
  image = load_img(file)
  image = np.array(image)
  plt.imshow(image)
  plt.title('Rowing')
  plt.axis('off')


# In[ ]:



plt.figure(figsize=(10,10))

j = training[training['label']==3]['image']

begin = random.randint(0, len(j))

z = j[begin:begin+9]

for i, file in enumerate(z):
  plt.subplot(3,3, i+1)
  image = load_img(file)
  image = np.array(image)
  plt.imshow(image)
  plt.title('Swimming')
  plt.axis('off')


# In[ ]:


plt.figure(figsize=(10,10))

j = training[training['label']==4]['image']

begin = random.randint(0, len(j))

z = j[begin:begin+9]

for i, file in enumerate(z):
  plt.subplot(3,3, i+1)
  iamge = load_img(file)
  iamge = np.array(iamge)
  plt.imshow(iamge)
  plt.title('Tennis')
  plt.axis('off')


# In[ ]:


plt.figure(figsize=(10,10))

j = training[training['label']==5]['image']

begin = random.randint(0, len(j))

z = j[begin:begin+9]

for i, file in enumerate(z):
  plt.subplot(3,3, i+1)
  iamge = load_img(file)
  iamge = np.array(iamge)
  plt.imshow(iamge)
  plt.title('Yoga')
  plt.axis('off')


# In[ ]:


sns.countplot(training['label'])


# In[ ]:


sns.countplot(testing['label'])


# In[5]:


def featureExtruction(images):
    f = []
    for image in tqdm(images):
        img = load_img(image, grayscale=True)
        img = img.resize((128, 128), Image.ANTIALIAS)
        img = np.array(img)
        f.append(img)
        
    f = np.array(f)
    new = f.reshape(len(f), 128, 128, 1)
    return f


# In[6]:


featurestrain = featureExtruction(training['image'])


# In[7]:


featurestest = featureExtruction(testing['image'])


# In[8]:


x_train = featurestrain/255.0
x_test = featurestest/255.0


# In[9]:


encode = LabelEncoder()
encode.fit(training['label'])
y_train = encode.transform(training['label'])
y_test = encode.transform(testing['label'])


# In[10]:


y_train = to_categorical(y_train, num_classes=6)
y_test = to_categorical(y_test, num_classes=6)


# # *CNN*

# In[39]:



CNNmodel = Sequential([

    # adding the first layer
    Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[128, 128, 1]),
    MaxPool2D(pool_size=2, strides=2),

    # adding the second layer
    Conv2D(filters=32, kernel_size=3, activation='relu'),
    MaxPool2D(pool_size=2, strides=2),

    # adding the third layer
    Flatten(),

    # full connection
    Dense(units=128, activation='relu'),

    # output layer
    Dense(units=6, activation='sigmoid')

])

print(CNNmodel.summary())


# In[43]:


# compiling the model
CNNmodel.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[45]:


ModelHistory = CNNmodel.fit(x=x_train, y=y_train, epochs=20, validation_data=(x_test, y_test))


# In[ ]:



CNNmodel.save('/content/drive/MyDrive/sports/cnn.h5')


# # ***VGG***

# In[48]:


model = Sequential()

input_shape = (128,128,1)
output_class = 6
# model layers
model.add(Conv2D(128, kernel_size=(3,3), activation='relu', input_shape=input_shape))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())

# fully connected layers
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))

# output layer
model.add(Dense(output_class, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
model.summary()


# In[ ]:


plot_model(model)


# In[49]:


#Training
ModelHistory = model.fit(x=x_train, y=y_train, batch_size=128, epochs=15, validation_data=(x_test, y_test))


# ## Plot the Relation between Validation and Accuracy 
# 
# 

# In[50]:


accuracy = ModelHistory.history['accuracy']
validationAcc = ModelHistory.history['val_accuracy']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'b', label='Training Accuracy')
plt.plot(epochs, validationAcc, 'r', label='Validation Accuracy')
plt.title('Accuracy Graph')
plt.legend()
plt.figure()

loss = ModelHistory.history['loss']
validationLoss = ModelHistory.history['val_loss']
plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, validationLoss, 'r', label='Validation Loss')
plt.title('Loss Graph')
plt.legend()
plt.show()


# In[ ]:


model.save("/content/drive/My Drive/DPprojects/sports/model.h5")


# # ***Prediction***
# 

# **cnn**

# In[51]:


def prediction():
  imgg = files.upload()
  inputImg = (featureExtruction(imgg))/255.
  reshapeImg = np.squeeze(inputImg, axis=(0,))
  i = reshapeImg
  pre = []

  reshaped_image = i.reshape((128, 128))

  pre.append(i)

  pre = np.array(pre)

  predictions = CNNmodel.predict(pre)

  imgClasses = np.argmax(predictions, axis = 1)
  pred = labels[int(imgClasses)]

  plt.imshow(reshaped_image)
  plt.title(f"Prediction: {pred}")
  plt.show()


# **cnn**

# In[56]:




def pred_from_test():
  use_samples = []
  pred = []
  samples_to_predict = []


  n = int(input("Enter number of sample test that you want to predict : "))
 
  for i in range(0, n):
    
    use_samples.append(i)
  
 
  for sample in use_samples:
    reshaped_image = x_test[sample].reshape((128, 128))
    samples_to_predict.append(x_test[sample])
    
    
    plt.imshow(reshaped_image)
    plt.show()

  
  samples_to_predict = np.array(samples_to_predict)


  predictions = CNNmodel.predict(samples_to_predict)

  
  classes = np.argmax(predictions, axis = 1)  
  

  for i in range(0,n):
    pred.append(labels[int(classes[i])])
    
    data = {'prediction': pred}
   
     
      
  df = pd.DataFrame(data)
  df.to_csv('cnn.csv')
    
       


# **VGG**

# In[57]:


def prediction2():
  imgg = files.upload()
  inputImg = (featureExtruction(imgg))/255.
  reshapeImg = np.squeeze(inputImg, axis=(0,))
  i = reshapeImg
  pre = []

  reshaped_image = i.reshape((128, 128))

  pre.append(i)

  pre = np.array(pre)

  predictions = model.predict(pre)

  imgClasses = np.argmax(predictions, axis = 1)
  pred = labels[int(imgClasses)]

  plt.imshow(reshaped_image)
  plt.title(f"Prediction: {pred}")
  plt.show()


# **VGG**

# In[58]:




def pred_from_test2():
  use_samples = []
  pred = []
  samples_to_predict = []


  n = int(input("Enter number of sample test that you want to predict : "))
 
  for i in range(0, n):
    
    use_samples.append(i)
  
 
  for sample in use_samples:
    reshaped_image = x_test[sample].reshape((128, 128))
    samples_to_predict.append(x_test[sample])
    
    
    plt.imshow(reshaped_image)
    plt.show()

  
  samples_to_predict = np.array(samples_to_predict)


  predictions = model.predict(samples_to_predict)

  
  classes = np.argmax(predictions, axis = 1)  
  

  for i in range(0,n):
    pred.append(labels[int(classes[i])])
    
    data = {'prediction': pred}
   
     
      
  df = pd.DataFrame(data)
  df.to_csv('vgg.csv')
    
       


# In[ ]:


labels = ['Basketball', 'Football','Rowing','Swimming','Tennis','Yoga']



opt = int(input("press one for predict 1 image and 2 for predict multiple images from testing set: "))
if opt == 1:
  print('cnn prediction ')
  prediction()
  print('vgg prediction ')
  prediction2()
else:
  print('cnn prediction ')
  pred_from_test()
  print('vgg prediction ')
  pred_from_test2()

