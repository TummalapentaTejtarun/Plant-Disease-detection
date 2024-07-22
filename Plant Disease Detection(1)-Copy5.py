#!/usr/bin/env python
# coding: utf-8

# ## Import necessary libraries

# In[1]:


import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt


# ## Data Preprocessing

# #### Load the dataset

# In[2]:


BATCH_SIZE = 32
IMAGE_SIZE = 256
CHANNELS=3
EPOCHS=20


# In[3]:


dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "Desktop/PlantVillage4/PlantVillage",
    seed=123,
    shuffle=True,
    image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE
)


# In[4]:


class_names = dataset.class_names
class_names


# #### Explore the dataset

# In[5]:


for image_batch, labels_batch in dataset.take(1):
    print(image_batch.shape)
    print(labels_batch.numpy())


# In[6]:


plt.figure(figsize=(18, 10))
for image_batch, labels_batch in dataset.take(1):
    for i in range(4):
        ax = plt.subplot(1, 4, i + 1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        plt.title(class_names[labels_batch[i]])
        plt.axis("off")


# In[7]:


len(dataset)


# #### Split the dataset into train, validation, test sets

# In[8]:


def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1
    
    ds_size = len(ds)
    
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds


# In[9]:


train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)


# In[10]:


len(train_ds)


# In[11]:


len(val_ds)


# In[12]:


len(test_ds)


# In[13]:


train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)


# #### Creating a Layer for Resizing and Normalization
# Before we feed our images to network, we should be resizing it to the desired size. Moreover, to improve model performance, we should normalize the image pixel value (keeping them in range 0 and 1 by dividing by 256). This should happen while training as well as inference. Hence we can add that as a layer in our Sequential Model.

# In[14]:


resize_and_rescale = tf.keras.Sequential([
  layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
  layers.Rescaling(1./255),
])


# #### Data Augmentation

# In[15]:


data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
])


# In[16]:


train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y)
).prefetch(buffer_size=tf.data.AUTOTUNE)


# # Model Building

# ### 1. Customized CNN model

# In[17]:


input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 15

model = models.Sequential([
    resize_and_rescale,
    layers.Conv2D(64, kernel_size = (3,3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax'),
])

model.build(input_shape=input_shape)


# In[18]:


input_shape


# In[19]:


model.summary()


# In[20]:


model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)


# In[21]:


model.compile


# #### Train the model

# In[23]:


history = model.fit(
    train_ds,
    batch_size=BATCH_SIZE,
    validation_data=val_ds,
    verbose=1,
    epochs=20,
)


# #### Plot the Training and Validation accuracy and loss

# In[24]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']


# In[25]:


plt.figure(figsize=(4, 4))
plt.plot(range(EPOCHS), acc, label='Training Accuracy')
plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')


# In[26]:


plt.figure(figsize=(4, 4))
plt.plot(range(EPOCHS), loss, label='Training Loss')
plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# #### Test the model on test set

# #### Function 

# In[27]:


import numpy as np
def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence


# #### Testing

# In[28]:


plt.figure(figsize=(20, 15))
for images, labels in test_ds.take(1):
    for i in range(4):
        ax = plt.subplot(1, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        predicted_class, confidence = predict(model, images[i].numpy())
        actual_class = class_names[labels[i]] 
        if(actual_class==predicted_class):
            medicine = {
                "Pepper__bell___Bacterial_spot": "copper-based products","Pepper__bell___healthy": "Good plant care practices","Potato___Early_blight": "Chlorothalonil","Potato___Late_blight": "Metalaxyl-M","Potato___healthy": "Crop rotation and resistant varieties","Tomato_Bacterial_spot": "Streptomycin","Tomato_Early_blight": "Azoxystrobin","Tomato_Late_blight": "Mancozeb","Tomato_Leaf_Mold": "Serenade","Tomato_Septoria_leaf_spot": "Biofungicides","Tomato_Spider_mites_Two_spotted_spider_mite": "Horticultural Oils","Tomato__Target_Spot": "Bacillus subtilis","Tomato__Tomato_YellowLeaf__Curl_Virus": "Sanitation","Tomato__Tomato_mosaic_virus": "Remove Infected Plants","Tomato_healthy": "Proper watering and fertilization"}
            color = medicine.get(predicted_class, "Unknown")
        plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class},\n Medicine: {color}.\n Confidence: {confidence}%")
        plt.axis("off")


# #### Evaluate the model

# In[29]:


model.evaluate(test_ds)


# ### 2. ResNet50V2 Model

# In[30]:


input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 15

model1 = models.Sequential(resize_and_rescale)
base_model1 = tf.keras.applications.ResNet50V2(
    include_top=False,
    input_shape = (256,256,3),
    pooling='avg',
    classes=15,
    weights="imagenet"
)

for layer in base_model1.layers:
    layer.trainable = False
    
model1.add(base_model1)
model1.add(layers.Flatten())
model1.add(layers.Dense(128,activation='relu'))
model1.add(layers.Dense(64,activation='relu'))
model1.add(layers.Dense(256,activation='relu'))
model1.add(layers.Dense(64,activation='relu'))
model1.add(layers.Dense(n_classes, kernel_regularizer = tf.keras.regularizers.l2(0.01), activation='softmax'))

model1.build(input_shape=input_shape)


# In[31]:


model1.summary()


# In[32]:


model1.compile(optimizer='adam',loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['accuracy'])


# #### Training

# In[33]:


history1=model1.fit(
  train_ds,
  validation_data=val_ds,
  epochs=20)


# #### Plot

# In[34]:


acc = history1.history['accuracy']
val_acc = history1.history['val_accuracy']

loss = history1.history['loss']
val_loss = history1.history['val_loss']


# In[35]:


plt.figure(figsize=(4, 4))
plt.plot(range(EPOCHS), acc, label='Training Accuracy')
plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy of ResNet50V2')


# In[36]:


plt.figure(figsize=(4, 4))
plt.plot(range(EPOCHS), loss, label='Training Loss')
plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss of ResNet50V2')
plt.show()


# #### Testing

# In[37]:


plt.figure(figsize=(20, 15))
for images, labels in test_ds.take(1):
    for i in range(4):
        ax = plt.subplot(1, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        predicted_class, confidence = predict(model, images[i].numpy())
        actual_class = class_names[labels[i]] 
        if(actual_class==predicted_class):
            medicine = {
                "Pepper__bell___Bacterial_spot": "copper-based products","Pepper__bell___healthy": "Good plant care practices","Potato___Early_blight": "Chlorothalonil","Potato___Late_blight": "Metalaxyl-M","Potato___healthy": "Crop rotation and resistant varieties","Tomato_Bacterial_spot": "Streptomycin","Tomato_Early_blight": "Azoxystrobin","Tomato_Late_blight": "Mancozeb","Tomato_Leaf_Mold": "Serenade","Tomato_Septoria_leaf_spot": "Biofungicides","Tomato_Spider_mites_Two_spotted_spider_mite": "Horticultural Oils","Tomato__Target_Spot": "Bacillus subtilis","Tomato__Tomato_YellowLeaf__Curl_Virus": "Sanitation","Tomato__Tomato_mosaic_virus": "Remove Infected Plants","Tomato_healthy": "Proper watering and fertilization"}
            color = medicine.get(predicted_class, "Unknown")
        plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class},\n Medicine: {color}.\n Confidence: {confidence}%")
        plt.axis("off")


# #### Evaluate

# In[38]:


model1.evaluate(test_ds)


# ### 3. MobileNetV2 Model

# In[39]:


input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 15

model2 = models.Sequential(resize_and_rescale)

base_model2 = tf.keras.applications.MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(256,256,3),
    pooling='max',
    classes=15,
)

for layer in base_model2.layers:
    layer.trainable = False

model2.add(base_model2)
model2.add(layers.Flatten())
model2.add(layers.Dense(128,activation='relu'))
model2.add(layers.Dense(64,activation='relu'))
model2.add(layers.Dense(128,activation='relu'))
model2.add(layers.Dense(64,activation='relu'))
model2.add(layers.Dense(n_classes, activation='softmax'))

model2.build(input_shape=input_shape)


# In[40]:


model2.summary()


# In[41]:


model2.compile(optimizer='adam',loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['accuracy'])


# #### Training

# In[42]:


history2 = model2.fit(
  train_ds,
  validation_data=val_ds,
  epochs=20)


# #### Plot

# In[43]:


acc = history2.history['accuracy']
val_acc = history2.history['val_accuracy']

loss = history2.history['loss']
val_loss = history2.history['val_loss']


# In[44]:


plt.figure(figsize=(4, 4))
plt.plot(range(EPOCHS), acc, label='Training Accuracy')
plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy of MobileNetV2')


# In[45]:


plt.figure(figsize=(4, 4))
plt.plot(range(EPOCHS), loss, label='Training Loss')
plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss of MobileNetV2')
plt.show()


# #### Testing

# In[46]:


plt.figure(figsize=(20, 15))
for images, labels in test_ds.take(1):
    for i in range(4):
        ax = plt.subplot(1, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        predicted_class, confidence = predict(model, images[i].numpy())
        actual_class = class_names[labels[i]] 
        if(actual_class==predicted_class):
            medicine = {
                "Pepper__bell___Bacterial_spot": "copper-based products","Pepper__bell___healthy": "Good plant care practices","Potato___Early_blight": "Chlorothalonil","Potato___Late_blight": "Metalaxyl-M","Potato___healthy": "Crop rotation and resistant varieties","Tomato_Bacterial_spot": "Streptomycin","Tomato_Early_blight": "Azoxystrobin","Tomato_Late_blight": "Mancozeb","Tomato_Leaf_Mold": "Serenade","Tomato_Septoria_leaf_spot": "Biofungicides","Tomato_Spider_mites_Two_spotted_spider_mite": "Horticultural Oils","Tomato__Target_Spot": "Bacillus subtilis","Tomato__Tomato_YellowLeaf__Curl_Virus": "Sanitation","Tomato__Tomato_mosaic_virus": "Remove Infected Plants","Tomato_healthy": "Proper watering and fertilization"}
            color = medicine.get(predicted_class, "Unknown")
        plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class},\n Medicine: {color}.\n Confidence: {confidence}%")
        plt.axis("off")


# #### Evaluate

# In[47]:


model2.evaluate(test_ds)


# ### 4. InceptionV3

# In[48]:


input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 15

model3 = models.Sequential(resize_and_rescale)

base_model3 = tf.keras.applications.InceptionV3(
    include_top=False,
    weights="imagenet",
    input_shape=(256,256,3),
    pooling='max',
    classes=15,
)

for layer in base_model3.layers:
    layer.trainable = False
    
model3.add(base_model3)
model3.add(layers.Flatten())
model3.add(layers.Dense(128,activation='relu'))
model3.add(layers.Dense(64,activation='relu'))
model3.add(layers.Dense(256,activation='relu'))
model3.add(layers.Dense(64,activation='relu'))
model3.add(layers.Dense(n_classes, kernel_regularizer = tf.keras.regularizers.l2(0.01), activation='softmax'))

model3.build(input_shape=input_shape)


# In[49]:


model3.summary()


# In[50]:


model3.compile(optimizer='adam',loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['accuracy'])


# #### Training

# In[51]:


history3=model3.fit(
  train_ds,
  validation_data=val_ds,
  epochs=20)


# #### Plot

# In[52]:


acc = history3.history['accuracy']
val_acc = history3.history['val_accuracy']

loss = history3.history['loss']
val_loss = history3.history['val_loss']


# In[53]:


plt.figure(figsize=(4, 4))
plt.plot(range(EPOCHS), acc, label='Training Accuracy')
plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy of InceptionV3')


# In[54]:


plt.figure(figsize=(4, 4))
plt.plot(range(EPOCHS), loss, label='Training Loss')
plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss of InceptionV3')
plt.show()


# #### Testing

# In[55]:


plt.figure(figsize=(20, 15))
for images, labels in test_ds.take(1):
    for i in range(4):
        ax = plt.subplot(1, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        predicted_class, confidence = predict(model, images[i].numpy())
        actual_class = class_names[labels[i]] 
        if(actual_class==predicted_class):
            medicine = {
                "Pepper__bell___Bacterial_spot": "copper-based products","Pepper__bell___healthy": "Good plant care practices","Potato___Early_blight": "Chlorothalonil","Potato___Late_blight": "Metalaxyl-M","Potato___healthy": "Crop rotation and resistant varieties","Tomato_Bacterial_spot": "Streptomycin","Tomato_Early_blight": "Azoxystrobin","Tomato_Late_blight": "Mancozeb","Tomato_Leaf_Mold": "Serenade","Tomato_Septoria_leaf_spot": "Biofungicides","Tomato_Spider_mites_Two_spotted_spider_mite": "Horticultural Oils","Tomato__Target_Spot": "Bacillus subtilis","Tomato__Tomato_YellowLeaf__Curl_Virus": "Sanitation","Tomato__Tomato_mosaic_virus": "Remove Infected Plants","Tomato_healthy": "Proper watering and fertilization"}
            color = medicine.get(predicted_class, "Unknown")
        plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class},\n Medicine: {color}.\n Confidence: {confidence}%")
        plt.axis("off")


# #### Evaluate

# In[56]:


model3.evaluate(test_ds)


# ### 5. Ensemble Model

# In[57]:


from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Average

models = [model2, model3]
model_input = Input(shape=(256,256,3))
model_outputs = [model(model_input) for model in models]
ensemble_output = Average()(model_outputs)
ensemble_model = Model(inputs=model_input, outputs=ensemble_output, name='ensemble')


# In[58]:


ensemble_model.compile(optimizer='adam',loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['accuracy'])


# #### Training

# In[59]:


history4=ensemble_model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=20)


# #### Plot

# In[60]:


acc = history4.history['accuracy']
val_acc = history4.history['val_accuracy']

loss = history4.history['loss']
val_loss = history4.history['val_loss']


# In[61]:


plt.figure(figsize=(4, 4))
plt.plot(range(EPOCHS), acc, label='Training Accuracy')
plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy of Ensemble model')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')


# In[62]:


plt.figure(figsize=(4, 4))
plt.plot(range(EPOCHS), loss, label='Training Loss')
plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss of Ensemble model')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.show()


# #### Testing

# In[63]:


plt.figure(figsize=(20, 15))
for images, labels in test_ds.take(1):
    for i in range(4):
        ax = plt.subplot(1, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        predicted_class, confidence = predict(model, images[i].numpy())
        actual_class = class_names[labels[i]] 
        if(actual_class==predicted_class):
            medicine = {
                "Pepper__bell___Bacterial_spot": "copper-based products","Pepper__bell___healthy": "Good plant care practices","Potato___Early_blight": "Chlorothalonil","Potato___Late_blight": "Metalaxyl-M","Potato___healthy": "Crop rotation and resistant varieties","Tomato_Bacterial_spot": "Streptomycin","Tomato_Early_blight": "Azoxystrobin","Tomato_Late_blight": "Mancozeb","Tomato_Leaf_Mold": "Serenade","Tomato_Septoria_leaf_spot": "Biofungicides","Tomato_Spider_mites_Two_spotted_spider_mite": "Horticultural Oils","Tomato__Target_Spot": "Bacillus subtilis","Tomato__Tomato_YellowLeaf__Curl_Virus": "Sanitation","Tomato__Tomato_mosaic_virus": "Remove Infected Plants","Tomato_healthy": "Proper watering and fertilization"}
            color = medicine.get(predicted_class, "Unknown")
        plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class},\n Medicine: {color}.\n Confidence: {confidence}%")
        plt.axis("off")


# #### Evaluate

# In[64]:


ensemble_model.evaluate(test_ds)


# ### 6. NasNetMobile

# In[65]:


print(type(models))
if type(models) == list:
    import tensorflow.keras as keras
    models = keras.models
input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 15

model5 = models.Sequential(resize_and_rescale)

base_model5 = tf.keras.applications.NASNetMobile(
    include_top=False,
    weights="imagenet",
    input_shape=(256,256,3),
    pooling='max',
    classes=15,
)

for layer in base_model5.layers:
    layer.trainable = False
    
model5.add(base_model5)
model5.add(layers.Flatten())
model5.add(layers.Dense(128,activation='relu'))
model5.add(layers.Dense(64,activation='relu'))
model5.add(layers.Dropout(0.3))
model5.add(layers.Dense(256,activation='relu'))
model5.add(layers.Dense(64,activation='relu'))
model5.add(layers.Dense(n_classes, activation='softmax'))

model5.build(input_shape=input_shape)


# In[66]:


model5.summary()


# In[67]:


model5.compile(optimizer='adam',loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['accuracy'])


# In[68]:


history5=model5.fit(
  train_ds,
  validation_data=val_ds,
  epochs=20)


# #### Plot

# In[69]:


acc = history5.history['accuracy']
val_acc = history5.history['val_accuracy']

loss = history5.history['loss']
val_loss = history5.history['val_loss']


# In[70]:


plt.figure(figsize=(4, 4))
plt.plot(range(EPOCHS), acc, label='Training Accuracy')
plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy of NasNetMobile')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')


# In[71]:


plt.figure(figsize=(4, 4))
plt.plot(range(EPOCHS), loss, label='Training Loss')
plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss of NasNetMobile')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


# #### Testing

# In[72]:


plt.figure(figsize=(20, 15))
for images, labels in test_ds.take(1):
    for i in range(4):
        ax = plt.subplot(1, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        predicted_class, confidence = predict(model, images[i].numpy())
        actual_class = class_names[labels[i]] 
        if(actual_class==predicted_class):
            medicine = {
                "Pepper__bell___Bacterial_spot": "copper-based products","Pepper__bell___healthy": "Good plant care practices","Potato___Early_blight": "Chlorothalonil","Potato___Late_blight": "Metalaxyl-M","Potato___healthy": "Crop rotation and resistant varieties","Tomato_Bacterial_spot": "Streptomycin","Tomato_Early_blight": "Azoxystrobin","Tomato_Late_blight": "Mancozeb","Tomato_Leaf_Mold": "Serenade","Tomato_Septoria_leaf_spot": "Biofungicides","Tomato_Spider_mites_Two_spotted_spider_mite": "Horticultural Oils","Tomato__Target_Spot": "Bacillus subtilis","Tomato__Tomato_YellowLeaf__Curl_Virus": "Sanitation","Tomato__Tomato_mosaic_virus": "Remove Infected Plants","Tomato_healthy": "Proper watering and fertilization"}
            color = medicine.get(predicted_class, "Unknown")
        plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class},\n Medicine: {color}.\n Confidence: {confidence}%")
        plt.axis("off")


# #### Evaluate

# In[73]:


model5.evaluate(test_ds)

