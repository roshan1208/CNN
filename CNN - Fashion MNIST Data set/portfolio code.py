#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from tensorflow.keras.datasets import fashion_mnist


# In[4]:


(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()


# In[5]:


X_train.shape


# In[6]:


from matplotlib.image import imread


# In[8]:


X_test[0]


# In[10]:


plt.imshow(X_train[1])


# In[12]:


y_test


# In[13]:


X_test.max()


# In[15]:


X_test = X_test/255
X_train = X_train/255


# In[16]:


len(X_train)


# In[17]:


len(X_test)


# In[18]:


X_test = X_test.reshape(10000, 28, 28, 1)
X_train = X_train.reshape(60000, 28, 28, 1)


# In[19]:


from tensorflow.keras.utils import to_categorical


# In[20]:


y_cat_test = to_categorical(y_test, 10)
y_cat_train = to_categorical(y_train, 10)


# In[22]:


X_test.shape


# In[23]:


X_train.shape


# In[25]:


y_cat_test.shape


# In[55]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D, MaxPool2D, Flatten


# In[56]:


model = Sequential()
model.add(Conv2D(filters=32 , kernel_size=(4,4), input_shape = (28,28,1), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(10, activation='softmax'))

model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam', metrics = ['accuracy'])


# In[57]:


from tensorflow.keras.callbacks import EarlyStopping


# In[58]:


early = EarlyStopping(monitor='val_loss', patience=2)


# In[59]:


model.fit(x = X_train, y= y_cat_train, epochs = 12, validation_data=(X_test, y_cat_test), callbacks = [early])


# In[60]:


loss = pd.DataFrame(model.history.history)


# In[61]:


loss


# In[62]:


loss[['loss', 'val_loss']].plot()


# In[63]:


loss[['accuracy', 'val_accuracy']].plot()


# In[64]:


predd = model.predict_classes(X_test)


# In[65]:


y_cat_test


# In[66]:


predd


# In[67]:


y_test


# In[68]:


from sklearn.metrics import confusion_matrix, classification_report


# In[69]:


print(confusion_matrix(y_test, predd))


# In[70]:


print(classification_report(y_test, predd))


# In[71]:


plt.figure(figsize = (12,6))
sns.heatmap(confusion_matrix(y_test, predd), annot = True)


# # lets try to predd new image

# In[72]:


new_image = X_test[12].reshape(1,28,28,1)


# In[73]:


model.predict_classes(new_image)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




