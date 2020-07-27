#!/usr/bin/env python
# coding: utf-8

# # NN手寫辨識

# In[1]:


# 架構keras環境
get_ipython().run_line_magic('env', 'KERAS_BACKEND=tensorflow')


# In[3]:


# 讀入數據分析套件
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt


# In[8]:


# 讀入mnist資料庫
from keras.datasets import mnist
(x_train,y_train), (x_test,y_test) = mnist.load_data()


# In[9]:


# 輸入格式整理
# 調整前資料形狀
print('x_train:',x_train.shape)
print('x_test:',x_test.shape)

# 標準神經網路只接受一維，因此要改形狀
print(28*28)
x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000,784)


# In[10]:


# 調整後資料形狀
print('x_train:',x_train.shape)
print('x_test:',x_test.shape)


# In[12]:


# 輸出格式調整
# 改成one-hot encoding格式
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train,10)
y_test = np_utils.to_categorical(y_test,10)


# In[13]:


# 決定神經網路架構
# 讀入相關套件
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD


# In[15]:


# 開啟空的神經網路
model = Sequential()

# 第一層
# 要手動輸入input_dim
model.add(Dense(500,input_dim=784))
model.add(Activation('sigmoid'))

# 第二層
model.add(Dense(500))
model.add(Activation('sigmoid'))

# 最後一層
model.add(Dense(10))
model.add(Activation('softmax'))


# In[17]:


# 組裝
# metrics=['accuracy']是方便檢視準確度
model.compile(loss='mse',
             optimizer=SGD(lr=0.1),
             metrics=['accuracy'])


# In[18]:


# 檢視成果
model.summary()


# In[19]:


# 訓練神經網路
model.fit(x_train, y_train, batch_size=100, epochs=20)


# In[20]:


# 檢視成果
score = model.evaluate(x_test, y_test)
print('loss:',score[0])
print('accuracy:',score[1])


# In[21]:


# 用互動式介面試用結果
from ipywidgets import interact_manual


# In[25]:


# 用訓練過的模型進行判讀預測
predict = model.predict_classes(x_test)

# 寫一個畫出題目與模型判讀結果的函數
def test(num):
    plt.imshow(x_test[num].reshape(28,28),cmap='Greys')
    print('神經網路判斷為:',predict[num])


# In[26]:


# 用互動套件配合test函數測試結果
interact_manual(test, num=(0,9999))


# In[27]:


# 儲存
model.json = model.to_json()
open('handwriting_model_nn.json','w').write(model.json)

# 儲存權重
model.save_weights('handwriting_model_weights.h5')


# In[ ]:




