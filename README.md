# practice14-NN
Neural Network to distinguish handwriting numbers from MNIST dataset

使用標準神經網路(Neural Network)辨別手寫數字(MNIST)
1. 初始準備 
   - 讀入tensorflow環境
        %env KERAS_BACKEND=tensorflow
2. 讀入MNIST資料庫
   - 輸入格式整理: 整理成一維
   - 輸出格式整理: 載入np_utils將答案轉換成1-hot encoding格式
3. 打造神經網路
   - 決定神經網路架構並讀入相關套件
   - 總共有2個隱藏層(hidden layers)
   - 每個隱藏層有500個神經元
   - Activation Function唯一指名sigmoid
4. 檢視成果
   - model.summary()
5. 訓練神經網路
   - 將x_train, y_train丟入模型中訓練 
        model.fit(x_train,y_train,batch_size=100,epochs=20)
   - batch_size 每次訓練的資料量
   - epochs 訓練次數
6. 試用成果
   - predict = model.predict_classes(x_test)
7. 將訓練好的神經網路分別存下來
   - 存本體 
        model_json = model.to_json()
        open('XXXX.json','w').write(model.json)
   - 存權重 
        model.save_weights('XXXX.h5')
    
