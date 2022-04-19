import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import sparse_categorical_crossentropy
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LeakyReLU
LeakyReLU = LeakyReLU(alpha=0.1)
import warnings
warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", None)
from fast_ml.model_development import train_valid_test_split

class authentication:

  def __init__(self,
               data_X,
               data_y,
               epoch:int,
               batch_size:int):
        self.__data_X = data_X   # extractFeatures result - independent variables
        self.__data_y = data_y   # extractFeatures result - dependent variables
        self.__epoch = epoch
        self.__batch_size = batch_size

  def __model (self,neurons):
      model = Sequential()
      #Step 1 - Convolution
      model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(10,6, 1)))
      #Step 2 - Pooling
      model.add(MaxPooling2D((2, 2)))
      #Step 3 - Flattening
      model.add(Flatten())
      #Step 4 - Hidden layer
      model.add(Dense(neurons, activation='relu', kernel_initializer='he_uniform'))
      #Step 5 - output
      model.add(Dense(22, activation='softmax'))
      #Step 6 - chose the optimizer
      opt = SGD(lr=0.001, momentum=0.9)
      #Step 7 - Compile the model
      model.compile(loss=sparse_categorical_crossentropy, optimizer=opt, metrics=['accuracy'])
      return model

  def opt_model(self):
    neurons = [16,32,64,128,256]
    model_result = []
    for i in neurons:
      model_result.append(self.__model(i))
    # split dataset
    X_train,X_test,y_train,y_test = train_test_split (self.__data_X,self.__data_y,test_size=0.4)
    # run the model into a loop in order to find best value for neurons
    history_v0 = []
    scores_v0 = []
    for i in range(len(model_result)):
        history_v0.append(model_result[i].fit(X_train, y_train, epochs=self.__epoch, batch_size=self.__batch_size, validation_data=(X_test, y_test), verbose=0))
        _, acc = model_result[i].evaluate(X_test, y_test, verbose=0)
        # append scores
        scores_v0.append(acc)
    # find the best number of neurons
    max_value = max(scores_v0)
    max_index = scores_v0.index(max_value)
    opt_neu = neurons[max_index]
    optModel = model_result[max_index]
    return X_train,X_test,y_train,y_test,optModel