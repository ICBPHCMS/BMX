import tensorflow as tf
import keras
from keras import backend as K

class Network:
    def __init__(self,regLoss=1e-4):
        self.convLayers = [
            keras.layers.Conv1D(32, 1, strides=1, activation=None,
                kernel_regularizer=keras.regularizers.l2(regLoss)
            ),
            keras.layers.LeakyReLU(alpha=0.1),
            #drop same features (=filters) for all combinations
            keras.layers.Dropout(0.2,noise_shape=[1,1,32]),
            keras.layers.Conv1D(32, 1, strides=1, activation=None,
                kernel_regularizer=keras.regularizers.l2(regLoss)
            ),
            keras.layers.LeakyReLU(alpha=0.1), 
            #drop same features (=filters) for all combinations
            keras.layers.Dropout(0.3,noise_shape=[1,1,32]),
            keras.layers.Conv1D(16, 1, strides=1, activation='tanh', 
                kernel_regularizer=keras.regularizers.l2(regLoss)
            ),
            keras.layers.Dropout(0.3),
        ]

        self.lstmLayers = [
            keras.layers.LSTM(50, activation='tanh', recurrent_activation='hard_sigmoid',
                kernel_regularizer=keras.regularizers.l2(regLoss),
                implementation=2,
                recurrent_dropout=0.1,
                go_backwards=True
            ),
            keras.layers.Dropout(0.2),
        ]

        self.bypassLayersLSTM = [
            keras.layers.Lambda(lambda l:l[:,0:1,:]),
            keras.layers.Flatten(),
            #drop complete bypass
            keras.layers.Dropout(0.8,noise_shape=[1,1]),
        ]
        
        self.bypassLayersDense = [
            keras.layers.Dropout(0.2),
            keras.layers.Dropout(0.8,noise_shape=[1,1]),
        ]
        
        self.predictLayers = [
            keras.layers.Dense(50, activation=None,
                kernel_regularizer=keras.regularizers.l2(regLoss)
            ),
            keras.layers.LeakyReLU(alpha=0.1),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(50, activation=None,
                kernel_regularizer=keras.regularizers.l2(regLoss)
            ),
            keras.layers.LeakyReLU(alpha=0.1),
            keras.layers.Dropout(0.2),
        ]
        
        self.finalLayers = [
            keras.layers.Dense(50, activation=None,
                kernel_regularizer=keras.regularizers.l2(regLoss)
            ),
            keras.layers.LeakyReLU(alpha=0.1),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(1, activation='sigmoid',
                kernel_regularizer=keras.regularizers.l2(regLoss)
            )
        ]

    def getFeatures(self,x):
        conv = self.convLayers[0](x)
        for layer in self.convLayers[1:]:
            conv = layer(conv)
        return conv

    def getDiscriminant(self,x):
    
        conv = self.getFeatures(x)
 
        lstm = self.lstmLayers[0](conv)
        for layer in self.lstmLayers[1:]:
            lstm = layer(lstm)
            
        bypassLSTM = self.bypassLayersLSTM[0](conv)
        for layer in self.bypassLayersLSTM[1:]:
            bypassLSTM = layer(bypassLSTM)
            
        features = keras.layers.Concatenate()([lstm,bypassLSTM])
        
        predict = self.predictLayers[0](features)
        for layer in self.predictLayers[1:]:
            predict = layer(predict)
        
        bypassDense = self.bypassLayersDense[0](features)
        for layer in self.bypassLayersDense[1:]:
            bypassDense = layer(bypassDense)
        
        final = keras.layers.Concatenate()([predict,bypassDense])
        final = self.finalLayers[0](final)
        for layer in self.finalLayers[1:]:
            final = layer(final)
        return final
        

