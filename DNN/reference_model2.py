import keras
from reverse_gradient import GradientReversal
from functools import partial

class BMXNetwork():
    def __init__(
        self,
        activation='selu',
        useLSTM = False,
        kernel_initializer = 'lecun_normal',
        dropout_rate = 0.1,
        l2loss = 1e-8,
        targets = 3,
        taget_activation = 'softmax',
        adversarial_targets = 2,
        adversarial_activation = 'relu'
    ):
        self.activation = activation
        self.useLSTM = useLSTM
        self.kernel_initializer = kernel_initializer
        self.dropout_rate = dropout_rate
        self.l2loss = l2loss
        self.targets = targets
        self.taget_activation = taget_activation
        self.adversarial_targets = adversarial_targets
        self.adversarial_activation = adversarial_activation
        self.layersBuild = False
    
    def buildLayers(self,combinationInputs,globalInputs=None):
        if not self.layersBuild:
            nconvNodes = int(combinationInputs.shape[2])
            
            
            self.normCombinationLayer = keras.layers.BatchNormalization(
                axis=2, #normalize per feature over combination/batch
                momentum=0.99, 
                epsilon=0.001, 
                center=True, 
                scale=True, 
            )
            if globalInputs!=None:
                self.normGlobalLayer = keras.layers.BatchNormalization(
                    axis=1, #normalize per feature over batch
                    momentum=0.99, 
                    epsilon=0.001, 
                    center=True, 
                    scale=True, 
                )

            self.convLayers = [
                keras.layers.Conv1D(int(round(nconvNodes*1.2)), 1,
                   activation=self.activation, 
                   kernel_regularizer=keras.regularizers.l2(self.l2loss),
                   kernel_initializer=self.kernel_initializer
                ),
                keras.layers.Conv1D(nconvNodes, 1,
                   activation=self.activation, 
                   kernel_regularizer=keras.regularizers.l2(self.l2loss),
                   kernel_initializer=self.kernel_initializer
                ),
                keras.layers.Dropout(self.dropout_rate,noise_shape=[1,1,nconvNodes]),
                
                keras.layers.Conv1D(nconvNodes, 1,
                   activation=self.activation, 
                   kernel_regularizer=keras.regularizers.l2(self.l2loss),
                   kernel_initializer=self.kernel_initializer
                ),
                keras.layers.Dropout(self.dropout_rate,noise_shape=[1,1,nconvNodes]),
                
                keras.layers.Conv1D(int(round(nconvNodes*2./3.)), 1,
                   activation='tanh' if self.useLSTM else self.activation, 
                   kernel_regularizer=keras.regularizers.l2(self.l2loss),
                   kernel_initializer=self.kernel_initializer
                ),
            ]
            
            if self.useLSTM:
                self.lstmLayer = keras.layers.LSTM(
                    50, 
                    activation='tanh', 
                    recurrent_activation='hard_sigmoid',
                    kernel_regularizer=keras.regularizers.l2(self.l2loss),
                    kernel_initializer=self.kernel_initializer,
                    implementation=2,
                    recurrent_dropout=self.dropout_rate,
                    go_backwards=True
                )
            
            self.denseLayers = []
            for _ in range(2):
                self.denseLayers.extend([
                    keras.layers.Dense(
                        50,
                        activation=self.activation,
                        kernel_regularizer=keras.regularizers.l2(self.l2loss),
                        kernel_initializer=self.kernel_initializer
                    ),
                    keras.layers.Dropout(self.dropout_rate)
                ])
            
            
            self.classPredictionLayers = []
            for _ in range(4):
                self.classPredictionLayers.extend([
                    keras.layers.Dense(
                        25,
                        activation=self.activation,
                        kernel_regularizer=keras.regularizers.l2(self.l2loss),
                        kernel_initializer=self.kernel_initializer
                    ),
                    keras.layers.Dropout(self.dropout_rate)
                ])
            self.classPredictionLayers.append(
                keras.layers.Dense(
                    self.targets,
                    activation=self.taget_activation,
                    kernel_regularizer=keras.regularizers.l2(self.l2loss),
                    kernel_initializer=self.kernel_initializer
                )
            )
            
            if self.adversarial_targets>0:
                self.adversarialPredictionLayers = [
                    GradientReversal()
                ]
                for _ in range(4):
                    self.adversarialPredictionLayers.extend([
                        keras.layers.Dense(
                            25,
                            activation=self.activation,
                            kernel_regularizer=keras.regularizers.l2(self.l2loss),
                            kernel_initializer=self.kernel_initializer
                        ),
                        keras.layers.Dropout(self.dropout_rate)
                    ])
                    
                self.adversarialPredictionLayers.append(
                    keras.layers.Dense(
                        self.adversarial_targets,
                        activation=self.adversarial_activation,
                        kernel_regularizer=keras.regularizers.l2(self.l2loss),
                        kernel_initializer=self.kernel_initializer
                    )
                )
            
            self.layersBuild = True
        
    def getFeatures(self,combinationInputs,globalInputs=None):
        self.buildLayers(combinationInputs,globalInputs)
        
        conv = self.normCombinationLayer(combinationInputs)
        for layer in self.convLayers:
            conv = layer(conv)
        
        if self.useLSTM:
            conv = self.lstmLayer(conv)
        else:
            conv = keras.layers.Flatten()(conv)
        
        if globalInputs!=None:
            globalInputsNorm = self.normGlobalLayer(globalInputs)
            features = keras.layers.Concatenate(axis=1)([globalInputsNorm,conv],)
        else:
            features = conv
            
        for layer in self.denseLayers:
            features = layer(features)
        return features
        
    def getClassPrediction(self,combinationInputs,globalInputs=None):
        prediction = self.getFeatures(combinationInputs,globalInputs)
        for layer in self.classPredictionLayers:
            prediction = layer(prediction)
        return prediction
        
    def getAdversarialPrediction(self,combinationInputs,globalInputs=None):
        prediction = self.getFeatures(combinationInputs,globalInputs)
        #add inversion
        for layer in self.adversarialPredictionLayers:
            prediction = layer(prediction)
        return prediction
        
    def getModel(self,predictionFcts,*args):
        inputLayers = []
        for inputArray in args:
            if type(inputArray)==type(None):
                continue
            inputLayers.append(keras.layers.Input(inputArray.shape[1:]))
        predictions = [predictionFct(*inputLayers) for predictionFct in predictionFcts]
        return keras.models.Model(inputs=inputLayers,outputs=predictions)
        
    def getClassModel(self,combinationInputsArray,globalInputsArray=None):
        return self.getModel(
            [partial(BMXNetwork.getClassPrediction,self)],
            combinationInputsArray,
            globalInputsArray
        )
        
    def getFeatureModel(self,combinationInputsArray,globalInputsArray=None):
        return self.getModel(
            [partial(BMXNetwork.getFeatures,self)],
            combinationInputsArray,
            globalInputsArray
        )
        
    def getAdversarialModel(self,combinationInputsArray,globalInputsArray=None):
        return self.getModel(
            [partial(BMXNetwork.getAdversarialPrediction,self)],
            combinationInputsArray,
            globalInputsArray
        )
        
    def getAdversarialCriticModel(self,featureInputsArray=None):
        return self.getModel(
            [partial(BMXNetwork.getAdversarialPrediction,self)],
            combinationInputsArray,
            globalInputsArray
        )
        
    def getFullModel(self,combinationInputsArray,globalInputsArray=None):
        return self.getModel(
            [partial(BMXNetwork.getClassPrediction,self),
             partial(BMXNetwork.getAdversarialPrediction,self)],
            combinationInputsArray,
            globalInputsArray
        )

