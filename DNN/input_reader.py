import os
import h5py
import re
#import matplotlib.pyplot as plt
import numpy as np
#from keras.layers import Input, Dense, Dropout, Conv1D, Flatten, Concatenate
#from keras.optimizers import Adam

#from reference_model2 import BMXNetwork

class HDF5Reader():
    def __init__(
        self,
        fileList,
        nCombinations=100,
        nVertices=10
    ):
        self.fileList = fileList
        self.nCombinations = nCombinations
        self.nVertices = nVertices
        
    @staticmethod
    def fromFolder(folder,filePattern="\w+.hdf5",maxFiles=-1,nCombinations=100,nVertices=10):
        fileList = []
        for fileName in os.listdir(folder):
            if re.match(filePattern,fileName):
                fileList.append(os.path.join(folder,fileName))
                if maxFiles>0 and len(fileList)==maxFiles:
                    return HDF5Reader(fileList,nCombinations,nVertices)
        return HDF5Reader(fileList,nCombinations,nVertices)

    def __call__(self):
        columns = {}
        for filePath in self.fileList:
            h5file = h5py.File(filePath,'r')
            for k in h5file.keys():
                if k == "combinationFeatures":
                    if not columns.has_key(k):
                        columns[k] = {feature: [] for feature in h5file[k].keys()}
                    for feature in h5file[k].keys():
                        if not columns[k].has_key(feature):
                            columns[k][feature] = []
                        columns[k][feature].append(h5file[k][feature][:,:self.nCombinations])
                elif k == "globalFeatures":
                    if not columns.has_key(k):
                        columns[k] = {feature: [] for feature in h5file[k].keys()}
                    for feature in h5file[k].keys():
                        if not columns[k].has_key(feature):
                            columns[k][feature] = []
                        columns[k][feature].append(h5file[k][feature][()])
                else:
                    if not columns.has_key(k):
                        columns[k] = []
                    if k in ["refSel","bmass","llmass"]:
                        columns[k].append(h5file[k][:,:self.nVertices])
                    elif k == "genIndex":
                        genIndex = h5file[k]
                        slicedGenIndex = genIndex[:,:min(self.nVertices+1,genIndex.shape[1])]
                        slicedGenIndex[:,-1] = 1-np.sum(slicedGenIndex[:,:-1],axis=1)
                        columns[k].append(slicedGenIndex)
                    else:
                        columns[k].append(h5file[k][()])
                        
        HDF5Reader._concatenatePerColumn(columns)
        return columns
                    
    @staticmethod
    def _concatenatePerColumn(columns):        
        for k in columns.keys():
            if type(columns[k])==type(dict()):
                HDF5Reader._concatenatePerColumn(columns[k])
            else:
                columns[k] = np.concatenate(columns[k],axis=0)
                
    @staticmethod      
    def merge_combination_features(columns,featureNames):
        features = []
        for k in featureNames:
            features.append(columns["combinationFeatures"][k])
        return np.stack(features,axis=2)
        
    @staticmethod
    def merge_global_features(columns,featureNames):
        features = []
        for k in featureNames:
            features.append(columns["globalFeatures"][k])
        return np.stack(features,axis=1)
                
    
'''
class ThreadsafeCache():
    def __init__(self):
        self.cache = {}
        self.size = 0
        self.lock = threading.Lock()
    
    def add(self,dataDict):
        self.lock.acquire()
        size = ThreadsafeCache.addToCache(self.cache,dataDict)
        self.size+=size
        self.lock.release()
        
    def dequeue(self,length):
        self.lock.acquire()
        data,size = getFromCache(self.cache,length)
        self.size-=sizes
        self.lock.release()
        return data
        
    @staticmethod
    def addToCache(cache,dataDict):
        size = -1
        for k in dataDict.keys():
            if type(dataDict[k])==type(dict()):
                if not cache.has_key(k):
                    cache[k] = {}
                newSize = ThreadsafeCache.addToCache(cache[k],dataDict[k])
                if size<-1:
                    size = newSize
                elif size!=newSize:
                    raise Exception("Adding element in '"+k+"' with size %i does not match other elements %i"%(
                        newSize,size
                    ))
            elif type(dataDict[k])==type(numpy.array()):
                if not cache.has_key(k):
                    cache[k] = dataDict[k]
                else:
                    cache[k] = np.concatenate(cache[k],dataDict[k],axis=0)
                    if size<0:
                        size = dataDict[k].shape[0]
                    elif size!=dataDict[k].shape[0]:
                        raise Exception("Adding element '"+k+"' with size %i does not match other elements %i"%(
                            dataDict[k].shape[0],size
                        ))
                
    @staticmethod
    def getFromCache(cache,length):
        data = {}
        size = -1
        for k in cache.keys():
            if type(cache[k])==type(dict()):
                data[k] = ThreadsafeCache.getFromCache(cache[k],length)
            elif type(cache[k])==type(numpy.array()):
                dequeueSize = min(length,cache[k].shape[0])
                data[k] = cache[k][0:dequeueSize]
                cache[k] = cache[k][dequeueSize:]
                if size <0:
                    size = dequeueSize
                elif size!=dequeueSize:
                    raise Exception("Removing element '"+k+"' with size %i does not match other elements %i"%(
                        dequeueSize,size
                    ))
                    
        return data,size #can be dict with empty arrays


class InputQueue():
    

    class ThreadedReader():
        def __init__(self,cache)


    def __init__(
        self,
        fileList,
        nThreads = 1
    ):
        self.fileList = fileList
        self.nVertices = nVertices
        self.size = -1
        
    def size(self):
        if self.size<0:
            self.size = 0
            for filePath in fileList:
                h5file = h5py.File(filePath,'r')
                self.size+=h5file['truth'].shape[0]
        return self.size
        
    def __len__(self):
        return self.size()
        
    def iterate(self):
        cache = {}
        self.threads = []
'''
    
    




'''    
training_files = list_files('../../mumut2/',"train\w+.hdf5")
testing_files = list_files('../../mumut2/',"test\w+.hdf5")
        
print "Found %i/%i train/test files"%(len(training_files),len(testing_files))
        
columns = read_inputs(training_files[0:5])

signalSelection = (columns['truth']>0)
matchSignalVertices = np.sum(columns['genIndex'][signalSelection][:,:-1],axis=0)
matchSignalVertices /= np.sum(matchSignalVertices)
print matchSignalVertices

fig = plt.figure()
ax = plt.bar(range(1,len(matchSignalVertices)+1), matchSignalVertices*100.)
for i, v in enumerate(matchSignalVertices):
    plt.text(i+0.85, max(1,v*100.+10), "%-4.1f%%"%(100.*v),rotation=90)
plt.xlabel("index of gen-matched vertex ordered by CL")
plt.ylabel("fraction (%)")
plt.ylim([0,100.])
plt.savefig('fraction.pdf')
 

fig = plt.figure()
binning = np.linspace(3, 8, 20)
plt.hist(columns['bmass'][
    np.arange(0,columns['genIndex'].shape[0])[signalSelection],
    np.argmax(columns['genIndex'],axis=1)[signalSelection]
],binning,histtype='bar',color='#CCCCCC',edgecolor='#666666',alpha=1.,label="gen-matched")
plt.hist(columns['bmass'][signalSelection][:,0],binning, linewidth=2.2,color='#9911EE',histtype='step',alpha=1.,label="1st vtx by CL")
plt.hist(columns['bmass'][signalSelection][:,1],binning, linewidth=1.4,color='#FF6600',histtype='step',alpha=1.,label="2nd vtx by CL")
plt.hist(columns['bmass'][signalSelection][:,2],binning, linewidth=1.7,color='#00CC44',linestyle='dashed',histtype='step',alpha=1.,label="3rd vtx by CL")


plt.legend(loc='upper left')
plt.yscale("log")
plt.xlabel("b mass (GeV)")
plt.ylabel("MC events")
plt.savefig('bmass.pdf')

print sorted(map(lambda x: str(x),columns['combinationFeatures'].keys()))
print columns['combinationFeatures']['k_mindz_wrtPU'][0:10,0]
print columns['combinationFeatures']['k_dz'][0:10,0]

fig = plt.figure()
binning = np.linspace(-5, 2, 30)
plt.hist(np.log10(columns['combinationFeatures']['k_mindz_wrtPU'][signalSelection][:,0]),binning,density=True, linewidth=2.2,color='#9911EE',label="signal",histtype='step',alpha=1.)
plt.hist(np.log10(columns['combinationFeatures']['k_mindz_wrtPU'][signalSelection==False][:,0]),binning,density=True, linewidth=2.2,color='#00CC44',label="background",histtype='step',alpha=1.)
plt.xlabel("Min delta Z (kaon,PU vtx)")
plt.ylabel("MC events")
#plt.xscale("log")
plt.legend(loc='upper left')
plt.savefig('deltaKPUZ.pdf')

fig = plt.figure()
binning = np.linspace(-5, 2, 20)
plt.hist(np.log10(np.fabs(columns['combinationFeatures']['k_dz'][signalSelection][:,0])),binning,density=True, linewidth=2.2,color='#9911EE',label="signal",histtype='step',alpha=1.)
plt.hist(np.log10(np.fabs(columns['combinationFeatures']['k_dz'][signalSelection==False][:,0])),binning,density=True, linewidth=2.2,color='#00CC44',label="background",histtype='step',alpha=1.)
plt.xlabel("Delta Z (kaon,PV)")
plt.ylabel("MC events")
#plt.xscale("log")
plt.legend(loc='upper left')
plt.savefig('deltaKPVZ.pdf')

fig = plt.figure()
binning = np.linspace(-5, 2, 20)
plt.hist(np.log10(np.fabs(columns['combinationFeatures']['lepton1_dz'][signalSelection][:,0])),binning,density=True, linewidth=2.2,color='#9911EE',label="signal",histtype='step',alpha=1.)
plt.hist(np.log10(np.fabs(columns['combinationFeatures']['lepton1_dz'][signalSelection==False][:,0])),binning,density=True, linewidth=2.2,color='#00CC44',label="background",histtype='step',alpha=1.)
plt.xlabel("Delta Z (mu1,PV)")
plt.ylabel("MC events")
#plt.xscale("log")
plt.legend(loc='upper left')
plt.savefig('deltaLepton1PVZ.pdf')

fig = plt.figure()
binning = np.linspace(-5, 2, 20)
plt.hist(np.log10(np.fabs(columns['combinationFeatures']['lepton2_dz'][signalSelection][:,0])),binning,density=True, linewidth=2.2,color='#9911EE',label="signal",histtype='step',alpha=1.)
plt.hist(np.log10(np.fabs(columns['combinationFeatures']['lepton2_dz'][signalSelection==False][:,0])),binning,density=True, linewidth=2.2,color='#00CC44',label="background",histtype='step',alpha=1.)
plt.xlabel("Delta Z (mu2,PV)")
plt.ylabel("MC events")
#plt.xscale("log")
plt.legend(loc='upper left')
plt.savefig('deltaLepton2PVZ.pdf')




for k in columns.keys():
    if k == "combinationFeatures" or k == "globalFeatures":
        for f in columns[k].keys():
            print k,f,columns[k][f].shape,columns[k][f].dtype
    else:
        print k,columns[k].shape,columns[k].dtype

combinationFeatures = merge_combination_features(columns,exclude="\w*deltaR\w*")
globalFeatures = merge_global_features(columns)

network = BMXNetwork()
classModel = network.getClassModel(combinationFeatures)#,globalFeatures)
classModel.summary()

classOpt = Adam(
    lr=0.001, 
    beta_1=0.9, 
    beta_2=0.999,
    epsilon=None, 
    decay=0.0, 
    amsgrad=False
)

classModel.compile(
    loss=['categorical_crossentropy'],
    optimizer=classOpt,
    metrics=['accuracy'],
)

classModel.fit(
    [combinationFeatures,globalFeatures],
    [columns['genIndex']],
    batch_size=10000, epochs=5,
    validation_split=0.0,
)
'''

#print combinationFeatures.shape,globalFeatures.shape

