import uproot

class Chain(object):
    def __init__(self,fileList):
        self._fileList = fileList
        self._nEvents = []
        
        self._currentFile = None
        self._currentTree = None
        self._currentFileName = ""
        self._currentEntry = 0
        self._currentOffset = 0
        self._currentSize = 0
        self._prefetchSize = 10000
        self._buffer = {}
        
        self._fileEventPairs = []
        for i,f in enumerate(self._fileList):
            try:
                print i,'/',len(self._fileList),'...',f
                rootFile = self.OpenFile(f)
                tree = rootFile["Events"]
                nevents = len(tree)
                self._fileEventPairs.append([f,nevents])
            except Exception,e:
                print "Error - cannot open file: ",f
                print e
                
        self._sumEvents = sum(map(lambda x:x[1],self._fileEventPairs))
        
    def OpenFile(self, path):
        return uproot.open(path,localsource=lambda f: 
            uproot.FileSource(f,chunkbytes=8*1024,limitbytes=1024**2)
        )
            
    def GetEntries(self):
        return self._sumEvents
        
    def GetCurrentFile(self):
        return self._currentFileName
            
    def GetEntry(self,i):
        if i<0:
            print "Error - event entry negative: ",i
            i=0
        if self._currentTree!=None and (i-self._currentOffset)<len(self._currentTree) and (i-self._currentOffset)>=0:
            self._currentEntry = i-self._currentOffset
        else:
            del self._currentTree
            self._currentTree = None
            del self._buffer
            self._buffer = {}
            s = 0
            i = i%self._sumEvents #loop
            for e in range(len(self._fileEventPairs)):
                if s<=i and (s+self._fileEventPairs[e][1])>i:
                    print "opening",self._fileEventPairs[e][0]
                    self._currentFile = self.OpenFile(self._fileEventPairs[e][0])
                    self._currentFileName = self._fileEventPairs[e][0]
                    self._currentTree = self._currentFile["Events"]
                    self._currentSize = len(self._currentTree)
                    self._currentOffset = s
                    self._currentEntry = i-self._currentOffset
                    break
                s+=self._fileEventPairs[e][1]
                
    def prefetchBranch(self,k):
        maxPrefetch = min(self._currentEntry+self._prefetchSize,self._currentSize)
        self._buffer[k] = {
            "data": self._currentTree[k].array(entrystart=self._currentEntry, entrystop=maxPrefetch),
            "min":self._currentEntry,
            "max":maxPrefetch
        }
        '''
        print "loading branch ",k,
        print " with entries ",len(self._buffer[k]["data"]),
        print " (tree len=%i, fetched=[%i, %i])"%(self._currentSize,self._currentEntry,maxPrefetch)
        '''
                    
    def __getattr__(self,k):
        if self._currentEntry>self._currentSize:
            print "Error - at end of file: ",self._currentEntry,self._currentSize
            return
            
        if not self._buffer.has_key(k):
            self.prefetchBranch(k)
        elif self._currentEntry<self._buffer[k]["min"] or self._currentEntry>=self._buffer[k]["max"]:
            self.prefetchBranch(k)
        
        bufferOffset = self._currentEntry-self._buffer[k]["min"]
        #print "reading ",k,"entry=%i, offset=%i"%(self._currentEntry,bufferOffset)
        return self._buffer[k]["data"][bufferOffset]

