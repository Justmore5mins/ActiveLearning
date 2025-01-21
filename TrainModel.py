from ultralytics import YOLO
from torch import cuda

class train:
    def __init__(self,BaseModel:str,data:str, device:int|list[int]|str = "cpu",epoch:int = 2000,MemoryLimit:int=-1):
        '''
        this class can train model, returns the training path, MemoryLimit means the training max memory usage in percent
        '''
        self.model = YOLO(BaseModel,"detect")
        if device != "cpu":
            self.model.to(device)
        self.data = data
        self.epochs = epoch
        if MemoryLimit != -1:
            cuda.set_per_process_memory_fraction(MemoryLimit,device)
    
    def train(self):
        self.model.train(data=self.data,epochs=self.epochs,cache=False)
        