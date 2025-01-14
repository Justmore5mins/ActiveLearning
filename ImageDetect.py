from os import listdir
from os.path import isdir
from cv2 import VideoCapture, imread
from ultralytics import YOLO
from PIL import Image

class ImageDetect:
    def __init__(self, model:str="yolo11n.pt",source:str|int = 0,device:int|str|list[int] = "cpu"):
        self.model = YOLO(model,task="detect",verbose=False)
        if type(source) is int:
            self.cam = self.__CamInit__(source)
        else:
            print("Detected item is static source, now change to static detect")
            self.source = source
            
        self.model.to(device) if device != "cpu" else None
        
    def static(self,save:bool = False):
        if isdir(self.source):
            items = listdir(self.source)
            nones:list[str] = []
            for item in items:
                item = f"{self.source}/{item}"
                width = Image.open(item).width
                height = Image.open(item).height
                results = self.model.predict(imread(item),stream=True)
                infos = []
                for res in results:
                    if not res.boxes:
                        print(f"{item} not detected")
                        nones.append(item)
                    for box in res.boxes:
                        info = box.cls.tolist()[0]+" "+box.xywh[0]/width+" "+box.xywh[1]/height+" "+box.xywh[2]/width+" "+box.xywh[3]/height
                        infos.append(info)
                if save:
                    with open(f"{item.split(".")[0]}.txt","w") as file:
                        file.writelines(infos)
                        
    def stream(self,conf:float = 0.8, cls:list[str] = None):
        if not (0 <= conf <= 1):
            raise ValueError("Confidence must be in 0 and 1.")
        
        if not cls:
            cls = self.model.names
            
        
        
    def __CamInit__(self,source:int) -> VideoCapture:
        return VideoCapture(source)