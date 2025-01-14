from os import listdir
from os.path import isdir
from cv2 import VideoCapture, imread,CAP_PROP_FRAME_HEIGHT,CAP_PROP_FRAME_WIDTH
from sympy import true
from ultralytics import YOLO
from PIL import Image
from ImageCollect import ImageCollect

class ImageDetect:
    def __init__(self, model:str="yolo11n.pt",source:str|int = 0,device:int|str|list[int] = "cpu"):
        self.model = YOLO(model,task="detect",verbose=False)
        if type(source) is int:
            self.cam = self.__CamInit__(source)
        else:
            print("Detected item is static source, now change to static detect")
            self.source = source
            
        self.model.to(device) if device != "cpu" else None
        self.collect = ImageCollect("detected")
        
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
                        
    def stream(self,conf:float = 0.8, cls:list[str] = None,threshold:float = 0.9,interval:int = 5):
        if not (0 <= conf <= 1) or (0 <= threshold <= 1):
            raise ValueError("Confidence must be in 0 and 1.")
        if not cls:
            cls = self.model.names
        
        state:bool = True    
        while state:
            state, image = self.cam.read()
            result = self.model.predict(image,True)
            items = []
            for res in result:
                if not res.boxes:
                    continue
                if max(res.boxes.conf.tolist()) >= threshold:
                    xywh = res.boxes.xywh
                    for item in zip(res.boxes.cls.tolist(), xywh[0].tolist(), xywh[1].tolist(), xywh[2].tolist(), xywh[3].tolist()):
                        items.apped(f"{item[0]} {item[1]/CAP_PROP_FRAME_WIDTH} {item[2]/CAP_PROP_FRAME_HEIGHT} {item[3]/CAP_PROP_FRAME_WIDTH} {item[4]/CAP_PROP_FRAME_HEIGHT}")
                    self.collect.save(image,True,items)
                else:
                    self.collect.save(image)
                    
        
    def __CamInit__(self,source:int) -> VideoCapture:
        return VideoCapture(source)