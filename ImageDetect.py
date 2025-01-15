from os import listdir
from os.path import isdir
from cv2 import VideoCapture, imread,CAP_PROP_FRAME_HEIGHT,CAP_PROP_FRAME_WIDTH
from ultralytics import YOLO
from PIL import Image
from ImageCollect import ImageCollect

class ImageDetect:
    def __init__(self, model:str="yolo11n.pt",source:str|int = 0,resolution:tuple[int,int]=(640,480),device:int|str|list[int] = "cpu"):
        self.model = YOLO(model,task="detect",verbose=False)
        self.resolution = resolution
        if type(source) is int:
            self.cam = self.__CamInit__(source,resolution)
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
        if not cls:
            cls = self.model.names
        
        state:bool = True  
        print("camera resolution: {}x{}".format(CAP_PROP_FRAME_WIDTH,CAP_PROP_FRAME_HEIGHT))
        print("init successfully, now start detecting")
        while state:
            state, image = self.cam.read()
            result = self.model.predict(image,True,conf=conf)
            items = []
            for res in result:
                if not res.boxes:
                    continue
                if max(res.boxes.conf.tolist()) >= threshold:
                    for box in res.boxes:
                        xywh = box.xywh.tolist()[0]
                        print(xywh)
                        items.append(f"{int(box.cls)} {xywh[0]/self.resolution[0]} {xywh[1]/self.resolution[1]} {xywh[2]/self.resolution[0]} {xywh[3]/self.resolution[1]}\n")
                    self.collect.save(image,True,items)

                else:
                    self.collect.save(image)
        
    def __CamInit__(self,source:int,resolution:tuple[int,int]) -> VideoCapture:
        cam = VideoCapture(source)
        cam.set(CAP_PROP_FRAME_WIDTH,resolution[0])
        cam.set(CAP_PROP_FRAME_HEIGHT,resolution[1])
        return cam
    
if __name__ == "__main__":
    ImageDetect("Note2025Alpha3a.pt",source=1).stream()