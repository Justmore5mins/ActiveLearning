from cv2 import VideoCapture, CAP_PROP_FRAME_HEIGHT,CAP_PROP_FRAME_WIDTH
from ultralytics import YOLO
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

    def stream(self,conf:float = 0.8, cls:list[str] = None,threshold:float = 0.9,interval:int = 5):
        if not cls:
            cls = self.model.names
        
        state:bool = True  
        print("camera resolution: {}x{}".format(CAP_PROP_FRAME_WIDTH,CAP_PROP_FRAME_HEIGHT))
        print("init successfully, now start detecting")
        n=0
        while state:
            state, image = self.cam.read()
            result = self.model.predict(image,True,conf=conf)
            items = []
            for res in result:
                if not res.boxes:
                    continue
                if max(res.boxes.conf.tolist()) >= threshold and n == interval:
                    for box in res.boxes:
                        xywh = box.xywh.tolist()[0]
                        print(xywh, flush=True)
                        items.append(f"{int(box.cls)} {xywh[0]/self.resolution[0]} {xywh[1]/self.resolution[1]} {xywh[2]/self.resolution[0]} {xywh[3]/self.resolution[1]}\n")
                    self.collect.save(image,True,items)
                    n = 0
                elif max(res.boxes.conf.tolist()) >= threshold and n != interval:
                    n += 1

                else:
                    self.collect.save(image)
        
    def __CamInit__(self,source:int,resolution:tuple[int,int]) -> VideoCapture:
        cam = VideoCapture(source)
        cam.set(CAP_PROP_FRAME_WIDTH,resolution[0])
        cam.set(CAP_PROP_FRAME_HEIGHT,resolution[1])
        return cam
    
if __name__ == "__main__":
    ImageDetect("Note2025Alpha3a.pt",source=1).stream()