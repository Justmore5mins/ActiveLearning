from hashlib import md5
from os import mkdir
from time import perf_counter
from cv2 import imwrite
from cv2.typing import MatLike

class ImageCollect:
    def __init__(self,BaseDir:str):
        '''
        This class can save image from sources, BaseDir is the file wanted to saved with
        '''
        self.BaseDir = BaseDir
        try:
            mkdir(self.BaseDir)
            mkdir(self.BaseDir+"/high_conf")
            mkdir(self.BaseDir+"/high_conf/images")
            mkdir(self.BaseDir+"high_conf/labels")
            mkdir(self.BaseDir+"/low_conf")
        except FileExistsError:
            pass
        
    def save(self, image:MatLike, high_conf:bool = False, items:list[str] = None):
        '''
        This function can save images, which items is yolo format(class x y w h)
        '''
        if high_conf and not items:
            raise ValueError("You need to provide the palce of the item, bro")
    
        filename = md5(bytes(perf_counter())).hexdigest()
        imwrite(f"{self.BaseDir}/{'high_conf/images' if high_conf else 'low_conf'}/{filename}.jpg",image)
        if high_conf:
            with open(f"{self.BaseDir}/high_conf/labels/{filename}.txt","a") as file:
                file.writelines(items)