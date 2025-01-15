from hashlib import md5
from os import system, path
from time import perf_counter
from cv2 import imwrite, typing

class ImageCollect:
    def __init__(self,BaseDir:str):
        '''
        This class can save image from sources, BaseDir is the file wanted to saved with
        '''
        self.BaseDir = BaseDir
        if not path.isdir(self.BaseDir):
            system(f"mkdir {self.BaseDir} {self.BaseDir}/high_conf {self.BaseDir}/high_conf/images {self.BaseDir}/high_conf/labels {self.BaseDir}/low_conf")
        else:
            system(f"rm -r {self.BaseDir}")
            self.__init__(BaseDir)
        
    def save(self, image:typing.MatLike, high_conf:bool = False, items:list[str] = None):
        '''
        This function can save images, which items is yolo format(class x y w h)
        '''
        filename = md5(f"{perf_counter()}".encode()).hexdigest()
        imwrite(f"{self.BaseDir}/{'high_conf/images/' if high_conf else 'low_conf/'}/{filename}.jpg",image)
        if high_conf:
            with open(f"{self.BaseDir}/high_conf/labels/{filename}.txt","a") as file:
                file.writelines(items)