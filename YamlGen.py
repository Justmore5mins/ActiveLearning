from os import chdir

class YamlGen:
    def __init__(self,dataset:str,names:list[str]):
        chdir(dataset)
        self.names = names
    def generate(self):
        with open("data.yaml","w") as file:
            file.write("train: ../train/images\n")
            file.write("val: ../val/images\n")
            file.write("test: ../test/images\n")
            file.write("\n")
            file.write(f"nc: {len(self.names)}\n")
            file.write(f"names: {self.names}")
            
if __name__ == "__main__":
    YamlGen('dir',["halo","word"]).generate()