import keras as k
import os

class Datgenerator(k.utils.Sequence):
    def __init__(self, input_size,path, batch_size):
        self.path = path
        self.input_size = input_size
        self.batch_size = batch_size
        self.images_path = os.path.join(self.path,'img')
        self.trans_path = os.path.join(self.path,'trans')
        self.images  = os.listdir(self.images_path)
        
        
    def __len__(self):
         return int(len(self.images)/self.batch_size)
    
    def __getitem__(self, idx):
        batch_x = self.images[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_y = self.images[idx*self.batch_size:(idx+1)*self.batch_size]
        
        batch_x = [os.path.join(self.images_path,x) for x in batch_x]
        batch_y = [os.path.join(self.trans_path,x.replace('png','pkl')) for x in batch_y]
        x = [k.utils.img_to_array(k.utils.load_img(x,target_size=(self.input_shape[0],self.input_shape[1])) )for x in batch_x]
        pass