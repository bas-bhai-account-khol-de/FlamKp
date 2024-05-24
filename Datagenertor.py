import keras as k
import os
import PointLocator
import numpy as np
import cv2

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
        batch_x_n = self.images[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_y_n = self.images[idx*self.batch_size:(idx+1)*self.batch_size]
    
        batch_x = [os.path.join(self.images_path,x) for x in batch_x_n]
        batch_y = [os.path.join(self.trans_path,x.replace('png','pkl')) for x in batch_y_n]
        x = np.array([k.utils.img_to_array(k.utils.load_img(x,target_size=self.input_size) )for x in batch_x])
        cp = np.array([PointLocator.load_and_display_trsnformed_image(os.path.join(self.images_path,x),os.path.join(self.trans_path,x.replace('png','pkl'))) for x in batch_x_n])
        dist=[]
        for i in range(len(cp)):
            dis_image =  np.ones(self.input_size)
            for j in range(self.input_size[0]):
                for l in range(self.input_size[1]):
                    dis_image[j][l] = min(0.3+((np.linalg.norm(cp[i]*self.input_size[0] - np.array([l,j])))/25),1)
                    

            dist.append(dis_image)
            # cv2.imshow('img', dis_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            
            
        cross_image = []
        for i in range(len(cp)):
            dis_image =  np.zeros(self.input_size)
            c  = (cp[i]*self.input_size[0]).astype(np.int32)
            dis_image[c[1]][c[0]] = 1.0
            cross_image.append(dis_image)
            
        
        return x,np.array(cross_image),np.array(dist)        
        
        
        pass
    
    
    
def test():
    gen = Datgenerator(input_size=(100,100),path='dataset',batch_size=1)
    a,b,c = gen.__getitem__(0)
    print(f"shapes are {a.shape}, {b.shape} , {c.shape} ")

    
test()