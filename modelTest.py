import keras as k
import random
import numpy as np
import cv2


def test():
    path = f"dataset/img/output{str(random.randint(0,399))}.png"
    img = np.array([k.utils.img_to_array(k.utils.load_img(path,target_size=(256,256)) )])
    model = k.models.load_model('model/m1.keras')
    
    output =model(img)
    print(output.shape)
    output =  np.array(output[0])
    cv2.imshow('op', output)
    cv2.imshow('image', cv2.imread(path))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

test()