import keras as k


def get_Prediction_head(inputs,num_points):
   
    x = k.layers.Conv2D(32, (3, 3), activation='relu',padding='same')(inputs)
    x = k.layers.Conv2D(num_points*3, (3, 3), activation='sigmoid',padding='same')(x)
    return x
    
    

def getModel(input_shape,num_points):
    input  = k.Input(input_shape)
    i =  k.layers.Conv2D(32, (3, 3), activation='relu',padding='same')(input)
    i =  k.layers.Conv2D(32, (3, 3), activation='relu',padding='same')(i)
    i =  k.layers.Conv2D(32, (3, 3), activation='relu',padding='same')(i)
    i =  k.layers.MaxPool2D((2,2))(i)
    
    
    x = k.layers.Conv2D(32, (3, 3), activation='relu',padding='same')(i)
    x = k.layers.Conv2D(32, (3, 3), activation='relu',padding='same')(x)
    x = k.layers.Conv2D(32, (3, 3), activation='relu',padding='same')(x)
    
    y = k.layers.Conv2D(32, (3, 3), activation='relu',dilation_rate=3,padding='same')(i)
    y = k.layers.Conv2D(32, (3, 3), activation='relu',dilation_rate=3,padding='same')(y)
    y = k.layers.Conv2D(32, (3, 3), activation='relu',dilation_rate=3,padding='same')(y)
    
    
    z = k.layers.Conv2D(32, (3, 3), activation='relu',dilation_rate=7,padding='same')(i)
    z = k.layers.Conv2D(32, (3, 3), activation='relu',dilation_rate=7,padding='same')(z)
    z = k.layers.Conv2D(32, (3, 3), activation='relu',dilation_rate=7,padding='same')(z)
    
    X = get_Prediction_head(y,num_points)
    Y = get_Prediction_head(y,num_points)
    Z = get_Prediction_head(z,num_points)
    
    
    
    model = k.Model(inputs=input, outputs=[X,Y,Z])
    return model
    
    

def testModel():
    input_shape =(256,256,3)
    model = getModel(input_shape,1)
    model(k.Input(input_shape))
    model.summary()
    
    
testModel()