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
    i = k.layers.BatchNormalization()(i)
    
    # i =  k.layers.MaxPool2D((2,2))(i)
    
    x = k.layers.Conv2D(32, (3, 3), activation='relu',padding='same')(i)
    x = k.layers.Conv2D(64, (3, 3), activation='relu',padding='same')(x)
    x = k.layers.Conv2D(32, (3, 3), activation='relu',padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x =k.layers.Dropout(0.2)(x)
    
    y = k.layers.Conv2D(32, (3, 3), activation='relu',dilation_rate=3,padding='same')(i)
    y = k.layers.Conv2D(64, (3, 3), activation='relu',dilation_rate=3,padding='same')(y)
    y = k.layers.Conv2D(32, (3, 3), activation='relu',dilation_rate=3,padding='same')(y)
    y = k.layers.BatchNormalization()(y)
    y = k.layers.Dropout(0.2)(y)
    
    z = k.layers.Conv2D(32, (3, 3), activation='relu',dilation_rate=7,padding='same')(i)
    z = k.layers.Conv2D(64, (3, 3), activation='relu',dilation_rate=7,padding='same')(z)
    z = k.layers.Conv2D(32, (3, 3), activation='relu',dilation_rate=7,padding='same')(z)
    z = k.layers.BatchNormalization()(z)
    z = k.layers.Dropout(0.2)(z)
    
    final = k.layers.concatenate([x,y,z])
    final = k.layers.Conv2D(32, (3, 3), activation='relu',padding='same')(final)
    final = k.layers.Conv2D(16, (3, 3), activation='relu',padding='same')(final)
    final = k.layers.Conv2D(8, (3, 3), activation='relu',padding='same')(final)
    final = k.layers.Conv2D(num_points, (3, 3), activation='relu',padding='same')(final)
    final = k.layers.Flatten()(final)
    final = k.activations.sigmoid(final)
    final = k.layers.Reshape((input_shape[0],input_shape[1],num_points))(final)
    
    
    
    
    model = k.Model(inputs=input, outputs=final)
    return model
    
    

def testModel():
    input_shape =(256,256,3)
    model = getModel(input_shape,1)
    model(k.Input(input_shape))
    model.summary()
    
    
testModel()