import keras  as k
from keras.src import ops
import Datagenertor
import Model
import tensorflow as tf
import Losses



def train():
    input_size =(256,256,3)
    dataset= 'dataset'
    epoch = 5
    model_path  = 'model/m1.keras'
    learning_rate =1e-5
    batch_size = 10
    
    
    optimizer = k.optimizers.Adam(learning_rate=learning_rate)
    gen = Datagenertor.Datgenerator(input_size[:-1],dataset,batch_size)
    
    model = Model.getModel(input_size,1)
    # model = k.models.load_model(model_path)
    
    for e in range(epoch):
        for i in range(gen.__len__()):
            x,cross,dist = gen.__getitem__(i)
            with tf.GradientTape() as tape:
                o = model(x)
                o =  k.layers.Reshape((input_size[0],input_size[1]))(o)
                o=1-o
                loss = Losses.binary_crossentropy(dist,o)
                loss = loss[0]
                # loss = loss*dist
                loss  = ops.mean(loss)
                print(loss)
                
                gradients = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(gradients, model.trainable_weights))
                if(i%2 ==0):
                    model.save(model_path)
            
            
           
           

  


train()
            