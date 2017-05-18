from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop,Adam
import parameters as parameter




def generate():
    layer1_dimention = parameter.layer1_dimention 
    layer2_dimention = parameter.layer2_dimention 
    layer3_dimention = parameter.layer3_dimention 
    
    numCep = parameter.numCep 
    frame_step = parameter.frame_step 
    frame_len = parameter.frame_len 
    fft_len = parameter.fft_len 
    no_frames = parameter.no_frames 
    
    input_dim = (no_frames+1) * (numCep + fft_len/2 +1) # +1 is for noise aware system
    output_dim = 2* (fft_len/2 +1) +numCep

    model = Sequential()
    model.add(Dense(layer1_dimention, input_shape=(16,)) )
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(.2))
    model.add(Dense(layer2_dimention ) )
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(.2))
    model.add(Dense(layer3_dimention ) )
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(output_dim))
    model.add(Activation('linear'))

    RMS = RMSprop(lr=0.001, rho=0.9, epsilon=1e-12)
    model.compile(loss = 'mse', optimizer = RMS)
    
    model.summary()
    return model

if __name__=='__main__':
    pass