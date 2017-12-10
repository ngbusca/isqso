
from keras.layers import Input, Add, Dense, Activation, BatchNormalization, Flatten, Conv1D, AveragePooling1D, MaxPooling1D
from keras.models import Model, load_model, save_model
from keras.preprocessing import image
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform

def SimpleNet(input_shape =  None, classes = 6):
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)
    X = X_input

    nlayers=5
    for stage in range(nlayers):
        X = Conv1D(2*stage+8, 2*(nlayers-stage), strides = 1,name = 'conv{}'.format(stage+1), kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=2)(X)
        X = Activation('relu')(X)
        X = MaxPooling1D(pool_size=10, strides = 2)(X)

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='SimpleNet')

    return model
