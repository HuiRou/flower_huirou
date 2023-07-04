import os
import sys
import numpy as np
import flwr as fl
import tensorflow as tf
from tensorflow.keras import Sequential, layers
#tf.keras.backend.set_image_data_format('channels_last')
# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
client_num = int(sys.argv[1])
client_id = int(sys.argv[2])
print(f'{client_id}: {os.getpid()}')

def model4cifar10():
    '''
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 32, 32, 32)        896       
    _________________________________________________________________
    batch_normalization (BatchNo (None, 32, 32, 32)        128       
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 32, 32, 32)        9248      
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 32, 32, 32)        128       
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 16, 16, 32)        0         
    _________________________________________________________________
    dropout (Dropout)            (None, 16, 16, 32)        0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 16, 16, 64)        18496     
    _________________________________________________________________
    batch_normalization_2 (Batch (None, 16, 16, 64)        256       
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 16, 16, 64)        36928     
    _________________________________________________________________
    batch_normalization_3 (Batch (None, 16, 16, 64)        256       
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 8, 8, 64)          0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 8, 8, 64)          0         
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 8, 8, 128)         73856     
    _________________________________________________________________
    batch_normalization_4 (Batch (None, 8, 8, 128)         512       
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 8, 8, 128)         147584    
    _________________________________________________________________
    batch_normalization_5 (Batch (None, 8, 8, 128)         512       
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 4, 4, 128)         0         
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 4, 4, 128)         0         
    _________________________________________________________________
    flatten (Flatten)            (None, 2048)              0         
    _________________________________________________________________
    dense (Dense)                (None, 128)               262272    
    _________________________________________________________________
    batch_normalization_6 (Batch (None, 128)               512       
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 128)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                1290      
    =================================================================
    Total params: 552,874
    Trainable params: 551,722
    Non-trainable params: 1,152 
    '''
    num_classes = 10
    model = Sequential()

    model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(32,32,3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.5))

    model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.5))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))    # num_classes = 10

    # Checking the model summary
    #model.summary()
    return model

def isNum(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def data_load(client_id):
    
    # Load CIFAR10
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data() #10 class 5000 instance each

    # Get all training targets and count the number of class instances
    classes, class_counts = np.unique(y_train , return_counts=True)
    nb_classes = len(classes)
    #print(class_counts)

    # Create artificial imbalanced class counts
    # imbal_class_counts = [500, 5000] * 5
    if(client_id < 2):
        imbal_class_counts = [3000, 3000, 0, 0, 0, 0, 0, 0, 0, 0]
    elif(client_id < 4):
        imbal_class_counts = [0, 0, 3000, 3000, 0, 0, 0, 0, 0, 0]
    elif(client_id < 6):
        imbal_class_counts = [0, 0, 0, 0, 3000, 3000, 0, 0, 0, 0]
    else:
        imbal_class_counts = [0, 0, 0, 0, 0, 0, 3000, 3000, 0, 0]
    print(imbal_class_counts)

    # Get class indices
    class_indices = [np.where(y_train == i)[0] for i in range(nb_classes)]
    #print(class_indices)

    # Get imbalanced number of instances
    imbal_class_indices = [class_idx[:class_count] for class_idx, class_count in   zip(class_indices, imbal_class_counts)]
    imbal_class_indices = np.hstack(imbal_class_indices)

    # Set target and data to dataset
    y_train = y_train[imbal_class_indices]
    x_train = x_train[imbal_class_indices]

    assert len(x_train) == len(y_train)
    return (x_train, y_train), (x_test, y_test)

# Load model and data (MobileNetV2, CIFAR-10)
#model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
model = model4cifar10()
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
(x_train, y_train), (x_test, y_test) = data_load(client_id)

# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        print(f'client_id: {client_id}')
        result = model.get_weights()        
        return result

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=2, batch_size=32)
        result = model.get_weights()
        return result, len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        print(f'client evaluate acc: {accuracy}')
        return loss, len(x_test), {"accuracy": accuracy}

# Start Flower client
fl.client.start_numpy_client(server_address="192.168.50.179:8080", client=CifarClient())
