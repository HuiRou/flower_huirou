import os
import sys
import numpy as np
import flwr as fl
import tensorflow as tf
from tensorflow.keras import Sequential, layers
import copy
#tf.keras.backend.set_image_data_format('channels_last')
# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
mode = str(sys.argv[1])
client_num = int(sys.argv[2])
client_id = int(sys.argv[3])
#print(f'{client_id}: {os.getpid()}')
warmup = 2
g_rounds = 0
past_acc = 0
g_model = None
init_acc = 0
done = False
g_episode = 0
g_step = 0

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

    ***NEW***

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)             (None, 30, 30, 32)        896       
                                                                    
    batch_normalization (BatchN  (None, 30, 30, 32)       128       
    ormalization)                                                   
                                                                    
    conv2d_1 (Conv2D)           (None, 28, 28, 32)        9248      
                                                                    
    batch_normalization_1 (Batc  (None, 28, 28, 32)       128       
    hNormalization)                                                 
                                                                    
    max_pooling2d (MaxPooling2D  (None, 14, 14, 32)       0         
    )                                                               
                                                                    
    dropout (Dropout)           (None, 14, 14, 32)        0         
                                                                    
    flatten (Flatten)           (None, 6272)              0         
                                                                    
    dense (Dense)               (None, 128)               802944    
                                                                    
    batch_normalization_2 (Batc  (None, 128)              512       
    hNormalization)                                                 
                                                                    
    dropout_1 (Dropout)         (None, 128)               0         
                                                                    
    dense_1 (Dense)             (None, 8)                 1032      
                                                                    
    =================================================================
    Total params: 814,888
    Trainable params: 814,504
    Non-trainable params: 384
    '''
    num_classes = 8 # 10
    model = Sequential()

    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3))) # padding='same', 
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3,3), activation='relu')) # padding='same', 
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.3))

    # model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
    # model.add(layers.BatchNormalization())
    # model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
    # model.add(layers.BatchNormalization())
    # model.add(layers.MaxPooling2D(pool_size=(2,2)))
    # model.add(layers.Dropout(0.5))

    # model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
    # model.add(layers.BatchNormalization())
    # model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
    # model.add(layers.BatchNormalization())
    # model.add(layers.MaxPooling2D(pool_size=(2,2)))
    # model.add(layers.Dropout(0.5))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))    # num_classes = 10

    # Checking the model summary
    # model.summary()
    return model

def isNum(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def data_load(client_id):

    user_test_data = True

    # Load CIFAR10
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data() #10 class 5000 instance each

    # Get all training targets and count the number of class instances
    classes, class_counts = np.unique(y_train , return_counts=True)
    nb_classes = len(classes)
    #print(class_counts

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
    imbal_class_indices = [class_idx[:class_count] for class_idx, class_count in zip(class_indices, imbal_class_counts)]
    imbal_class_indices = np.hstack(imbal_class_indices)

    # Set target and data to dataset
    y_train = y_train[imbal_class_indices]
    x_train = x_train[imbal_class_indices]

    # test data
    if user_test_data == True:
        class_indices = [np.where(y_test == i)[0] for i in range(nb_classes)]
        test_imbal_class_counts = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 0, 0]
        test_imbal_class_indices = [class_idx[:class_count] for class_idx, class_count in zip(class_indices, test_imbal_class_counts)]
        test_imbal_class_indices = np.hstack(test_imbal_class_indices)
        y_test = y_test[test_imbal_class_indices]
        x_test = x_test[test_imbal_class_indices]

    assert len(x_train) == len(y_train)
    return (x_train, y_train), (x_test, y_test)

# Load model and data (MobileNetV2, CIFAR-10)
#model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
model = model4cifar10()
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
(x_train, y_train), (x_test, y_test) = data_load(client_id)
g_model = model
# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        print(f'client_id: {client_id}')
        result = model.get_weights()        
        return result

    def fit(self, parameters, config):
        global warmup, g_rounds, g_model
        #print(f'Client Round: {rounds}')

        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=2, batch_size=32)
        result = model.get_weights()
        if mode == "train":
            if warmup > 0:
                g_model.set_weights(model.get_weights())
                warmup -= 1
        
        # if mode == "train" and done: #reset
        #     print('Fit Reset Model')
        #     model.set_weights(g_model.get_weights())
                
        
        return result, len(x_train), {}

    def evaluate(self, parameters, config):
        global g_model, past_acc, init_acc, g_rounds, done, g_episode, g_step, warmup
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        # print(f'client evaluate acc: {accuracy}')  
        if mode == "train":     
            if warmup == 0:
                init_acc = accuracy
                past_acc = accuracy
                #reset(rounds)
                if client_id == 0:
                    print(f'init_acc on client: {init_acc}, w={warmup}')
                warmup -= 1   

            else:         
                r = (accuracy - past_acc) * 100 -100

                if client_id == 0:
                    print(f'Client: r={g_rounds}, e={g_episode}, s={g_step}, acc={accuracy}, pa={past_acc}, rw={r}')

                if accuracy >= (1 + g_rounds)/100:#r >= 0 or g_step >= 9: #reset step-1
                    done = True
                else:               
                    done = False
                
                if done:
                    if client_id == 0:
                        print(f'Reset Model, reward = {r}')
                    model.set_weights(g_model.get_weights())
                    #past_acc = init_acc
                    past_acc = accuracy
                    done = False
                    g_episode += 1
                    g_step = 0
                else:         
                    past_acc = accuracy
                    g_step += 1       

                g_rounds += 1      
        return loss, len(x_test), {"accuracy": accuracy}

    # not for work ==
    def reset(self, config):
        global g_model
        print('Reset Model')
        model.set_weights(g_model.get_weights())
        #result = model.get_weights()        
        #return result

# Start Flower client
fl.client.start_numpy_client(server_address="192.168.50.179:8080", client=CifarClient())
