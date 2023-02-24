import os
import sys
import numpy as np
import flwr as fl
import tensorflow as tf

#tf.keras.backend.set_image_data_format('channels_last')
# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
client_num = int(sys.argv[1])
client_id = int(sys.argv[2])

def isNum(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def data_load(client_id):
    print(client_id)
    
    # Load CIFAR10
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data() #10 class 5000 instance each

    # Get all training targets and count the number of class instances
    classes, class_counts = np.unique(y_train , return_counts=True)
    nb_classes = len(classes)
    #print(class_counts)

    # Create artificial imbalanced class counts
    # imbal_class_counts = [500, 5000] * 5
    if(client_id < client_num/2):
        imbal_class_counts = [5000, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    else:
        imbal_class_counts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 5000]
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
model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
(x_train, y_train), (x_test, y_test) = data_load(client_id)

# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        # model.get_weights() = list[numpy.ndarray]
        #id_byte  = str(client_id).encode()
        #print(type(np.array(id_byte)), np.array(id_byte)) #<class 'numpy.ndarray'> b'1'
        result = model.get_weights()
        return result

    def fit(self, parameters, config):
        #print(type(parameters[-1][), str(parameters[-1]), client_id)
        #if(isNum(str(parameters[-1][0]))):
            #parameters.pop()
        #print(type(parameters[-1]), str(parameters[-1]), client_id)
        #print(f'parameters\' length: {len(parameters)}')
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=2, batch_size=32)
        result = model.get_weights()
        #result.append(client_id)
        #result.append(np.array(str(client_id).encode()))
        #print(result[-1], client_id)
        return result, len(x_train), {}

    def evaluate(self, parameters, config):
        #print(str(parameters[-1]))
        #if(str(parameters[-1]).isdigit()):
            #parameters.pop()
        #print(f'parameters\' length: {len(parameters)}')
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(server_address="192.168.50.179:8080", client=CifarClient())
