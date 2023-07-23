#from sklearn.cluster import KMeans, AgglomerativeClustering
#from sklearn.decomposition import PCA
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
#from matplotlib import pyplot as plt
#from scipy.cluster.hierarchy import dendrogram
#from sklearn.feature_selection import RFECV
from sklearn.metrics import pairwise_distances
from gym import Env
from gym.spaces import Discrete, Box
import random

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from functools import reduce
from typing import List, Tuple
from flwr.common import (
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

#env = gym.make('CartPole-v0')
client_num = 3
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

class SelectEnv(Env):
    def __init__(self):
        # Actions we can take, down, stay, up
        self.action_space = [[0, 1, 1], [1, 0, 1], [1 ,1 , 0], [1 ,1 ,1]]
        # Temperature array
        #self.observation_space = Box(low=np.array([0]), high=np.array([100]))
        # Set start temp
        self.state = [1]*client_num
        # Set shower length
        self.acc = 0.4 #use to control done's condition
        self.round = 5
        
    def step(self, action):
        global models
        actions = self.action_space[action]
        print(actions)
        agg_models = [models[i] for i in range(len(actions)) if actions[i] == 1]
        parameters_aggregated = aggregate(agg_models)     
        global global_model   
        global_model.set_weights(parameters_aggregated)
        (x_train, y_train), (x_test, y_test) = data_load(999)
        loss, accuracy = global_model.evaluate(x_test, y_test)

        print(f'Aggregate, loss: {loss}, accuracy: {accuracy}')
        
        # Calculate reward
        #if accuracy >= self.acc and self.state <=39: 
        #    reward =1 
        #else: 
        #    reward = -1 
        reward = accuracy*100
        
        self.round -= 1
        # Check if is done
        if accuracy >= self.acc or self.round == 0: 
            done = True
        else:
            done = False
        
        # Apply temperature noise
        #self.state += random.randint(-1,1)
        # Set placeholder for info
        info = {}
        models = []
        for client_id in range(client_num):
            weight, loss, accuracy = client_fit(client_id)
            models.append(weight)

        states = build_distance_matrix(models)[0]
        # Return step information
        return states, reward, done, info

    def render(self):
        # Implement viz
        pass
    
    def reset(self):
        # Reset shower temperature
        #self.state = 38 + random.randint(-3,3)
        # Reset shower time
        #self.shower_length = 60 
        return self.state

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
    # conv2d_2 (Conv2D)            (None, 16, 16, 64)        18496     
    # _________________________________________________________________
    # batch_normalization_2 (Batch (None, 16, 16, 64)        256       
    # _________________________________________________________________
    # conv2d_3 (Conv2D)            (None, 16, 16, 64)        36928     
    # _________________________________________________________________
    # batch_normalization_3 (Batch (None, 16, 16, 64)        256       
    # _________________________________________________________________
    # max_pooling2d_1 (MaxPooling2 (None, 8, 8, 64)          0         
    # _________________________________________________________________
    # dropout_1 (Dropout)          (None, 8, 8, 64)          0         
    # _________________________________________________________________
    # conv2d_4 (Conv2D)            (None, 8, 8, 128)         73856     
    # _________________________________________________________________
    # batch_normalization_4 (Batch (None, 8, 8, 128)         512       
    # _________________________________________________________________
    # conv2d_5 (Conv2D)            (None, 8, 8, 128)         147584    
    # _________________________________________________________________
    # batch_normalization_5 (Batch (None, 8, 8, 128)         512       
    # _________________________________________________________________
    # max_pooling2d_2 (MaxPooling2 (None, 4, 4, 128)         0         
    # _________________________________________________________________
    # dropout_2 (Dropout)          (None, 4, 4, 128)         0         
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
    Total params: 552,874 -> 815.146(no padding)
    Trainable params: 551,722
    Non-trainable params: 1,152 
    '''
    num_classes = 10
    model = Sequential()

    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3))) # padding='same', 

    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    #model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3,3), activation='relu')) # padding='same', 
    #model.add(layers.BatchNormalization())
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
    #model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))    # num_classes = 10
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    # Checking the model summary
    model.summary()
    return model

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def plot_acc_loss(loss, acc):
    host = host_subplot(111)  # row=1 col=1 first pic
    plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window
    par1 = host.twinx()   # 共享x軸

    # set labels
    host.set_xlabel("steps")
    host.set_ylabel("test-loss")
    par1.set_ylabel("test-accuracy")

    # plot curves
    p1, = host.plot(range(len(loss)), loss, label="loss")
    p2, = par1.plot(range(len(acc)), acc, label="accuracy")

    # set location of the legend,
    # 1->rightup corner, 2->leftup corner, 3->leftdown corner
    # 4->rightdown corner, 5->rightmid ...
    host.legend(loc=5)

    # set label color
    host.axis["left"].label.set_color(p1.get_color())
    par1.axis["right"].label.set_color(p2.get_color())

    # set the range of x axis of host and y axis of par1
    # host.set_xlim([-200, 5200])
    # par1.set_ylim([-0.1, 1.1])

    plt.draw()
    plt.show()

def data_load(client_id):
    
    # Load CIFAR10
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data() #10 class 5000 instance each

    # Get all training targets and count the number of class instances
    classes, class_counts = np.unique(y_train , return_counts=True)
    nb_classes = len(classes)
    #print(class_counts)

    # Create artificial imbalanced class counts
    # imbal_class_counts = [500, 5000] * 5
    if(client_id == 999):
        imbal_class_counts = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
    elif(client_id < client_num/2):
        imbal_class_counts = [1000, 1000, 1000, 1000, 1000, 0, 0, 0, 0, 0]
    else:
        imbal_class_counts = [0, 0, 0, 0, 0, 1000, 1000, 1000, 1000, 1000]
    #print(client_id, imbal_class_counts)

    # Get class indices
    class_indices = [np.where(y_train == i)[0] for i in range(nb_classes)]
    #print(class_indices)

    # Get imbalanced number of instances
    imbal_class_indices = [class_idx[:class_count] for class_idx, class_count in zip(class_indices, imbal_class_counts)]
    imbal_class_indices = np.hstack(imbal_class_indices)

    # Set target and data to dataset
    y_train = y_train[imbal_class_indices]
    x_train = x_train[imbal_class_indices]
    #print(client_id, 'len: ', len(x_train), len(y_train))
    assert len(x_train) == len(y_train)

    #new = model.fit(x_train, y_train, epochs=2, batch_size=32)
    #cols = new.get_support(indices=True)
    #X_1=x_train.iloc[:, cols]
    #print(X_1)

    return (x_train, y_train), (x_test, y_test)

def get_gradients(model):
    grads = []
    for param in model.parameters():
        g = param.grad.view(-1).tolist()
        grads.append(g)
    return grads

def aggregate(results) -> NDArrays:
    """Compute weighted average."""
    # Calculate the total number of examples used during training#

    # Create a list of weights, each multiplied by the related number of examples
    #weighted_weights = [
    #    [layer for layer in weights] for weights in results
    #]

    # Compute average weights of each layer
    weighted_weights = [
        [layer for layer in weights] for weights in results
    ]

    #print(len(weighted_weights))
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / len(results)
        for layer_updates in zip(*weighted_weights)
    ]
    #print(len(weights_prime))
    #weights: NDArrays = [reduce(np.add, re) for re in zip(*results)]
    
    #weights_prime = [x/len(results) for x in weights]
    return weights_prime

def build_distance_matrix(models):
    weights = []
    result_list = []
    for weight in models:
        result_list = []
        for i in range(len(weight)):
            result = weight[i].flatten() # to 1D
            result_list.extend(result)
        weights.append(result_list)
    distance_matrix = pairwise_distances(weights, metric='euclidean')
    print(distance_matrix)
    print(type(distance_matrix))
    print(len(distance_matrix))
    return distance_matrix

def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1,states)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn

def client_fit(client_id):
    # Load model and data (MobileNetV2, CIFAR-10)
    #model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
    global global_model
    model = global_model

    (x_train, y_train), (x_test, y_test) = data_load(client_id)

    model.fit(x_train, y_train, epochs=1, batch_size=128)
    
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'Client: {client_id}, loss: {loss}, accuracy: {accuracy}')
    return model.get_weights(), loss, accuracy


    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=10000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn

if __name__ == '__main__':
    weights = []
    gradients = []
    models = []
    global_model = model4cifar10()

    for client_id in range(client_num):
        weight, loss, accuracy = client_fit(client_id)
        models.append(weight)

    env = SelectEnv()
    env.state = build_distance_matrix(models)[0]
    states = client_num

    '''episodes = 10
    for episode in range(1, episodes+1):
        state = states
        done = False
        score = 0 
        
        while not done:
            #env.render()

            action = random.sample(env.action_space, 1)[0] #DQN decide
            n_state, reward, done, info = env.step(action)
            score+=reward
        print('Episode:{} Score:{}'.format(episode, score)) '''
    actions = len(env.action_space)
    fl_model = build_model(states, actions)

    dqn = build_agent(fl_model, actions)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    dqn.fit(env, nb_steps=10, visualize=False, verbose=1)
    print("SCORES")
    scores = dqn.test(env, nb_episodes=3, visualize=False)
    print(np.mean(scores.history['episode_reward']))



'''
    result = model.get_weights()
    #print(model.trainable_variables)
    
    #print('np.shape(result) = ', np.shape(result))
    #print('\n', list(np.shape(x) for x in result))
    r_list = []
    for i in range(len(result)):
        #if i < 3:
            #print(i, np.shape(result[i]), "\n", result[i])
        r = result[i].flatten()
            #print(i, np.shape(r), "\n", r)
        r_list.extend(r)
    x = tf.Variable(r_list)
    #with tf.GradientTape(persistent=True) as g:
        #g.watch(x)
        #y = x**2
    #print(g.gradient(y, x))
    #gradients.append(g.gradient(y, x))
    weights.append(r_list)
'''

#print('np.shape(weights) = ', np.shape(weights))
'''
# Clustering
#cluster = AgglomerativeClustering(n_clusters=2, distance_threshold=None, compute_distances=True).fit(gradients)
cluster = AgglomerativeClustering(n_clusters=None, distance_threshold=0, compute_distances=True).fit(weights)
#cluster = KMeans(n_clusters=2, n_init='auto').fit(weights)
labels = cluster.labels_
#centers = kmeans.cluster_centers_
print(labels)
#print(cluster.n_leaves_)
#print(cluster.children_)
#print(cluster.distances_)
plt.title('Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
plot_dendrogram(cluster, truncate_mode='level', p=10)
plt.xlabel("Client Id")
plt.show()
'''