from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

class SelectEnv(Env):
    def __init__(self):
        # Actions we can take, down, stay, up
        self.action_space = []
        # Temperature array
        #self.observation_space = Box(low=np.array([0]), high=np.array([100]))
        # Set start temp
        self.action = []
        self.state = []
        # Set shower length
        self.acc = 0.4 #use to control done's condition
        self.round = 5

        
    def step(self, action):
        print(action)
        actions = self.action_space[action]
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

    def build_model(self, states, actions):
        model = Sequential()
        model.add(Flatten(input_shape=(1,states)))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(actions, activation='linear'))
        return model

    def build_agent(self, model, actions):
        policy = BoltzmannQPolicy()
        memory = SequentialMemory(limit=50000, window_length=1)
        dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                    nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
        return dqn