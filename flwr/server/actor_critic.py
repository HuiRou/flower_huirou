from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Add, Input
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# setting seed for reproducibility of results. This is not super important.
tf.set_random_seed(2212)

class Actor:
    def __init__(self, sess, action_dim, observation_dim):
        # setting our created session as default session
        self.sess = sess
        K.set_session(sess)
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.state_input, self.output, self.model = self.create_model()
        model_weights = self.model.trainable_weights
        # Placeholder for critic gradients with respect to action_input.
        self.actor_critic_grads = tf.placeholder(tf.float32, [None, action_dim])
        # Adding small number inside log to avoid log(0) = -infinity
        log_prob = tf.math.log(self.output + 10e-10)
        # Multiply log by -1 to convert the optimization problem as minimization problem.
        # This step is essential because apply_gradients always do minimization.
        neg_log_prob = tf.multiply(log_prob, -1)
        # Calulate and update the weights of the model to optimize the actor
        self.actor_grads = tf.gradients(neg_log_prob, model_weights, self.actor_critic_grads)
        grads = zip(self.actor_grads, model_weights)
        self.optimize = tf.train.AdamOptimizer(0.001).apply_gradients(grads)

    def create_model(self):
        state_input = Input(shape=self.observation_dim)
        state_h1 = Dense(24, activation='relu')(state_input)
        state_h2 = Dense(24, activation='relu')(state_h1)
        output = Dense(self.action_dim, activation='softmax')(state_h2)
        model = Model(inputs=state_input, outputs=output)
        adam = Adam(lr=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=adam)
        return state_input, output, model

    def train(self, critic_gradients_val, X_states):
        self.sess.run(self.optimize, feed_dict={self.state_input:X_states, self.actor_critic_grads:critic_gradients_val})



# setting seed for reproducibility of results. This is not super important.
tf.set_random_seed(2212)

class Critic:
    def __init__(self, sess, action_dim, observation_dim):
        # setting our created session as default session
        K.set_session(sess)
        self.sess = sess
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.state_input, self.action_input, self.output, self.model = self.create_model()
        self.critic_gradients = tf.gradients(self.output, self.action_input)

    def create_model(self):
        state_input = Input(shape=self.observation_dim)
        state_h1 = Dense(24, activation='relu')(state_input)
        state_h2 = Dense(24, activation='relu')(state_h1)

        action_input = Input(shape=(self.action_dim,))
        action_h1 = Dense(24, activation='relu')(action_input)
        action_h2 = Dense(24, activation='relu')(action_h1)

        state_action = Add()([state_h2, action_h2])
        state_action_h1 = Dense(24, activation='relu')(state_action)
        output = Dense(1, activation='linear')(state_action_h1)

        model = Model(inputs=[state_input, action_input], outputs=output)
        model.compile(loss="mse", optimizer=Adam(lr=0.005))
        return state_input, action_input, output, model

    def get_critic_gradients(self, X_states, X_actions):
        # critic gradients with respect to action_input to feed in the weight updation of actor
        critic_gradients_val = self.sess.run(self.critic_gradients, feed_dict={self.state_input:X_states, self.action_input:X_actions})
        return critic_gradients_val[0]