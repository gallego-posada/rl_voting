import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import matplotlib.pyplot as plt
import os
import numpy as np
from voting import *
import pylab 

try:
    xrange = xrange
except:
    xrange = range

RENDER = 1

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

class Net(object):

    def __init__(self, lr, state_dim,action_dim,architecture, activations,id=''):

        tf.reset_default_graph()

        self.name = 'nn_' + str(state_dim) + '_' + '_'.join([str(_) for _ in architecture]) +'_' + str(action_dim)+'__'+id
        self.folder_name = 'models/' + self.name + '/'
        self.file_name = self.folder_name + self.name
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)


        #These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        self.state_in= tf.placeholder(shape=[None,state_dim],dtype=tf.float32,name='state_in')

        prev_layer = self.state_in
        for i in range(len(architecture)):
            hidden = slim.fully_connected(prev_layer, architecture[i], activation_fn = activations[i])
            prev_layer = hidden

        self.output = slim.fully_connected(hidden, action_dim, activation_fn = tf.nn.softmax)
        self.chosen_action = tf.argmax(self.output,1)

        #The next six lines establish the training proceedure. We feed the reward and chosen action into the network
        #to compute the loss, and use it to update the network.
        self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32,name='reward')
        self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32,name='action')

        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)

        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx,var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)

        self.gradients = tf.gradients(self.loss,tvars)
        self.update_batch = self.optimizer.apply_gradients(zip(self.gradient_holders,tvars))
        self.saver = tf.train.Saver()
        self.sess = tf.Session()

    def train(self, total_episodes = 5000, max_ep = 999, update_frequency = 5,saveModel=True):

        init = tf.global_variables_initializer()

        # Launch the tensorflow graph
        with self.sess as sess:
            sess.run(init)
            i = 0
            total_reward = []
            total_lenght = []

            gradBuffer = sess.run(tf.trainable_variables())
            for ix,grad in enumerate(gradBuffer):
              gradBuffer[ix] = grad * 0

            while i < total_episodes:

                # Get first observation and initialize
                observation = env.reset()
                running_reward = 0
                ep_history = []

                for j in range(max_ep):

                    if (len(total_reward) > 100 and np.mean(total_reward[-100:]) > 300) and RENDER>0:
                        env.render()

                    action = self.action(observation)

                    # Execute action and store observation and reward
                    new_obs, rwd, done, _ = env.step(action)
                    ep_history.append([observation, action, rwd, new_obs])
                    observation = new_obs
                    running_reward += rwd

                    # Update network
                    if done:
                        ep_history = np.array(ep_history)
                        ep_history[:,2] = discount_rewards(ep_history[:,2])
                        feed_dict={self.reward_holder:ep_history[:,2],
                              self.action_holder:ep_history[:,1],self.state_in:np.vstack(ep_history[:,0])}

                        grads = sess.run(self.gradients, feed_dict=feed_dict)
                        for idx,grad in enumerate(grads):
                          gradBuffer[idx] += grad

                        if i % update_frequency == 0 and i != 0:
                            feed_dict= dictionary = dict(zip(self.gradient_holders, gradBuffer))
                            _ = sess.run(self.update_batch, feed_dict=feed_dict)
                            for ix,grad in enumerate(gradBuffer):
                                gradBuffer[ix] = grad * 0

                        total_reward.append(running_reward)
                        total_lenght.append(j)
                        break

                #Update our running tally of scores.
                if i % 100 == 0 and len(total_reward) >= 100:
                    print("Avg reward: %f" % np.mean(total_reward[-100:]))
                i += 1
			#save model
            if saveModel:
                if not os.path.exists(self.folder_name):
                    os.makedirs(self.folder_name)
                self.saver.save(sess, self.file_name)


    def action(self, state_obs):
        #Probabilistically pick an action given our network outputs.
        a_dist = self.sess.run(self.output,feed_dict={self.state_in:[state_obs]})
        return a_dist[0]

    def run(self, total_episodes = 100):

        # Launch the tensorflow graph
        with self.sess as sess:

            # Obtain an initial observation of the environment
            observation = env.reset()

            eps_num = 0
            reward_sum = 0

            while eps_num < total_episodes:

                # Render according to flag
                if RENDER > 0:
                    env.render()

                action = self.action(observation)

                observation, reward, done, _ = env.step(action)

                reward_sum += reward
                if done:
                    eps_num += 1
                    print ("Reward for this episode was:",reward_sum)
                    reward_sum = 0
                    env.reset()
            env.render(close=True)

    def load_network(self, sess, path):
        t_vars = tf.trainable_variables()
        #self.own_vars = [var for var in t_vars if self.name == var.name.split("/")[0]]
        with tf.variable_scope(self.name):
            new_saver = tf.train.Saver(t_vars)
            new_saver.restore(sess, tf.train.latest_checkpoint(path))

def load(state_dim, action_dim, architecture, activations, id):
    agent = Net(1e-2, state_dim, action_dim, architecture, activations,id)
    agent.load_network(agent.sess, agent.folder_name)
    return agent

if __name__ == "__main__":

    # Definition of the enviroment
    env = gym.make('CartPole-v1')
    gamma = 0.99

    # Clear TF graph
    tf.reset_default_graph()

    TRAIN = False
    TEST = True

    # Create agent
    if(TRAIN):
        state_dim = 4
        action_dim = 11
        architecture = [5,5]
        activations = [tf.nn.tanh, tf.nn.tanh]

	    #Create an agent and train it
        #agent = Net(1e-2, state_dim, action_dim, architecture, activations)
        #agent.train(total_episodes=5000,saveModel=True)

	    #Load and agent and test it
        #network = load(state_dim,action_dim,architecture,activations)
        #network.run(total_episodes=50)

        all_architecures=[[3],[5],[10],[3,3],[5,5]]
        all_activations = [[tf.nn.tanh],[tf.nn.tanh],[tf.nn.tanh],[tf.nn.tanh, tf.nn.tanh],[tf.nn.tanh, tf.nn.tanh]]
        nb_model=2
        for j in range (0,nb_model):
            for i in range(0,len(all_architecures)):
                print(str(j)+'_'+str(i))
                #Create an agent, train it and save it
                agent = Net(1e-2, state_dim, action_dim, all_architecures[i], all_activations[i],str(j))
                agent.train()
    elif(TEST):
        state_dim = 4
        action_dim = 11
        all_architecures=[[3],[5],[10],[3,3],[5,5]]
        all_activations = [[tf.nn.tanh],[tf.nn.tanh],[tf.nn.tanh],[tf.nn.tanh, tf.nn.tanh],[tf.nn.tanh, tf.nn.tanh]]
        nb_model=2

        total_episodes = 200
        test_alone_index = -1 #index of the network to be tested alone, without voting rule, -1 for none

        #TODO: ADD SOME NOISE TO THE OBSERVATIONS

        # Possible voting rules: plurality, borda, hundred_points, copeland
        voting_rule = copeland

        # Load all networks
        networks = []
        all_rewards=[]

        voting_rule = plurality

        for i in range(0,len(all_architecures)):
            for j in range(0,nb_model):
                networks.append(load(state_dim,action_dim,all_architecures[i],all_activations[i],str(j)))

        # Launch the tensorflow graph
        with tf.Session() as sess:

            # Obtain an initial observation of the environment
            observation = env.reset()

            eps_num = 0
            reward_sum = 0

            while eps_num < total_episodes:

                observation += np.random.normal(0.0, 0.3, size=observation.shape)

                # Render according to flag
                if RENDER > 0:
                    env.render()

                # Just use one agent
                if (test_alone_index != -1) :
                    a_dist = networks[test_alone_index].action(observation)
                    a = np.random.choice(len(a_dist),1,p = a_dist)
                    action = a[0]

                # Actually vote
                else:

                    Q_function_list= []
                    for i in range(len(networks)):
                            Q_function_list.append(networks[i].action(observation))

                    # Get the index of the action
                    action = int(voting_rule(np.array(Q_function_list)))

                observation, reward, done, _ = env.step(action)

                reward_sum += reward
                if done:
                    eps_num += 1
                    print ("Reward for this episode was:",reward_sum)
                    all_rewards.append(reward_sum)
                    reward_sum = 0
                    env.reset()
            env.render(close=True)
            x = np.linspace(1, total_episodes, num=total_episodes)
            pylab.plot(x,all_rewards)
            pylab.xlim(0,total_episodes)
            pylab.ylim(0,500)
            pylab.show()

