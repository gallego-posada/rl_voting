# Imports
import os
import datetime
import scipy as sp
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import gym
import tensorflow as tf
import tensorflow.contrib.slim as slim
from voting import *

plt.set_cmap('hot')

# Python 2 and 3 compatibility
try:
    xrange = xrange
except:
    xrange = range

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

    def train(self, total_episodes = 5000, max_ep = 999, update_frequency = 5, saveModel = True):

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

                    if (len(total_reward) > 100 and np.mean(total_reward[-100:]) > 300):
                        print("Avg reward goal achieved. Stopping.")
                        # Increase episode number to break outer loop
                        i  = total_episodes + 1
                        break

                        if RENDER > 0:
                            env.render()

                    a_dist = self.action(observation)
                    a = np.random.choice(len(a_dist),1,p = a_dist)
                    action = a[0]

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

                # Increase episode number
                i += 1

			#save model
            if saveModel:
                if not os.path.exists(self.folder_name):
                    os.makedirs(self.folder_name)
                self.saver.save(sess, self.file_name)

    def action(self, state_obs):
        #Probabilistically pick an action given our network outputs.
        a_dist = self.sess.run(self.output,feed_dict={self.state_in:[state_obs]})
        return a_dist[0] #This was a list with a list inside

    def load_network(self, sess, path):
        t_vars = tf.trainable_variables()
        #self.own_vars = [var for var in t_vars if self.name == var.name.split("/")[0]]
        with tf.variable_scope(self.name):
            new_saver = tf.train.Saver(t_vars)
            new_saver.restore(sess, tf.train.latest_checkpoint(path))

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def load(state_dim, action_dim, architecture, activations, id):
    agent = Net(1e-2, state_dim, action_dim, architecture, activations,id)
    agent.load_network(agent.sess, agent.folder_name)
    return agent

def train_agents(state_dim, action_dim, all_architecures, all_activations, nb_model, total_episodes):
    for j in range (0,nb_model):
        for i in range(0,len(all_architecures)):
            print("Agent number: " + str(j)+'_'+str(i))
            #Create an agent, train it and save it
            agent = Net(1e-2, state_dim, action_dim, all_architecures[i], all_activations[i],str(j))
            agent.train(total_episodes = total_episodes)

def noise_obs(sigma_frac, observation, fixed_noise = None):
    # Covariance matrix
    if (fixed_noise is not None):
        return observation + fixed_noise
    else:
        cov_mat = sigma_frac * np.diag(np.array([ 0.50549634,  0.29165111,  0.03309178,  0.18741141]))**2
        return observation + np.random.multivariate_normal(np.zeros(len(observation)), cov_mat)

def run_simulation(voting_rule, networks, sigma_frac, total_episodes, test_alone_index, noise_type, eps_length = 999, verbose = False):

    all_rewards = []

    # Collect Statistics: Kendall's Tau, Spearman's Rho, Top 1 Match
    all_stats = {}
    for rule in set([borda, plurality, copeland, hundred_points]) - {voting_rule}:
        all_stats[rule.__name__] = [[], [], []]

    # If no noise, then take 0 variance for the noise:
    if noise_type == -1:
        sigma_frac = 0

    # Launch the tensorflow graph
    with tf.Session() as sess:

        # Current episode counter
        eps_num = 0


        # #Precompute fixed noise (noise_type= -1 or 2 or 3)
        # #-1: No noise
        # #0: Global noise changing every episods, fixed every timestep
        # #1: Noise per networks changing every episods, fixed every timestep
        # #2: Global fixed noise: fixed every episod and every timestep
        # #3: Noise per network fixed: fixed every episod and every timestep
        # #Computing the fixed noise (fixed with respect to the episods and the time step): this is as if agents had a blurry version of the world
        # fixed_noise = []
        # observation = env.reset()
        #
        # for i in range(len(networks)):
        #     if(noise_type==-1):
        #         #Noise is null
        #         fixed_noise.append(0)
        #     elif(noise_type==2):
        #         #Noise is the same for all networks
        #         if(i==0):
        #             fixed_noise.append(noise_obs(sigma_frac, 0*observation))
        #         else:
        #             fixed_noise.append(fixed_noise[0])
        #     elif(noise_type==3):
        #         #Noise is different per networks
        #         fixed_noise.append(noise_obs(sigma_frac, 0*observation))


        while eps_num < total_episodes:

            # Obtain an initial observation of the environment
            observation = env.reset()

            # Accumulates the reward for the current episode
            reward_sum = 0

            # #Computing the noise that changes every episod (but that is the same for every timestep)
            # if (noise_type == 0):
            #     #Noise is the same for all networks
            #     if(i==0):
            #         fixed_noise.append(noise_obs(sigma_frac, 0*observation))
            #     else:
            #         fixed_noise.append(fixed_noise[0])
            #
            # if (noise_type == 1):
            #     #Noise is different per networks
            #     for i in range(len(networks)):
            #         fixed_noise.append(noise_obs(sigma_frac, 0*observation))

            for time_step in range(eps_length):

                # Render according to flag
                if RENDER > 0:
                    env.render()

                # Just use one agent
                if (test_alone_index != -1) :
                    curr_obs = noise_obs(sigma_frac, observation) #,fixed_noise[0])
                    a_dist = networks[test_alone_index].action(curr_obs)
                    a = np.random.choice(len(a_dist), 1, p = a_dist)
                    action = a[0]
                # Actually vote
                else:
                    Q_function_list= []
                    for i in range(len(networks)):
                        curr_obs = noise_obs(sigma_frac, observation) #,fixed_noise[i])
                        ballot = networks[i].action(curr_obs)
                        Q_function_list.append(ballot)

                    # Get the index of the action
                    social_ranking = vote(np.array(Q_function_list), voting_rule)
                    action = int(social_ranking[0])

                    # Calculate Kendall's Tau
                    for rule in set([borda, plurality, copeland, hundred_points]) - {voting_rule}:
                        comparison_ranking = vote(np.array(Q_function_list), rule)
                        tau, _ = scipy.stats.kendalltau(social_ranking, comparison_ranking)
                        all_stats[rule.__name__][0].append(tau)
                        rho, _ = scipy.stats.spearmanr(social_ranking, comparison_ranking)
                        all_stats[rule.__name__][1].append(rho)
                        all_stats[rule.__name__][2].append(1. * (action == int(comparison_ranking[0])))

                # Execute selected action
                observation, reward, done, _ = env.step(action)

                reward_sum += reward

                if done or (time_step == eps_length-1):
                    eps_num += 1
                    if verbose:
                        print ("Reward for this episode was:",reward_sum)
                    all_rewards.append(reward_sum)
                    observation = env.reset()
                    reward_sum = 0
                    break

            env.render(close=True)

    return all_rewards, all_stats

def mean_confidence_interval(data, confidence=0.95):
    # Author: https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m-h, m, m+h


if __name__ == "__main__":

    # Definition of the enviroment
    env = gym.make('CartPole-v1')
    gamma = 0.99

    # Clear TF graph
    tf.reset_default_graph()

    RENDER = 0
    TRAIN = False
    TEST = True

    # Networks parameters
    state_dim = 4
    action_dim = 11
    all_architecures = [[3],[5],[10],[3,3],[5,5]]
    all_activations = [[tf.nn.tanh],[tf.nn.tanh],[tf.nn.tanh],[tf.nn.tanh, tf.nn.tanh],[tf.nn.tanh, tf.nn.tanh]]
    nb_model = 2

    if TRAIN:
        train_agents(state_dim, action_dim, all_architecures, all_activations, nb_model, total_episodes = 1000)

    if TEST:

        # ============= HYPERPARMETERS ===================
        #-1: No noise
        #0: Global noise changing every episods
        #1: Noise per networks changing every episods
        #2: Global fixed noise
        #3: Noise per network fixed
        noise_type = -1
        sigma_frac = 20
        #test_alone_index
        total_episodes = 100 #default 100
        all_rules = [borda, plurality, copeland, hundred_points]
        eps_len = 500 #default 999
        # ============= END HYPERPARMETERS ===================

        for test_alone_index in range(-1,0): #index of the network to be tested alone from 0 to 9 , without voting rule, -1 for all networks that vote

            # Optimize the testing
            if (test_alone_index > -1) :
                print("Testing Network: %d\n" % test_alone_index)
                all_rules = [plurality]
            else:
                print("Testing Ensemble\n")

            # Load all networks
            networks = []
            for i in range(0,len(all_architecures)):
                for j in range(0,nb_model):
                    networks.append(load(state_dim,action_dim,all_architecures[i],all_activations[i],str(j)))

            # Execute simulation with each of the rules
            list_rewards = []
            for voting_rule in all_rules:
                print("\nRule: %s" % voting_rule.__name__)
                l_rwds, dict_stats = run_simulation(voting_rule, networks, sigma_frac, total_episodes, test_alone_index, noise_type, eps_length = eps_len)
                list_rewards.append(l_rwds)
                for rule in set([borda, plurality, copeland, hundred_points]) - {voting_rule}:
                    print("\t%s - Tau: %f" % (rule.__name__, np.mean(dict_stats[rule.__name__][0])))
                    print("\t%s - Rho: %f" % (rule.__name__, np.mean(dict_stats[rule.__name__][1])))
                    print("\t%s - Top 1 Match: %f" % (rule.__name__, np.mean(dict_stats[rule.__name__][2])))
                #print(l_stats)

            # Run t tests for difference of means.
            print('\nStatistical Tests\n')
            # H0 means equal means, if we reject H0 we are concluding that the
            # two samples come from populations with different means, i.e., the
            # two rules have significantly different average performance
            for ix, rule_1 in enumerate(all_rules):
                for jx, rule_2 in enumerate(all_rules):
                    if ix < jx:
                        t, p = scipy.stats.ttest_ind(list_rewards[ix], list_rewards[jx], equal_var=False)
                        if p < 0.05:
                            print("Reject H0 for %s vs %s - p-value: %f" % (rule_1.__name__, rule_2.__name__, p))
                        else:
                            print("Can NOT Reject H0 for %s vs %s - p-value: %f" % (rule_1.__name__, rule_2.__name__, p))

            # Plot results, print confidence intervals
            print('\nConfidence Intervals\n')
            marker_list = ['ro', 'g*', 'bv', 'k.']
            for ix, voting_rule in enumerate(all_rules):
                curr_rewards = list_rewards[ix]
                print(voting_rule.__name__ + ": " + str(mean_confidence_interval(curr_rewards)))
                plt.plot(range(1, len(curr_rewards) + 1), list_rewards[ix],
                        marker = marker_list[ix][1], color = marker_list[ix][0],
                        label = voting_rule.__name__)
                plt.xlim(1, len(curr_rewards))
                #plt.ylim(0, 510)
                plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=4, mode="expand", borderaxespad=0.)

            plt.savefig('plots/' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '.png')
