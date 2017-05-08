import gym
import numpy as np
from gym import wrappers

NUM_EPISODES = 100
STEP_GOAL = 450
env = gym.make('CartPole-v1')
#env = wrappers.Monitor(env, '/tmp/cartpole-experiment-2')


###########################################
# SOLUTION 1
# Author: athuldevin
# Source: https://gym.openai.com/evaluations/eval_B2SpXVpzQJqyXw0jCYPIg
###########################################

def sol1(observation):
	
    R=np.array([.05,.05,.05,.05])
    L=np.array([-.05,-.05,-.05,-.05])
    RR=[False,False,False,False]
    LR=[False,False,False,False]
    obse = observation

    for i in range(4):
        if obse[i]>R[i]:
            RR[i]=True
        else:
            RR[i]=False
        if obse[i]<L[i]:
            LR[i]=True
        else:
            LR[i]=False

    # Initialize action
    action = 0

    if RR[1]==True:
        action=0
    if LR[1]==True:
        action=1
    if RR[0]==True:
        action=1
    if LR[0]==True:
        action=0
    if RR[3]==True:
        action=1
    if LR[3]==True:
        action=0
    if RR[2]==True:
        action=1
    if LR[2]==True:
        action=0

    return action

###########################################
# SOLUTION 2
# Author: luongminh97
# Source: https://gym.openai.com/evaluations/eval_OMmLPsSQ7a1bs5zEMraQ
###########################################

class Net(object):
    def __init__(self, input, output):
        # W = 0.01 * np.random.randn(input, output)
        # b = np.zeros([1, output])  # Random initialization
        W = np.array([[-0.60944746, 0.45539405],
                      [0.53731965, -0.15138026],
                      [-2.38568372, 1.7827912],
                      [-5.02650308, 5.4449609]])
        b = np.array([[0.39728291, 2.86320634]])  # Optimized parameters
        self.params = [W, b]
        self.grad = [np.zeros_like(W), np.zeros_like(b)]
        self.lr = 0.005
        self.gamma = 0.95

    def forward(self, x):
        W, b = self.params
        out = x.dot(W) + b
        return out

    def update_params(self, x, dout):
        #  SGD + Momentum
        W, b = self.params
        db = dout.sum(axis=0)
        dW = x.T.dot(dout)
        grad = [dW, db]
        for g, gc in zip(grad, self.grad):
            gc *= self.gamma
            gc += self.lr * g
        for w, d in zip(self.params, self.grad):
            w -= d

def sol2(observation):
    actor = Net(4, 2)
    
    obse = observation      
    
    # Normalize state
    obse -= np.array([0.00076127, 0.01893811, 0.00292497, -0.01291666])
    obse /= np.array([0.09564344, 0.57818071, 0.10437309, 0.87035585])
    action = np.argmax(actor.forward(obse[None, :]), axis=1)[0]
    
    return action

###########################################
# RUN GAME
###########################################

def run_game(agent1, agent2):
    
    for i_episode in range(NUM_EPISODES):
        env.reset()

        #Set arbitrary initial action
        action = 0

        for t in range(STEP_GOAL):
        	# Render environment, comment to reduce simulation overhead
            env.render()

            # Execute previously chosen action and observe environment
            observation, reward, done, info = env.step(action)

            # Add noise signal
            obs_noise = 0.1 * (np.random.rand(len(observation)) - 0.5)
            observation += obs_noise

            if np.random.rand() > 0.5:
            	action = agent1(observation)
            else:
            	action = agent2(observation)

            # If we lose or reach the goal
            if done or t == STEP_GOAL - 1:

            	# If we finish because we reach the goal, indicate with +
            	if t == STEP_GOAL - 1:
            		print'+ Episode finished after {} timesteps'.format(t+1)
                
                # Else it is because we loose, indicate with -
                else:
                    print'- Episode finished after {} timesteps'.format(t+1)
                break
               
run_game(sol1, sol2)