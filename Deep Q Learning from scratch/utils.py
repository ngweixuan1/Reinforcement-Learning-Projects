import torch
import torch.nn as nn
import gym
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)
torch.manual_seed(1)

class TwoLayerNet(nn.Module):
    def __init__(self, inputdim, layer1dim,layer2dim, n_action):
        super(TwoLayerNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(inputdim, layer1dim),
            nn.ReLU(),
            nn.Linear(layer1dim, layer2dim),
            nn.ReLU(),
            nn.Linear(layer2dim, n_action))

        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters())
        

    def forward(self, x):
        output = self.layer(x)
        return output
    
    
class QLearningAgent():
    def __init__(self, gamma, epsilon, epsilon_decay, batch_size, layer1dim, layer2dim, early_stopping_score, environ = "LunarLander-v2"):
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.env = gym.make(environ)
        self.total_states = self.env.observation_space.shape[0]
        self.total_actions = self.env.action_space.n

        self.Q =  torch.zeros((self.total_states, self.total_actions))  #Initialize Q(s,a)
        self.dnn = TwoLayerNet(self.total_states, layer1dim, layer2dim, self.total_actions)
        self.dnn2 = TwoLayerNet(self.total_states, layer1dim, layer2dim, self.total_actions)
        self.score= []
        self.buffer_size = 100000
        self.observation_buffer= np.zeros((self.buffer_size, self.total_states), dtype=np.float64)
        self.reward_buffer = np.zeros(self.buffer_size , dtype=np.float64)
        self.observation_next_buffer = np.zeros((self.buffer_size, self.total_states) , dtype=np.float64)
        self.action_buffer = np.zeros(self.buffer_size , dtype=np.float64)
        self.done = np.zeros(self.buffer_size , dtype=np.float64)
        self.current_buffer = 0
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        self.replace_count = 0
        self.target_replace = 1000
        self.index = 0
        self.step_counter =0
        self.score_all = None
        self.early_stopping_score = early_stopping_score
        print("Initialized agent. Gamma = {}, decay= {}, batch_size = {}". format(self.gamma, self.epsilon_decay, self.batch_size))

    def train(self, episodes):
        score_all = []
        ## algorithm per 6.5
        for ep in range(episodes): #Loop for each episode:
            observation = self.env.reset() # Initialize S per Gym documentation
            observation = observation.reshape(1,-1) 
            done = False
            score = 0
            
            while done == False: # Loop for each step until S is terminal
                # Choose A from S using policy derived from Q (e.g., epilson-greedy)
                
                if np.random.rand() >= self.epsilon:
                    action = torch.argmax(self.dnn.forward(torch.tensor(observation).float().reshape((1,-1)))).item()
                else:
                    action = np.random.randint(self.total_actions)
                
                observation_next, reward, done, _ = self.env.step(action) # Take action A, observe R, S'
                score += reward
                self.index = self.current_buffer % self.buffer_size                
                self.observation_buffer[self.index] = torch.tensor(observation)
                self.reward_buffer[self.index] = reward
                self.observation_next_buffer[self.index] = torch.tensor(observation_next)
                self.action_buffer[self.index] = action
                self.optimize()
                observation = observation_next
                self.current_buffer +=1
                self.index +=1
                self.current_buffer = min(self.index, self.current_buffer)
                self.step_counter +=1 
            self.epsilon = max(self.epsilon_decay*self.epsilon, self.epsilon_min)
            score_all.append(score)
            if ep % 50 == 0 and ep > 95:
                print("Mean score across past 100 episodes at step {step} is {score}".format(step = ep, score = np.mean(score_all[-100:])))
            if ep > 130:
                if sum(score_all[-125:])/125 >= self.early_stopping_score:
                    print("Score above {} in last 125 episodes. Stopping Training.".format(self.early_stopping_score))
                    break
        return score_all

    def optimize(self):

        if self.current_buffer >= self.batch_size:
            self.dnn.optimizer.zero_grad()
            if self.replace_count% self.target_replace ==0:
                self.dnn2.load_state_dict(self.dnn.state_dict())
            self.replace_count +=1
            
            sample_index = np.random.choice(min(self.step_counter, self.buffer_size), self.batch_size, replace=False)
            
            predictor = torch.tensor(self.observation_buffer[sample_index]).float()
            predictor = torch.autograd.Variable(predictor.data, requires_grad=False)

            predictor_next = torch.tensor(self.observation_next_buffer[sample_index]).float()
            predictor_next = torch.autograd.Variable(predictor_next.data, requires_grad=False)

            done_next_state = torch.tensor(self.done[sample_index])

            action = self.action_buffer[sample_index]
            
            forwardpass = self.dnn.forward(predictor) 
            predict = forwardpass[range(forwardpass.shape[0]),action]
            q_nextstate = torch.max(self.dnn2.forward(predictor_next),dim=1)[0] 
            
            target = (torch.tensor(self.reward_buffer[sample_index]).float() + self.gamma*q_nextstate*(done_next_state==0).type_as(q_nextstate)).float()
            target = torch.autograd.Variable(target.data, requires_grad=True).float()
            
            loss = self.dnn.loss(target, predict)
            
            loss.backward()
            self.dnn.optimizer.step()
            
    def test(self):
        score_all = []
        for _ in range(100):
            score = 0
            observation = self.env.reset()
            done = False
            while done == False:
                action = torch.argmax(self.dnn.forward(torch.tensor(observation).float().reshape(1,-1)), dim =1) ####THIIIS
                observation_next, reward, done, _ = self.env.step(action.numpy()[0])
                score += reward
                observation = observation_next 
            score_all.append(score)
        print( "Mean test score across 100 episodes is {}".format(sum(score_all)/len(score_all)))
        print("The standard deviation is {}".format(np.std(score_all)))
        return score_all

def plot_train(score_all, plot_moving):
    plt.plot(list(range(len(score_all))), score_all, color="blue")
    if plot_moving:
        moving_average = np.convolve(score_all, np.ones(100), 'valid') / 100
        plt.plot(np.array(list(range(len(score_all))))[99:], moving_average, color="red")
        plt.legend(["Training Score", "Moving Average 100 eps"])
    plt.title("Training Score")
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.savefig('training_score.png')
    plt.clf()


def plot_test(score_all, legend, filename):
    
    for i in score_all:
        plt.plot(list(range(100)), i,linewidth= 0.5)
        plt.annotate("Mean:" + str((sum(i)/len(i))), xy=(101, i[-1]))
    plt.title("Test Score")
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.legend(legend, loc = "upper right")
    plt.savefig(filename)
    plt.clf()

def plot_train2(score_all, legend, filename, filename2):
    for i in score_all:
        plt.plot(list(range(len(i))), i,linewidth= 0.3)
    plt.title("Training Score")
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.legend(legend)
    plt.savefig(filename)
    plt.clf()

    for i in score_all:
        moving_average = np.convolve(i, np.ones(100), 'valid') / 100
        plt.plot(np.array(list(range(len(i))))[99:], moving_average)
    plt.title("Training Score Moving Average")
    plt.ylabel('Average Score')
    plt.xlabel('Episode')
    plt.legend(legend)
    plt.savefig(filename2)
    plt.clf()