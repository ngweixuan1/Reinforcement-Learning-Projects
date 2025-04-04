# -*- coding: utf-8 -*-
"""CEQ use expected value.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1AUCse2pdd9pU0jJAm0_Bxc5JGWzJzQQg
"""

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from cvxopt import matrix, solvers ##Used cvxopt due to slow speed of cvxpy implementation (non-vectorizable)
np.random.seed(1)

def plot_error(errors, title):
    plt.plot(range(len(errors)), errors)
    plt.title(title)
    plt.xlabel('Simulation Iteartion')
    plt.ylabel('Q-value Difference')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
    plt.ylim(0, 0.5)

    plt.show()
    #plt.savefig(title+'.png')
    plt.clf()

class SoccerGame():
    def __init__(self):
        '''
        Grid:
        |0 | 2 | 4 | 6|
        |1 | 3 | 5 | 7|
        '''
        self.observation_space_n = 8
        self.action_space_n = 5
        player_a_start = 4
        player_b_start = 2
        self.player_a = player(player_a_start, has_ball=False)
        self.player_b = player(player_b_start, has_ball=True)
        self.goal_a = [0,1]
        self.goal_b = [6,7]
        
    def reset(self):
        player_a_start = 4
        player_b_start = 2
        self.player_a = player(player_a_start, has_ball=False)
        self.player_b = player(player_b_start, has_ball=True)
        ball_possession = 0 if self.player_a.has_ball else 1
        return self.player_a.position, self.player_b.position, ball_possession
    
    def step(self, action_a, action_b):
        move_first = "a" if np.random.random() < 0.5 else "b"
        done = False
        reward_a = 0
        reward_b = 0
        new_pos_a = self.move(self.player_a, action_a)
        new_pos_b = self.move(self.player_b, action_b)
        
            
        if move_first == "a":
            if new_pos_a == self.player_b.position and new_pos_a != new_pos_b:
                if self.player_a.has_ball:
                    self.player_a.has_ball == False
                    self.player_b.has_ball == True
                if new_pos_b == self.player_a.position:
                    if self.player_b.has_ball:
                        self.player_a.has_ball == True
                        self.player_b.has_ball == False
                else:
                    self.player_b.position = new_pos_b
            elif new_pos_a == new_pos_b and new_pos_a != self.player_b.position:
                if self.player_b.has_ball:
                    self.player_a.has_ball == True
                    self.player_b.has_ball == False
                self.player_a.position = new_pos_a
            elif new_pos_a != self.player_b.position and new_pos_a != new_pos_b:
                self.player_a.position = new_pos_a
                self.player_b.position = new_pos_b
        elif move_first == "b":
            if new_pos_b == self.player_a.position and new_pos_b != new_pos_a:
                if self.player_b.has_ball:
                    self.player_b.has_ball == False
                    self.player_a.has_ball == True
                if new_pos_a == self.player_b.position:
                    if self.player_a.has_ball:
                        self.player_b.has_ball == True
                        self.player_a.has_ball == False
                else:
                    self.player_a.position = new_pos_a
            elif new_pos_b == new_pos_a and new_pos_b != self.player_a.position:
                if self.player_a.has_ball:
                    self.player_b.has_ball == True
                    self.player_a.has_ball == False
                self.player_b.position = new_pos_b
            else:
                self.player_b.position = new_pos_b
                self.player_a.position = new_pos_a
        if (self.player_a.has_ball) == True:
            if (self.player_a.position in self.goal_a):
                done = True                
                reward_a = 100
                reward_b = -100
            elif (self.player_a.position in self.goal_b):
                done = True                
                reward_a = -100
                reward_b = 100            
        elif (self.player_b.has_ball) == True:
            if (self.player_b.position in self.goal_b):
                done = True                
                reward_b = 100
                reward_a = -100
            elif (self.player_b.position in self.goal_a):
                done = True                
                reward_b = -100
                reward_a = 100                
        assert self.player_a.has_ball != self.player_b.has_ball    
        ball_possession = 0 if self.player_a.has_ball else 1
        return self.player_a.position, self.player_b.position, ball_possession, reward_a, reward_b, done
        
    def move(self, player, action):
        '''
        North: 0, South: 1, Right: 2, Left: 3, Stick: 4
        '''
        new_position = player.position
        if action == 0:
            if player.position in [1,3,5,7]:
                new_position =  player.position - 1
        elif action == 1:
            if player.position in [0,2,4,6]:
                new_position =  player.position + 1
        elif action == 2:
            if player.position not in [6,7]:
                new_position =  player.position + 2
        elif action == 3:
            if player.position not in [0,1]:
                new_position =  player.position - 2
        return new_position
                
class player():
    def __init__(self, position, has_ball = False):
        self.position = position
        self.has_ball = has_ball

class CEQAgent():
    def __init__(self, iterations, alpha, alpha_decay, alpha_min, gamma, epsilon, epsilon_decay, epsilon_min, environment):
        self.alpha = alpha
        self.gamma = gamma
        self.env = environment
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.alpha_min = alpha_min
        self.iterations = iterations

    def solve(self):
        
        total_states = self.env.observation_space_n
        total_actions = self.env.action_space_n
        Q_a = np.ones((total_states, total_states, 2, total_actions, total_actions))
        Q_b = np.ones((total_states, total_states, 2, total_actions, total_actions))
        prob = np.ones((total_actions, total_actions)) / total_actions**2
        ev_a_all = np.ones((total_states, total_states, 2, total_actions *total_actions))/total_actions**2
        ev_b_all = np.ones((total_states, total_states, 2, total_actions *total_actions))/total_actions**2
        errors = []
        done = False
        for ep in range(self.iterations):
            if ep%10000 ==0:
                print(ep)
            if done or ep == 0:
                S_a, S_b, S_ball = self.env.reset()
                
                if done:
                    self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)
                done = False 

            old_Q_a = Q_a[4,2,1,1,4]
            if np.random.random() < self.epsilon:
                action_a = np.random.randint(total_actions)
                action_b = np.random.randint(total_actions)
            else:
                if ep > 0:
                    temp_var = self.LP(Q_a[S_a, S_b, S_ball], Q_b[S_a, S_b, S_ball])
                    if temp_var is not None:
                        prob, ev_a, ev_b = temp_var
                        ev_a_all[S_a, S_b, S_ball] = ev_a
                        ev_b_all[S_a, S_b, S_ball] = ev_b
                if prob is not None:
                    action_a_b =  np.random.choice(list(range(total_actions**2)), 1, p=prob)
                    action_a = action_a_b //5
                    action_b = action_a_b%5
                else:
                    action_a = np.random.randint(total_actions)
                    action_b = np.random.randint(total_actions)            

            S_a_next, S_b_next, S_ball_next, reward_a, reward_b, done = self.env.step(action_a, action_b)
            
            if done:
                Q_a[S_a, S_b, S_ball, action_a, action_b] =   (1-self.alpha)*Q_a[S_a, S_b, S_ball, action_a, action_b] + self.alpha*(reward_a)
                Q_b[S_a, S_b, S_ball, action_a, action_b] =  (1-self.alpha)*Q_b[S_a, S_b, S_ball, action_a, action_b] + self.alpha*(reward_b)                   
            else:

                Q_a[S_a, S_b, S_ball, action_a, action_b] =   (1-self.alpha)*Q_a[S_a, S_b, S_ball, action_a, action_b] + self.alpha*(reward_a + self.gamma * np.sum(ev_a_all[S_a_next, S_b_next, S_ball_next]) )
                Q_b[S_a, S_b, S_ball, action_a, action_b] = (1-self.alpha)*Q_b[S_a, S_b, S_ball, action_a, action_b] + self.alpha*(reward_b + self.gamma * np.sum(ev_b_all[S_a_next, S_b_next, S_ball_next]) )
            S_a = S_a_next
            S_b = S_b_next
            S_ball = S_ball_next
            new_Q_a = Q_a[4,2,1,1,4]
            errors.append(np.abs(new_Q_a - old_Q_a))
            self.alpha = max(self.alpha*self.alpha_decay, self.alpha_min)
        return Q_a, Q_b, errors 
    def LP(self, Q_a, Q_b):
        H, W = Q_a.shape
        N = H*W
        constraints = []
        for action_a in range(H):
            for other_action_a in range(W):
                if action_a!=other_action_a:
                    temp= np.zeros(N)
                    for action_b in range(H):
                        temp[H*action_a+action_b] = Q_a[action_a,action_b] - Q_a[other_action_a,action_b]
                    constraints.append(temp)
        for action_b in range(H):
            for other_action_b in range(W):
                if action_b!=other_action_b:
                    temp= np.zeros(N)
                    for action_a in range(H):
                        temp[H*action_a+action_b] = Q_b[action_a,action_b] - Q_b[action_a,other_action_b]
                    constraints.append(temp)
        G = matrix(np.concatenate((-np.array(constraints),-np.eye(N)), axis=0))
        h = matrix(np.zeros(np.array(constraints).shape[0] + N))
        A = np.ones((1, N))
        A[0,0] = 0
        A = matrix(A)
        b = matrix(1.0)        
        c = matrix((Q_a + Q_b).reshape(H*W))  
        solvers.options['show_progress'] = False
        prob = solvers.lp(c=c, G=G, h=h, A=A, b=b)

        if np.array(prob['x']).all() is not None:

            solution= np.array(prob['x']).reshape(N)
            result = np.abs(solution)
            sum_of_x = sum(result)
            a = 1/sum_of_x
            x_scaled = [e*a for e in result]
            EV_a =  Q_a.reshape(N)*x_scaled
            EV_b =  Q_b.reshape(N)*x_scaled

            return x_scaled, EV_a, EV_b



iterations = 1000000
alpha = 1
alpha_decay = 0.999992
alpha_min = 0.001
gamma = 0.99
epsilon = 1
epsilon_decay = 0.999995
epsilon_min = 0.001
environment = SoccerGame()
qlearn = CEQAgent(iterations, alpha, alpha_decay, alpha_min, gamma, epsilon, epsilon_decay, epsilon_min, environment)
Q_a, Q_b, errors = qlearn.solve()
plot_error(errors, 'CE-Q-learning')

iterations = 1000000
alpha = 1
alpha_decay = 0.9999
alpha_min = 0.001
gamma = 0.99
epsilon = 1
epsilon_decay = 0.999995
epsilon_min = 0.001
environment = SoccerGame()
qlearn = CEQAgent(iterations, alpha, alpha_decay, alpha_min, gamma, epsilon, epsilon_decay, epsilon_min, environment)
Q_a, Q_b, errors = qlearn.solve()
plot_error(errors, 'CE-Q-learning')