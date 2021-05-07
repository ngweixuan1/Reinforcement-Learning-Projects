import torch
import torch.nn as nn
import numpy as np
import torchvision
import gym
import argparse
import yaml
from utils import QLearningAgent, plot_train, plot_test, plot_train2
np.random.seed(1)
torch.manual_seed(1)
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config.yaml')

if __name__ == '__main__':
    global args
    print("Load configs...")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)
    for k, v in config.items():
        setattr(args, k, v)
    gamma = args.gamma
    epsilon = args.epsilon
    batch_size = args.batch_size
    episodes = args.episodes
    epsilon_decay = args.epsilon_decay
    layer1dim = args.layer1dim
    layer2dim = args.layer2dim
    plot_moving = args.plot_moving
    plot_one = args.plot_one
    plot_gamma_tune = args.plot_gamma_tune
    gamma_list = args.gamma_list
    plot_decay_tune = args.plot_decay_tune
    decay_list =args.decay_list
    plot_batch_tune = args.plot_batch_tune
    batch_size_list = args.batch_size_list
    early_stopping_score = args.early_stopping_score
    plot_gamma_decay = args.plot_gamma_decay
    environment  = args.environment 


    if plot_one:
        agent = QLearningAgent(gamma = gamma, epsilon = epsilon, epsilon_decay = epsilon_decay, batch_size = batch_size, layer1dim = layer1dim, layer2dim = layer2dim, early_stopping_score = early_stopping_score,
        environ = environment
        )
        score_all = agent.train(episodes = episodes)
        plot_train(score_all, plot_moving)
        total_score = agent.test()
        torch.save(agent.dnn, 'dnn.pt')
        torch.save(agent.dnn2, 'dnn2.pt')
        score_list = []
        score_list.append(total_score)
        plot_test(score_list, ["Test Score"], "test_score.png")

    if plot_gamma_tune:
        print("Tuning gamma...")
        score_train = []
        score_test = []
        for i in gamma_list:
            agent = QLearningAgent(gamma = i, epsilon = epsilon, epsilon_decay = epsilon_decay, batch_size = batch_size, layer1dim = layer1dim, layer2dim = layer2dim, early_stopping_score = early_stopping_score, environ = environment)
            score_train.append(agent.train(episodes = episodes))
            score_test.append(agent.test())
            torch.save(agent.dnn, 'dnn_gamma' + str(i)+ '.pt')
            torch.save(agent.dnn2, 'dnn2_gamma' + str(i)+ '.pt')
        plot_train2(score_train, gamma_list, "train_gamma.png", "train_gamma_avg.png")
        plot_test(score_test, gamma_list, "test_gamma.png")

    if plot_decay_tune:
        print("Tuning decay...")
        score_train = []
        score_test = []
        for i in decay_list:
            agent = QLearningAgent(gamma = gamma, epsilon = epsilon, epsilon_decay = i, batch_size = batch_size, layer1dim = layer1dim, layer2dim = layer2dim, early_stopping_score = early_stopping_score, environ = environment)
            score_train.append(agent.train(episodes = episodes))
            score_test.append(agent.test())
            torch.save(agent.dnn, 'dnn_decay' + str(i)+ '.pt')
            torch.save(agent.dnn2, 'dnn2_decay' + str(i)+ '.pt')
        plot_train2(score_train, decay_list, "train_decay.png", "train_decay_avg.png")
        plot_test(score_test, decay_list, "test_decay.png")

    if plot_batch_tune:
        print("Tuning batch size...")
        score_train = []
        score_test = []
        for i in batch_size_list:
            agent = QLearningAgent(gamma = gamma, epsilon = epsilon, epsilon_decay = epsilon_decay, batch_size = i, layer1dim = layer1dim, layer2dim = layer2dim, early_stopping_score = early_stopping_score, environ = environment)
            score_train.append(agent.train(episodes = episodes))
            score_test.append(agent.test())
            torch.save(agent.dnn, 'dnn_batch' + str(i)+ '.pt')
            torch.save(agent.dnn2, 'dnn2_batch' + str(i)+ '.pt')
        plot_train2(score_train, batch_size_list, "train_batch_size.png", "train_batch_size_avg.png")
        plot_test(score_test, batch_size_list, "test_batch_size.png")

    if plot_gamma_decay:
        print("Tuning batch size...")
        score_test = np.zeros((len(decay_list), len(gamma_list)))
        for i in range(len(decay_list)):
            for j in range(len(gamma_list)):
                agent = QLearningAgent(gamma = gamma_list[j], epsilon = epsilon, epsilon_decay = decay_list[i], batch_size = batch_size, layer1dim = layer1dim, layer2dim = layer2dim, early_stopping_score = early_stopping_score, environ = environment)
                score_train = agent.train(episodes = episodes)
                score_all = agent.test()
                score_test[i,j] = sum(score_all)/len(score_all)

        np.savetxt("gamma_decay_test.csv", score_test, delimiter=",")
        