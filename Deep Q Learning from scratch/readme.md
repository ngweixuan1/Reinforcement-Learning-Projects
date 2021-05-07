# Reinforcement Learning Project 2


## Libraries
Python 3.7.7

numpy 

matplotlib

yaml

argparse

torch

torchvision

gym

## Running the code

1. Place your configuarations in config.yaml file. The default values are listed here:


Default Values:

gamma: 0.99

epsilon: 1

batch_size: 64

episodes: 2000

epsilon_decay: 0.995

layer1dim: 128

layer2dim: 64


No hyperparameters tuning:

plot_one: True # plot graph with no hyperparameter tuning

plot_moving: True # To determine if moving average is to be plotted


Gamma Tuning:

plot_gamma_tune: True

gamma_list: [0.5, 0.7, 0.8, 0.9] # List of gamma values

Epsilon Decay Tuning:

plot_decay_tune: True

decay_list: [0.8, 0.99, 0.995, 0.998]


Batch Size Tuning:

plot_batch_tune: True

batch_size_list: [32, 64, 128, 256]


Gamma and Epsilon Decay Values Tuning:True

plot_gamma_decay: True


Other Values:

early_stopping_score: 225

environment: "LunarLander-v2" 


2. Run main.py using python main.py


3. Output of the code would be printed on the console or saved in the same directory.

Other folders:

Please refer to the graphs folder for all output graphs, notebook folder for all notebooks for individual experiments.