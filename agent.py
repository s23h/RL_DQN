############################################################################
############################################################################
# THIS IS THE ONLY FILE YOU SHOULD EDIT
#
#
# Agent must always have these five functions:
#     __init__(self)
#     has_finished_episode(self)
#     get_next_action(self, state)
#     set_next_state_and_distance(self, next_state, distance_to_goal)
#     get_greedy_action(self, state)
#
#
# You may add any other functions as you wish
############################################################################
############################################################################

import numpy as np
import torch
import random
from collections import deque


# The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=100)
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=100)
        self.layer_3 = torch.nn.Linear(in_features=100, out_features=100)
        self.output_layer = torch.nn.Linear(in_features=100, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        layer_3_output = torch.nn.functional.relu(self.layer_3(layer_2_output))
        output = self.output_layer(layer_3_output)
        return output

# The DQN class determines how to train the above neural network.
class DQN:

    # The class initialisation function.
    def __init__(self, config):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=4)
#         # Create target Q-network, which predicts the q-value for a particular state.
        self.target_network = Network(input_dimension=2, output_dimension=4)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=config['learning_rate'])
        #
        self.config = config

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, transition):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss(transition)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item()

    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, transition):
        mini_batch = np.array(transition, dtype=object)
        minibatch_states_tensor = torch.tensor(np.stack(mini_batch[:, 0]), dtype=torch.float32)
        minibatch_actions_tensor = torch.tensor(mini_batch[:, 1].astype(np.int64), dtype=torch.int64)
        minibatch_reward_tensor = torch.tensor(mini_batch[:, 2].astype(np.float32), dtype=torch.float32).unsqueeze(-1)
        minibatch_next_states_tensor = torch.tensor(np.stack(mini_batch[:, 3]), dtype=torch.float32).detach()
        network_prediction = self.q_network.forward(minibatch_states_tensor).gather(dim=1, index=minibatch_actions_tensor.unsqueeze(1))
        state_q_values = self.target_network.forward(minibatch_next_states_tensor).detach()
        action_tensor = state_q_values.argmax(1).detach()
        state_action_q_values = state_q_values.gather(dim=1, index=action_tensor.unsqueeze(-1)).squeeze(1).unsqueeze(-1).detach()
        loss = torch.nn.MSELoss()(minibatch_reward_tensor + self.config['gamma']*state_action_q_values, network_prediction)
        return loss


    def update_target(self):
        network_weights = self.q_network.state_dict()
        self.target_network.load_state_dict(network_weights)


class ReplayBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=5000)

    def add_transition(self, t):
        self.buffer.append(t)

    def get_sample(self, n):
        return random.sample(self.buffer, n)


class Agent:

    # Function to initialise the agent
    def __init__(self, config):
        # Set the episode length
        self.episode_length = config['episode_length']
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None
        # epsilon to get epsilon-greedy actions
        self.epsilon = 1
        # q_network
        self.dqn = DQN(config)
        # ReplayBuffer
        self.buffer = ReplayBuffer()
        # Batch size
        self.batch_size = config['batch_size']
        #
        self.config = config


    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        if self.num_steps_taken % self.episode_length == 0:
            return True
        else:
            return False

    def _discrete_action_to_continuous(self, discrete_action):
        if discrete_action == 0:
            # Move 0.1 to the right, and 0 upwards
            continuous_action = np.array([0.02, 0], dtype=np.float32)
        if discrete_action == 1:
            # Move 0 to the right, and 0.1 down
            continuous_action = np.array([0, -0.02], dtype=np.float32)
        if discrete_action == 2:
            # Move 0.1 to the left, and 0 upwards
            continuous_action = np.array([-0.02, 0], dtype=np.float32)
        if discrete_action == 3:
            # Move 0 to the right, and 0.1 upwards
            continuous_action = np.array([0, 0.02], dtype=np.float32)
        return continuous_action

    # Function to get the next action, using whatever method you like
    def get_next_action(self, state):
        # Here, the action is random, but you can change this
        # action = np.random.uniform(low=-0.01, high=0.01, size=2).astype(np.float32)
        action_space = 4
        action = random.choice(list(range(action_space)))

        # Epsilon greedy
        p = random.random()
        if p>self.epsilon:
            action = self.get_greedy_action(state, ret_int=True)
        else:
            action = random.choice(list(range(action_space)))

        # Update the number of steps which the agent has taken
        self.num_steps_taken += 1
        # Store the state; this will be used later, when storing the transition
        self.state = state
        # Store the action; this will be used later, when storing the transition
        self.action = action
        return self._discrete_action_to_continuous(action)

    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        # Convert the distance to a reward
        reward = 1 - distance_to_goal
        # Create a transition
        transition = (self.state, self.action, reward, next_state)
        self.buffer.add_transition(transition)
        if self.num_steps_taken>=self.config['batch_size']:
            minibatch_inputs = self.buffer.get_sample(self.batch_size)
            loss = self.dqn.train_q_network(minibatch_inputs)

            if self.num_steps_taken>self.config['batch_size']*2 and self.num_steps_taken % self.config['network_update'] == 0:
                self.dqn.update_target()
                if self.epsilon>0.30:
                    self.epsilon = self.epsilon*self.config['epsilon_decay']

    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state, ret_int=False):
        # Here, the greedy action is fixed, but you should change it so that it returns the action with the highest Q-value
        # action = np.array([0.02, 0.0], dtype=np.float32)
        state_q_values = self.dqn.q_network.forward(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).detach().numpy()
        action = np.argmax(state_q_values)
        if ret_int:
            return action
        else:
            return self._discrete_action_to_continuous(action)
