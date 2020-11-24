import time
import numpy as np
import wandb

from random_environment import Environment
from agent import Agent

wandb.init(project='RL_Part_2')

sweep_config = {
    'method': 'random', #grid, random
    'metric': {
      'name': 'distance_to_goal',
      'goal': 'minimize'
    },
    'parameters': {
        'batch_size': {
            'values': [32, 64, 128, 256, 512, 1024]
        },

        'learning_rate': {
            'values': [0.1, 0.05, 0.01, 0.005]
        },

        'epsilon_decay': {
            'values': [0.999, 0.995, 0.99]
        },

        'gamma': {
            'values': [0.99, 0.95, 0.9]
        },
        'episode_length': {
            'values': [100, 200, 500, 1000, 2000]
        },
        'network_update': {
            'values': [50, 100, 250, 500]
        }

    }
}

sweep_id = wandb.sweep(sweep_config, project="RL_Part_2")

def train():

    config_defaults = {
        'batch_size': 400,
        'learning_rate': 0.01,
        'gamma': 0.95,
        'epsilon_decay': 0.993,
        'episode_length': 1000,
        'network_update': 50
    }

    # Initialize a new wandb run
    wandb.init(config=config_defaults)

    # Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config

    # This determines whether the environment will be displayed on each each step.
    # When we train your code for the 10 minute period, we will not display the environment.
    display_on = False

    # Create a random seed, which will define the environment
    random_seed = int(time.time())
    np.random.seed(random_seed)

    # Create a random environment
    environment = Environment(magnification=500)

    # Create an agent
    agent = Agent(config_defaults)

    # Get the initial state
    state = environment.init_state

    # Determine the time at which training will stop, i.e. in 10 minutes (600 seconds) time
    start_time = time.time()
    end_time = start_time + 600

    # Train the agent, until the time is up
    while time.time() < end_time:
        # If the action is to start a new episode, then reset the state
        if agent.has_finished_episode():
            state = environment.init_state
        # Get the state and action from the agent
        action = agent.get_next_action(state)
        # Get the next state and the distance to the goal
        next_state, distance_to_goal = environment.step(state, action)
        wandb.log({"loss":distance_to_goal})
        # Return this to the agent
        agent.set_next_state_and_distance(next_state, distance_to_goal)
        # Set what the new state is
        state = next_state
        # Optionally, show the environment
        if display_on:
            environment.show(state)
        if agent.has_finished_episode():
            wandb.log({"episode_end_distance":distance_to_goal})

    # Test the agent for 100 steps, using its greedy policy
    state = environment.init_state
    has_reached_goal = False
    for step_num in range(100):
        action = agent.get_greedy_action(state)
        next_state, distance_to_goal = environment.step(state, action)
        # The agent must achieve a maximum distance of 0.03 for use to consider it "reaching the goal"
        if distance_to_goal < 0.03:
            has_reached_goal = True
            wandb.log({"goal_reached":has_reached_goal})

            break
        state = next_state

    # Print out the result
    if has_reached_goal:
        print('Reached goal in ' + str(step_num) + ' steps.')
    else:
        print('Did not reach goal. Final distance = ' + str(distance_to_goal))
    wandb.log({"distance_to_goal":distance_to_goal})


wandb.agent(sweep_id, train)
