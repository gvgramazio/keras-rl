import numpy as np
import gym
import gym_foosball

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

import string
import random

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

class FoosballProcessor(Processor):
    def process_reward(self, reward):
        """Processes the reward as obtained from the environment for use in an agent and
        returns it.

        # Arguments
            reward (float): A reward as obtained by the environment

        # Returns
            Reward obtained by the environment processed
        """
        return float(reward - 1)


rand_str = id_generator()

ENV_NAME = 'FoosballDiscrete_sp-v0'
LOG_FILEPATH = 'dqn_{}_'.format(ENV_NAME)+rand_str+'.json'
WINDOW_LENGTH = 4


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(WINDOW_LENGTH,) + env.observation_space.shape))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
policy = BoltzmannQPolicy()
processor = FoosballProcessor()
agent = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=200,
               target_model_update=1e-2, policy=policy, processor=processor, batch_size=128)
agent.compile(Adam(lr=1e-4), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
callbacks = [FileLogger(LOG_FILEPATH, interval=1)]
agent.fit(env, callbacks=callbacks, nb_steps=500000, nb_max_episode_steps=200, visualize=False, verbose=0)

# After training is done, we save the final weights.
agent.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
# agent.test(env, nb_episodes=5, visualize=True)
