import numpy as np
import gym
import gym_foosball

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.agents import ContinuousDQNAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

class FoosballProcessor(Processor):
    def process_step(self, observation, reward, done, info):
        """Processes an entire step by applying the processor to the observation, reward, and info arguments.

        # Arguments
            observation (object): An observation as obtained by the environment.
            reward (float): A reward as obtained by the environment.
            done (boolean): `True` if the environment is in a terminal state, `False` otherwise.
            info (dict): The debug info dictionary as obtained by the environment.

        # Returns
            The tupel (observation, reward, done, reward) with with all elements after being processed.
        """
        self.done = done
        observation = self.process_observation(observation)
        reward = self.process_reward(reward)
        info = self.process_info(info)
        return observation, reward, done, info

    def process_reward(self, reward):
        """Processes the reward as obtained from the environment for use in an agent and
        returns it.

        # Arguments
            reward (float): A reward as obtained by the environment

        # Returns
            Reward obtained by the environment processed
        """
        if self.done:
            return reward / 100.
        else:
            return (reward - 1) / 100.

import string
import random

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

rand_str = id_generator()

ENV_NAME = 'Foosball_sp-v0'
LOG_FILEPATH = 'cdqn_{}_'.format(ENV_NAME)+rand_str+'.json'
WINDOW_LENGTH = 4

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

# Build all necessary models: V, mu, and L networks.
V_model = Sequential()
V_model.add(Flatten(input_shape=(WINDOW_LENGTH,) + env.observation_space.shape))
V_model.add(Dense(512))
V_model.add(Activation('relu'))
V_model.add(Dense(512))
V_model.add(Activation('relu'))
V_model.add(Dense(256))
V_model.add(Activation('relu'))
V_model.add(Dense(128))
V_model.add(Activation('relu'))
V_model.add(Dense(1))
V_model.add(Activation('linear'))
print(V_model.summary())

mu_model = Sequential()
mu_model.add(Flatten(input_shape=(WINDOW_LENGTH,) + env.observation_space.shape))
mu_model.add(Dense(512))
mu_model.add(Activation('relu'))
mu_model.add(Dense(512))
mu_model.add(Activation('relu'))
mu_model.add(Dense(256))
mu_model.add(Activation('relu'))
mu_model.add(Dense(128))
mu_model.add(Activation('relu'))
mu_model.add(Dense(nb_actions))
mu_model.add(Activation('linear'))
print(mu_model.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(WINDOW_LENGTH,) + env.observation_space.shape, name='observation_input')
x = Concatenate()([action_input, Flatten()(observation_input)])
x = Dense(512)(x)
x = Activation('relu')(x)
x = Dense(512)(x)
x = Activation('relu')(x)
x = Dense(256)(x)
x = Activation('relu')(x)
x = Dense(128)(x)
x = Activation('relu')(x)
x = Dense(((nb_actions * nb_actions + nb_actions) // 2))(x)
x = Activation('linear')(x)
L_model = Model(inputs=[action_input, observation_input], outputs=x)
print(L_model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
processor = FoosballProcessor()
memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.3, size=nb_actions)
agent = ContinuousDQNAgent(nb_actions=nb_actions, V_model=V_model, L_model=L_model, mu_model=mu_model,
                 memory=memory, nb_steps_warmup=200, random_process=random_process,
                 gamma=.99, target_model_update=1e-3, processor=processor, batch_size=64)
agent.compile(Adam(lr=1e-4, clipnorm=1.), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
callbacks = [FileLogger(LOG_FILEPATH, interval=1)]
agent.fit(env, nb_steps=250000, visualize=False, verbose=0, nb_max_episode_steps=200, callbacks=callbacks)

# After training is done, we save the final weights.
agent.save_weights('cdqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
# agent.test(env, nb_episodes=10, visualize=True, nb_max_episode_steps=200)
