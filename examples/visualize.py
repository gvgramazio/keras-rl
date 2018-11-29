import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Filepath to .json file
# e.g.
# file_hashes = [
#     'A0YU0J',
#     'ADAW5Y',
#     'DA6HAW',
#     'XBOOX9',
#     'YK608P']
# agent = 'duel_dqn'
# env = 'CartPole-v0'
file_hashes = []
agent = ''
env = ''

assert len(file_hashes), "You need at least one file hash!"

filenames = ['{}_{}_{}.json'.format(agent, env, h) for h in file_hashes]

figsize = {}
figsize['x'] = 12.
figsize['y'] = 8.

subfigsize = {}
subfigsize['x'] = 12.
subfigsize['y'] = 4.5

# Load all the simulations in a list of dataframes
dfs = [pd.read_json(fn, orient='columns') for fn in filenames]

# Concatenate them
df = pd.concat([d for d in dfs], ignore_index=True)

# Extract data to plot
episode = df.episode.unique()
episode_reward = {}
mean_q = {}
mean_absolute_error = {}
loss = {}
episode_reward['mean'] = df.groupby('episode').episode_reward.mean().rolling(10).mean()
episode_reward['std'] = df.groupby('episode').episode_reward.std().rolling(10).mean()
mean_q['mean'] = df.groupby('episode').mean_q.mean().rolling(10).mean()
mean_q['std'] = df.groupby('episode').mean_q.std().rolling(10).mean()
mean_absolute_error['mean'] = df.groupby('episode').mean_absolute_error.mean().rolling(10).mean()
mean_absolute_error['std'] = df.groupby('episode').mean_absolute_error.std().rolling(10).mean()
loss['mean'] = df.groupby('episode').loss.mean().rolling(10).mean()
loss['std'] = df.groupby('episode').loss.std().rolling(10).mean()

# Configure figures
plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')
# ax = plt.subplot(111)
# ax.spines["top"].set_visible(False)
# ax.spines["bottom"].set_visible(False)
# ax.spines["right"].set_visible(False)
# ax.spines["left"].set_visible(False)
# ax.get_xaxis().tick_bottom()
# ax.get_yaxis().tick_left()
figure = {}

# Plot episode reward
figure['episode_reward'] = plt.figure(figsize=(figsize['x'], figsize['y']))
plt.fill_between(
    episode,
    episode_reward['mean'] - episode_reward['std'],
    episode_reward['mean'] + episode_reward['std'],
    color="#3F5D7D")
plt.plot(episode, episode_reward['mean'], color="white", lw=2)
plt.xlabel("episode", fontsize=32)
plt.ylabel("mean episode reward", fontsize=32)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
figure['episode_reward'].tight_layout()
figure['episode_reward'].savefig("plots/{}/{}/episode_reward.pdf".format(env, agent), bbox_inches="tight", format="pdf");
figure['episode_reward'].savefig("plots/{}/{}/episode_reward.png".format(env, agent), bbox_inches="tight", format="png");

# Plot mean q
figure['mean_q'] = plt.figure(figsize=(figsize['x'], figsize['y']))
plt.fill_between(
    episode,
    mean_q['mean'] - mean_q['std'],
    mean_q['mean'] + mean_q['std'],
    color="#3F5D7D")
plt.plot(episode, mean_q['mean'], color="white", lw=2)
plt.xlabel("episode", fontsize=32)
plt.ylabel("mean $Q$", fontsize=32)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
figure['mean_q'].tight_layout()
figure['mean_q'].savefig("plots/{}/{}/mean_q.pdf".format(env, agent), bbox_inches="tight", format="pdf");
figure['mean_q'].savefig("plots/{}/{}/mean_q.png".format(env, agent), bbox_inches="tight", format="png");

# Plot mean absolute error
figure['mean_absolute_error'] = plt.figure(figsize=(figsize['x'], figsize['y']))
plt.fill_between(
    episode,
    mean_absolute_error['mean'] - mean_absolute_error['std'],
    mean_absolute_error['mean'] + mean_absolute_error['std'],
    color="#3F5D7D")
plt.plot(episode, mean_absolute_error['mean'], color="white", lw=2)
plt.xlabel("episode", fontsize=32)
plt.ylabel("mean absolute error", fontsize=32)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
figure['mean_absolute_error'].tight_layout()
figure['mean_absolute_error'].savefig("plots/{}/{}/mean_absolute_error.pdf".format(env, agent), bbox_inches="tight", format="pdf");
figure['mean_absolute_error'].savefig("plots/{}/{}/mean_absolute_error.png".format(env, agent), bbox_inches="tight", format="png");

# Plot loss
figure['loss'] = plt.figure(figsize=(figsize['x'], figsize['y']))
plt.fill_between(
    episode,
    loss['mean'] - loss['std'],
    loss['mean'] + loss['std'],
    color="#3F5D7D")
plt.plot(episode, loss['mean'], color="white", lw=2)
plt.xlabel("episode", fontsize=32)
plt.ylabel("loss", fontsize=32)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
figure['loss'].tight_layout()
figure['loss'].savefig("plots/{}/{}/loss.pdf".format(env, agent), bbox_inches="tight", format="pdf");
figure['loss'].savefig("plots/{}/{}/loss.png".format(env, agent), bbox_inches="tight", format="png");

# Plot everything
figure['everything'], axes = plt.subplots(4, figsize=(subfigsize['x'], 4 * subfigsize['y']))
# Episode reward
axes[0].fill_between(
    episode,
    episode_reward['mean'] - episode_reward['std'],
    episode_reward['mean'] + episode_reward['std'],
    color="#3F5D7D")
axes[0].plot(episode, episode_reward['mean'], color="white", lw=2)
# axes[0].set_xlabel("episode", fontsize=32)
axes[0].set_ylabel("episode reward", fontsize=32)
axes[0].tick_params(labelsize=20)
# Mean q
axes[1].fill_between(
    episode,
    mean_q['mean'] - mean_q['std'],
    mean_q['mean'] + mean_q['std'],
    color="#3F5D7D")
axes[1].plot(episode, mean_q['mean'], color="white", lw=2)
# axes[1].set_xlabel("episode", fontsize=32)
axes[1].set_ylabel("mean $Q$", fontsize=32)
axes[1].tick_params(labelsize=20)
# Mean absolute error
axes[2].fill_between(
    episode,
    mean_absolute_error['mean'] - mean_absolute_error['std'],
    mean_absolute_error['mean'] + mean_absolute_error['std'],
    color="#3F5D7D")
axes[2].plot(episode, mean_absolute_error['mean'], color="white", lw=2)
# axes[2].set_xlabel("episode", fontsize=32)
axes[2].set_ylabel("mean absolute error", fontsize=32)
axes[2].tick_params(labelsize=20)
# Mean loss
axes[3].fill_between(
    episode,
    loss['mean'] - loss['std'],
    loss['mean'] + loss['std'],
    color="#3F5D7D")
axes[3].plot(episode, loss['mean'], color="white", lw=2)
axes[3].set_xlabel("episode", fontsize=32)
axes[3].set_ylabel("loss", fontsize=32)
axes[3].tick_params(labelsize=20)
figure['everything'].tight_layout()
figure['everything'].savefig("plots/{}/{}/everything.pdf".format(env, agent), bbox_inches="tight", format="pdf");
figure['everything'].savefig("plots/{}/{}/everything.png".format(env, agent), bbox_inches="tight", format="png");
