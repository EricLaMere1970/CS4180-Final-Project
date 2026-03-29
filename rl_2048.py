
import sys
import time
import pickle
import numpy as np
from tqdm import tqdm
from vis_2048 import *
import matplotlib
matplotlib.rcParams['agg.path.chunksize'] = 10000  # fix for large dataset plotting (1000000+)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

BOLD = '\033[1m'  # ANSI escape sequence for bold text
RESET = '\033[0m' # ANSI escape sequence to reset text formatting
train_flag = 'train' in sys.argv
gui_flag = 'gui' in sys.argv

# setup gui if needed
setup(GUI=gui_flag)
env = game

# define refresh function for GUI updates (used ai to help with this)
def refresh(env, delay=0.1):
    draw_grid(env)
    import pygame
    pygame.display.flip()
    if 'clock' in globals() and clock:
        clock.tick(60)
    time.sleep(delay)

# define hashing function for state representation
def hash_state(obs):
    # Convert to tuple and use Python's built-in hash
    return hash(tuple(obs.flatten()))

# define Q-Learning agent
def Q_learning(num_episodes=1000, decay_rate=0.999, gamma=0.9, epsilon=1): 

    # initialize Q-table
    Q_table = {}

    # count number of updates
    N_table = {}

    # intialize epsilon decay
    current_epsilon = epsilon
    
    # track training metrics
    episode_rewards = []
    episode_lengths = []
    max_tiles_achieved = []

    # training loop
    for episode in tqdm(range(num_episodes), desc="Training"):
        # reset environment
        obs = env.reset()
        total_reward = 0
        steps = 0
        
        while not env.game_over and not env.reached_2048 and steps < env.max_steps:
            # get current state
            state = hash_state(obs)
            
            # initalize state if not seen before
            if state not in Q_table:
                Q_table[state] = np.zeros(env.action_space.n)
            
            # epsilon-greedy action selection
            if np.random.random() < current_epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q_table[state])
            
            # take action
            next_obs, reward, done, info = env.step(action)
            next_state = hash_state(next_obs)
            
            # calculate reward and update counters
            total_reward += reward
            steps += 1
            
            # initialize next state if not seen before
            if next_state not in Q_table:
                Q_table[next_state] = np.zeros(env.action_space.n)
            
            # Q-learning update
            if done:
                target = reward
            else:
                target = reward + gamma * np.max(Q_table[next_state])
            
            # update visit counter and learning rate
            state_action_key = (state, action)
            N_table[state_action_key] = N_table.get(state_action_key, 0) + 1
            eta = 1 / (1 + N_table[state_action_key])
            
            # update q-value
            Q_table[state][action] += eta * (target - Q_table[state][action])
            
            # move to next state
            obs = next_obs
            
            if done:
                break
        
        # record episode statistics
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        max_tiles_achieved.append(env.get_max_tile())
        
        # decay epsilon
        current_epsilon *= decay_rate
        
    # create high-resolution training rewards plot (used ai to help format the plot)
    plt.figure(figsize=(12, 8), dpi=300)
    episodes = range(1, len(episode_rewards) + 1)
    plt.plot(episodes, episode_rewards, alpha=0.3, color='lightblue', linewidth=0.5, label='Episode Rewards')
    # moving average
    window_size = max(1, num_episodes // 50)
    if len(episode_rewards) >= window_size:
        moving_avg = []
        for i in range(len(episode_rewards)):
            start_idx = max(0, i - window_size + 1)
            end_idx = i + 1
            moving_avg.append(np.mean(episode_rewards[start_idx:end_idx]))
        plt.plot(episodes, moving_avg, color='darkblue', linewidth=2, label=f'Moving Average (window={window_size})')
    plt.title(f'Q-Learning Training Progress: Rewards per Episode\n(Episodes: {num_episodes}, ε decay: {decay_rate})', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Episode', fontsize=14, fontweight='bold')
    plt.ylabel('Total Reward', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xlim(1, num_episodes)
    plt.ylim(-1000, 2000)
    plt.tight_layout()
    plt.savefig(f'training_rewards_{num_episodes}_{decay_rate}.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # final statistics
    print(f"\nTraining completed!")
    print(f"Total states discovered: {len(Q_table)}")
    print(f"Average final reward: {np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards):.2f}")
    print(f"Max tile achieved: {np.max(max_tiles_achieved)}")
    
    return Q_table, N_table


def prune_qtable(Q_table, N_table, min_visits=5):
    print(f"Pruning Q-table")
    print(f"Original states: {len(Q_table)}")

    visited_states = set()
    for (state, action), count in N_table.items():
        if count >= min_visits:
            visited_states.add(state)

    # only keep state-action pairs visited 5+ times
    pruned_Q = {state: actions for state, actions in Q_table.items() if state in visited_states}

    print(f"Pruned states: {len(pruned_Q)} ({len(pruned_Q) / len(Q_table) * 100:.1f}% kept)")

    return pruned_Q


num_episodes = 100000
decay_rate = 0.9

# train agent
if train_flag: 
    Q_table, N_table = Q_learning(num_episodes=num_episodes, decay_rate=decay_rate, gamma=0.9, epsilon=1)

    # prune Q-table only if num_episodes > 100000
    if num_episodes > 100000:
        Q_table = prune_qtable(Q_table, N_table, min_visits=5)

    # always save Q-table after training
    with open('Q_table_'+str(num_episodes)+'_'+str(decay_rate)+'.pickle', 'wb') as handle:
        pickle.dump(Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)

# evaluate agent
if not train_flag:

    rewards = []
    episode_lengths = []
    max_tiles_achieved = []
    episode_times = []
    seen_states = set()
    total_actions = 0
    actions_using_Q = 0

    # load q-table
    filename = 'Q_table_'+str(num_episodes)+'_'+str(decay_rate)+'.pickle'
    input(f"\n{BOLD}Currently loading Q-table from "+filename+f"{RESET}.  \n\nPress Enter to confirm, or Ctrl+C to cancel and load a different Q-table file.\n(set num_episodes and decay_rate in rl_2048.py).")
    Q_table = np.load(filename, allow_pickle=True)

    eval_episodes = 1 if gui_flag else 10000
    for episode in tqdm(range(eval_episodes), desc="Evaluating"):
        obs = env.reset()
        total_reward = 0
        steps = 0
        start_time = time.time()

        while not env.game_over and not env.reached_2048 and steps < env.max_steps:
            state = hash_state(obs)
            seen_states.add(state)

            # select action using Q-table
            if state in Q_table:
                action = np.argmax(Q_table[state])
                actions_using_Q += 1
            else:
                action = env.action_space.sample()

            total_actions += 1

            next_obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            obs = next_obs

            if gui_flag:
                refresh(env, delay=0.1)
                pygame.event.pump()
            if done:
                break

        end_time = time.time()
        rewards.append(total_reward)
        episode_lengths.append(steps)
        max_tiles_achieved.append(env.get_max_tile())
        episode_times.append(end_time - start_time)

    # calculate final statistics
    avg_reward = np.mean(rewards)
    avg_length = np.mean(episode_lengths)
    total_time = np.sum(episode_times)

    # print evaluation results
    print("Unique states in Q-table:", len(Q_table))
    print(f"Average reward over 10,000 episodes: {avg_reward:.2f}")
    print(f"Average episode length: {avg_length:.2f} steps")
    print(f"Total time to play 10,000 episodes: {total_time:.2f} seconds")
    print(f"Max tile achieved: {np.max(max_tiles_achieved)}")

    # states unseen in training
    unique_unseen_states = seen_states - set(Q_table.keys())
    print("Unique states unseen during training:", len(unique_unseen_states))

    # percent of actions driven by Q-table
    percent_Q_usage = (actions_using_Q / total_actions) * 100
    print(f"Percentage of actions chosen using Q-table: {percent_Q_usage:.2f}%")


    # Create evaluation visualizations
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    fig.suptitle(f'Evaluation Results: {num_episodes} Training Episodes, {decay_rate} Decay', fontsize=16, fontweight='bold', y=1.02)

    # subplot 1: rolling avg reward 
    window = 100
    rolling_avg = pd.Series(rewards).rolling(window=window, min_periods=1).mean()
    axes[0].plot(range(len(rewards)), rewards, alpha=0.2, color='lightblue', linewidth=0.5, label='Episode Reward')
    axes[0].plot(range(len(rolling_avg)), rolling_avg, color='darkblue', linewidth=2, label=f'{window}-Episode Moving Avg')
    axes[0].axhline(avg_reward, color='red', linestyle='--', linewidth=1.5, label=f'Overall Mean: {avg_reward:.1f}')
    axes[0].set_xlabel('Evaluation Episode', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Total Reward', fontsize=12, fontweight='bold')
    axes[0].set_title('Reward Progression During Evaluation', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # subplot 2: max tile distribution
    tile_counts = pd.Series(max_tiles_achieved).value_counts().sort_index()
    bars = axes[1].bar(range(len(tile_counts)), tile_counts.values, color='coral', edgecolor='black', alpha=0.7)
    axes[1].set_xticks(range(len(tile_counts)))
    axes[1].set_xticklabels([int(tile) for tile in tile_counts.index], rotation=45)
    axes[1].set_xlabel('Max Tile Achieved', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1].set_title(f'Max Tile Distribution (Best: {np.max(max_tiles_achieved)})', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    # add percentage labels on bars
    for i, (bar, count) in enumerate(zip(bars, tile_counts.values)):
        height = bar.get_height()
        pct = (count / len(max_tiles_achieved)) * 100
        axes[1].text(bar.get_x() + bar.get_width() / 2., height, f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)

    axes[0].set_ylim(-1000, 2000)
    plt.tight_layout()
    eval_plot_filename = f'evaluation_results_{num_episodes}_{decay_rate}.png'
    plt.savefig(eval_plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nEvaluation plots saved to: {eval_plot_filename}")
    plt.close()