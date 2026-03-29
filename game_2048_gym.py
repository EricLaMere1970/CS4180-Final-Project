import gym
from gym import spaces
import numpy as np
import random
import copy

# 2048 game enviroment
class Game2048Env(gym.Env):
    
    # metadata for rendering (taken from PA2)
    metadata = {'render.modes': ['human']}

    # initalize the environment
    def __init__(self):
        # initialize the parent class
        super(Game2048Env, self).__init__()
        
        # game parameters
        self.grid_size = 4
        self.max_steps = 10000 # prevent infinite games
        
        # define the action space with 4 possible moves
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.action_space = spaces.Discrete(len(self.actions))
        
        # define the observation space
        self.observation_space = spaces.Box(
            low=0, 
            high=17,  # log2(131072) = 17
            shape=(self.grid_size, self.grid_size), 
            dtype=int
        )
        
        # define rewards
        self.rewards = {
            'invalid_move': -10,    # moderate penalty for moves that don't change grid
            'valid_move': 1,        # small reward for any valid move
            'tile_merge': 0,        # base reward for merging
            'game_over': -100,      # moderate penalty for game over
            'reach_2048': 2048,     # large reward for reaching 2048
            'step': -0.1            # small penalty for each step to encourage faster games
        }
        
        # reset the game to start
        self.reset()

    # reset the game to initial state
    def reset(self):
        # initialize the grid, score, and game state
        self.grid = [[0] * self.grid_size for _ in range(self.grid_size)]
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.reached_2048 = False
        
        # add two initial tiles
        self.add_new_tile()
        self.add_new_tile()
        
        # return the initial observation
        return self.get_observation()

    # define function to add a new tile
    def add_new_tile(self):
        # get a list of empty cells
        empty_cells = [(i, j) for i in range(self.grid_size) 
                      for j in range(self.grid_size) if self.grid[i][j] == 0]
        # randomly select an empty cell
        if empty_cells:
            i, j = random.choice(empty_cells)
            # add a 2 at a 90% chance, else add a 4
            self.grid[i][j] = 2 if random.random() < 0.9 else 4

    # define function to get observation
    def get_observation(self):
        # initially set array to zeroes
        obs = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        # for each cell, store log2 value or 0 if empty
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i][j] > 0:
                    obs[i][j] = int(np.log2(self.grid[i][j]))
                else:
                    obs[i][j] = 0
        return obs

    # define function to slide tiles to the left
    def slide_tiles(self, row):
        # slide non-zero tiles sto the left
        new_row = [num for num in row if num != 0]

        # pad with zeros on the right
        new_row += [0] * (self.grid_size - len(new_row))
        return new_row

    # define a function to merge tiles
    def merge(self, row):
        # define score gained from merges
        score_gained = 0

        # merge tiles
        for i in range(self.grid_size - 1):
            # check if two adjacent tiles can be merged (not zero and equal)
            if row[i] != 0 and row[i] == row[i + 1]:
                # multiply the tile by 2 and set the next tile to 0
                row[i] *= 2
                row[i + 1] = 0

                # update score gained
                score_gained += row[i]
                
                # check if 2048 tile is reached
                if row[i] == 2048:
                    self.reached_2048 = True
        return row, score_gained

    # define function to move left
    def move_left(self):
        # initialize change flag and score gained
        changed = False
        total_score_gained = 0
        
        # process each row
        for i in range(self.grid_size):
            original = self.grid[i][:]
            
            # slide left, try and merge, slide left again
            self.grid[i] = self.slide_tiles(self.grid[i])
            self.grid[i], score_gained = self.merge(self.grid[i])
            self.grid[i] = self.slide_tiles(self.grid[i])
            
            # update score gained
            total_score_gained += score_gained

            # check if any change occurred
            if original != self.grid[i]:
                changed = True
                
        return changed, total_score_gained

    # define function to reverse the grid
    def reverse_grid(self):
        # reverse the grid
        self.grid = [row[::-1] for row in self.grid]

    # define function to transpose the grid
    def transpose_grid(self):
        # transpose the grid
        self.grid = [list(row) for row in zip(*self.grid)]

    # define function to check if any moves are possible
    def can_move(self):
        # check for empty cells
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i][j] == 0:
                    return True
        
        # check for possible merges
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                current = self.grid[i][j]
                # check right neighbor
                if j < self.grid_size - 1 and current == self.grid[i][j + 1]:
                    return True
                # check bottom neighbor  
                if i < self.grid_size - 1 and current == self.grid[i + 1][j]:
                    return True
        
        return False

    # define the step function
    def step(self, action):
        # convert action to index if needed
        if isinstance(action, str):
            action = self.actions.index(action)
        
        # check if game is already over
        if self.game_over:
            return self.get_observation(), 0, True, {'message': 'Game already over'}
        
        # store previous state for comparison
        prev_grid = copy.deepcopy(self.grid)
        prev_score = self.score
        
        # execute move
        action_name = self.actions[action]
        changed, score_gained = self.execute_move(action_name)
        
        # calculate reward
        reward = self.calculate_reward(changed, score_gained, prev_grid)
        
        # update game state
        self.score += score_gained
        self.steps += 1
        
        # add new tile if the grid changed
        if changed:
            self.add_new_tile()
        
        # check terminal conditions
        done = False
        info = {'action': action_name, 'score_gained': score_gained, 'changed': changed}
        
        # check if game is over
        if not self.can_move():
            self.game_over = True
            done = True
            reward += self.rewards['game_over']
            info['message'] = 'Game Over - No moves available'

        # check if 2048 tile is reached
        elif self.reached_2048:
            done = True
            reward += self.rewards['reach_2048']
            info['message'] = 'Reached 2048!'

        # check if max steps reached
        elif self.steps >= self.max_steps:
            done = True
            info['message'] = 'Max steps reached'
            
        return self.get_observation(), reward, done, info

    # define function to execute a move
    def execute_move(self, direction):
        # move left if direction is left
        if direction == 'LEFT':
            return self.move_left()
        
        # move right if direction is right
        elif direction == 'RIGHT':
            # reverse, move left, reverse back
            self.reverse_grid()
            changed, score = self.move_left()
            self.reverse_grid()
            return changed, score
        
        # move up if direction is up
        elif direction == 'UP':
            # transpose, move left, transpose back
            self.transpose_grid()
            changed, score = self.move_left()
            self.transpose_grid()
            return changed, score
        
        # move down if direction is down
        elif direction == 'DOWN':
            # transpose, reverse, move left, reverse back, transpose back
            self.transpose_grid()
            self.reverse_grid()
            changed, score = self.move_left()
            self.reverse_grid()
            self.transpose_grid()
            return changed, score
        else:
            return False, 0

    # define function to calculate reward
    def calculate_reward(self, changed, score_gained, prev_grid):
        # initialize reward
        reward = 0
        
        # check if move was invalid
        if not changed:
            # penalize invalid moves
            reward += self.rewards['invalid_move']

        # otherwise, calculate reward for valid move
        else:
            # reward for making a valid move
            reward += self.rewards['valid_move']
            
            # add score gained from merges
            reward += score_gained
            
            # additional reward for creating higher value tiles
            max_tile_prev = max(max(row) for row in prev_grid) if any(any(row) for row in prev_grid) else 0
            max_tile_current = max(max(row) for row in self.grid) if any(any(row) for row in self.grid) else 0
            
            # reward for achieving new max tile
            if max_tile_current > max_tile_prev:
                reward += np.log2(max_tile_current) * 10 # * 10 to encourage higher merges
        
        # apply step penalty to encourage faster games
        reward += self.rewards['step']
        
        return reward

    # define function to render the game state (used AI to format this well)
    def render(self, mode='human'):
        if mode == 'human':
            print(f"\nScore: {self.score}")
            print(f"Steps: {self.steps}")
            print("-" * (self.grid_size * 6 + 1))
            
            # print the grid
            for row in self.grid:
                print("|", end="")
                for cell in row:
                    if cell == 0:
                        print(f"{'':>5}", end="|")
                    else:
                        print(f"{cell:>5}", end="|")
                print()
                print("-" * (self.grid_size * 6 + 1))
            
            # check if game over
            if self.game_over:
                print("GAME OVER!")
            # check if 2048 reached
            elif self.reached_2048:
                print("YOU WIN! Reached 2048!")
            print()

    # define function to create a unique state hash
    def get_state_hash(self):
        # convert grid to tuple for hashing
        return hash(tuple(tuple(row) for row in self.grid))

    # define function to get max tile value
    def get_max_tile(self):
        return max(max(row) for row in self.grid) if any(any(row) for row in self.grid) else 0

    # define function to get number of empty cells
    def get_empty_cells(self):
        return sum(row.count(0) for row in self.grid)

    # define function to clone the current game state
    def clone(self):
        new_env = Game2048Env()
        new_env.grid = copy.deepcopy(self.grid)
        new_env.score = self.score
        new_env.steps = self.steps
        new_env.game_over = self.game_over
        new_env.reached_2048 = self.reached_2048
        return new_env