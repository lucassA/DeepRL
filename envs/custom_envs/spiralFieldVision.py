import math

import numpy as np
from gymnasium import spaces

from envs.hideSeekEnv import HideSeekEnv

class SpiralFieldVisionEnv(HideSeekEnv):
    """
    This class represents a concrete hide & seek environment
    It defines an environment where the observations are an "agent near-vision" set, i.e, a list of tiles ordered in spiraling order starting from the agent
    Example of spiraling order, X is the agent:
                          9 10 11 12
                          8  1  2 13
                          7  X  3 14
                          6  5  4 ...

    If the map around the agent looked something like this (with 1 = empty tiles and 2 = block tiles):
                          2  2  2  2
                          1  1  1  1
                          1  X  1  1
                          2  2  2  ...
    The observation vector would look like: (1, 1, 1, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, ...)

    """

    def __init__(self, map_file="",
                 enemy_placement="static", player_placement="static", opti=False):
        super(SpiralFieldVisionEnv, self).__init__(map_file=map_file,
                                                   enemy_placement=enemy_placement, player_placement=player_placement, opti=opti)


        # As explained above, the observation is a serialized spiral vector corresponding to a square of 9 * 9 tiles centered around the agent.
        self.size_obs = 9 * 9
        self.observation_space = spaces.Box(
            low=0, high=7, shape=(self.size_obs,), dtype=np.int32
        )

    def reset(self, seed=None, options=None):
        """
        Function inherited from the gym environment, subclasses have to implement this
        @param self:
        @return:
        """
        super().reset(seed=seed, options=options)

        # New round of gameplay, we initialize the map, the positions of the unit and their vision
        self.initialize_map_pos_vision()

        # We compute the intersection of the agent's and enemy's visions and update the map accordingly: the agent does not have a complete view of the enemy's vision
        intersect_vision = list(set(self.enemy.vision) & set(self.agent.vision))
        for i in range(0, self.grid_size):
            for j in range(0, self.grid_size):
                if self.playing_map.current_map[i][j] == self.INTEREST_POINT:
                    self.playing_map.current_map[i][j] = self.EMPTY

                if (i, j) in intersect_vision and self.playing_map.current_map[i][j] != self.AGENT:
                    self.playing_map.current_map[i][j] = self.ENEMY_vision

        self.reward = 0.0
        self.prev_reward = 0.0
        self.n_step = 0

        # We compute the agent's interest points
        self.compute_interest_points()
        # And update the map accordingly
        if len(self.agent.interest_points) > 0:
            self.playing_map.current_map[self.agent.interest_points[0][0]][self.agent.interest_points[0][1]] = self.INTEREST_POINT

        # We compute the agent's spiral observations
        list_block_in_vision = self.find_blocks_near_me_spiral()
        return np.array(list_block_in_vision, dtype=np.int32), {}

    def step(self, action):
        """
        Function inherited from the gym environment, subclasses have to implement this
        @param self:
        @return:
        """
        terminated = False
        decides_to_stop = False

        self.agent.previous_xcoord = self.agent.xcoord
        self.agent.previous_ycoord = self.agent.ycoord

        # If the agent decides to move left
        if action == self.LEFT:
            self.move_agent_left()

        # If the agent decides to move right
        elif action == self.RIGHT:
            self.move_agent_right()

        # If the agent decides to move up
        elif action == self.UP:
            self.move_agent_up()

        # If the agent decides to move down
        elif action == self.DOWN:
            self.move_agent_down()

        # If the agent decides to stop
        elif action == self.STOP:
            terminated = True
            decides_to_stop = True

        else:
            raise ValueError(
                f"Received invalid action={action} which is not part of the action space"
            )

        truncated = False

        if self.n_step > 50:
            self.n_step = 0
            truncated = True


        # Update the agent's vision again after he moved (also updates the map accordingly)
        self.update_agent_vison_and_map()

        # We compute the agent's point of interest after he moved
        self.compute_interest_points()

        # Compute and update the rewards
        self.compute_rewards()

        self.n_step += 1

        # And update the map accordingly
        if len(self.agent.interest_points) > 0:
            self.playing_map.current_map[self.agent.interest_points[0][0]][self.agent.interest_points[0][1]] = self.INTEREST_POINT

        # We compute the agent's spiral observations
        list_block_in_vision = self.find_blocks_near_me_spiral()
        list_block_in_vision = np.array(list_block_in_vision, dtype=np.int32)

        return (
            list_block_in_vision,
            self.reward, # step_reward,
            terminated,
            truncated,
            {},
        )

    def render(self):
        """
        Function inherited from the gym environment, subclasses have to implement this
        @param self:
        @return:
        """
        for i in range(0, self.grid_size):
            print('\n', end="")
            for j in range(0, self.grid_size):
                if self.playing_map.current_map[i][j] == self.BLOCK:
                    print("O ", end="")
                elif self.playing_map.current_map[i][j] == self.EMPTY:
                    print(". ", end="")
                elif self.playing_map.current_map[i][j] == self.AGENT:
                    print("X ", end="")
                elif self.playing_map.current_map[i][j] == self.ENEMY:
                    print("Y ", end="")
                elif self.playing_map.current_map[i][j] == self.ENEMY_vision:
                    print("- ", end="")
                elif self.playing_map.current_map[i][j] == self.INTEREST_POINT:
                    print("$ ", end="")
        print('')
        print('_' * 12)

    def find_blocks_near_me_spiral(self):
        """
        This function computes a serialized vector of 9 by 9 in the order of a spiral starting from the center (i.e the agent's positionà
        @param self:
        @return: a list of tiles
        """
        left = self.agent.ycoord - 4
        right = self.agent.ycoord + 4
        top = self.agent.xcoord - 4
        bottom = self.agent.xcoord + 4
        list_blocks = []

        while left <= right and top <= bottom:

            #Process the first row from the remaining rows
            for i in range(left, right + 1):
                if 0 <= top < self.grid_size and 0 <= i < self.grid_size:
                    if (top, i) in self.enemy.vision:
                        list_blocks.append(self.ENEMY_vision)
                        self.playing_map.current_map[top][i] = self.ENEMY_vision
                    else:
                        list_blocks.append(self.playing_map.current_map[top][i])
                    # list_blocks.append((top, i))
                else:
                    list_blocks.append(self.OUTSIDE)
            top += 1

            #Process the last column from the remaining columns
            for i in range(top, bottom + 1):
                if 0 <= right < self.grid_size and 0 <= i < self.grid_size:
                    if (i, right) in self.enemy.vision:
                        list_blocks.append(self.ENEMY_vision)
                        self.playing_map.current_map[i][right] = self.ENEMY_vision
                    else:
                        list_blocks.append(self.playing_map.current_map[i][right])
                    # list_blocks.append((i, right))
                else:
                    list_blocks.append(self.OUTSIDE)
            right -= 1

            #Print the last row from the remaining rows
            if top <= bottom:
                for i in range(right, left - 1, -1):
                    if 0 <= bottom < self.grid_size and 0 <= i < self.grid_size:
                        if (bottom, i) in self.enemy.vision:
                            list_blocks.append(self.ENEMY_vision)
                            self.playing_map.current_map[bottom][i] = self.ENEMY_vision
                        else:
                            list_blocks.append(self.playing_map.current_map[bottom][i])
                        # list_blocks.append((bottom, i))
                    else:
                        list_blocks.append(self.OUTSIDE)
                bottom -= 1

            #Print the first column from the remaining columns
            if left <= right:
                for i in range(bottom, top - 1, -1):
                    if 0 <= left < self.grid_size and 0 <= i < self.grid_size:
                        if (i, left) in self.enemy.vision:
                            list_blocks.append(self.ENEMY_vision)
                            self.playing_map.current_map[i][left] = self.ENEMY_vision
                        else:
                            list_blocks.append(self.playing_map.current_map[i][left])
                        # list_blocks.append((i, left))
                    else:
                        list_blocks.append(self.OUTSIDE)
                left += 1

        list_blocks.reverse()

        # We consider that the tile the agent is on is "empty"
        if list_blocks[0] == self.AGENT:
            list_blocks[0] = self.EMPTY

        # We check that units are still on the map
        self.playing_map.current_map[self.agent.xcoord][self.agent.ycoord] = self.AGENT
        self.playing_map.current_map[self.enemy.xcoord][self.enemy.ycoord] = self.ENEMY

        return list_blocks




