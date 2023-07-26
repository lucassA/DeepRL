import math

import numpy as np
from gymnasium import spaces

from envs.hideSeekEnv import HideSeekEnv
from misc.utils import shortest_paths_expensive


class FullDirectionOnFieldEnv(HideSeekEnv):
    """
    This class represents a concrete hide & seek environment
    It defines an environment where the observation is a vector of five values, representing the direction towards the agent's point of interest
    """


    def __init__(self, map_file="",
                 enemy_placement="static", player_placement="static", opti=False):
        super(FullDirectionOnFieldEnv, self).__init__(map_file=map_file,
                                                  enemy_placement=enemy_placement, player_placement=player_placement,
                                                  opti=opti)



        # The observation space is 5 values, that differs according to the direction the agent has to go in order to get closer to its point of interest
        # Example: left -> (1, 0, 0, 0, 0)
        # Example: right -> (0, 1, 0, 0, 0)
        # etc
        self.size_obs = 5
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.size_obs,), dtype=np.int32
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

        # defined for observations
        left = right = down = up = stop = 0

        # We compute the agent's interest points
        self.compute_interest_points()
        # We update the map accordingly, and compute the path from the agent toward his point of interest
        if len(self.agent.interest_points) > 0:
            self.playing_map.current_map[self.agent.interest_points[0][0]][self.agent.interest_points[0][1]] = self.INTEREST_POINT

            path_to_interest_tile = shortest_paths_expensive((self.agent.xcoord, self.agent.ycoord),
                                                          (self.agent.interest_points[0][0], self.agent.interest_points[0][1]),
                                                          self.playing_map.current_map, 1, to_avoid=self.BLOCK)

            full_directions = self.full_direction_towards_tile(path_to_interest_tile)

            final_direction = full_directions[0]
            if final_direction == 0:
                left = 1
            elif final_direction == 1:
                right = 1
            elif final_direction == 2:
                up = 1
            elif final_direction == 3:
                down = 1
        else:
            # if the agent has no point of interest, the direction stop is set to 1
            stop = 1

        final_direction = [left, right, up, down, stop]

        return np.array(final_direction, dtype=np.int32), {}

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


        # defined for observations
        left = right = down = up = stop = 0

        # We update the map accordingly, and compute the path from the agent toward his point of interest
        if len(self.agent.interest_points) > 0:
            path_to_interest_tile = shortest_paths_expensive((self.agent.xcoord, self.agent.ycoord),
                                                             (self.agent.interest_points[0][0],
                                                              self.agent.interest_points[0][1]),
                                                             self.playing_map.current_map, 1, to_avoid=self.BLOCK)

            full_directions = self.full_direction_towards_tile(path_to_interest_tile)


            self.playing_map.current_map[self.agent.interest_points[0][0]][self.agent.interest_points[0][1]] = self.INTEREST_POINT
            final_direction = full_directions[0]
            if final_direction == 0:
                left = 1
            elif final_direction == 1:
                right = 1
            elif final_direction == 2:
                up = 1
            elif final_direction == 3:
                down = 1
        else:
            # if the agent has no point of interest, the direction stop is set to 1
            stop = 1

        final_direction = [left, right, up, down, stop]

        list_block_in_vision = np.array(final_direction, dtype=np.int32)

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
                    print(". ", end="")
        print('')
        print('_' * 12)

    def direction_towards_tile(self, starting_x, starting_y, path_to_tile=None):
        """
        Function computing the direction from a specific position to the first tile in a path of tiles
        @param self:
        @param starting_x: x coordinates of the starting position
        @param starting_y: y coordinates of the starting position
        @param path_to_tile: path of tiles
        @return: the desired direction (integer)
        """
        empty_tile_direction = -1
        # UP
        if path_to_tile[0][1][0] < starting_x:
            empty_tile_direction = self.UP
        # DOWN
        elif path_to_tile[0][1][0] > starting_x:
            empty_tile_direction = self.DOWN
        # RIGHT
        elif path_to_tile[0][1][1] > starting_y:
            empty_tile_direction = self.RIGHT
        # LEFT
        elif path_to_tile[0][1][1] < starting_y:
            empty_tile_direction = self.LEFT

        return empty_tile_direction


    def full_direction_towards_tile(self, path_to_empty_tile):
        """
        Function transforming a path of tiles into a list of directions that one would have to follow to walk down that path of tiles
        @param self:
        @param path_to_empty_tile: path of tiles to turn into a list of directions
        @return: a list of integers representing a list of directions
        """
        full_directions = []
        i, j = self.agent.xcoord, self.agent.ycoord
        for p in path_to_empty_tile[0][1:]:
            direction = self.direction_towards_tile(i, j, [[(0,0),p]])
            full_directions.append(direction)
            i, j = p

        return full_directions


