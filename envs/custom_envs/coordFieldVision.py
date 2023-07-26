import math
import numpy as np
from gymnasium import spaces
from envs.hideSeekEnv import HideSeekEnv
from misc.utils import shortest_paths_expensive


class CoordFieldVisionEnv(HideSeekEnv):
    """
    This class represents a concrete hide & seek environment
    It defines an environment where the observation is composed of two positions and one distance: the coordinates of the agent and the coordinates of his point of interest ; and the distance between the two
    """

    # Constant
    SEP = -1
    def __init__(self, map_file="",
                 enemy_placement="static", player_placement="static", opti="False"):
        super(CoordFieldVisionEnv, self).__init__(map_file,
                                                   enemy_placement, player_placement, opti)

        # The observation space is a composed of 7 values, as such:
        # (Agent_x,Agent_y,SEP,Point_of_interest_x,Point_of_interest_y,SEP,distance)
        # SEP being a special separator
        self.size_obs = 7
        self.observation_space = spaces.Box(
            low=-1, high=30, shape=(self.size_obs,), dtype=np.int32
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

        # We update the map accordingly, and compute the path from the agent toward his point of interest
        path_to_interest_tile = None
        if len(self.agent.interest_points) > 0:
            self.playing_map.current_map[self.agent.interest_points[0][0]][self.agent.interest_points[0][1]] = self.INTEREST_POINT
            path_to_interest_tile = shortest_paths_expensive((self.agent.xcoord, self.agent.ycoord),
                                                             (self.agent.interest_points[0][0],
                                                              self.agent.interest_points[0][1]),
                                                             self.playing_map.current_map, 1, to_avoid=self.BLOCK)


        # We only consider the coordinates of the agent, the first tile in the path and the total length of the path
        if path_to_interest_tile is None:
            vector_to_return = [self.agent.xcoord, self.agent.ycoord, self.SEP, self.agent.xcoord,
                                self.agent.ycoord, -1, -1]
        else:
            length_path = len(path_to_interest_tile[0])
            vector_to_return = [self.agent.xcoord, self.agent.ycoord, self.SEP, path_to_interest_tile[0][1][0], path_to_interest_tile[0][1][1], self.SEP, length_path]

        return np.array(vector_to_return, dtype=np.int32), {}

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


        self.n_step += 1

        # Update the agent's vision again after he moved (also updates the map accordingly)
        self.update_agent_vison_and_map()

        # We compute the agent's point of interest after he moved
        self.compute_interest_points()

        # Compute and update the rewards
        self.compute_rewards()

        # We update the map accordingly, and compute the path from the agent toward his point of interest
        path_to_interest_tile = None
        if len(self.agent.interest_points) > 0:
            self.playing_map.current_map[self.agent.interest_points[0][0]][self.agent.interest_points[0][1]] = self.INTEREST_POINT
            path_to_interest_tile = shortest_paths_expensive((self.agent.xcoord, self.agent.ycoord),
                                                             (self.agent.interest_points[0][0],
                                                              self.agent.interest_points[0][1]),
                                                             self.playing_map.current_map, 1, to_avoid=self.BLOCK)

        # We only consider the coordinates of the agent, the first tile in the path and the total length of the path
        if path_to_interest_tile is None:
            vector_to_return = [self.agent.xcoord, self.agent.ycoord, self.SEP, self.agent.xcoord,
                                self.agent.ycoord, -1, -1]
        else:
            length_path = len(path_to_interest_tile[0])
            vector_to_return = [self.agent.xcoord, self.agent.ycoord, self.SEP, path_to_interest_tile[0][1][0], path_to_interest_tile[0][1][1], self.SEP, length_path]

        vector_to_return = np.array(vector_to_return, dtype=np.int32)

        return (
            vector_to_return,
            self.reward,
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


