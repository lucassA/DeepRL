import numpy as np
from gymnasium import spaces

from envs.hideSeekEnv import HideSeekEnv


class FullFieldVisionEnv(HideSeekEnv):
    """
    This class represents a concrete hide & seek environment
    It defines an environment where the observations are an RGB 32x32 image representing the map.
    This environment proposes either a "static/fixed" image representing the map, or a "dynamic" image which is centered around and follows the agent
    """

    # Constant used to transform a map matrix into an RGB image
    EMPTY_img = (255, 255, 255)
    BLOCK_img = (150, 150, 0)
    OUTSIDE_img = (1, 1, 1)
    AGENT_img = (0, 250, 0)
    ENEMY_img = (255, 0, 0)
    ENEMY_vision_img = (255, 0, 255)
    INTEREST_POINT_img = (255, 200, 0)

    def __init__(self, map_file="",
                  enemy_placement="static", player_placement="static", opti=False, mode_vision="static"):
        super(FullFieldVisionEnv, self).__init__(map_file=map_file,
                                                   enemy_placement=enemy_placement, player_placement=player_placement, opti=opti)


        # The observation space is an RGB 32*32 image:
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(3, self.grid_size * 3, self.grid_size * 3), dtype=np.uint8
        )


        # This defines whether or not the image is "static" or if its is centered around the player
        self.mode_vision = mode_vision

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

        # We update the map accordingly
        if len(self.agent.interest_points) > 0:
            self.playing_map.current_map[self.agent.interest_points[0][0]][self.agent.interest_points[0][1]] = self.INTEREST_POINT

        # These function turn a matrix-based map into a 32*32 image
        if self.mode_vision == "static":
            img_format_map = self.prepare_map_img_CNN(self.playing_map.current_map)
            return img_format_map, {}

        elif self.mode_vision == "dynamic":
            img_format_map = self.prepare_map_img_CNN(self.center_map_around_player())
            return img_format_map, {}

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

        # We update the map accordingly
        if len(self.agent.interest_points) > 0:
            self.playing_map.current_map[self.agent.interest_points[0][0]][self.agent.interest_points[0][1]] = self.INTEREST_POINT

        # These function turn a matrix-based map into a 32*32 image
        if self.mode_vision == "static":
            img_format_map = self.prepare_map_img_CNN(self.playing_map.current_map)
            return (
                img_format_map,
                self.reward, # step_reward,
                terminated,
                truncated,
                {},
            )

        elif self.mode_vision == "dynamic":
            img_format_map = self.prepare_map_img_CNN(self.center_map_around_player())
            return (
                img_format_map,
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

    def close(self):
        pass


    def prepare_map_img_CNN(self, current_map):
        """
        Function computing the direction from a specific position to the first tile in a path of tiles
        @param self:
        @param current_map: current matrix representation of the map
        @return: a new 3*32*32 numpy array, representing the map as a RGB image
        """

        # Since the current matrix is 12*12, we need to expand it to 32*32 by "tripling" its content

        # We triple the rows
        first_new_map = np.zeros((self.grid_size * 3, self.grid_size))
        for i in range(0, self.grid_size):
            first_new_map[3 * i] = current_map[i]
            first_new_map[3 * i + 1] = current_map[i]
            first_new_map[3 * i + 2] = current_map[i]

        # We triple the columns
        second_new_map = np.zeros((self.grid_size * 3, self.grid_size * 3))
        for i in range(0, self.grid_size):
            second_new_map[3 * i] = first_new_map[:, i]
            second_new_map[3 * i + 1] = first_new_map[:, i]
            second_new_map[3 * i + 2] = first_new_map[:, i]

        # We create three version of the 32*32 matrix, corresponding to the three RGB channels
        first_channel = np.array(second_new_map, dtype=np.uint8)
        first_channel[first_channel == self.EMPTY] = self.EMPTY_img[0]
        first_channel[first_channel == self.OUTSIDE] = self.OUTSIDE_img[0]
        first_channel[first_channel == self.BLOCK] = self.BLOCK_img[0]
        first_channel[first_channel == self.AGENT] = self.AGENT_img[0]
        first_channel[first_channel == self.ENEMY] = self.ENEMY_img[0]
        first_channel[first_channel == self.ENEMY_vision] = self.ENEMY_vision_img[0]
        first_channel[first_channel == self.INTEREST_POINT] = self.INTEREST_POINT_img[0]

        second_channel = np.array(second_new_map, dtype=np.uint8)
        second_channel[second_channel == self.EMPTY] = self.EMPTY_img[1]
        second_channel[second_channel == self.OUTSIDE] = self.OUTSIDE_img[1]
        second_channel[second_channel == self.BLOCK] = self.BLOCK_img[1]
        second_channel[second_channel == self.AGENT] = self.AGENT_img[1]
        second_channel[second_channel == self.ENEMY] = self.ENEMY_img[1]
        second_channel[second_channel == self.ENEMY_vision] = self.ENEMY_vision_img[1]
        second_channel[second_channel == self.INTEREST_POINT] = self.INTEREST_POINT_img[1]

        third_channel = np.array(second_new_map, dtype=np.uint8)
        third_channel[third_channel == self.EMPTY] = self.EMPTY_img[2]
        third_channel[third_channel == self.OUTSIDE] = self.OUTSIDE_img[2]
        third_channel[third_channel == self.BLOCK] = self.BLOCK_img[2]
        third_channel[third_channel == self.AGENT] = self.AGENT_img[2]
        third_channel[third_channel == self.ENEMY] = self.ENEMY_img[2]
        third_channel[third_channel == self.ENEMY_vision] = self.ENEMY_vision_img[2]
        third_channel[third_channel == self.INTEREST_POINT] = self.INTEREST_POINT_img[2]

        # We stack the three channels, yielding a 3*32*32 image
        new_map = np.stack((first_channel, second_channel, third_channel), axis=0)

        return new_map


    def center_map_around_player(self):
        """
        Function computing an agent-centered version of the current map
        @param self:
        @return: a new 12*12 matrix representing the current map centered around the agent
        """
        agent_x = self.agent.xcoord
        agent_y = self.agent.ycoord

        NO = (agent_x - 6 + 6, agent_y - 6 + 6)
        NE = (agent_x - 6 + 6, agent_y + 6 + 6)
        SO = (agent_x + 6 + 6, agent_y - 6 + 6)

        vision_map = self.augment_current_map()[np.ix_(list(range(NO[0], SO[0])), list(range(NO[1], NE[1])))]
        return vision_map

    def augment_current_map(self):
        """
        Augment the current map by adding a padding of 6 on each side (so that when the agent stand near the border of the map, the game still allows for an agent-centered 12*12 matrix)
        @param self:
        @return: a new padded matrix map representation
        """
        return np.pad(np.array(self.playing_map.current_map, dtype=np.uint8), 6, constant_values=self.OUTSIDE)

