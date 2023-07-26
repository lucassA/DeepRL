from __future__ import annotations

import math
import random
from typing import Any, SupportsFloat
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ActType, ObsType, RenderFrame
from gymnasium.utils import seeding
from numpy import int64
from copy import deepcopy

from data.dataloader import Dataloader
from envs.unit import Unit
from misc.utils import compute_optimized_training_maps, shortest_paths_naive, find_path_to_nearest, \
    find_path_to_not_yet_seen, shortest_paths_expensive


class HideSeekEnv(gym.Env):

    """
    This class represents an abstract hide & seek environment
    It defines variables and functions that can be used by subclasses to represent a concrete game of hide & seek
    It is built using : a textual map filename
                        an enemy_placement mode, either "static", "moves", or "random" representing how the enemy is initially placed on the map
                        a player_placement mode, either "static", "moves", or "random" representing how the player is initially placed on the map
                        an optimisation boolean, useful fo training purposes only
    """

    metadata = {"render_modes": ["console"]}

    # Define constants for directions
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    STOP = 4

    # Define constants for tiles in the matrix map representation
    EMPTY = 1
    BLOCK = 2
    OUTSIDE = 3
    AGENT = 4
    ENEMY = 5
    ENEMY_vision = 6
    INTEREST_POINT = 7

    def __init__(self, map_file="",
                 enemy_placement="static", player_placement="static", opti=False):
        super(HideSeekEnv, self).__init__()

        # Dataloader to load the maps from textual files
        self.dataloader = Dataloader()
        if map_file != "":
            self.playing_map = self.dataloader.load_map_from_file(map_file)
        else:
            self.playing_map = self.dataloader.load_map_from_file("map_v1")

        # Modes of unit placement
        self.player_placement = player_placement
        self.enemy_placement = enemy_placement

        # Size of the map
        self.grid_size = len(self.playing_map.current_map)

        # Two units are created, one for the agent (hides) and one for the enemy (stays in place)
        self.agent = Unit()
        self.enemy = Unit()

        # In this game, the agent (player) moves to hide himself. His actions are to move 1 tile at a time through the matrix map representation
        # The five actions of the agent are UP, DOWN, LEFT, RIGHT, STOP
        n_actions = 5
        self.action_space = spaces.Discrete(n_actions)

        # Define observation space, specific to subclasses
        observation_space: spaces.Space[ObsType]

        # These tracks the length of a round of gameplay, as well as the rewards associated with each steap
        self.n_step = 0
        self.reward = 0.0
        self.prev_reward = 0.0

        # Defines if optimization learning is activated
        self.optimized_maps = None
        if opti:
            self.optimized_maps = compute_optimized_training_maps()

    #Function inherited from the gym environment, subclasses have to implement this
    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        raise NotImplementedError

    #Function inherited from the gym environment, subclasses have to implement this
    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:  # type: ignore
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)


    def render(self) -> RenderFrame | list[RenderFrame] | None:
        raise NotImplementedError


    #Function inherited from the gym environment, subclasses have to implement this
    def close(self):
        pass

    def move_agent_left(self):
        """
        Moves the agent left on the current map
        @param self:
        @return:
        """
        if self.agent.ycoord - 1 >= 0:
            if self.playing_map.current_map[self.agent.xcoord][self.agent.ycoord - 1] != self.BLOCK and \
                    self.playing_map.current_map[self.agent.xcoord][self.agent.ycoord - 1] != self.ENEMY:

                self.playing_map.current_map[self.agent.xcoord][self.agent.ycoord] = self.EMPTY
                self.playing_map.current_map[self.agent.xcoord][self.agent.ycoord - 1] = self.AGENT

                # Update the agent position
                self.agent.set_coord(self.agent.xcoord, self.agent.ycoord - 1)

            else:  # The agent tried to move into a block or the enemy
                self.reward = self.reward - 50
        else:  # The agent tried to move outside of the map
            self.reward = self.reward - 50

    def move_agent_right(self):
        """
        Moves the agent right on the current map
        @param self:
        @return:
        """
        if self.agent.ycoord + 1 < self.grid_size:
            if self.playing_map.current_map[self.agent.xcoord][self.agent.ycoord + 1] != self.BLOCK and \
                    self.playing_map.current_map[self.agent.xcoord][self.agent.ycoord + 1] != self.ENEMY:

                self.playing_map.current_map[self.agent.xcoord][self.agent.ycoord] = self.EMPTY
                self.playing_map.current_map[self.agent.xcoord][self.agent.ycoord + 1] = self.AGENT

                # update agent position
                self.agent.set_coord(self.agent.xcoord, self.agent.ycoord + 1)

            else:  # The agent tried to move into a block or the enemy
                self.reward = self.reward - 50
        else:  # The agent tried to move outside of the map
            self.reward = self.reward - 50

    def move_agent_up(self):
        """
        Moves the agent up on the current map
        @param self:
        @return:
        """
        if self.agent.xcoord - 1 >= 0:
            if self.playing_map.current_map[self.agent.xcoord - 1][self.agent.ycoord] != self.BLOCK and \
                    self.playing_map.current_map[self.agent.xcoord - 1][self.agent.ycoord] != self.ENEMY:

                self.playing_map.current_map[self.agent.xcoord][self.agent.ycoord] = self.EMPTY
                self.playing_map.current_map[self.agent.xcoord - 1][self.agent.ycoord] = self.AGENT

                # update agent position
                self.agent.set_coord(self.agent.xcoord - 1, self.agent.ycoord)

            else:  # The agent tried to move into a block or the enemy
                self.reward = self.reward - 50
        else:  # The agent tried to move outside of the map
            self.reward = self.reward - 50

    def move_agent_down(self):
        """
        Moves the agent down on the current map
        @param self:
        @return:
        """
        if self.agent.xcoord + 1 < self.grid_size:
            if self.playing_map.current_map[self.agent.xcoord + 1][self.agent.ycoord] != self.BLOCK and \
                    self.playing_map.current_map[self.agent.xcoord + 1][self.agent.ycoord] != self.ENEMY:

                self.playing_map.current_map[self.agent.xcoord][self.agent.ycoord] = self.EMPTY
                self.playing_map.current_map[self.agent.xcoord + 1][self.agent.ycoord] = self.AGENT

                # update agent position
                self.agent.set_coord(self.agent.xcoord + 1, self.agent.ycoord)

            else:  # The agent tried to move into a block or the enemy
                self.reward = self.reward - 50
        else:  # The agent tried to move outside of the map
            self.reward = self.reward - 50

    def compute_rewards(self):
        """
        This function computes the rewards associated with the movement of the agent
        @param self:
        @return:
        """

        # This whole section has been used for experimentations
        """
        # First reward, if the agent is hidden from the enemy
        hidden = ((self.agent.xcoord, self.agent.ycoord) not in self.enemy.vision)
        if hidden:
            self.reward = self.reward + 200

        # Second reward, if there are several blocks between the agent and the enemy
        nb_blocks_through = self.nb_block_in_line_of_sight((self.agent.xcoord, self.agent.ycoord),
                                                           (self.enemy.xcoord, self.enemy.ycoord))
        well_hidden = (nb_blocks_through > 4)
        if well_hidden:
            self.reward = self.reward + 50

        # Third reward, if the agent is adjacent to more than one block ("harder to find")
        nb_blocks_next = self.nb_next_to_block(self.agent.xcoord, self.agent.ycoord)
        harder_to_find = (nb_blocks_next > 1)
        if harder_to_find:
            self.reward = self.reward + 75

        # Fourth reward, penalizes the agent if he takes too much time
        self.reward = self.reward - self.n_step / 2

        # Fifth reward, penalizes the agent if he moves away from his point of interest
        distance_poi = 0
        if len(self.agent.interest_points) > 0:
            path_to_interest_tile = shortest_paths_expensive((self.agent.xcoord, self.agent.ycoord),
                                                             (self.agent.interest_points[0][0],
                                                              self.agent.interest_points[0][1]),
                                                             self.playing_map.current_map, 1, to_avoid=self.BLOCK)

            if len(path_to_interest_tile[0]) > 1:
                distance_poi = len(path_to_interest_tile[0])

        self.reward = self.reward - distance_poi * 20
        """

        # This is how we compute rewards
        self.reward = 0.0
        if len(self.agent.interest_points) > 0: # If the agent is still interested in something

            if self.agent.previous_interest_points is not None:
                    # We compute the distance to the point of interest from both the current and previous coordinates of the agent
                    distance_previous_coordinates = math.sqrt((self.agent.previous_xcoord - self.agent.previous_interest_points[0]) ** 2 + (self.agent.previous_ycoord - self.agent.previous_interest_points[1]) ** 2)
                    distance_present_coordinates =  math.sqrt((self.agent.xcoord - self.agent.previous_interest_points[0]) ** 2 + (self.agent.ycoord - self.agent.previous_interest_points[1]) ** 2)

                    if distance_previous_coordinates > distance_present_coordinates: # The agent got closer to its interest point
                        self.reward = self.reward + 20
                    else: # The agent got away from its interest point
                        self.reward = self.reward - 100

            else: #In this case, the agent had no previous reward. If he has one now, he took a wrong decision
                self.reward = self.reward - 400

        else: # Else, we assume he is hidden
            self.reward = 200

    def update_agent_vison_and_map(self):
        """
        This function is called whenever the agent moves: it computes his vision, and updates the map according to the intersection between his vison and the enemy's vision
        @param self:
        @return:
        """
        # We compute the agent's vision
        self.agent.compute_vision(self.playing_map.current_map, mode="naive")

        # We compute the intersection of his vision with the enemy's vision
        intersect_vision = list(set(self.enemy.vision) & set(self.agent.vision))
        # And update the map accordingly
        for i in range(0, self.grid_size):
            for j in range(0, self.grid_size):
                if self.playing_map.current_map[i][j] == self.INTEREST_POINT:
                    self.playing_map.current_map[i][j] = self.EMPTY

                if (i, j) in intersect_vision and self.playing_map.current_map[i][j] != self.AGENT:
                    self.playing_map.current_map[i][j] = self.ENEMY_vision

    def place_npcs(self, coord_agent=(6, 7), coord_enemy=(6, 5)):
        """
        Places the units (both agent and enemy) on the map
        @param self:
        @param coord_agent: pair of coordinates representing the agent position in the matrix map representation
        @param coord_enemy: pair of coordinates representing the enemy position in the matrix map representation
        @return:
        """
        # Places the units on the map
        self.playing_map.current_map[coord_agent[0]][coord_agent[1]] = self.AGENT
        self.playing_map.current_map[coord_enemy[0]][coord_enemy[1]] = self.ENEMY

        # Modifies the coordinates of the units
        self.agent.xcoord = coord_agent[0]
        self.agent.ycoord = coord_agent[1]
        self.enemy.xcoord = coord_enemy[0]
        self.enemy.ycoord = coord_enemy[1]


    def initialize_pos(self):
        """
        Initialize the position of the units for a new round of hide & seek gameplay
        @param self:
        @return: an integer representing which enemy position has been chosen if it has been chosen randomly
        """
        rand_pos = None
        coord_agent = None
        coord_enemy = None
        agent_no_placed = True
        enemy_no_placed = True

        # We pick position coordinates for the agent first, depending on the player_placement mode
        if self.player_placement == "random":
            while agent_no_placed:
                i, j = np.random.randint(self.grid_size - 1, dtype=int64), np.random.randint(self.grid_size - 1,
                                                                                             dtype=int64)
                if self.playing_map.current_map[i][j] != self.EMPTY and self.playing_map.current_map[i][j] != self.AGENT:
                    pass
                else:
                    coord_agent = (i, j)
                    self.playing_map.current_map[i][j] = self.AGENT
                    agent_no_placed = False

        elif self.player_placement == "static":
            coord_agent = self.playing_map.agent_initial_position[0]

        else:
            coord_agent = random.choice(self.playing_map.agent_initial_position)

        # We pick position coordinates for the enemy second, depending on the enemy_placement mode
        if self.enemy_placement == "random":
            while enemy_no_placed:
                i, j = np.random.randint(self.grid_size - 1, dtype=int64), np.random.randint(self.grid_size - 1,
                                                                                             dtype=int64)
                if self.playing_map.current_map[i][j] != self.EMPTY and self.playing_map.current_map[i][j] != self.AGENT:
                    pass
                else:
                    coord_enemy = (i, j)
                    self.playing_map.current_map[i][j] = self.ENEMY
                    enemy_no_placed = False

        elif self.enemy_placement == "static":
            coord_enemy = self.playing_map.enemy_initial_position[0]

        else:
            rand_pos = random.randint(0,len(self.playing_map.enemy_initial_position)-1)
            coord_enemy = self.playing_map.enemy_initial_position[rand_pos]

        # Actual placement of the nits depending on their picked coordinates
        self.place_npcs(coord_agent, coord_enemy)

        return rand_pos


    def initialize_map_pos_vision(self):
        """
        Calls other functions to initialize the position of the units and their visions
        @param self:
        @return:
        """
        #self.initialize_map()
        self.playing_map.current_map = deepcopy(self.playing_map.initial_map)
        random_enemy_pos = self.initialize_pos()
        self.initialize_vision(random_enemy_pos)


    def initialize_vision(self, random_enemy_pos):
        """
        Initialize the vision of the units for a new round of hide & seek gameplay
        @param self:
        @param random_enemy_pos: integer representing which enemy position has been chosen IF chosen at random, useful for optimisation during training
        @return:
        """
        # We compute the vision of the enemy first
        # If optimization is activated, we load enemy vision from precomputed sources
        if self.optimized_maps is not None:
            if self.enemy_placement == "static":
                self.enemy.vision = self.optimized_maps[int(self.playing_map.map_name[-1]) - 1][0]

            else:
                if random_enemy_pos is not None:
                    self.enemy.vision = self.optimized_maps[int(self.playing_map.map_name[-1]) - 1][random_enemy_pos]
                else:
                    print("Error, random_enemy_pos should not be None")

        # Else, we compute it from scratch
        else:
            self.enemy.compute_vision(self.playing_map.current_map, mode="clever")

        # We compute the vision of the player second
        # Since the player is moving, his vision needs to be computed at every movement which is why we use a more naive (and quicker) method
        self.agent.previously_viewed = []
        self.agent.compute_vision(self.playing_map.current_map, mode="naive")


    def nb_block_in_line_of_sight(self, start, end):
        """
        Computes the number of blocks in the "line of sight" between two tiles
        @param self:
        @param start: Pair of coordinates corresponding to the starting position of the "line of sight"
        @param end: Pair of coordinates corresponding to the ending position of the "line of sight"
        @return: Number of block tiles between two positions
        """
        path = shortest_paths_naive(start, end)
        # Compute the path between the two positions and check the number of block tiles in that path
        nb_blocks = 0
        for p in path:
            if self.playing_map.current_map[p[0]][p[1]] == self.BLOCK:
                nb_blocks += 1
        return nb_blocks


    def nb_next_to_block(self, x, y):
        """
        Computes the number of blocks adjacent to a specific position
        @param self:
        @param x: X coordinate of the specific position
        @param y: Y coordinate of the specific position
        @return: Number of block tiles adjacent to the specific position
        """
        nb_next_to_block = 0
        if x > 0:
            if self.playing_map.current_map[x - 1][y] == self.BLOCK:
                nb_next_to_block += 1
        if x + 1 < self.grid_size:
            if self.playing_map.current_map[x + 1][y] == self.BLOCK:
                nb_next_to_block += 1
        if y > 0:
            if self.playing_map.current_map[x][y - 1] == self.BLOCK:
                nb_next_to_block += 1
        if y + 1 < self.grid_size:
            if self.playing_map.current_map[x][y + 1] == self.BLOCK:
                nb_next_to_block += 1
        return nb_next_to_block


    def look_for_nearest_block(self, mode="distance"):
        """
        Retrieves the block that is the closest to the agent
        @param self:
        @param mode: Mode specifying the return value of the function, either the distance to the agent, or the coordinate of the block
        @return: Either the distance between the Agent and the nearest block,or the coordinate of the nearest block
        """
        nearest_value = 100
        nearest_x = None
        nearest_y = None
        # We process the whole matrix map representation
        for i in range(0, self.grid_size):
            for j in range(0, self.grid_size):
                if self.playing_map.current_map[i][j] == self.BLOCK: # If a tile is a block
                    temp = math.sqrt((self.agent.xcoord - i) ** 2 + (self.agent.ycoord - j) ** 2)
                    if temp < nearest_value:
                        nearest_value = temp
                        nearest_x = i
                        nearest_y = j

        if mode == "distance":
            return nearest_value
        else:
            return nearest_x, nearest_y


    def compute_interest_points(self):
        """
        Function computing "interest points" for the agent, based on its vision and on handcrafted heuristics
        @param self:
        @return:
        """
        if len(self.agent.interest_points) > 0:
            self.agent.previous_interest_points = self.agent.interest_points[0]
        else:
            self.agent.previous_interest_points = None
        self.agent.interest_points = []

        # First heuristic : if the agent is seen by the enemy, his interest point will be the closest tile that he thinks is not seen by the enemy
        # This is a priority heuristic
        if (self.agent.xcoord, self.agent.ycoord) in self.enemy.vision:
            nearest_empty_tile_x, nearest_empty_tile_y = find_path_to_nearest(self.playing_map.current_map, self.agent.xcoord,
                                                                              self.agent.ycoord, self.EMPTY, self.BLOCK)
            self.agent.interest_points.append((nearest_empty_tile_x, nearest_empty_tile_y))

        else: # If we are not seen by the enemy

            # Second heuristic : if the agent notices a spot that is adjacent to several blocks (in his vision field), or "a good place to hide", he will consider it as an interest point
            if self.nb_next_to_block(self.agent.xcoord, self.agent.ycoord) <= 1: # If we are not in a perfect spot already
                for (i, j) in self.agent.previously_viewed:
                    if self.playing_map.current_map[i][j] == self.EMPTY:
                        if self.nb_next_to_block(i, j) > 1:
                            if (i, j) not in self.agent.interest_points:
                                self.agent.interest_points.append((i, j))
                                break

                # Third heuristic : if the agent considers he has not see enough of the map, he will keep exploring the nearest. The nearest "unseen" tile is its interest point
                if len(self.agent.interest_points) == 0 and len(self.agent.previously_viewed) < (
                        (self.grid_size * self.grid_size) / 1.5):
                    nearest_empty_tile_x, nearest_empty_tile_y = find_path_to_not_yet_seen(self.playing_map.current_map, self.agent.xcoord,
                                                                                           self.agent.ycoord,
                                                                                           self.agent.previously_viewed,
                                                                                           self.BLOCK)
                    self.agent.interest_points.append((nearest_empty_tile_x, nearest_empty_tile_y))

                # Fourth heuristic : if the agent is not next to any block, his interest point will be the closest block in order to be better hidden
                if len(self.agent.interest_points) == 0 and self.nb_next_to_block(self.agent.xcoord, self.agent.ycoord) == 0:
                    closest_block = self.look_for_nearest_block(mode="coord")
                    next_to_nearest_x, next_to_nearest_y = self.find_appropriate_tile_next_to_block(closest_block)

                    if next_to_nearest_x != 0 and next_to_nearest_y != 0:
                        self.agent.interest_points.append((next_to_nearest_x, next_to_nearest_y))

            else: # We are in a perfect stop, no need to have any more interests
                pass


    def find_appropriate_tile_next_to_block(self, closest_block):
        """
        Function the computes the most appropriate tile to move to, next to a specific block tile, according to the position of the enemy
        @param self:
        @param closest_block: Block tile to move next to
        @return: pair of coordinates of the block-adjacent tile which is deemed as the most appropriate to move to
        """

        # The search for the most appropriate tile is done as such:
        # We try to move to the block-adjacent tile that is the farthest from the enemy
        # If this tile is not available, we try with the second farthest, etc.
        next_to_nearest_x, next_to_nearest_y = 0, 0
        if abs(self.agent.xcoord - self.enemy.xcoord) > abs(self.agent.ycoord - self.enemy.ycoord):
            if self.agent.xcoord > self.enemy.xcoord:
                if closest_block[0] + 1 < self.grid_size and self.playing_map.current_map[closest_block[0] + 1][
                    closest_block[1]] == self.EMPTY:
                    next_to_nearest_x, next_to_nearest_y = closest_block[0] + 1, closest_block[1]
                else:
                    if self.agent.ycoord > self.enemy.ycoord:
                        if closest_block[1] + 1 < self.grid_size and self.playing_map.current_map[closest_block[0]][
                            closest_block[1] + 1] == self.EMPTY:
                            next_to_nearest_x, next_to_nearest_y = closest_block[0], closest_block[1] + 1
                    else:
                        if closest_block[1] - 1 > 0 and self.playing_map.current_map[closest_block[0]][
                            closest_block[1] - 1] == self.EMPTY:
                            next_to_nearest_x, next_to_nearest_y = closest_block[0], closest_block[1] - 1
            else:
                if closest_block[0] > 0 and self.playing_map.current_map[closest_block[0] - 1][closest_block[1]] == self.EMPTY:
                    next_to_nearest_x, next_to_nearest_y = closest_block[0] - 1, closest_block[1]
                else:
                    if self.agent.ycoord > self.enemy.ycoord:
                        if closest_block[1] + 1 < self.grid_size and self.playing_map.current_map[closest_block[0]][
                            closest_block[1] + 1] == self.EMPTY:
                            next_to_nearest_x, next_to_nearest_y = closest_block[0], closest_block[1] + 1
                    else:
                        if closest_block[1] - 1 > 0 and self.playing_map.current_map[closest_block[0]][
                            closest_block[1] - 1] == self.EMPTY:
                            next_to_nearest_x, next_to_nearest_y = closest_block[0], closest_block[1] - 1
        else:
            if self.agent.ycoord > self.enemy.ycoord:
                if closest_block[1] + 1 < self.grid_size and self.playing_map.current_map[closest_block[0]][
                    closest_block[1] + 1] == self.EMPTY:
                    next_to_nearest_x, next_to_nearest_y = closest_block[0], closest_block[1] + 1
                else:
                    if self.agent.xcoord > self.enemy.xcoord:
                        if closest_block[0] + 1 < self.grid_size and self.playing_map.current_map[closest_block[0] + 1][
                            closest_block[1]] == self.EMPTY:
                            next_to_nearest_x, next_to_nearest_y = closest_block[0] + 1, closest_block[1]
                    else:
                        if closest_block[0] - 1 > 0 and self.playing_map.current_map[closest_block[0] - 1][
                            closest_block[1]] == self.EMPTY:
                            next_to_nearest_x, next_to_nearest_y = closest_block[0] - 1, closest_block[1]
            else:
                if closest_block[1] > 0 and self.playing_map.current_map[closest_block[0]][closest_block[1] - 1] == self.EMPTY:
                    next_to_nearest_x, next_to_nearest_y = closest_block[0], closest_block[1] - 1
                else:
                    if self.agent.xcoord > self.enemy.xcoord:
                        if closest_block[0] + 1 < self.grid_size and self.playing_map.current_map[closest_block[0] + 1][
                            closest_block[1]] == self.EMPTY:
                            next_to_nearest_x, next_to_nearest_y = closest_block[0] + 1, closest_block[1]
                    else:
                        if closest_block[0] - 1 > 0 and self.playing_map.current_map[closest_block[0] - 1][
                            closest_block[1]] == self.EMPTY:
                            next_to_nearest_x, next_to_nearest_y = closest_block[0] - 1, closest_block[1]

        return next_to_nearest_x, next_to_nearest_y
