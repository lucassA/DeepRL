from data.opti_misc import map_v1_ennemy_vision_1, map_v4_ennemy_vision_3, map_v4_ennemy_vision_2, \
    map_v4_ennemy_vision_1, map_v4_ennemy_vision_4, map_v3_ennemy_vision_4, map_v3_ennemy_vision_3, \
    map_v3_ennemy_vision_2, map_v3_ennemy_vision_1, map_v2_ennemy_vision_4, map_v2_ennemy_vision_3, \
    map_v2_ennemy_vision_2, map_v2_ennemy_vision_1, map_v1_ennemy_vision_4, map_v1_ennemy_vision_3, \
    map_v1_ennemy_vision_2

"""
This file contains functions that, while not specifically designed for the game of hide & seek, are used by the different environments.
"""

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


def shortest_paths(start, end, len_map, nb_paths):
    """
    Function that computes and retrieves a number of shortest path (in a matrix) from start to end
    @param start: pair of coordinates from where to start the path
    @param end: pair of coordinates where to end the path
    @param len_map: length of the matrix that we search the path in
    @param nb_paths: number of desired "shortest paths" to return
    @return: a set of shortest paths (list of pairs of coordinates) from start to end
    """

    # first, let's look for the beginning position, there is better but it works
    (i_start, j_start) = start
    # and take the goal position (used in the heuristic)
    (i_end, j_end) = end

    width = height = len_map

    heuristic = lambda i, j: abs(i_end - i) + abs(j_end - j)
    comp = lambda state: state[2] + state[3]  # get the total cost

    # small variation for easier code, state is (coord_tuple, previous, path_cost, heuristic_cost)
    fringe = [((i_start, j_start), list(), 0, heuristic(i_start, j_start))]
    visited = {}  # empty set

    return_set = []

    # maybe limit to prevent too long search
    limit = 0
    while True and limit <= 10000:
        limit = limit + 1

        # get first state (least cost)
        state = fringe.pop(0)

        # goal check
        (i, j) = state[0]
        if i == i_end and j == j_end:
            path = [state[0]] + state[1]
            path.reverse()
            return_set.append(path)
            if len(return_set) == nb_paths:
                return return_set

        # set the cost (path is enough since the heuristic won't change)
        visited[(i, j)] = state[2]

        # explore neighbor
        neighbor = list()
        if i > 0 and i_end <= i:  # top
            neighbor.append((i - 1, j))
        if i + 1 < height and i_end >= i:
            neighbor.append((i + 1, j))
        if j > 0 and j >= j_end:
            neighbor.append((i, j - 1))
        if j + 1 < width and j <= j_end:
            neighbor.append((i, j + 1))

        for n in neighbor:
            next_cost = state[2] + 1
            if n in visited and visited[n] >= next_cost:
                continue
            fringe.append((n, [state[0]] + state[1], next_cost, heuristic(n[0], n[1])))

        # resort the list (SHOULD use a priority queue here to avoid re-sorting all the time)
        fringe.sort(key=comp)
    return return_set


def find_path_to_nearest(current_map, xcoord, ycoord, type_tile=1, tile_to_avoid=2):
    """
    Function that find the tile of type type_tile which is nearest to the position (xcoord, ycoord), while avoiding tiles of type tile_to_avoid.
    @param current_map: Current matrix map representation
    @param xcoord: x coordinate from which to start searching
    @param ycoord: y coordinate from which to start searching
    @param type_tile: type of tile to look for
    @param tile_to_avoid: type of tle to avoid during the search
    @return: a path (list of pairs of coordinates) to the nearest tile of type type_tile
    """
    i, j = xcoord, ycoord
    list_tile = []
    len_map = len(current_map)
    already_visited = [(i, j)]
    while True:
        if i > 0 and current_map[i-1][j] != tile_to_avoid:  # top
            if current_map[i-1][j] == type_tile:
                return i - 1, j
            else:
                if (i - 1, j) not in already_visited:
                    list_tile.append((i - 1, j))
                    already_visited.append((i - 1, j))

        if i + 1 < len_map and current_map[i+1][j] != tile_to_avoid:
            if current_map[i+1][j] == type_tile:
                return i + 1, j
            else:
                if (i + 1, j) not in already_visited:
                    list_tile.append((i + 1, j))
                    already_visited.append((i + 1, j))

        if j > 0 and current_map[i][j-1] != tile_to_avoid:
            if current_map[i][j-1] == type_tile:
                return i, j - 1
            else:
                if (i, j - 1) not in already_visited:
                    list_tile.append((i, j - 1))
                    already_visited.append((i, j - 1))
                list_tile.append((i, j - 1))

        if j + 1 < len_map and current_map[i][j+1] != tile_to_avoid:
            if current_map[i][j+1] == type_tile:
                return i, j + 1
            else:
                if (i, j + 1) not in already_visited:
                    list_tile.append((i, j + 1))
                    already_visited.append((i, j + 1))
        if len(list_tile)>0:
            i, j = list_tile.pop(0)
        else:
            return None


def find_path_to_not_yet_seen(current_map, xcoord, ycoord, agent_previously_seen, tile_to_avoid=2):
    """
    Function that find the path from the agent to a tile that he has not yet seen
    @param current_map: Current matrix map representation
    @param xcoord: x coordinate from which to start searching
    @param ycoord: y coordinate from which to start searching
    @param agent_previously_seen: list of tiles (pairs of coordinates) that the agent has already seen
    @param tile_to_avoid: type of tle to avoid during the search
    @return: a path (list of pairs of coordinates) to the nearest tile that the agent has not seen
    """
    i, j = xcoord, ycoord
    list_tile = []
    len_map = len(current_map)
    already_visited = [(i, j)]
    while True:
        if i > 0 and current_map[i-1][j] != tile_to_avoid:  # top
            if (i - 1, j) not in agent_previously_seen:
                return i - 1, j
            else:
                if (i - 1, j) not in already_visited:
                    list_tile.append((i - 1, j))
                    already_visited.append((i - 1, j))

        if i + 1 < len_map and current_map[i+1][j] != tile_to_avoid:
            if (i + 1, j) not in agent_previously_seen:
                return i + 1, j
            else:
                if (i + 1, j) not in already_visited:
                    list_tile.append((i + 1, j))
                    already_visited.append((i + 1, j))

        if j > 0 and current_map[i][j-1] != tile_to_avoid:
            if (i, j - 1) not in agent_previously_seen:
                return i, j - 1
            else:
                if (i, j - 1) not in already_visited:
                    list_tile.append((i, j - 1))
                    already_visited.append((i, j - 1))
                list_tile.append((i, j - 1))

        if j + 1 < len_map and current_map[i][j+1] != tile_to_avoid:
            if (i, j + 1) not in agent_previously_seen:
                return i, j + 1
            else:
                if (i, j + 1) not in already_visited:
                    list_tile.append((i, j + 1))
                    already_visited.append((i, j + 1))
        if len(list_tile)>0:
            i, j = list_tile.pop(0)
        else:
            return None


def shortest_paths_expensive(start, end, current_map, nb_paths, to_avoid=2):
    """
    Function that computes and retrieves a number of shortest path (in a matrix) from start to end. This function is more effective, but more time-consuming
    @param start: pair of coordinates from which to start the search
    @param end: pair of coordinates to end the search on
    @param current_map: Current matrix map representation
    @param nb_paths: number of paths to return
    @param to_avoid: type of tile to avoid during the search
    @return: a set of shortest paths (list of pairs of coordinates) from start to end
    """
    # first, let's look for the beginning position, there is better but it works
    (i_start, j_start) = start
    # and take the goal position (used in the heuristic)
    (i_end, j_end) = end

    width = height = len(current_map)

    heuristic = lambda i, j: abs(i_end - i) + abs(j_end - j)
    comp = lambda state: state[2] + state[3]  # get the total cost

    # small variation for easier code, state is (coord_tuple, previous, path_cost, heuristic_cost)
    fringe = [((i_start, j_start), list(), 0, heuristic(i_start, j_start))]
    visited = {}  # empty set

    return_set = []

    # maybe limit to prevent too long search
    limit = 0
    while True and limit <= 10000:
        limit = limit + 1

        # get first state (least cost)
        state = fringe.pop(0)

        # goal check
        (i, j) = state[0]
        if i == i_end and j == j_end:
            path = [state[0]] + state[1]
            path.reverse()
            return_set.append(path)
            if len(return_set) == nb_paths:
                return return_set

        # set the cost (path is enough since the heuristic won't change)
        visited[(i, j)] = state[2]

        # explore neighbor
        neighbor = list()
        if i > 0 and current_map[i-1][j] != to_avoid:  # top
            neighbor.append((i - 1, j))
        if i + 1 < height and current_map[i+1][j] != to_avoid:
            neighbor.append((i + 1, j))
        if j > 0 and current_map[i][j-1] != to_avoid:
            neighbor.append((i, j - 1))
        if j + 1 < width and current_map[i][j+1] != to_avoid:
            neighbor.append((i, j + 1))

        for n in neighbor:
            next_cost = state[2] + 1
            if n in visited and visited[n] >= next_cost:
                continue
            fringe.append((n, [state[0]] + state[1], next_cost, heuristic(n[0], n[1])))

        # resort the list (SHOULD use a priority queue here to avoid re-sorting all the time)
        fringe.sort(key=comp)
    return return_set


def shortest_paths_naive( start, end):
    """
    Function that computes and retrieves a number of shortest path (in a matrix) from start to end. This function is less effective, but less time-consuming
    @param start: pair of coordinates from which to start the search
    @param end: pair of coordinates to end the search on
    @return: a list of pairs of coordinates, representing the shortest path between start and end
    """
    (i_start, j_start) = start
    (i_end, j_end) = end

    path_visited = []
    i, j = i_start, j_start
    while i_end > i and j_end > j:
        path_visited.append((i + 1, j))
        path_visited.append((i, j + 1))
        path_visited.append((i + 1, j + 1))
        i = i + 1
        j = j + 1

    while i_end > i and j_end < j:
        path_visited.append((i + 1, j))
        path_visited.append((i, j - 1))
        path_visited.append((i + 1, j - 1))
        i = i + 1
        j = j - 1

    while i_end < i and j_end > j:
        path_visited.append((i - 1, j))
        path_visited.append((i, j + 1))
        path_visited.append((i - 1, j + 1))
        i = i - 1
        j = j + 1

    while i_end < i and j_end < j:
        path_visited.append((i - 1, j))
        path_visited.append((i, j - 1))
        path_visited.append((i - 1, j - 1))
        i = i - 1
        j = j - 1

    while i_end == i and j_end > j:
        path_visited.append((i, j + 1))
        j = j + 1

    while i_end == i and j_end < j:
        path_visited.append((i, j - 1))
        j = j - 1

    while i_end > i and j_end == j:
        path_visited.append((i + 1, j))
        i = i + 1

    while i_end < i and j_end == j:
        path_visited.append((i - 1, j))
        i = i - 1

    return path_visited


def compute_optimized_training_maps():
    """
    Function that loads pre-computed enemy vision for speed during training
    @return: a set of pre-computed visions (list of coordinates)
    """
    optimized_maps = [[]]
    optimized_maps[0].append(map_v1_ennemy_vision_1)
    optimized_maps[0].append(map_v1_ennemy_vision_2)
    optimized_maps[0].append(map_v1_ennemy_vision_3)
    optimized_maps[0].append(map_v1_ennemy_vision_4)
    optimized_maps.append([])
    optimized_maps[1].append(map_v2_ennemy_vision_1)
    optimized_maps[1].append(map_v2_ennemy_vision_2)
    optimized_maps[1].append(map_v2_ennemy_vision_3)
    optimized_maps[1].append(map_v2_ennemy_vision_4)
    optimized_maps.append([])
    optimized_maps[2].append(map_v3_ennemy_vision_1)
    optimized_maps[2].append(map_v3_ennemy_vision_2)
    optimized_maps[2].append(map_v3_ennemy_vision_3)
    optimized_maps[2].append(map_v3_ennemy_vision_4)
    optimized_maps.append([])
    optimized_maps[3].append(map_v4_ennemy_vision_1)
    optimized_maps[3].append(map_v4_ennemy_vision_2)
    optimized_maps[3].append(map_v4_ennemy_vision_3)
    optimized_maps[3].append(map_v4_ennemy_vision_4)
    return optimized_maps