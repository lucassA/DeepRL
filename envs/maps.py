

class Maps:
    """
    This class represents a map on which a game of hide & seek takes place
    It contains: a map filename
                 a representation of the initial, unchanged map
                 a representation of the current in-game map
                 a set of initial agent positions
                 a set of initial enemy positions

    """
    def __init__(self, map_name, initial_map, agent_positions, enemy_positions):
        super(Maps, self).__init__()

        self.map_name = map_name
        self.initial_map = initial_map
        self.current_map = initial_map
        self.agent_initial_position = agent_positions
        self.enemy_initial_position = enemy_positions

