from misc.utils import shortest_paths, EMPTY, AGENT, ENEMY, shortest_paths_naive


class Unit:
    """
    This class represents a unit (either agent or enemy) playing the game of hide & seek
    It contains: a set of coordinates (x, y)
                 a vision space represented as a list of tiles coordinates
                 a "memory" of previously seen tiles represented as a list of tiles coordinates
                 a set of points of interest to the unit
    """

    def __init__(self):
        super(Unit, self).__init__()

        self.xcoord = None
        self.ycoord = None
        self.vision = []
        self.previously_viewed = []
        self.interest_points = []



    def set_coord(self, x, y):
        """
        Modifies the position of the unit
        @param self:
        @param x: coordinate x to switch to
        @param y: coordinate y to switch to
        @return:
        """
        self.xcoord = x
        self.ycoord = y



    def compute_vision(self, current_map, mode="naive"):
        """
          Computes the list of tiles currently seen by the unit, set the "vision" attribute of the unit
          @param self:
          @param current_map: matrix representation of the current state of the map
          @param mode: mode representing wether the vision is estimated using a "naive" method or a more clever and expensive one
          @return:
          """
        vision = []
        len_map = len(current_map)
        if mode == "naive":
            already_checked = []

            # We process every tile across the map
            for i in range(0, len_map):
                for j in range(0, len_map):
                    if (i, j) not in already_checked:
                        clear = True
                        path = shortest_paths_naive((self.xcoord, self.ycoord), (i, j))
                        for p in path:  # We process the path (list of tile) from the unit, to (i,j)
                            if clear:  # If there is no block between the unit and this tile
                                if current_map[p[0]][p[1]] == EMPTY or current_map[p[0]][p[1]] == AGENT:
                                    vision.append(p)

                                else:  # That tile was not empty, it is thus blocking vision of further tiles down the path
                                    clear = False

                            already_checked.append(p)

            vision.append((self.xcoord, self.ycoord))

            # We also add the tile to the "memory" (the previously_viewed set of tiles)
            for v in vision:
                if v not in self.previously_viewed:
                    self.previously_viewed.append(v)

            self.vision = vision


        else:
            # We process every tile across the map
            for i in range(0, len_map):
                for j in range(0, len_map):
                    clear = True
                    paths = shortest_paths((self.xcoord, self.ycoord), (i, j), len_map, 5)
                    for path in paths:
                        for p in path:  # We process the path (list of tile) from the unit, to (i,j)
                            if clear:  # If there is no block between the unit and this tile
                                if current_map[p[0]][p[1]] == EMPTY or current_map[p[0]][p[1]] == AGENT or \
                                        current_map[p[0]][p[1]] == ENEMY:
                                    vision.append(p)

                                else:  # That tile was not empty, it is thus blocking vision of further tiles down
                                    # the path
                                    clear = False
            self.vision = vision
