from envs.maps import Maps


class Dataloader:
    """
    This class is a simple dataloader transforming map files into Maps object
    The format for map files is first a matrix composed of 1 (empty tiles) and 2 (block tiles)
    Then a line indicating the possible starting positions (coordinates) for an agent (player), starting with "A:"
    Then a line indicating the possible starting positions (coordinates) for an enemy, starting with "E:"
    Examples can be seen in the data repository
    """
    def __init__(self):
        super(Dataloader, self).__init__()

    """
    @param self:
    @param file: the path of the (textual) file containing information about a map
    @return: a Maps object
    """
    def load_map_from_file(self, file):
        map_matrix = []
        agent_positions = []
        enemy_positions = []
        with open(file, 'r') as f:
            for line in f:
                if line[0].isdigit():
                    matrix_row = [int(num) for num in line.split(',')]
                    map_matrix.append(matrix_row)
                elif line[0] == 'A':
                    list_pos = line[2:].split('|')
                    for pos in list_pos:
                        agent_x = int(pos.split(',')[0])
                        agent_y = int(pos.split(',')[1])
                        agent_positions.append((agent_x, agent_y))
                elif line[0] == 'E':
                    list_pos = line[2:].split('|')
                    for pos in list_pos:
                        enemy_x = int(pos.split(',')[0])
                        enemy_y = int(pos.split(',')[1])
                        enemy_positions.append((enemy_x, enemy_y))

        return Maps(file, map_matrix, agent_positions, enemy_positions)

