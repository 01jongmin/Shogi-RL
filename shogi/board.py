from .piece import *
import numpy as np
ROW_COUNT = 4
COL_COUNT = 3

class Board:
    def __init__(self):
       state = np.empty((ROW_COUNT, COL_COUNT), dtype=object)

       state[0][0] = JANG_GREEN
       state[0][1] = WANG_GREEN
       state[0][2] = SANG_GREEN
       state[1][1] = JA_GREEN

       state[2][1] = JA_RED
       state[3][0] = SANG_RED
       state[3][1] = WANG_RED
       state[3][2] = JANG_RED

       self.state = state

       self.prisoners = { \
           0: {"JA": 0, "SANG": 0, "JANG": 0}, \
           1: {"JA": 0, "SANG": 0, "JANG": 0}} 

       self.move_count = 0

    def get_perspective_state(self, agent_idx):
        return self.state if agent_idx == 0 else np.rot90(self.state, k=2)

    def legal_moves(self, agent_idx):
        state = self.get_perspective_state(agent_idx)
        legal_moves_list = []
        for row in range(ROW_COUNT):
            for col in range(COL_COUNT):
                piece = state[row][col]
                if piece and piece.color == agent_idx:
                    legal_moves_list.extend(self.legal_moves_idx(agent_idx, state, piece, row, col))
                elif not piece and (row * 3 + col) >= 3:
                    for prisoner_symbol, count in self.prisoners[agent_idx].items():
                        if count > 0:
                            prisoner_idx = PRISONER_SYMBOL.index(prisoner_symbol)
                            legal_moves_list.append(96 + prisoner_idx * 9 + row * 3 + col - 3)

        return np.uint8(legal_moves_list)

    def legal_moves_idx(self, agent_idx, state, piece, row, col):
        ls = []
        for dir_idx in piece.directions:
            new_row = row + DIRECTION[dir_idx][0]
            new_col = col + DIRECTION[dir_idx][1]
            if new_row >= 0 and new_row < ROW_COUNT \
                and new_col >= 0 and new_col < COL_COUNT \
                and (state[new_row][new_col] is None or  \
                        state[new_row][new_col].color != agent_idx):
                    pos_id = 3 * row + col
                    ls.append(pos_id * 8 + dir_idx)

        return ls

    def step(self, action_idx, agent_idx):
        self.move_count += 1
        state = self.get_perspective_state(agent_idx)

        if action_idx < 96:
            cell_id, direction_idx = divmod(action_idx, 8)
            direction = DIRECTION[direction_idx]
            row, col = divmod(cell_id, 3)
            new_row = row + direction[0]
            new_col = col + direction[1]

            won = None

            if state[new_row][new_col]:
                captured_piece = state[new_row][new_col]
                if captured_piece.symbol == "JA" or captured_piece.symbol == "HU":
                    self.prisoners[agent_idx]["JA"] += 1
                elif captured_piece.symbol == "JANG":
                    self.prisoners[agent_idx]["JANG"] += 1
                elif captured_piece.symbol == "SANG":
                    self.prisoners[agent_idx]["SANG"] += 1
                else:
                    won = agent_idx

            state[new_row][new_col] = state[row][col]
            state[row][col] = None

            if won != None:
                return won

            if new_row == 0 and state[new_row][new_col].symbol == "JA":
                if agent_idx == 1:
                    state[new_row][new_col] = HU_GREEN
                else:
                    state[new_row][new_col] = HU_RED
            elif new_row == 0 and state[new_row][new_col].symbol == "WANG":
                return (not agent_idx) if self.can_be_captured(new_col, agent_idx) else agent_idx

        else:
            piece_idx, cell = divmod(action_idx - 96, 9)
            row, col = divmod(cell + 3, 3)

            piece_symbol = PRISONER_SYMBOL[piece_idx]
            self.prisoners[agent_idx][piece_symbol] -= 1

            if agent_idx == 0:
                if piece_symbol == "JANG":
                    state[row][col] = JANG_RED
                elif piece_symbol == "SANG":
                    state[row][col] = SANG_RED
                elif piece_symbol == "JA":
                    state[row][col] = JA_RED
            else:
                if piece_symbol == "JANG":
                    state[row][col] = JANG_GREEN
                elif piece_symbol == "SANG":
                    state[row][col] = SANG_GREEN
                elif piece_symbol == "JA":
                    state[row][col] = JA_GREEN

        if self.move_count >= 50:
            return -1

    def render(self, agent_idx):
        #state = self.get_perspective_state(agent_idx)
        print("")
        print("------------------------------")
        for row in self.state:
            print(*np.where(row == None, " ", row), sep="\t")
        print("--------------------------------")
        print("")

    def can_be_captured(self, king_col, agent_idx):
        if king_col == 0:
            king_col = 2
        elif king_col == 1:
            king_col = 1
        elif king_col == 2:
            king_col = 0

        opponent_state = self.get_perspective_state(not agent_idx)
        col_occupied = set()

        for row in range(2, 4):
            for col in range(3):
                if opponent_state[row][col] and opponent_state[row][col].color != agent_idx:
                    for direction_idx in opponent_state[row][col].directions:
                        direction = DIRECTION[direction_idx]
                        col_occupied.add((row + direction[0], col + direction[1]))

        return (3, king_col) in col_occupied

