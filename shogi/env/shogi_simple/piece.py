class Piece:
    def __init__(self, directions, symbol, color):
        self.directions = directions
        self.symbol = symbol
        self.color = color

    def __str__(self):
        color_indicator = '\033[31m' if self.color == 0 else '\033[32m'
        if self.symbol == "WANG":
            return color_indicator + "王" + '\033[0m'
        elif self.symbol == "JANG":
            return color_indicator + "將" + '\033[0m'
        elif self.symbol == "SANG":
            return color_indicator + "相" + '\033[0m'
        elif self.symbol == "HU":
            return color_indicator + "侯" + '\033[0m'
        elif self.symbol == "JA":
            return color_indicator + "子" + '\033[0m'
        else:
            raise Exception("Invalid piece symbol")

# row column movement
WANG_RED = Piece({0, 1, 2, 3, 4, 5, 6, 7}, 'WANG', 0)
JANG_RED = Piece({1, 3, 4, 6}, 'JANG', 0)
SANG_RED = Piece({0, 2, 5, 7}, 'SANG', 0)
HU_RED = Piece({0, 1, 2, 3, 4, 6}, 'HU', 0)
JA_RED = Piece({1}, 'JA', 0)

WANG_GREEN = Piece({0, 1, 2, 3, 4, 5, 6, 7}, 'WANG', 1)
JANG_GREEN = Piece({1, 3, 4, 6}, 'JANG', 1)
SANG_GREEN = Piece({0, 2, 5, 7}, 'SANG', 1)
HU_GREEN = Piece({0, 1, 2, 3, 4, 6}, 'HU', 1)
JA_GREEN = Piece({1}, 'JA', 1)

DIRECTION = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]

PRISONER_SYMBOL = ["JA", "SANG", "JANG"]


