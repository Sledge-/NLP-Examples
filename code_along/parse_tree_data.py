# Placehoder file - going to dev in jupyter

class Tree:
    def __init__(self, data):
        self.data = data
        self.tree = {}
        self.parse_chars = ['(',')']
        self.parse_data()

    def parse_data(self):
        for c, i in enumerate(self.data):
            if c == '(':
                data = self.data[i:]
                next_index = data.find()
                curr_tupe = self.data[i+1:]

if __name__ == '__main__':
