import pickle
# from dlsys import autodiff as ad
import numpy as np


class Saver:

    def __init__(self, file_name):
        self.file_name = file_name
        self.output_file = open(file_name, "wb")
        self.KV_map = {}
        # self.keys = []

    def append(self, key, value):
        # self.keys.append(key)
        self.KV_map.update({key:value})

    def save(self):
        pickle.dump(self.KV_map, self.output_file)

    def KV_print(self):
        print self.KV_map

    def close(self):
        self.output_file.close()

class Loader:

    def __init__(self, file_name):
        self.file_name = file_name
        self.input_file = open(file_name, "rb")
        self.KV_map = pickle.load(self.input_file)

    def load(self, key):
        return self.KV_map[key]

    def Key_print(self):
        keys = []
        for key in self.KV_map:
            keys.append(key)
        print keys

    def KV_print(self):
        for key, value in self.KV_map.items():
            print key, value

    def close(self):
        self.input_file.close()

# if __name__ == "__main__":
#     file_name = "data.pkl"
#     s = Saver(file_name)

#     s.append(3, np.zeros([10]))
#     s.append(1, np.ones([10]))
#     # s.KV_print()

#     s.save()
#     s.close()
#     l = Loader(file_name)
#     l.KV_print()
