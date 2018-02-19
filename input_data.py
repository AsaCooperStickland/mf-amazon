import numpy as np

class DataInput:

    def __init__(self, data, batch_size):
        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):

        if self.i == self.epoch_size:
            raise StopIteration
        ts = self.data[self.i * self.batch_size: min((self.i+1) * self.batch_size,
                       len(self.data))]
        self.i += 1

        u, i_next, r_next, sl = [], [], [], []
        for t in ts:
            u.append(t[0])
            i_next.append(t[3])
            r_next.append(t[4])
            sl.append(len(t[1]))

        max_sl = max(sl)

        hist_i = np.zeros([len(ts), max_sl], np.int64)
        hist_r = np.zeros([len(ts), max_sl], np.int64)

        k = 0
        for t in ts:
            for l in range(len(t[1])):
                hist_i[k][l] = t[1][l]
                hist_r[k][l] = t[2][l]
            k += 1
        return self.i, (u, i_next, r_next, hist_i, hist_r,  sl)

class DataInputTest:

    def __init__(self, data, batch_size):
        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):

        if self.i == self.epoch_size:
            raise StopIteration
        ts = self.data[self.i * self.batch_size: min((self.i+1) * self.batch_size,
                       len(self.data))]
        self.i += 1

        u, i_next, r_next, sl = [], [], [], []
        for t in ts:
            u.append(t[0])
            i_next.append(t[3])
            r_next.append(t[4])
            sl.append(len(t[1]))

        max_sl = max(sl)

        hist_i = np.zeros([len(ts), max_sl], np.int64)
        hist_r = np.zeros([len(ts), max_sl], np.int64)

        k = 0
        for t in ts:
            for l in range(len(t[1])):
                hist_i[k][l] = t[1][l]
                hist_r[k][l] = t[2][l]
            k += 1
        return self.i, (u, i_next, r_next, hist_i, hist_r,  sl)
