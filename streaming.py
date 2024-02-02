
import seaborn as sns
import numpy.random as npr
sns.color_palette("bright")
import time

from torch.utils.data import DataLoader , TensorDataset


class StreamingDataLoaderold:
    def __init__(self, dataset, batch_size, simulate_stream_speed=1.0):
        self.dataset = dataset
        self.data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.simulate_stream_speed = simulate_stream_speed

    def __iter__(self):
        return self

    def __next__(self):
        time.sleep(self.simulate_stream_speed)  # Simulate real-time data arrival
        return next(iter(self.data_loader))

class StreamingDataLoader:
    def __init__(self, dataset, batch_size, simulate_stream_speed=1.0):
        self.dataset = dataset
        self.data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.simulate_stream_speed = simulate_stream_speed
        self.data_iter = iter(self.data_loader)

    def __iter__(self):
        return self

    def __next__(self):
        time.sleep(self.simulate_stream_speed)  # Simulate real-time data arrival
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.data_loader)
            batch = next(self.data_iter)
        return batch #next(iter(self.data_loader))