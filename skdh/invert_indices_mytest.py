import numpy as np 

def invert_indices(starts, stops, zero_index, end_index):

    if starts.size != stops.size:
        raise ValueError("starts and stops indices arrays must be the same size")
    if starts.size == 0:
        return np.array([zero_index]), np.array([end_index])

    inv_stops = np.insert(np.roll(starts, -1), 0, starts[0])
    inv_stops[-1] = end_index

    inv_starts = np.insert(stops, 0, 0)
    mask = inv_starts != inv_stops

    return inv_starts[mask], inv_stops[mask]





# Test inputs
starts = np.array([3, 10])
stops = np.array([5, 15])
zero_index = 0
end_index = 20

inv_starts, inv_stops = invert_indices(starts, stops, zero_index, end_index)
print("", inv_starts, "\n", inv_stops)