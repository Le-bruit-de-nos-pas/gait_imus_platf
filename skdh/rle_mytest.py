from numpy import nonzero, diff, insert, asarray

def rle(to_encode):
    starts = nonzero(diff(to_encode))[0] + 1
    print(starts)

    starts = insert(starts, (0, starts.size), (0, len(to_encode)))
    print(starts)
    
    lengths = diff(starts)
    print(lengths)
    starts = starts[:-1]  # remove that last index which isn't actually a start
    print(starts)
    values = asarray(to_encode)[starts]
    print(values)
    return lengths, starts, values


rle([0, 0, 0, 1, 1, 0, 2, 2, 2])
