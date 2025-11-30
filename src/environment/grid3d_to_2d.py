# src/environment/grid3d_to_2d.py

def to_2d(floor, x, y, W):
    """
    Convert (floor, x, y) to unfolded 2D coordinates (gx, gy).
    floor: which floor (0..F-1)
    x: row index on that floor
    y: column index on that floor
    W: width of each floor
    """
    gx = x
    gy = y + floor * W
    return gx, gy


def to_3d(gx, gy, W):
    """
    Convert unfolded 2D coordinate (gx, gy) back to (floor, x, y).
    gx: row index in the unfolded grid
    gy: column index in the unfolded grid
    W: width of each floor
    """
    floor = gy // W
    y = gy % W
    x = gx
    return floor, x, y
