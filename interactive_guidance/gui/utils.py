import numpy as np


def is_point_within_area(points, p1, p2, r):
    """
    Checks if points are within a given radius r of a line defined by points p1 and p2.
    """
    points = np.array(points)
    p1 = np.array(p1)
    p2 = np.array(p2)

    d = p2 - p1

    # Check if p1 and p2 are the same
    if not np.any(d):
        P_closest = p1 + np.zeros_like(points)
    else:
        f = points - p1

        t = np.dot(f, d) / np.dot(d, d)
        t = np.clip(t, 0, 1)
        t = np.stack([t, t], axis=2)

        # Closest point on the line to points
        P_closest = [[p1]] + t * [[d]]

    # Closest distance from each point to the line
    distance = np.linalg.norm(points - P_closest, axis=2)

    return distance <= r
