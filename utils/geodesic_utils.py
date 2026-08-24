import numpy as np


def fix_geodesic_distance_for_disconnected_components(geodesic_distance_matrix):
    """Fix geodesic distance for disconnected components, when computed with potpourri3d

    Detects unreachable points by a very simple heuristic. If there exist more then 100 values, with exactly the same
    distance within the matrix, we set all occurances of the corresponding value to `np.inf`.

    Args:
        geodesic_distance_matrix: a N x N matrix with precomputed geodesic distances

    Returns:
        N x N matrix with distances between unreachable points set to `np.inf` as a distance

    """
    dis, cnt = np.unique(geodesic_distance_matrix, return_counts=True)
    candidate_val = dis[np.argmax(cnt)]
    repeated_vals = [d for d, c in zip(dis, cnt) if c > 100]
    if candidate_val in repeated_vals:
        geodesic_distance_matrix[geodesic_distance_matrix == candidate_val] = np.inf
    return geodesic_distance_matrix
