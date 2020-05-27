from .imports import *

def get_clusters(kps, labels):
    clusters = defaultdict(list)
    for pt, c in zip(kps, labels):
        clusters[c].append(pt)
    return clusters

def get_cluster_bbox(cluster):
    xmin, xmax, ymin, ymax = float('inf'), 0, float('inf'), 0
    for pt in cluster:
        x, y = pt
        if x < xmin:
            xmin = x
        if x > xmax:
            xmax = x
        if y < ymin:
            ymin = y
        if y > ymax:
            ymax = y
    xmin = np.round(xmin).astype(int)
    ymin = np.round(ymin).astype(int)
    xmax = np.round(xmax).astype(int)
    ymax = np.round(ymax).astype(int)
    return xmin, ymin, xmax, ymax