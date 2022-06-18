from scipy.spatial import distance
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import jensenshannon

def compute_distance_for_points(a, b, requested_distance_metrics):
    
    if "euclidean" in requested_distance_metrics:
        return distance.euclidean(a, b) * 1e6

    if "cosine" in requested_distance_metrics:
        return distance.cosine(a, b) * 1e6
    return 0
    
def compute_distances_for_sets(a, b, requested_distance_metrics):
    if len(a) == 0 or len(b) == 0:
        return 0

    if requested_distance_metrics == 'cluster_count':
        a_clustering = AffinityPropagation(random_state=9).fit(a)
        b_clustering = AffinityPropagation(random_state=9).fit(b)
        
        if requested_distance_metrics == 'cluster_count':
            return max(a_clustering.labels_) != max(b_clustering.labels_)
        
    if requested_distance_metrics == 'pointwise':
        return pairwise_distances(a,b)
    if requested_distance_metrics == 'jsd':
        return jensenshannon(a,b)

    return 0