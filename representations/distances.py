from scipy.spatial import distance
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import jensenshannon

def compute_distance_for_points(a, b, requested_distance_metric):
    
    if "euclidean" in requested_distance_metric:
        return distance.euclidean(a, b) * 1e6

    if "cosine" in requested_distance_metric:
        return distance.cosine(a, b) * 1e6
    return 0
    
def compute_distances_for_sets(a, b, requested_distance_metric):

    print(requested_distance_metric)
    a = a.to_numpy()
    b = b.to_numpy()

    if requested_distance_metric == 'cluster_count':
        a_clustering = AffinityPropagation(random_state=9).fit(a)
        b_clustering = AffinityPropagation(random_state=9).fit(b)
        
        if requested_distance_metric == 'cluster_count':
            rez = max(a_clustering.labels_) != max(b_clustering.labels_)
            print(rez)
            return rez
        
    if requested_distance_metric == 'pointwise':
        rez=pairwise_distances(a,b)
        print(rez)
        return rez
    if requested_distance_metric == 'jsd':
        rez = jensenshannon(a,b, axis=0)
        print(rez)
        return rez

    return 0