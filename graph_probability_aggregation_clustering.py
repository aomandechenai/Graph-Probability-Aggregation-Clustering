import faiss
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from tqdm import tqdm
from kmpp import kmeans_plusplus

def split_list(lst, size):
    """Splits a list into chunks of a given size."""
    return [lst[i:i + size] for i in range(0, len(lst), size)]

class GPAC:
    """
    Graph Probability Aggregation Clustering (GPAC)
    """

    def __init__(self, fuzziness=1.05, num_clusters=8, knn_neighbors=20, alpha=1, beta='step', batch_size=10,
                 init_method='k-means++', max_iterations=50, convergence_threshold=1e-3):
        self.fuzziness = fuzziness
        self.num_clusters = num_clusters
        self.knn_neighbors = knn_neighbors
        self.alpha = alpha
        self.beta = beta
        self.init_method = init_method
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

    def initialize_matrices(self, data, num_samples, init_method):
        """Initializes the hard assignment matrix."""
        hard_assignments = np.zeros((num_samples, self.num_clusters), dtype=int)
        if init_method == 'k-means++':
            cluster_centers, _ = kmeans_plusplus(data, n_clusters=self.num_clusters)
            labels = np.argmin(cdist(data, cluster_centers), axis=1)
            for i, label in enumerate(labels):
                hard_assignments[i, label] = 1
        elif init_method == 'random':
            labels = np.random.randint(0, self.num_clusters, num_samples)
            for i, label in enumerate(labels):
                hard_assignments[i, label] = 1
        probabilities = np.ones_like(hard_assignments) / self.num_clusters

        return hard_assignments, probabilities

    @staticmethod
    def build_knn_graph(data, knn_neighbors=10):
        """Builds the k-NN graph using FAISS."""
        data = data.astype('float32')
        index = faiss.IndexFlatIP(data.shape[1])
        index.add(data)
        distances, indices = index.search(data, knn_neighbors)
        distances = np.exp(-(2 - distances) ** 2 / 0.1)
        return distances, indices

    def knn_predict(self, data):
        """Runs the GPAC clustering algorithm."""
        data /= np.linalg.norm(data, axis=1, keepdims=True)
        num_samples = data.shape[0]
        hard_assignments, probabilities = self.initialize_matrices(data, num_samples, self.init_method)
        knn_distances, knn_indices = self.build_knn_graph(data, knn_neighbors=self.knn_neighbors + 1)

        indptr = np.arange(num_samples + 1) * (self.knn_neighbors + 1)
        indices = knn_indices.ravel()
        adjacency_matrix = csr_matrix((np.ones_like(indices), indices, indptr), shape=(num_samples, num_samples), dtype=bool)
        adjacency_matrix = (adjacency_matrix + adjacency_matrix.transpose())
        knn_matrix = adjacency_matrix.copy()
        propagation_steps = max(1, int(np.ceil(np.log(num_samples / self.num_clusters) / np.log(self.knn_neighbors)) - 1))
        for _ in range(propagation_steps):
            adjacency_matrix = adjacency_matrix.dot(knn_matrix)
        adjacency_matrix = adjacency_matrix.toarray()
        np.fill_diagonal(adjacency_matrix, 0)

        for iteration in tqdm(range(self.max_iterations), desc="GPAC Training"):
            shuffled_indices = np.random.permutation(num_samples)
            batch_indices = split_list(shuffled_indices, self.batch_size * self.num_clusters)
            prev_probabilities = probabilities.copy()
            lambda_factor = iteration / self.max_iterations if self.beta == 'step' else 0.9

            for batch in batch_indices:
                cluster_prob_sums = probabilities[batch].sum(0)
                cluster_counts = hard_assignments[batch].sum(0)

                for sample_idx in batch:
                    sample_knn = knn_indices[sample_idx, 1:]
                    sample_distances = knn_distances[sample_idx, 1:]
                    sample_adjacency = adjacency_matrix[sample_idx, batch].astype(bool)

                    cluster_prob_sums -= probabilities[sample_idx]
                    cluster_score = cluster_prob_sums - self.alpha * hard_assignments[batch[sample_adjacency]].sum(0)
                    cluster_score = cluster_score - cluster_score.min() + 1
                    score1 = np.einsum('i,ik->k', sample_distances, probabilities[sample_knn])
                    score2 = np.power(cluster_score, int(-1 / (self.fuzziness - 1)))
                    probabilities[sample_idx] = lambda_factor * score1 / score1.sum() + (1 - lambda_factor) * score2 / score2.sum()
                    cluster_counts -= hard_assignments[sample_idx]
                    best_cluster = np.argmin(cluster_counts - self.alpha * np.einsum('ik->k', probabilities[batch[sample_adjacency]]**self.fuzziness))
                    cluster_counts[best_cluster] += 1
                    hard_assignments[sample_idx] = 0
                    hard_assignments[sample_idx, best_cluster] = 1
                    cluster_prob_sums += probabilities[sample_idx]

            if np.sum(np.abs(probabilities - prev_probabilities)) / num_samples <= self.convergence_threshold:
                break

        return probabilities
