import numpy as np
from graph_probability_aggregation_clustering import GPAC
import time
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment

acc_min = 1
acc_max = 0
nmi_min = 1
nmi_max = 0
ari_min = 1
ari_max = 0


class metrics:
    ari = adjusted_rand_score
    nmi = normalized_mutual_info_score

    @staticmethod
    def acc(y_true, y_pred):
        y_true = y_true.astype(np.int64)
        y_pred = y_pred.astype(np.int64)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        row, col = linear_sum_assignment(w.max() - w)
        return sum([w[i, j] for i, j in zip(row, col)]) * 1.0 / y_pred.size



if __name__ == '__main__':
    # Load dataset
    dataset = 'usps'
    X = np.load(f"{dataset}.npy")
    Y = np.load(f"{dataset}_label.npy")
    print(f"Load {dataset} dataset")

    results = {'acc': [], 'nmi': [], 'ari': []}
    run_time = []
    c = int(Y.max() + 1)

    # Predefined GPAC parameters for different datasets
    gpac_params = {
        'CIFAR-10': {'num_clusters': 10, 'batch_size': 100},
        'CIFAR-20': {'num_clusters': 20, 'batch_size': 100},
        'STL-10': {'num_clusters': 10, 'batch_size': 100},
        'mnist': {'num_clusters': 10, 'batch_size': 100},
        'emnist': {'num_clusters': 47, 'batch_size': 100},
        'pendigits': {'num_clusters': 10, 'batch_size': 10},
        'isolet': {'num_clusters': 26, 'batch_size': 10},
        'usps': {'num_clusters': 10, 'batch_size': 10},
        'coil': {'num_clusters': 100, 'batch_size': 10},
    }

    # Retrieve the parameters for the selected dataset, defaulting to general values if not found
    gpac_config = gpac_params.get(dataset, {'n_clusters': c, 'batch_size': 10})
    gpac = GPAC(fuzziness=1.05, knn_neighbors=10, **gpac_config)

    # Initialize min/max values for performance tracking
    acc_min, acc_max = float('inf'), float('-inf')
    nmi_min, nmi_max = float('inf'), float('-inf')
    ari_min, ari_max = float('inf'), float('-inf')

    # Run the clustering experiment 50 times
    for i in range(50):
        start_time = time.time()

        # Perform clustering prediction
        p = gpac.knn_predict(X)
        prediction = np.argmax(p, axis=1)

        # Evaluate clustering performance
        acc, nmi, ari = (metrics.acc(Y.astype(int), prediction), metrics.nmi(Y.astype(int), prediction),
                         metrics.ari(Y.astype(int), prediction))
        print(f"Run: {i + 1}/50 | ACC: {acc:.4f} | NMI: {nmi:.4f} | ARI: {ari:.4f}")

        # Update min/max values for tracking best and worst performance
        acc_min, acc_max = min(acc_min, acc), max(acc_max, acc)
        nmi_min, nmi_max = min(nmi_min, nmi), max(nmi_max, nmi)
        ari_min, ari_max = min(ari_min, ari), max(ari_max, ari)

        # Store results
        results['acc'].append(acc)
        results['nmi'].append(nmi)
        results['ari'].append(ari)

        # Record runtime
        end_time = time.time()
        run_time.append(end_time - start_time)

    # Compute and display statistical results
    print(f"ACC: mean={np.mean(results['acc']):.4f}, std={np.std(results['acc']):.4f}, best={acc_max:.4f}")
    print(f"NMI: mean={np.mean(results['nmi']):.4f}, std={np.std(results['nmi']):.4f}, best={nmi_max:.4f}")
    print(f"ARI: mean={np.mean(results['ari']):.4f}, std={np.std(results['ari']):.4f}, best={ari_max:.4f}")



