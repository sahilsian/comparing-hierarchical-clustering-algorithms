from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import BisectingKMeans
import time
import matplotlib.pyplot as plt

clusters_experiment_2 = [2,5,10,20,50]
n_samples = 5000
trials = 10

# Agglomeritive Ward Clustering
def experiment_2():
    print("------ Experiment 2 ------")

    ward_means_results = []
    bisecting_k_means_results = []

    for trial in range(0, trials):
        print ('Trial ', trial+1)

        print("(Agglomerative) Ward Clustering:")
        for n_cluster in clusters_experiment_2:
            print("N Clusters = ", n_cluster)

            # Generate dataset
            X, _ = make_blobs(n_samples, n_cluster, cluster_std=1.0, random_state=0)

            start = time.time()
            ward_model = AgglomerativeClustering( n_clusters=n_cluster,metric='euclidean',linkage='ward')
            labels = ward_model.fit_predict(X)

            ward_means_results.append({
                'sample_size': n_samples,
                'n_clusters': n_cluster,
                'labels': labels,
                'runtime': time.time() - start,
                'trial': trial
            })

        print("(Divisive) Bisecting K Means Clustering:")
        for n_cluster in clusters_experiment_2:
            print("N Clusters = ", n_cluster)

            # Generate dataset
            X, _ = make_blobs(n_samples, n_cluster, cluster_std=1.0, random_state=0)

            start = time.time()
            bisecting_k_means_model = BisectingKMeans( n_clusters=n_cluster, random_state=0)
            labels = bisecting_k_means_model.fit_predict(X)

            bisecting_k_means_results.append({
                'sample_size': n_samples,
                'n_clusters': n_cluster,
                'labels': labels,
                'runtime': time.time() - start,
                'trial': trial
            })

    ward_runtime_average = []
    for n_cluster in clusters_experiment_2:
        runtimes = [r['runtime'] for r in ward_means_results if r['n_clusters'] == n_cluster]
        ward_runtime_average.append({'n_clusters': n_cluster, 'avg_runtime': sum(runtimes) / len(runtimes)})

    k_means_runtime_average = []
    for n_cluster in clusters_experiment_2:
        runtimes = [r['runtime'] for r in bisecting_k_means_results if r['n_clusters'] == n_cluster]
        k_means_runtime_average.append({'n_clusters': n_cluster, 'avg_runtime': sum(runtimes) / len(runtimes)})

    return ward_means_results, bisecting_k_means_results, ward_runtime_average, k_means_runtime_average

ward_means_results, bisecting_k_means_results, ward_avg, k_means_avg = experiment_2()

print("\n\n-------- Experiment 2 Results --------\n\n")

print("--- Agglomerative Ward Results ---")
for res in ward_means_results:
    print(f"  N Clusters: {res['n_clusters']:>6}  |  Runtime: {res['runtime']:.4f}s")

print("--- Agglomerative Ward Results Averages ---")
for r in ward_avg:
    print(f"  N Clusters: {r['n_clusters']:>6}  |  Avg Runtime: {r['avg_runtime']:.4f}s")

print("--- Divisive Bisecting K Means Results ---")
for res in bisecting_k_means_results:
    print(f"  N Clusters: {res['n_clusters']:>6}  |  Runtime: {res['runtime']:.4f}s")

print("--- Divisive Bisecting K Means Averages ---")
for r in k_means_avg:
    print(f"  N Clusters: {r['n_clusters']:>6}  |  Avg Runtime: {r['avg_runtime']:.4f}s")



def plot_experiment_2(ward_means_results, bisecting_k_means_results, ward_avg, k_means_avg):

    # ── Scatter plot of all trial results ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    ward_clusters = [r['n_clusters'] for r in ward_means_results]
    ward_runtimes = [r['runtime'] for r in ward_means_results]

    bkm_clusters = [r['n_clusters'] for r in bisecting_k_means_results]
    bkm_runtimes = [r['runtime'] for r in bisecting_k_means_results]

    ax.scatter(ward_clusters, ward_runtimes, label='Ward (all trials)', alpha=0.5, color='blue')
    ax.scatter(bkm_clusters, bkm_runtimes, label='Bisecting KMeans (all trials)', alpha=0.5, color='orange')

    ax.set_title('Experiment 2: Runtime vs Cluster Count (All Trials)')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Runtime (s)')
    ax.legend()
    plt.tight_layout()
    plt.savefig('experiment_2_scatter.png', dpi=150)
    plt.show()

    # ── Line plot of averages ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    ward_avg_clusters = [r['n_clusters'] for r in ward_avg]
    ward_avg_runtimes = [r['avg_runtime'] for r in ward_avg]

    bkm_avg_clusters = [r['n_clusters'] for r in k_means_avg]
    bkm_avg_runtimes = [r['avg_runtime'] for r in k_means_avg]

    ax.plot(ward_avg_clusters, ward_avg_runtimes, marker='o', label='Ward (avg)', color='blue')
    ax.plot(bkm_avg_clusters, bkm_avg_runtimes, marker='o', label='Bisecting KMeans (avg)', color='orange')

    ax.set_title('Experiment 2: Mean Runtime vs Cluster Count')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Mean Runtime (s)')
    ax.legend()
    plt.tight_layout()
    plt.savefig('experiment_2_averages.png', dpi=150)
    plt.show()

plot_experiment_2(ward_means_results, bisecting_k_means_results, ward_avg, k_means_avg)