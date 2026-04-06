from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import BisectingKMeans
import time
import matplotlib.pyplot as plt

samples_experiment_1 = [50, 100, 2000, 5000, 25000]
n_clusters = 10
trials = 10

# Agglomeritive Ward Clustering
def experiment_1():
    print("------ Experiment 1 ------")

    ward_means_results = []
    bisecting_k_means_results = []

    for trial in range(0, trials):
        print ('Trial ', trial+1)

        print("(Agglomerative) Ward Clustering:")
        for sample in samples_experiment_1:
            print("Sample Size = ", sample)

            # Generate dataset
            X, _ = make_blobs(sample, n_clusters, cluster_std=1.0, random_state=0)

            start = time.time()
            ward_model = AgglomerativeClustering( n_clusters=n_clusters,metric='euclidean',linkage='ward')
            labels = ward_model.fit_predict(X)

            ward_means_results.append({
                'sample_size': sample,
                'labels': labels,
                'runtime': time.time() - start,
                'trial': trial
            })

        print("(Divisive) Bisecting K Means Clustering:")
        for sample in samples_experiment_1:
            print("Sample Size = ", sample)

            # Generate dataset
            X, _ = make_blobs(sample, n_clusters, cluster_std=1.0, random_state=0)

            start = time.time()
            bisecting_k_means_model = BisectingKMeans( n_clusters=n_clusters, random_state=0)
            labels = bisecting_k_means_model.fit_predict(X)

            bisecting_k_means_results.append({
                'sample_size': sample,
                'labels': labels,
                'runtime': time.time() - start,
                'trial': trial
            })

    ward_runtime_average = []
    for sample in samples_experiment_1:
        runtimes = [r['runtime'] for r in ward_means_results if r['sample_size'] == sample]
        ward_runtime_average.append({'sample_size': sample, 'avg_runtime': sum(runtimes) / len(runtimes)})

    k_means_runtime_average = []
    for sample in samples_experiment_1:
        runtimes = [r['runtime'] for r in bisecting_k_means_results if r['sample_size'] == sample]
        k_means_runtime_average.append({'sample_size': sample, 'avg_runtime': sum(runtimes) / len(runtimes)})

    return ward_means_results, bisecting_k_means_results, ward_runtime_average, k_means_runtime_average

ward_means_results, bisecting_k_means_results, ward_avg, k_means_avg = experiment_1()

print("\n \n -------- Experiment 1 Results -------- \n \n")

print("--- Agglomerative Ward Results ---")
for res in ward_means_results:
    print(f"Sample Size: {res['sample_size']:>6}  |  Runtime: {res['runtime']:.4f}s")

print("--- Agglomerative Ward Results Averages ---")
for r in ward_avg:
    print(f"  Sample Size: {r['sample_size']:>6}  |  Avg Runtime: {r['avg_runtime']:.4f}s")


print("--- Divisive Bisecting K Means Results ---")
for res in bisecting_k_means_results:
    print(f"Sample Size: {res['sample_size']:>6}  |  Runtime: {res['runtime']:.4f}s")

print("--- Divisive Bisecting K Means Averages ---")
for r in k_means_avg:
    print(f"  Sample Size: {r['sample_size']:>6}  |  Avg Runtime: {r['avg_runtime']:.4f}s")



# additional plotting
def plot_experiment_1(ward_means_results, bisecting_k_means_results, ward_avg, k_means_avg):

    fig, ax = plt.subplots(figsize=(10, 6))

    ward_sizes = [r['sample_size'] for r in ward_means_results]
    ward_runtimes = [r['runtime'] for r in ward_means_results]

    bkm_sizes = [r['sample_size'] for r in bisecting_k_means_results]
    bkm_runtimes = [r['runtime'] for r in bisecting_k_means_results]

    ax.scatter(ward_sizes, ward_runtimes, label='Ward (all trials)', alpha=0.5, color='blue')
    ax.scatter(bkm_sizes, bkm_runtimes, label='Bisecting KMeans (all trials)', alpha=0.5, color='orange')

    ax.set_title('Experiment 1: Runtime vs Sample Size (All Trials)')
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Runtime (s)')
    ax.legend()
    plt.tight_layout()
    plt.savefig('experiment_1_scatter.png', dpi=150)
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))

    ward_avg_sizes = [r['sample_size'] for r in ward_avg]
    ward_avg_runtimes = [r['avg_runtime'] for r in ward_avg]

    bkm_avg_sizes = [r['sample_size'] for r in k_means_avg]
    bkm_avg_runtimes = [r['avg_runtime'] for r in k_means_avg]

    ax.plot(ward_avg_sizes, ward_avg_runtimes, marker='o', label='Ward (avg)', color='blue')
    ax.plot(bkm_avg_sizes, bkm_avg_runtimes, marker='o', label='Bisecting KMeans (avg)', color='orange')

    ax.set_title('Experiment 1: Mean Runtime vs Sample Size')
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Mean Runtime (s)')
    ax.legend()
    plt.tight_layout()
    plt.savefig('experiment_1_averages.png', dpi=150)
    plt.show()

plot_experiment_1(ward_means_results, bisecting_k_means_results, ward_avg, k_means_avg)