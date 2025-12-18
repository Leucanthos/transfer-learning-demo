import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

def air_hpg_visualization(
        air:pd.DataFrame, 
        hpg:pd.DataFrame, 
        cols:list[str],
        random_state: int = 42,
        sample_size: int = 2000,
    )->None:
    np.random.seed(random_state)

    """
    先构建便准化数据集
    """
    # 对两个数据集进行抽样
    air_cnt, hpg_cnt = len(air), len(hpg)
    adj_coef = air_cnt / hpg_cnt

    air_sample = air[cols].sample(
        n=min(int(sample_size * adj_coef), air_cnt),
        random_state=random_state
    )
    hpg_sample = hpg[cols].sample(
        n=min(int(sample_size), hpg_cnt),
        random_state=random_state
    )

    # 标准化特征
    scaler = StandardScaler()
    xy_air = scaler.fit_transform(air_sample)
    xy_hpg = scaler.transform(hpg_sample)

    """
    构建t-SNE
    """
    # 合并数据用于t-SNE
    combined_xy = np.vstack([xy_air, xy_hpg])
    dataset_labels_xy = np.array(['air'] * len(xy_air) + ['hpg'] * len(xy_hpg))

    # 使用t-SNE降维到2D
    print("Running t-SNE for (X,y) joint distribution...")
    tsne_xy = TSNE(n_components=2, random_state=random_state, perplexity=30, verbose=2)
    xy_tsne = tsne_xy.fit_transform(combined_xy)

    # 分离t-SNE结果
    air_xy_tsne, hpg_xy_tsne = xy_tsne[dataset_labels_xy == 'air'], xy_tsne[dataset_labels_xy == 'hpg']

    """
    K-means
    """
    optimal_k = elbow_kmeans(hpg_sample)

    kmeans_hpg = KMeans(n_clusters=optimal_k, init='k-means++', random_state=random_state, n_init='auto')
    kmeans_hpg.fit(hpg_sample)

    # 获取质心
    centroids = kmeans_hpg.cluster_centers_
    print(f"Using K={optimal_k} for HPG centroids.")

    

    # 假设 xy_air 是 AIR 的特征矩阵 (N_samples, N_features)
    X_air = xy_air

    # 创建一个 NearestNeighbors 模型来查找最近的质心
    nbrs_centroids = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(centroids)
    distances_to_centroids, _ = nbrs_centroids.kneighbors(X_air)

    # -----------------------------
    # 步骤 5: 找出距离最近的 20% 的 AIR 样本
    # -----------------------------

    # 计算 30% 和 70% 分位数并按分位数分组
    p30 = np.percentile(distances_to_centroids.ravel(), 30)
    p70 = np.percentile(distances_to_centroids.ravel(), 70)
    threshold_distance = p30  # 保留用于后续输出（设为30%分位数）

    # 分组掩码：前30%，30%-70%，其他(>70%)
    mask_top30 = distances_to_centroids.ravel() <= p30
    mask_30_70 = (distances_to_centroids.ravel() > p30) & (distances_to_centroids.ravel() <= p70)
    mask_rest = distances_to_centroids.ravel() > p70

    # 各组对应的 t-SNE 点
    air_top30_points = air_xy_tsne[mask_top30]
    air_mid30_70_points = air_xy_tsne[mask_30_70]
    air_rest_points = air_xy_tsne[mask_rest]

    return air_top30_points, air_mid30_70_points, air_rest_points, hpg_xy_tsne


    # -----------------------------
    # 步骤 6: 可视化结果
    # -----------------------------

    plt.figure(figsize=(14, 6))

    # 子图 1: 原始的基于最近邻的方法 (如果存在)
    # 注意：你需要保留原始的 `close_air_points` 和 `far_air_points` 变量
    # 如果不存在，可以注释掉这部分
    try:
        plt.subplot(1, 2, 1)
        plt.scatter(hpg_xy_tsne[:, 0], hpg_xy_tsne[:, 1], alpha=0.6, label='HPG (t-SNE)', s=5, color='blue')
        if 'far_air_points' in locals():
            plt.scatter(far_air_points[:, 0], far_air_points[:, 1], alpha=0.6, label='AIR (far from HPG)', s=5, color='lightcoral')
        if 'close_air_points' in locals():
            plt.scatter(close_air_points[:, 0], close_air_points[:, 0], alpha=0.8, label='AIR closest to HPG (Nearest Neighbor top 20%)', s=5, color='red')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title('(Original Method) Closest AIR to HPG Samples')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
    except Exception as e:
        print(f"Could not plot original method subplot: {e}")
        plt.clf() # 清除当前 figure
        plt.figure(figsize=(7, 6))
        plt.subplot(1, 1, 1) # 只画一个子图

    # 子图 2: 基于 KMeans 质心的新方法
    plt.subplot(1, 1, 1) # 如果上面的 try 失败了，这个会是第一个也是唯一一个子图
    plt.scatter(hpg_xy_tsne[:, 0], hpg_xy_tsne[:, 1], alpha=0.6, label='HPG (t-SNE)', s=5, color='blue')
    plt.scatter(far_air_points_elbow[:, 0], far_air_points_elbow[:, 1], alpha=0.6, label='AIR (far from HPG Centroids)', s=5, color='lightcoral')
    plt.scatter(close_air_points_elbow[:, 0], close_air_points_elbow[:, 1], alpha=0.8, label=f'AIR closest to HPG Centroids (KMeans top 20%, K={optimal_k})', s=5, color='red')

    # （可选）可视化质心在 t-SNE 空间的位置（需要将质心投影到 t-SNE 空间，这里简化处理，
    # 假设你有办法将 centroids 映射到 t-SNE 空间，或者用原始特征空间的均值近似）
    # 例如，如果你有 t-SNE 模型，可以用 `tsne_model.transform(centroids)` 来获得近似的 t-SNE 坐标。
    # 这里我们不画质心，因为没有直接的 t-SNE 坐标。

    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('(New Method) Closest AIR to HPG Cluster Centroids')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # 步骤 7: 输出统计信息
    # -----------------------------

    print("\n--- Summary ---")
    print(f"Total AIR samples analyzed: {len(air_sample)}") # 假设 air_sample 是 AIR 的索引或标识符
    print(f"Number of HPG clusters (K) determined by Elbow Method: {optimal_k}")
    print(f"Distance threshold for top 20% closest AIR samples: {threshold_distance:.4f}")
    print(f"Number of AIR samples identified as 'close' (top 20%): {len(close_air_points_elbow)}")


def elbow_kmeans(data, max_k=10)->int:

    """
    计算不同 K 值下的 WCSS (Within-Cluster Sum of Squares)，用于肘部法则。
    """
    k_range = range(1, min(max_k + 1, len(data)))
    wcss_values = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init='auto')
        kmeans.fit(data)
        wcss_values.append(kmeans.inertia_)


    # -----------------------------
    # 步骤 2: 自动选择“肘部”点作为最优 K 值
    # -----------------------------

    # 一种简单的方法：寻找二阶导数最大的点
    if len(wcss_values) > 2:
        # 计算一阶差分（斜率变化）
        slopes = np.diff(wcss_values)
        # 计算二阶差分
        second_diff = np.diff(slopes)
        # 肘部点通常是二阶差分最大的点
        optimal_k_index = np.argmax(second_diff) + 1  # +1 because diff reduces length by 1
        optimal_k = k_range[optimal_k_index]
    else:
        optimal_k = 2 # 默认值

    return optimal_k