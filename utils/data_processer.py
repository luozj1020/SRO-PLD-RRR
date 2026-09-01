import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_samples, pairwise_distances
from scipy.spatial.distance import mahalanobis
import joblib
import logging
import warnings
from typing import Union, Dict, Tuple, Optional, List
import os

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# ================== 定义边界 ==================
BOUND_2 = {
    'Oxygen pressure': [-3, -1],
    'Laser energy density': [1, 3],
    'Temperature': [650, 750],
    'Frequency': [2, 6],
    'Thickness': [10, 100]
}
BOUND_1 = {
    'Oxygen pressure': [-4, 0],
    'Laser energy density': [0.5, 5],
    'Temperature': [500, 900],
    'Frequency': [0, 10],
    'Thickness': [0, 200]
}


def calculate_boundary_penalty_weights(X_raw: np.ndarray, feature_names: List[str],
                                       bound_1: Dict = BOUND_1, bound_2: Dict = BOUND_2,
                                       penalty_sharpness: float = 2.0,
                                       use_mahalanobis: bool = True,
                                       min_penalty: float = 0.0,
                                       max_penalty: float = 1.0,
                                       gaussian_sigma: float = 0.5,
                                       missing_penalty_rate: float = 0.95,
                                       fixed_covariance_inv: Optional[np.ndarray] = None,
                                       weight_threshold: float = 0.1) -> np.ndarray:
    """
    计算边界惩罚权重，支持使用固定的协方差矩阵以避免数据泄露
    参数中增加了 fixed_covariance_inv 和 weight_threshold
    """
    # 标准化特征名称以确保匹配
    feature_names_standardized = [name.strip().lower() for name in feature_names]
    bound_1_standardized = {k.strip().lower(): v for k, v in bound_1.items()}
    bound_2_standardized = {k.strip().lower(): v for k, v in bound_2.items()}

    # 初始化惩罚权重
    penalty_weights = np.ones(X_raw.shape[0])

    # 预处理数据：对第一个特征(Oxygen pressure)取对数
    X_processed = X_raw.copy()
    # 获取相关特征的索引
    relevant_features = []
    relevant_indices = []
    for j, feature in enumerate(feature_names_standardized):
        if feature in bound_1_standardized and feature in bound_2_standardized:
            relevant_features.append(feature)
            relevant_indices.append(j)
            # 特殊处理第一个分量：使用对数计算
            if j == 0:  # Oxygen pressure
                # 确保值大于0才能计算对数
                valid_mask = X_processed[:, j] > 0
                X_processed[valid_mask, j] = np.log10(X_processed[valid_mask, j])
                # 对于无效值，设置一个极小的正数的对数
                X_processed[~valid_mask, j] = np.log10(1e-10)

    if len(relevant_indices) == 0:
        # 如果没有相关特征，直接返回初始权重并应用阈值
        penalty_weights = np.where(penalty_weights < weight_threshold, 0.0, penalty_weights)
        return penalty_weights

    # 【修复】使用固定的协方差矩阵或计算新的协方差矩阵
    covariance_inv = None
    if use_mahalanobis and len(relevant_indices) > 1:
        if fixed_covariance_inv is not None:
            # 使用传入的固定协方差矩阵
            covariance_inv = fixed_covariance_inv
            logger.debug("使用传入的固定协方差矩阵计算马氏距离")
        else:
            # 原始计算逻辑（用于首次计算）
            relevant_data = X_processed[:, relevant_indices]
            valid_rows = ~np.any(np.isnan(relevant_data), axis=1)
            if np.sum(valid_rows) > len(relevant_indices):
                valid_data = relevant_data[valid_rows]
                try:
                    covariance_matrix = np.cov(valid_data.T)
                    # 增加更稳健的矩阵条件检查
                    cond_number = np.linalg.cond(covariance_matrix)
                    if cond_number > 1e10 or np.linalg.det(covariance_matrix) <= 1e-12:
                        # 使用更稳健的正则化方法
                        regularization = np.eye(len(relevant_indices)) * 1e-4
                        covariance_matrix += regularization
                    covariance_inv = np.linalg.inv(covariance_matrix)
                    logger.debug("基于当前数据计算协方差矩阵")
                except (np.linalg.LinAlgError, ValueError):
                    logger.warning("马氏距离计算失败，使用欧几里得距离代替")
                    use_mahalanobis = False
                    covariance_inv = None
            else:
                logger.warning("有效数据行数不足以计算协方差矩阵，使用欧几里得距离")
                use_mahalanobis = False
                covariance_inv = None

    # 对每个样本计算权重
    for i in range(X_processed.shape[0]):
        # 检查缺失特征情况
        sample_features = X_processed[i, relevant_indices]
        non_missing_mask = ~np.isnan(sample_features)
        if np.sum(non_missing_mask) == 0:
            # 如果没有非缺失特征，设置中等权重
            penalty_weights[i] = 0.5
            continue

        # 提取非缺失的特征和对应的边界
        valid_features = []
        valid_indices_local = []
        valid_feature_values = []
        for j, feature_idx in enumerate(relevant_indices):
            if non_missing_mask[j]:
                feature = relevant_features[j]
                valid_features.append(feature)
                valid_indices_local.append(j)
                valid_feature_values.append(sample_features[j])

        if len(valid_features) == 0:
            penalty_weights[i] = 0.5
            continue

        # 检查边界条件并计算距离
        in_small_boundary = True
        outside_large_boundary = False
        normalized_distances = []
        for j, feature in enumerate(valid_features):
            value = valid_feature_values[j]
            low_1, high_1 = bound_1_standardized[feature]  # 大边界
            low_2, high_2 = bound_2_standardized[feature]  # 小边界

            # 检查是否在小边界内
            if not (low_2 <= value <= high_2):
                in_small_boundary = False
            # 检查是否在大边界外
            if value < low_1 or value > high_1:
                outside_large_boundary = True

            # 计算归一化距离（从小边界到大边界）
            if low_2 <= value <= high_2:
                # 在小边界内，距离为0
                normalized_dist = 0.0
            elif low_1 <= value <= high_1:
                # 在过渡区域内，计算归一化距离
                if value < low_2:
                    # 在小边界左侧
                    transition_span = low_2 - low_1
                    dist_from_small = low_2 - value
                    normalized_dist = dist_from_small / transition_span if transition_span > 0 else 1.0
                else:
                    # 在小边界右侧
                    transition_span = high_1 - high_2
                    dist_from_small = value - high_2
                    normalized_dist = dist_from_small / transition_span if transition_span > 0 else 1.0
            else:
                # 在大边界外，距离为1（最远）
                normalized_dist = 1.0

            normalized_distances.append(min(1.0, max(0.0, normalized_dist)))

        # 根据边界条件设置权重
        if outside_large_boundary:
            # 在大边界外，权重为0
            penalty_weights[i] = min_penalty
            continue
        if in_small_boundary:
            # 在小边界内，权重为1
            penalty_weights[i] = max_penalty
            continue

        # 在过渡区域，计算基于距离的权重
        if len(normalized_distances) == 0:
            penalty_weights[i] = 0.5
            continue

        # 计算综合距离
        if use_mahalanobis and covariance_inv is not None and len(valid_features) > 1:
            # 使用马氏距离
            try:
                # 选择固定协方差矩阵的对应子块
                if fixed_covariance_inv is not None:
                    # 使用固定的协方差矩阵子块
                    valid_covariance_inv = covariance_inv[np.ix_(valid_indices_local, valid_indices_local)]
                else:
                    # 使用当前计算的协方差矩阵子块
                    valid_covariance_inv = covariance_inv[np.ix_(valid_indices_local, valid_indices_local)]

                distance_vector = np.array(normalized_distances)
                mahal_dist = np.sqrt(np.dot(np.dot(distance_vector, valid_covariance_inv), distance_vector))
                # 改进马氏距离归一化方式
                # 使用更合理的归一化因子
                max_possible_dist = np.sqrt(len(valid_features))  # 理论最大距离
                combined_distance = min(1.0, mahal_dist / max_possible_dist)
            except:
                # 如果马氏距离计算失败，使用欧几里得距离
                combined_distance = np.sqrt(np.mean([d ** 2 for d in normalized_distances]))
        else:
            # 使用欧几里得距离
            combined_distance = np.sqrt(np.mean([d ** 2 for d in normalized_distances]))

        # 使用高斯分布函数计算权重
        # 高斯函数: exp(-x^2 / (2*sigma^2))
        # combined_distance = 0 时权重 = 1，combined_distance = 1 时权重接近 0
        gaussian_weight = np.exp(-combined_distance ** 2 / (2 * gaussian_sigma ** 2))

        # 应用锐化参数
        if penalty_sharpness != 1.0:
            gaussian_weight = np.power(gaussian_weight, penalty_sharpness)

        # 确保权重在指定范围内
        weight = np.clip(gaussian_weight, min_penalty, max_penalty)

        # 改进缺失特征的惩罚策略
        missing_count = len(relevant_indices) - len(valid_features)
        if missing_count > 0:
            # 使用更温和的缺失惩罚：每缺失一个特征，权重乘以missing_penalty_rate
            missing_penalty = missing_penalty_rate ** missing_count
            weight *= missing_penalty
            # 为缺失特征过多的样本设置最小权重阈值
            # 如果缺失特征超过一半，确保权重不会太小
            missing_ratio = missing_count / len(relevant_indices)
            if missing_ratio > 0.5:
                min_weight_for_missing = 0.1  # 设置一个合理的最小权重
                weight = max(weight, min_weight_for_missing)

        penalty_weights[i] = weight

    # 【修复】确保在所有代码路径后应用权重阈值
    penalty_weights = np.where(penalty_weights < weight_threshold, 0.0, penalty_weights)

    return penalty_weights

def compute_fixed_covariance_matrix(X_raw: np.ndarray, feature_names: List[str],
                                    bound_1: Dict = BOUND_1, bound_2: Dict = BOUND_2) -> Optional[np.ndarray]:
    """
    计算并返回固定的协方差矩阵，用于后续权重计算
    """
    # logger.info("开始计算固定的协方差矩阵...")
    # 标准化特征名称以确保匹配
    feature_names_standardized = [name.strip().lower() for name in feature_names]
    bound_1_standardized = {k.strip().lower(): v for k, v in bound_1.items()}
    bound_2_standardized = {k.strip().lower(): v for k, v in bound_2.items()}

    # 预处理数据：对第一个特征(Oxygen pressure)取对数
    X_processed = X_raw.copy()
    # 获取相关特征的索引
    relevant_indices = []
    for j, feature in enumerate(feature_names_standardized):
        if feature in bound_1_standardized and feature in bound_2_standardized:
            relevant_indices.append(j)
            # 特殊处理第一个分量：使用对数计算
            if j == 0:  # Oxygen pressure
                # 确保值大于0才能计算对数
                valid_mask = X_processed[:, j] > 0
                X_processed[valid_mask, j] = np.log10(X_processed[valid_mask, j])
                # 对于无效值，设置一个极小的正数的对数
                X_processed[~valid_mask, j] = np.log10(1e-10)

    if len(relevant_indices) == 0:
        logger.warning("没有找到相关特征用于计算协方差矩阵")
        return None

    relevant_data = X_processed[:, relevant_indices]
    valid_rows = ~np.any(np.isnan(relevant_data), axis=1)
    if np.sum(valid_rows) < len(relevant_indices):
        logger.warning("有效数据行数不足以计算协方差矩阵")
        return None

    valid_data = relevant_data[valid_rows]
    try:
        covariance_matrix = np.cov(valid_data.T)
        # 增加更稳健的矩阵条件检查
        cond_number = np.linalg.cond(covariance_matrix)
        if cond_number > 1e10 or np.linalg.det(covariance_matrix) <= 1e-12:
            # logger.info("协方差矩阵条件数过大或行列式过小，进行正则化")
            # 使用更稳健的正则化方法
            regularization = np.eye(len(relevant_indices)) * 1e-4
            covariance_matrix += regularization
        covariance_inv = np.linalg.inv(covariance_matrix)
        # logger.info(f"固定协方差矩阵计算成功，形状: {covariance_inv.shape}")
        return covariance_inv
    except (np.linalg.LinAlgError, ValueError) as e:
        logger.warning(f"计算固定协方差矩阵失败: {e}")
        return None

class ClusteringBasedWeightCalculator:
    """
    Calculate sample weights using multiple clustering strategies
    to identify and downweight noisy/outlier samples
    """

    def __init__(
            self,
            n_clusters: int = None,
            min_cluster_size: int = 5,
            noise_percentile: float = 10,
            methods: list = None,
            feature_names: list = None
    ):
        """
        Args:
            n_clusters: Number of clusters (auto if None)
            min_cluster_size: Minimum samples per cluster
            noise_percentile: Percentile for noise threshold (lower = more aggressive)
            methods: Clustering methods to use ['kmeans', 'dbscan', 'hierarchical', 'density', 'residual']
            feature_names: Names of features for analysis
        """
        self.n_clusters = n_clusters
        self.min_cluster_size = min_cluster_size
        self.noise_percentile = noise_percentile
        self.methods = methods or ['kmeans', 'density', 'residual']
        self.feature_names = feature_names

        self.scaler = StandardScaler()
        self.cluster_info = {}
        self.method_weights = {}

    def fit_calculate_weights(
            self,
            X: np.ndarray,
            y: np.ndarray,
            return_diagnostics: bool = False
    ) -> np.ndarray:
        """
        Calculate sample weights based on clustering analysis

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
            return_diagnostics: Whether to return detailed diagnostics

        Returns:
            weights: Sample weights (n_samples,)
            diagnostics: (optional) Dictionary with detailed information
        """
        n_samples = X.shape[0]

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # Auto-determine number of clusters if not specified
        if self.n_clusters is None:
            self.n_clusters = self._estimate_optimal_clusters(X_scaled, y)

        # Calculate weights using different methods
        all_weights = []

        if 'kmeans' in self.methods:
            weights = self._kmeans_based_weights(X_scaled, y)
            all_weights.append(weights)
            self.method_weights['kmeans'] = weights

        if 'dbscan' in self.methods:
            weights = self._dbscan_based_weights(X_scaled, y)
            all_weights.append(weights)
            self.method_weights['dbscan'] = weights

        if 'hierarchical' in self.methods:
            weights = self._hierarchical_based_weights(X_scaled, y)
            all_weights.append(weights)
            self.method_weights['hierarchical'] = weights

        if 'density' in self.methods:
            weights = self._local_density_weights(X_scaled)
            all_weights.append(weights)
            self.method_weights['density'] = weights

        if 'residual' in self.methods:
            weights = self._residual_cluster_weights(X_scaled, y)
            all_weights.append(weights)
            self.method_weights['residual'] = weights

        # Ensemble: combine weights from different methods
        combined_weights = self._combine_weights(all_weights)

        # Normalize weights
        final_weights = self._normalize_weights(combined_weights)

        if return_diagnostics:
            diagnostics = self._generate_diagnostics(X, y, final_weights)
            return final_weights, diagnostics

        return final_weights

    def _estimate_optimal_clusters(self, X: np.ndarray, y: np.ndarray) -> int:
        """Estimate optimal number of clusters using elbow method and silhouette"""
        n_samples = X.shape[0]
        max_k = min(10, n_samples // self.min_cluster_size)

        if max_k < 2:
            return 2

        inertias = []
        silhouettes = []

        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            inertias.append(kmeans.inertia_)

            # Skip silhouette if too few samples per cluster
            if n_samples / k >= self.min_cluster_size:
                silhouettes.append(np.mean(silhouette_samples(X, labels)))
            else:
                silhouettes.append(0)

        # Find elbow using second derivative
        if len(inertias) > 2:
            diffs = np.diff(inertias)
            second_diffs = np.diff(diffs)
            elbow_k = np.argmax(second_diffs) + 2
        else:
            elbow_k = 2

        # Choose k with best silhouette around elbow
        search_range = range(max(2, elbow_k - 1), min(max_k + 1, elbow_k + 2))
        best_k = max(search_range, key=lambda k: silhouettes[k - 2])

        return best_k

    def _kmeans_based_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Weight calculation based on K-means clustering
        Samples far from cluster centers and in noisy clusters get lower weights
        """
        n_samples = X.shape[0]

        # Fit K-means
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        centers = kmeans.cluster_centers_

        weights = np.ones(n_samples)

        for cluster_id in range(self.n_clusters):
            cluster_mask = labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_indices) < self.min_cluster_size:
                # Small clusters are suspicious - reduce weight
                weights[cluster_indices] *= 0.5
                continue

            # Calculate distances to cluster center
            cluster_X = X[cluster_mask]
            distances = np.linalg.norm(cluster_X - centers[cluster_id], axis=1)

            # Calculate target variance within cluster
            cluster_y = y[cluster_mask]
            y_std = np.std(cluster_y)
            y_residuals = np.abs(cluster_y - np.mean(cluster_y))

            # Combine spatial distance and target deviation
            # Samples far from center OR with unusual target values get lower weight
            distance_threshold = np.percentile(distances, 100 - self.noise_percentile)
            residual_threshold = np.percentile(y_residuals, 100 - self.noise_percentile)

            for i, idx in enumerate(cluster_indices):
                dist_weight = 1.0 if distances[i] <= distance_threshold else \
                    np.exp(-(distances[i] - distance_threshold) ** 2 / (2 * np.var(distances)))

                residual_weight = 1.0 if y_residuals[i] <= residual_threshold else \
                    np.exp(-(y_residuals[i] - residual_threshold) ** 2 / (2 * y_std ** 2))

                weights[idx] = dist_weight * residual_weight

        self.cluster_info['kmeans'] = {
            'labels': labels,
            'centers': centers,
            'n_clusters': self.n_clusters
        }

        return weights

    def _dbscan_based_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Weight calculation using DBSCAN
        Samples labeled as noise get very low weights
        """
        n_samples = X.shape[0]

        # Estimate eps using k-distance graph
        k = max(5, int(np.log(n_samples)))
        nbrs = NearestNeighbors(n_neighbors=k).fit(X)
        distances, _ = nbrs.kneighbors(X)
        eps = np.percentile(distances[:, -1], 90)

        # Fit DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=max(3, self.min_cluster_size // 2))
        labels = dbscan.fit_predict(X)

        weights = np.ones(n_samples)

        # Noise points (label = -1) get very low weight
        noise_mask = labels == -1
        weights[noise_mask] = 0.1

        # For cluster points, weight by cluster quality
        unique_labels = set(labels) - {-1}
        for cluster_id in unique_labels:
            cluster_mask = labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            # Calculate cluster cohesion (average distance to cluster members)
            cluster_X = X[cluster_mask]
            if len(cluster_X) > 1:
                pairwise_dist = pairwise_distances(cluster_X)
                avg_distance = np.mean(pairwise_dist, axis=1)

                # Samples far from cluster members get lower weight
                distance_threshold = np.percentile(avg_distance, 75)
                for i, idx in enumerate(cluster_indices):
                    if avg_distance[i] > distance_threshold:
                        weights[idx] *= 0.7

        self.cluster_info['dbscan'] = {
            'labels': labels,
            'eps': eps,
            'noise_count': np.sum(noise_mask)
        }

        return weights

    def _hierarchical_based_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Weight calculation using hierarchical clustering
        Uses cluster stability and compactness
        """
        n_samples = X.shape[0]

        # Fit hierarchical clustering
        hierarchical = AgglomerativeClustering(n_clusters=self.n_clusters)
        labels = hierarchical.fit_predict(X)

        weights = np.ones(n_samples)

        for cluster_id in range(self.n_clusters):
            cluster_mask = labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_indices) < self.min_cluster_size:
                weights[cluster_indices] *= 0.6
                continue

            cluster_X = X[cluster_mask]
            cluster_y = y[cluster_mask]

            # Calculate cluster compactness
            center = np.mean(cluster_X, axis=0)
            distances = np.linalg.norm(cluster_X - center, axis=1)

            # Calculate target homogeneity
            y_variance = np.var(cluster_y)

            # Samples in loose or heterogeneous clusters get lower weight
            distance_threshold = np.percentile(distances, 75)
            for i, idx in enumerate(cluster_indices):
                if distances[i] > distance_threshold:
                    weights[idx] *= 0.8

            # If cluster has high target variance, reduce all weights slightly
            if y_variance > np.var(y) * 1.5:
                weights[cluster_indices] *= 0.9

        self.cluster_info['hierarchical'] = {
            'labels': labels,
            'n_clusters': self.n_clusters
        }

        return weights

    def _local_density_weights(self, X: np.ndarray) -> np.ndarray:
        """
        Weight based on local density
        Samples in low-density regions get lower weights
        """
        n_samples = X.shape[0]
        k = min(10, max(3, n_samples // 10))

        # Calculate local density using k-nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
        distances, _ = nbrs.kneighbors(X)

        # Local density = 1 / average distance to k nearest neighbors
        avg_distances = np.mean(distances[:, 1:], axis=1)  # Exclude self
        densities = 1.0 / (avg_distances + 1e-6)

        # Normalize densities to [0, 1]
        densities_norm = (densities - np.min(densities)) / (np.max(densities) - np.min(densities) + 1e-6)

        # Weight = density^0.5 (less aggressive than linear)
        # Low density points get lower weights
        weights = np.power(densities_norm, 0.5)

        # Ensure minimum weight
        weights = np.maximum(weights, 0.3)

        return weights

    def _residual_cluster_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Cluster in the residual space (feature + target jointly)
        Identify samples that don't fit local patterns
        """
        n_samples = X.shape[0]

        # Create augmented feature space [X, y]
        y_scaled = (y - np.mean(y)) / (np.std(y) + 1e-6)
        X_augmented = np.column_stack([X, y_scaled.reshape(-1, 1)])

        # Cluster in augmented space
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_augmented)

        weights = np.ones(n_samples)

        for cluster_id in range(self.n_clusters):
            cluster_mask = labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_indices) < self.min_cluster_size:
                weights[cluster_indices] *= 0.5
                continue

            # For each sample, check consistency with cluster
            cluster_X_aug = X_augmented[cluster_mask]
            cluster_center = np.mean(cluster_X_aug, axis=0)

            # Calculate Mahalanobis distance in augmented space
            try:
                cov = np.cov(cluster_X_aug.T)
                if np.linalg.det(cov) > 1e-6:
                    inv_cov = np.linalg.inv(cov)
                    for i, idx in enumerate(cluster_indices):
                        dist = mahalanobis(X_augmented[idx], cluster_center, inv_cov)
                        # Convert to weight using exponential decay
                        weights[idx] = np.exp(-dist / 5.0)
                else:
                    # Fallback to Euclidean
                    distances = np.linalg.norm(cluster_X_aug - cluster_center, axis=1)
                    threshold = np.percentile(distances, 75)
                    for i, idx in enumerate(cluster_indices):
                        if distances[i] > threshold:
                            weights[idx] *= 0.7
            except np.linalg.LinAlgError:
                # If covariance is singular, use Euclidean distance
                distances = np.linalg.norm(cluster_X_aug - cluster_center, axis=1)
                threshold = np.percentile(distances, 75)
                for i, idx in enumerate(cluster_indices):
                    if distances[i] > threshold:
                        weights[idx] *= 0.7

        self.cluster_info['residual'] = {
            'labels': labels,
            'n_clusters': self.n_clusters
        }

        return weights

    def _combine_weights(self, all_weights: list) -> np.ndarray:
        """
        Combine weights from different methods
        Uses geometric mean for robustness
        """
        if len(all_weights) == 1:
            return all_weights[0]

        # Stack weights
        weight_matrix = np.column_stack(all_weights)

        # Geometric mean (more robust to extreme values than arithmetic mean)
        combined = np.exp(np.mean(np.log(weight_matrix + 1e-6), axis=1))

        return combined

    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """
        Normalize weights to have mean = 1 and apply minimum threshold
        """
        # Apply minimum weight threshold
        min_weight = 0.05
        weights = np.maximum(weights, min_weight)

        # Normalize to mean = 1
        weights = weights / np.mean(weights)

        # Clip extreme values
        weights = np.clip(weights, 0.05, 3.0)

        return weights

    def _generate_diagnostics(
            self,
            X: np.ndarray,
            y: np.ndarray,
            final_weights: np.ndarray
    ) -> dict:
        """Generate diagnostic information"""
        diagnostics = {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_clusters_used': self.n_clusters,
            'weight_stats': {
                'mean': np.mean(final_weights),
                'std': np.std(final_weights),
                'min': np.min(final_weights),
                'max': np.max(final_weights),
                'q25': np.percentile(final_weights, 25),
                'q50': np.percentile(final_weights, 50),
                'q75': np.percentile(final_weights, 75)
            },
            'low_weight_samples': {
                'count_below_0.3': np.sum(final_weights < 0.3),
                'count_below_0.5': np.sum(final_weights < 0.5),
                'percentage_below_0.5': 100 * np.sum(final_weights < 0.5) / len(final_weights)
            },
            'method_agreement': self._calculate_method_agreement(),
            'cluster_info': self.cluster_info
        }

        # Feature-specific diagnostics
        if self.feature_names is not None:
            feature_diagnostics = {}
            for i, feat_name in enumerate(self.feature_names):
                # Correlation between feature and weights
                feat_vals = X[:, i]
                corr = np.corrcoef(feat_vals, final_weights)[0, 1]
                feature_diagnostics[feat_name] = {
                    'weight_correlation': corr,
                    'mean_value_low_weight': np.mean(feat_vals[final_weights < 0.5]),
                    'mean_value_high_weight': np.mean(feat_vals[final_weights >= 0.5])
                }
            diagnostics['feature_diagnostics'] = feature_diagnostics

        return diagnostics

    def _calculate_method_agreement(self) -> dict:
        """Calculate agreement between different weighting methods"""
        if len(self.method_weights) < 2:
            return {'agreement': 'N/A - only one method used'}

        # Calculate correlation between methods
        method_names = list(self.method_weights.keys())
        n_methods = len(method_names)
        correlations = {}

        for i in range(n_methods):
            for j in range(i + 1, n_methods):
                name1, name2 = method_names[i], method_names[j]
                corr = np.corrcoef(
                    self.method_weights[name1],
                    self.method_weights[name2]
                )[0, 1]
                correlations[f'{name1}_vs_{name2}'] = corr

        return {
            'correlations': correlations,
            'mean_correlation': np.mean(list(correlations.values())),
            'interpretation': 'High correlation (>0.7) indicates methods agree on noise samples'
        }

    def visualize_clusters(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray):
        """
        Visualize clustering results and weight distribution
        (Requires matplotlib)
        """
        try:
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA
        except ImportError:
            print("Matplotlib or sklearn not available for visualization")
            return

        # Reduce to 2D for visualization
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(self.scaler.transform(X))
            explained_var = pca.explained_variance_ratio_.sum()
        else:
            X_2d = self.scaler.transform(X)
            explained_var = 1.0

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Clusters with weights as size
        ax = axes[0, 0]
        if 'kmeans' in self.cluster_info:
            labels = self.cluster_info['kmeans']['labels']
            scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1],
                                 c=labels, s=weights * 50,
                                 cmap='tab10', alpha=0.6)
            ax.set_title(f'K-means Clusters (size=weight)\nExplained var: {explained_var:.2%}')

        # Plot 2: Weight distribution
        ax = axes[0, 1]
        ax.hist(weights, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(weights), color='red', linestyle='--', label=f'Mean={np.mean(weights):.2f}')
        ax.axvline(np.median(weights), color='green', linestyle='--', label=f'Median={np.median(weights):.2f}')
        ax.set_xlabel('Weight')
        ax.set_ylabel('Frequency')
        ax.set_title('Weight Distribution')
        ax.legend()

        # Plot 3: Target vs predicted cluster membership
        ax = axes[1, 0]
        if 'kmeans' in self.cluster_info:
            labels = self.cluster_info['kmeans']['labels']
            scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1],
                                 c=y, s=weights * 50,
                                 cmap='viridis', alpha=0.6)
            plt.colorbar(scatter, ax=ax, label='Target value')
            ax.set_title('Target Values (size=weight)')

        # Plot 4: Low weight samples highlighted
        ax = axes[1, 1]
        low_weight_mask = weights < np.percentile(weights, 25)
        ax.scatter(X_2d[~low_weight_mask, 0], X_2d[~low_weight_mask, 1],
                   c='blue', alpha=0.3, s=30, label='Normal weight')
        ax.scatter(X_2d[low_weight_mask, 0], X_2d[low_weight_mask, 1],
                   c='red', alpha=0.7, s=50, label='Low weight', marker='x')
        ax.set_title(f'Low Weight Samples (bottom 25%)')
        ax.legend()

        plt.tight_layout()
        return fig

class EnhancedFeatureProcessor:
    """增强特征处理器 - 修复版本"""

    def __init__(self, base_features=None, substrate_column='Substrate',
                 scaler_type='robust', interpolation_method='knn', n_neighbors=3):

        self.scaler_type = scaler_type
        self.interpolation_method = interpolation_method
        self.n_neighbors = n_neighbors
        self.scaler = self._get_scaler()
        self.imputer = None
        self.feature_stats = {}
        self.is_fitted = False
        self.selected_features = base_features
        self.substrate_column = substrate_column

        # NEW: Track original NaN mask during fitting
        self.fitted_nan_mask = None
        self.fitted_imputed_data = None

    def _get_scaler(self):
        from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

        if self.scaler_type == 'standard':
            return StandardScaler()
        elif self.scaler_type == 'robust':
            return RobustScaler()
        elif self.scaler_type == 'minmax':
            return MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")

    def _select_numerical_features(self, df: pd.DataFrame) -> list:
        if self.selected_features:
            available_features = [f for f in self.selected_features if f in df.columns and f != self.substrate_column]
            return available_features
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if self.substrate_column in numeric_cols:
                numeric_cols.remove(self.substrate_column)
            return numeric_cols

    def _fast_impute(self, X: np.ndarray, is_training=True) -> np.ndarray:
        """快速填充缺失值"""
        if self.interpolation_method == 'linear':
            return self._linear_interpolation_vectorized(X)
        else:
            if is_training:
                return self.imputer.fit_transform(X)
            else:
                return self.imputer.transform(X)

    def _linear_interpolation_vectorized(self, X: np.ndarray) -> np.ndarray:
        """向量化线性插值"""
        X_result = X.copy()
        n_rows, n_cols = X.shape
        for col in range(n_cols):
            col_data = X[:, col]
            nan_mask = np.isnan(col_data)
            if not nan_mask.any():
                continue
            if nan_mask.all():
                logger.warning(f"All values in column {col} are NaN.")
                continue
            temp_df = pd.DataFrame({f'col_{col}': col_data})
            temp_df[f'col_{col}'] = temp_df[f'col_{col}'].interpolate(method='linear', limit_direction='both')
            X_result[:, col] = temp_df[f'col_{col}'].values
        return X_result

    def _record_stats(self, X: np.ndarray):
        """记录原始数据统计信息"""
        for i, feat in enumerate(self.selected_features):
            col_data = X[:, i]
            self.feature_stats[feat] = {
                'mean': np.nanmean(col_data),
                'std': np.nanstd(col_data),
                'min': np.nanmin(col_data),
                'max': np.nanmax(col_data),
                'median': np.nanmedian(col_data),
                'missing_rate': np.isnan(col_data).mean()
            }

    def fit(self, data) -> 'EnhancedFeatureProcessor':
        """Fit processor and store NaN mask"""
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            self.selected_features = self._select_numerical_features(data)
            X = data[self.selected_features].values
        else:
            X = data.copy()
            if self.selected_features is None:
                self.selected_features = [f'feature_{i}' for i in range(X.shape[1])]

        # Convert to float64
        X = X.astype(np.float64)

        # NEW: Store original NaN mask (1 = valid, 0 = NaN/imputed)
        self.fitted_nan_mask = ~np.isnan(X)

        # Setup imputer
        if self.interpolation_method == 'knn':
            from sklearn.impute import KNNImputer
            self.imputer = KNNImputer(n_neighbors=self.n_neighbors)
        elif self.interpolation_method == 'median':
            from sklearn.impute import SimpleImputer
            self.imputer = SimpleImputer(strategy='median')
        elif self.interpolation_method == 'mean':
            from sklearn.impute import SimpleImputer
            self.imputer = SimpleImputer(strategy='mean')

        # Impute and scale
        X_imputed = self.imputer.fit_transform(X)
        self.fitted_imputed_data = X_imputed.copy()
        self.scaler.fit(X_imputed)

        self.is_fitted = True
        return self

    def transform(self, data, return_mask: bool = False):
        if not self.is_fitted:
            raise ValueError("Processor must be fitted first")

        # 获取数值列
        if isinstance(data, pd.DataFrame):
            X = data[self.selected_features].values
        else:
            X = data.copy()

        # 确保数据是浮点类型
        X = X.astype(np.float64)

        # 计算NaN掩码（1=有效，0=NaN）
        try:
            current_nan_mask = (~np.isnan(X)).astype(np.float32)
        except Exception as e:
            logger.warning(f"NaN mask calculation failed: {e}")
            current_nan_mask = np.ones(X.shape, dtype=np.float32)

        # 填充和标准化
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)

        if return_mask:
            return X_scaled, current_nan_mask
        else:
            return X_scaled

    def fit_transform(self, data, return_mask: bool = False):
        """Fit and transform"""
        self.fit(data)
        return self.transform(data, return_mask=return_mask)

    def _select_numerical_features(self, df):
        """Select numerical features"""

        if self.selected_features:
            available = [f for f in self.selected_features
                         if f in df.columns and f != self.substrate_column]
            return available
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if self.substrate_column in numeric_cols:
                numeric_cols.remove(self.substrate_column)
            return numeric_cols

    def inverse_transform(self, X_scaled: np.ndarray, restore_nan: bool = True) -> np.ndarray:
        """逆转换"""
        if not self.is_fitted:
            raise ValueError("Processor must be fitted before inverse transform.")
        X_original = self.scaler.inverse_transform(X_scaled)
        if restore_nan and self.nan_mask is not None:
            if X_original.shape == self.nan_mask.shape:
                X_original[self.nan_mask] = np.nan
        return X_original

class DataAugmentationSuite:
    """数据增强套件"""
    def __init__(self, feature_processor: EnhancedFeatureProcessor = None):
        self.feature_processor = feature_processor or EnhancedFeatureProcessor(
            scaler_type='robust',
            interpolation_method='knn',
            n_neighbors=3
        )
        self.original_data = None
        self.feature_names = []
        self.target_name = None
        self.missing_patterns = None
        self.original_missing_rates = {}
        self.feature_ranges = {}

    def _ensure_value_in_range(self, value: float, feature: str) -> float:
        """确保值在合理范围内"""
        if feature == 'Substrate':
            return value
        if value <= 0:
            original_positive_values = self.original_data[self.original_data[feature] > 0][feature]
            original_positive_values = original_positive_values[original_positive_values > 0]
            if len(original_positive_values) > 0:
                min_positive = original_positive_values.min()
                value = max(min_positive * 0.1, 0.001)
            else:
                value = 0.001
        if feature in self.feature_ranges:
            min_val, max_val = self.feature_ranges[feature]
            value = max(min_val, min(max_val, value))
        return value

    def _preserve_missing_pattern(self, base_sample: pd.Series, new_sample: Dict) -> Dict:
        """保留缺失模式"""
        base_missing = base_sample.isna()
        for feature in self.feature_names:
            if feature in new_sample and feature != 'Substrate':
                if base_missing[feature] and np.random.random() < 0.7:
                    new_sample[feature] = np.nan
                elif not base_missing[feature] and np.random.random() < self.original_missing_rates[feature] * 0.5:
                    new_sample[feature] = np.nan
                elif not pd.isna(new_sample[feature]):
                    new_sample[feature] = self._ensure_value_in_range(new_sample[feature], feature)
        return new_sample

    def fit(self, data: pd.DataFrame, target_column: str):
        """拟合增强套件"""
        self.original_data = data.copy()
        self.target_name = target_column
        self.feature_names = [col for col in data.columns if col != target_column and col != 'Substrate']

        if not self.feature_processor.is_fitted:
            feature_data = data[self.feature_names]
            self.feature_processor.fit(feature_data)

        for feature in self.feature_names:
            self.original_missing_rates[feature] = data[feature].isna().mean()

        self.missing_patterns = data[self.feature_names].isna()
        for feature in self.feature_names:
            min_val = data[feature].min()
            max_val = data[feature].max()
            self.feature_ranges[feature] = (min_val, max_val)

        # logger.info(f"Augmentation suite fitted. Features: {self.feature_names}, Target: {self.target_name}")
        return self

    def gaussian_noise_augmentation(self, n_samples: int = 100, noise_factor: float = 0.02) -> pd.DataFrame:
        """高斯噪声增强"""
        if self.original_data is None:
            raise ValueError("Must call fit() first")

        samples = []
        for _ in range(n_samples):
            base_idx = np.random.randint(0, len(self.original_data))
            base_sample = self.original_data.iloc[base_idx].copy()
            new_sample = base_sample.to_dict()

            for feature in self.feature_names:
                if not pd.isna(new_sample[feature]):
                    base_val = new_sample[feature]
                    noise = np.random.normal(0, noise_factor * abs(base_val))
                    new_val = base_val + noise
                    new_sample[feature] = self._ensure_value_in_range(new_val, feature)
                else:
                    new_sample[feature] = np.nan

            if not pd.isna(new_sample[self.target_name]):
                base_target = new_sample[self.target_name]
                target_noise = np.random.normal(0, noise_factor * abs(base_target))
                new_sample[self.target_name] = self._ensure_value_in_range(base_target + target_noise, self.target_name)
            else:
                new_sample[self.target_name] = np.nan

            new_sample['Substrate'] = np.nan
            new_sample = self._preserve_missing_pattern(base_sample, new_sample)
            samples.append(new_sample)

        df = pd.DataFrame(samples)
        df['augmentation_strategy'] = 'gaussian_noise'
        return df

    def interpolation_augmentation(self, n_samples: int = 100,
                                   alpha_range: Tuple[float, float] = (0.2, 0.8)) -> pd.DataFrame:
        """插值增强"""
        if self.original_data is None:
            raise ValueError("Must call fit() first")

        samples = []
        for _ in range(n_samples):
            idx1, idx2 = np.random.choice(len(self.original_data), 2, replace=False)
            sample1 = self.original_data.iloc[idx1]
            sample2 = self.original_data.iloc[idx2]
            alpha = np.random.uniform(*alpha_range)
            new_sample = {}

            for feature in self.feature_names + [self.target_name]:
                val1, val2 = sample1[feature], sample2[feature]
                if pd.isna(val1) and pd.isna(val2):
                    new_sample[feature] = np.nan
                elif pd.isna(val1):
                    if np.random.random() < 0.6:
                        new_sample[feature] = np.nan
                    else:
                        new_sample[feature] = self._ensure_value_in_range(val2, feature)
                elif pd.isna(val2):
                    if np.random.random() < 0.6:
                        new_sample[feature] = np.nan
                    else:
                        new_sample[feature] = self._ensure_value_in_range(val1, feature)
                else:
                    interpolated_value = alpha * val1 + (1 - alpha) * val2
                    new_sample[feature] = self._ensure_value_in_range(interpolated_value, feature)

            base_sample = sample1 if np.random.random() < 0.5 else sample2
            new_sample = self._preserve_missing_pattern(base_sample, new_sample)
            new_sample['Substrate'] = np.nan
            samples.append(new_sample)

        df = pd.DataFrame(samples)
        df['augmentation_strategy'] = 'interpolation'
        return df

    def knn_augmentation(self, n_samples: int = 100, k: int = 3, perturbation: float = 0.03) -> pd.DataFrame:
        """KNN增强"""
        if self.original_data is None:
            raise ValueError("Must call fit() first")

        X = self.original_data[self.feature_names]
        X_for_knn = X.fillna(X.median())
        X_processed = self.feature_processor.transform(X_for_knn)

        knn = NearestNeighbors(n_neighbors=min(k + 1, len(X_processed)), metric='euclidean')
        knn.fit(X_processed)

        samples = []
        for _ in range(n_samples):
            base_idx = np.random.randint(0, len(X_processed))
            base_point = X_processed[base_idx:base_idx + 1]
            distances, indices = knn.kneighbors(base_point)
            neighbor_indices = indices[0][1:k + 1] if len(indices[0]) > 1 else indices[0]

            if len(neighbor_indices) == 0:
                neighbor_idx = base_idx
            else:
                neighbor_idx = np.random.choice(neighbor_indices)

            base_sample = self.original_data.iloc[base_idx]
            neighbor_sample = self.original_data.iloc[neighbor_idx]

            new_features = {}
            for feature in self.feature_names + [self.target_name]:
                base_val = base_sample[feature]
                neighbor_val = neighbor_sample[feature]
                if pd.isna(base_val) and pd.isna(neighbor_val):
                    new_features[feature] = np.nan
                elif pd.isna(base_val):
                    if np.random.random() < 0.6:
                        new_features[feature] = np.nan
                    else:
                        new_features[feature] = self._ensure_value_in_range(neighbor_val, feature)
                elif pd.isna(neighbor_val):
                    if np.random.random() < 0.6:
                        new_features[feature] = np.nan
                    else:
                        new_features[feature] = self._ensure_value_in_range(base_val, feature)
                else:
                    blend_factor = np.random.uniform(0.4, 0.6)
                    blended = blend_factor * base_val + (1 - blend_factor) * neighbor_val
                    perturbed = blended + np.random.normal(0, perturbation * abs(blended))
                    new_features[feature] = self._ensure_value_in_range(perturbed, feature)

            new_features['Substrate'] = np.nan
            new_features = self._preserve_missing_pattern(base_sample, new_features)
            samples.append(new_features)

        df = pd.DataFrame(samples)
        df['augmentation_strategy'] = 'knn'
        return df

    def smote_like_augmentation(self, n_samples: int = 100, k: int = 3) -> pd.DataFrame:
        """SMOTE风格增强"""
        if self.original_data is None:
            raise ValueError("Must call fit() first")

        X = self.original_data[self.feature_names]
        X_for_knn = X.fillna(X.median())
        X_processed = self.feature_processor.transform(X_for_knn)

        knn = NearestNeighbors(n_neighbors=min(k + 1, len(X_processed)), metric='euclidean')
        knn.fit(X_processed)

        samples = []
        for _ in range(n_samples):
            base_idx = np.random.randint(0, len(X_processed))
            base_point = X_processed[base_idx:base_idx + 1]
            distances, indices = knn.kneighbors(base_point)
            neighbor_indices = indices[0][1:k + 1] if len(indices[0]) > 1 else indices[0]

            if len(neighbor_indices) == 0:
                neighbor_idx = base_idx
            else:
                neighbor_idx = np.random.choice(neighbor_indices)

            base_sample = self.original_data.iloc[base_idx]
            neighbor_sample = self.original_data.iloc[neighbor_idx]

            new_sample = {}
            for feature in self.feature_names:
                base_val = base_sample[feature]
                neighbor_val = neighbor_sample[feature]
                if pd.isna(base_val):
                    if np.random.random() < 0.6:
                        new_sample[feature] = np.nan
                    else:
                        new_sample[feature] = self._ensure_value_in_range(neighbor_val, feature)
                elif pd.isna(neighbor_val):
                    if np.random.random() < 0.6:
                        new_sample[feature] = np.nan
                    else:
                        new_sample[feature] = self._ensure_value_in_range(base_val, feature)
                else:
                    lambda_val = np.random.uniform(0, 1)
                    interpolated = base_val + lambda_val * (neighbor_val - base_val)
                    new_sample[feature] = self._ensure_value_in_range(interpolated, feature)

            base_target = base_sample[self.target_name]
            neighbor_target = neighbor_sample[self.target_name]
            if pd.isna(base_target):
                if np.random.random() < 0.6:
                    new_sample[self.target_name] = np.nan
                else:
                    new_sample[self.target_name] = self._ensure_value_in_range(neighbor_target, self.target_name)
            elif pd.isna(neighbor_target):
                if np.random.random() < 0.6:
                    new_sample[self.target_name] = np.nan
                else:
                    new_sample[self.target_name] = self._ensure_value_in_range(base_target, self.target_name)
            else:
                lambda_val_target = np.random.uniform(0, 1)
                interpolated_target = base_target + lambda_val_target * (neighbor_target - base_target)
                new_sample[self.target_name] = self._ensure_value_in_range(interpolated_target, self.target_name)

            new_sample['Substrate'] = np.nan
            new_sample = self._preserve_missing_pattern(base_sample, new_sample)
            samples.append(new_sample)

        df = pd.DataFrame(samples)
        df['augmentation_strategy'] = 'smote_like'
        return df

    def ensemble_augmentation(self, n_samples: int = 100, strategy_weights: Dict[str, float] = None) -> pd.DataFrame:
        """集成增强"""
        if strategy_weights is None:
            strategy_weights = {
                'gaussian_noise': 0.35,
                'interpolation': 0.30,
                'knn': 0.20,
                'smote_like': 0.15,
            }

        # 处理小样本情况：当n_samples小于策略数量时，只使用权重最大的策略
        if n_samples < len(strategy_weights):
            # 选择权重最大的策略
            main_strategy = max(strategy_weights.items(), key=lambda x: x[1])[0]
            # logger.info(f"样本数较少({n_samples})，仅使用{main_strategy}策略")
            if main_strategy == 'gaussian_noise':
                return self.gaussian_noise_augmentation(n_samples)
            elif main_strategy == 'interpolation':
                return self.interpolation_augmentation(n_samples)
            elif main_strategy == 'knn':
                return self.knn_augmentation(n_samples)
            elif main_strategy == 'smote_like':
                return self.smote_like_augmentation(n_samples)

        total_weight = sum(strategy_weights.values())
        normalized_weights = {k: v / total_weight for k, v in strategy_weights.items()}

        samples_needed = n_samples
        all_augmented_samples = []

        # 确保每种策略至少分配1个样本（如果权重不为0）
        strategy_samples = {}
        remaining_samples = samples_needed

        # 第一轮：为每种策略分配至少1个样本（如果权重不为0）
        for strategy, weight in normalized_weights.items():
            if weight > 0 and remaining_samples > 0:
                strategy_samples[strategy] = 1
                remaining_samples -= 1

        # 第二轮：按权重分配剩余样本
        if remaining_samples > 0:
            for strategy, weight in normalized_weights.items():
                if weight > 0:
                    additional_samples = int(weight * remaining_samples)
                    if additional_samples > 0:
                        strategy_samples[strategy] = strategy_samples.get(strategy, 0) + additional_samples
                        remaining_samples -= additional_samples

        # 处理剩余样本（由于取整可能还有剩余）
        if remaining_samples > 0:
            # 分配给权重最大的策略
            main_strategy = max(normalized_weights.items(), key=lambda x: x[1])[0]
            strategy_samples[main_strategy] = strategy_samples.get(main_strategy, 0) + remaining_samples

        # 生成样本
        for strategy, n_strat in strategy_samples.items():
            if n_strat > 0:
                # logger.info(f"Generating {n_strat} samples using {strategy}")
                try:
                    if strategy == 'gaussian_noise':
                        df_strat = self.gaussian_noise_augmentation(n_strat)
                    elif strategy == 'interpolation':
                        df_strat = self.interpolation_augmentation(n_strat)
                    elif strategy == 'knn':
                        df_strat = self.knn_augmentation(n_strat)
                    elif strategy == 'smote_like':
                        df_strat = self.smote_like_augmentation(n_strat)
                    else:
                        logger.warning(f"Unknown strategy: {strategy}")
                        continue

                    if len(df_strat) > 0:
                        all_augmented_samples.append(df_strat)
                    else:
                        logger.warning(f"Strategy {strategy} generated 0 samples")
                except Exception as e:
                    logger.error(f"Error in {strategy} augmentation: {e}")
                    continue

        if not all_augmented_samples:
            logger.warning("No samples generated by any strategy, using fallback gaussian noise")
            # 备用方案：使用高斯噪声
            fallback_samples = self.gaussian_noise_augmentation(n_samples)
            if len(fallback_samples) > 0:
                all_augmented_samples.append(fallback_samples)
            else:
                logger.error("Fallback strategy also failed")
                return pd.DataFrame()

        combined_df = pd.concat(all_augmented_samples, ignore_index=True)

        # 如果生成的样本多于需求，随机选择所需数量
        if len(combined_df) > n_samples:
            combined_df = combined_df.sample(n=n_samples, random_state=42).reset_index(drop=True)
        else:
            combined_df = combined_df.sample(frac=1).reset_index(drop=True)

        return combined_df

def validate_no_leakage(train_raw: pd.DataFrame, val_raw: pd.DataFrame,
                       current_train_data: pd.DataFrame, processed_val_df: pd.DataFrame) -> bool:
    """验证没有数据泄露"""
    checks = []

    # 检查1: 验证集样本不应出现在训练集中 (通过原始索引)
    # 注意：增强后的数据索引会改变，所以主要检查原始训练/验证集的划分逻辑
    # 这里主要是逻辑检查，确保增强只发生在训练集
    original_train_indices = set(train_raw.index)
    original_val_indices = set(val_raw.index)
    overlap = original_train_indices.intersection(original_val_indices)
    checks.append(("原始训练验证集索引不重叠", len(overlap) == 0))

    # 检查2: 验证集大小应保持不变
    checks.append(("验证集大小不变", len(val_raw) == len(processed_val_df)))

    # 检查3: 增强数据应该只在训练集
    original_train_size = len(train_raw)
    augmented_train_size = len(current_train_data)
    checks.append(("仅训练集增强", augmented_train_size >= original_train_size))

    # 检查4: 验证集应该没有"augmentation_strategy"标记
    has_aug_col = 'augmentation_strategy' in processed_val_df.columns
    checks.append(("验证集无增强标记", not has_aug_col))

    # 检查5: 验证集的增强策略标记（如果存在）
    has_aug_col_raw = 'augmentation_strategy' in val_raw.columns
    checks.append(("原始验证集无增强标记", not has_aug_col_raw))

    # 输出检查结果
    # logger.info("=" * 60)
    # logger.info("数据泄露验证检查:")
    for check_name, passed in checks:
        status = "✓ 通过" if passed else "✗ 失败"
        # logger.info(f"  {status} - {check_name}")
    # logger.info("=" * 60)

    return all(check[1] for check in checks)

def main():
    """
    修复版主函数 - 解决数据泄露问题和未定义变量错误
    """
    # --- 0. 创建输出文件夹 ---
    output_dir = 'augmented_output_fixed'
    os.makedirs(output_dir, exist_ok=True)
    # logger.info(f"创建输出文件夹: {output_dir}")

    # --- 1. 读取原始数据 ---
    input_file = 'converted_file.xlsx'
    try:
        # logger.info(f"正在读取文件: {input_file}")
        df = pd.read_excel(input_file)
        original_count = len(df)
        # logger.info(f"原始数据读取成功，共有 {original_count} 行。")
    except Exception as e:
        logger.error(f"读取文件错误: {e}")
        return None, None, None, None, None, None  # 修改返回值数量

    # --- 2. 数据预处理和选择 ---
    numeric_columns = ['Oxygen pressure', 'Laser energy density', 'Temperature', 'Frequency', 'Thickness']
    target_column = 'rrr'
    required_columns = numeric_columns + [target_column]

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        logger.error(f"原始数据中缺少必要列: {missing_cols}")
        return None, None, None, None, None, None  # 修改返回值数量

    selected_df = df[required_columns].copy()

    if 'Frequency' in selected_df.columns:
        selected_df['Frequency'] = selected_df['Frequency'].apply(lambda x: round(x) if not pd.isna(x) else x)
        # logger.info("已确保Frequency列的值为整数")

    # logger.info(f"使用以下列进行数据增强: {numeric_columns}")
    # logger.info("原始数据缺失情况:")
    for col in numeric_columns:
        missing_count = selected_df[col].isna().sum()
        missing_rate = missing_count / len(selected_df)
        # logger.info(f" {col}: {missing_count}/{len(selected_df)} ({missing_rate:.1%})")

    # --- 3. 数据分割 (原始数据分割) ---
    # logger.info("开始分割原始数据为训练集和验证集...")
    test_size = 0.3
    random_state = 42
    train_raw, val_raw = train_test_split(
        selected_df,
        test_size=test_size,
        random_state=random_state
    )
    # logger.info(f"原始训练集大小: {len(train_raw)}")
    # logger.info(f"原始验证集大小: {len(val_raw)}")

    # --- 4. 【关键修复】仅在原始训练集上拟合 Processor ---
    # logger.info("【修复】仅在原始训练集上拟合 processor，避免数据泄露...")
    processor = EnhancedFeatureProcessor(
        base_features=numeric_columns,
        scaler_type='robust',
        interpolation_method='knn',
        n_neighbors=3
    )
    X_train_features_for_fit = train_raw[numeric_columns]
    processor.fit(X_train_features_for_fit)
    # logger.info("✓ Processor 拟合完成（仅基于原始训练集，无数据泄露）")

    # --- 5. 【关键修复】预先计算训练集的固定协方差矩阵 ---
    # logger.info("【修复】计算原始训练集的固定协方差矩阵...")
    X_train_raw_initial = train_raw[numeric_columns].values
    train_covariance_inv = compute_fixed_covariance_matrix(
        X_train_raw_initial,
        numeric_columns
    )
    if train_covariance_inv is not None:
        # logger.info("✓ 已基于原始训练集计算固定协方差矩阵")
        pass
    else:
        logger.warning("⚠️ 计算固定协方差矩阵失败，将使用欧几里得距离")

    # --- 6. 训练集数据增强（修复版本）---
    # logger.info("开始对训练集进行数据增强...")
    target_train_size = 1050
    max_iterations = 20
    iteration = 0
    weight_threshold = 0.3  # 新增：定义权重阈值

    # 【修复】仅在原始训练集上初始化一次增强器，避免多次拟合
    feature_processor_for_aug = processor
    augmenter = DataAugmentationSuite(feature_processor=feature_processor_for_aug)
    augmenter.fit(train_raw, target_column)  # 只使用原始训练集拟合

    current_train_data = train_raw.copy()
    feature_names_for_weights = numeric_columns

    while len(current_train_data) < target_train_size and iteration < max_iterations:
        iteration += 1
        # logger.info(f"--- 训练集增强迭代 {iteration} ---")
        # logger.info(f"当前训练集大小: {len(current_train_data)}")

        # 计算当前训练数据的权重（使用原始训练集的固定协方差矩阵）
        X_raw = current_train_data[numeric_columns].values
        weights = calculate_boundary_penalty_weights(
            X_raw,
            feature_names_for_weights,
            gaussian_sigma=0.8,
            missing_penalty_rate=0.95,
            penalty_sharpness=1.5,
            fixed_covariance_inv=train_covariance_inv,
            weight_threshold=weight_threshold  # 新增：传入权重阈值
        )
        current_train_data['weight'] = weights

        # 移除权重低于阈值的样本（不仅仅是权重为0的样本）
        low_weight_mask = current_train_data['weight'] < weight_threshold  # 修改：使用阈值而不是0.1
        low_weight_count = low_weight_mask.sum()
        # logger.info(f"当前训练集权重低于{weight_threshold}的样本数: {low_weight_count}")

        if low_weight_count == 0 and len(current_train_data) >= target_train_size:
            # logger.info(f"训练集已达到目标: 没有权重低于{weight_threshold}的样本且总样本数≥1050")
            break

        if low_weight_count > 0:
            current_train_data = current_train_data[~low_weight_mask].copy()
            # logger.info(f"移除权重低于{weight_threshold}的样本后训练集大小: {len(current_train_data)}")

        # 如果数据量仍不足，进行增强
        if len(current_train_data) < target_train_size:
            samples_needed = target_train_size - len(current_train_data)
            # logger.info(f"训练集需要增强 {samples_needed} 个样本")

            # 【修复】使用预先拟合的增强器，避免重新拟合
            ensemble_augmented = augmenter.ensemble_augmentation(
                n_samples=samples_needed,
                strategy_weights={
                    'gaussian_noise': 0.35,
                    'interpolation': 0.30,
                    'knn': 0.20,
                    'smote_like': 0.15,
                }
            )

            # 合并数据
            current_train_data = pd.concat(
                [current_train_data, ensemble_augmented.drop(columns=['augmentation_strategy'], errors='ignore')],
                ignore_index=True)
            # logger.info(f"增强后训练集大小: {len(current_train_data)}")
        else:
            # logger.info("训练集数据量已达到或超过目标，无需增强")
            pass

    # --- 7. 最终训练集权重计算和清理 ---
    X_raw_train = current_train_data[numeric_columns].values
    final_train_weights = calculate_boundary_penalty_weights(
        X_raw_train,
        feature_names_for_weights,
        gaussian_sigma=0.8,
        missing_penalty_rate=0.95,
        penalty_sharpness=1.5,
        fixed_covariance_inv=train_covariance_inv,
        weight_threshold=weight_threshold  # 新增：传入权重阈值
    )
    current_train_data['weight'] = final_train_weights

    # 最终移除权重低于阈值的样本
    final_low_weight_mask = current_train_data['weight'] < weight_threshold  # 修改：使用阈值
    final_low_weight_count = final_low_weight_mask.sum()
    if final_low_weight_count > 0:
        # logger.info(f"最终移除训练集中 {final_low_weight_count} 个权重低于{weight_threshold}的样本")
        current_train_data = current_train_data[~final_low_weight_mask].copy()

    # logger.info(f"最终增强后训练集大小: {len(current_train_data)}")

    # --- 8. 转换增强后的训练集和原始验证集 ---
    # logger.info("使用原始训练集拟合的 processor 转换增强后的训练集和原始验证集...")

    # 转换增强后的训练集
    X_train_augmented_features = current_train_data[numeric_columns]
    processed_train_features = processor.transform(X_train_augmented_features)
    processed_train_df = pd.DataFrame(
        processed_train_features,
        columns=[f"{feat}" for feat in processor.selected_features]
    )

    # 【修复】为训练集计算最终权重（使用固定的协方差矩阵）
    X_raw_train_final = current_train_data[numeric_columns].values
    final_train_weights = calculate_boundary_penalty_weights(
        X_raw_train_final,
        feature_names_for_weights,
        gaussian_sigma=0.8,
        missing_penalty_rate=0.95,
        penalty_sharpness=1.5,
        fixed_covariance_inv=train_covariance_inv # 使用固定的协方差矩阵
    )
    processed_train_df['weight'] = final_train_weights

    # 添加非特征列
    if 'Substrate' in current_train_data.columns:
        processed_train_df['Substrate'] = current_train_data['Substrate'].values
    if 'rrr' in current_train_data.columns:
        processed_train_df['rrr'] = current_train_data['rrr'].values

    # 转换原始验证集
    X_val_features = val_raw[numeric_columns]
    processed_val_features = processor.transform(X_val_features)
    processed_val_df = pd.DataFrame(
        processed_val_features,
        columns=[f"{feat}" for feat in processor.selected_features]
    )

    # 【关键修复】为验证集也计算边界惩罚权重（使用训练集的固定协方差矩阵）
    X_val_raw = val_raw[numeric_columns].values
    val_weights = calculate_boundary_penalty_weights(
        X_val_raw,
        feature_names_for_weights,
        gaussian_sigma=0.8,
        missing_penalty_rate=0.95,
        penalty_sharpness=1.5,
        fixed_covariance_inv=train_covariance_inv # 使用训练集的固定协方差矩阵
    )
    processed_val_df['weight'] = val_weights

    # 添加非特征列
    if 'Substrate' in val_raw.columns:
        processed_val_df['Substrate'] = val_raw['Substrate'].values
    if 'rrr' in val_raw.columns:
        processed_val_df['rrr'] = val_raw['rrr'].values

    # --- 9. 保存处理后的数据集 ---
    # logger.info("保存处理后的训练集和验证集...")
    output_file_processed_train = os.path.join(output_dir, 'augmented_dataset_processed_train.xlsx')
    output_file_processed_val = os.path.join(output_dir, 'augmented_dataset_processed_val.xlsx')
    output_file_raw_train = os.path.join(output_dir, 'augmented_dataset_raw_train.xlsx')
    output_file_raw_val = os.path.join(output_dir, 'augmented_dataset_raw_val.xlsx')

    processed_train_df.to_excel(output_file_processed_train, index=False)
    processed_val_df.to_excel(output_file_processed_val, index=False)
    current_train_data.to_excel(output_file_raw_train, index=False)  # 保存增强后的原始训练集
    val_raw.to_excel(output_file_raw_val, index=False)  # 保存原始验证集

    # logger.info(f"已保存处理后训练集: {output_file_processed_train}")
    # logger.info(f"已保存处理后验证集: {output_file_processed_val}")
    # logger.info(f"已保存原始增强训练集: {output_file_raw_train}")
    # logger.info(f"已保存原始验证集: {output_file_raw_val}")

    # --- 9.5 逆变换检查：保存restored数据文件 ---
    # logger.info("进行逆变换检查，保存restored数据文件...")
    # 对处理后的训练集进行逆变换
    try:
        # 只对特征列进行逆变换
        processed_train_features_only = processed_train_df[numeric_columns].values
        restored_train_features = processor.inverse_transform(processed_train_features_only)

        # 创建restored数据框
        restored_train_df = pd.DataFrame(
            restored_train_features,
            columns=[f"{feat}_restored" for feat in numeric_columns]
        )
        # 添加原始列用于比较
        for col in numeric_columns:
            restored_train_df[f"{col}_original"] = current_train_data[col].values

        # 添加非特征列
        if 'weight' in current_train_data.columns:
            restored_train_df['weight'] = current_train_data['weight'].values
        if 'Substrate' in current_train_data.columns:
            restored_train_df['Substrate'] = current_train_data['Substrate'].values
        if 'rrr' in current_train_data.columns:
            restored_train_df['rrr'] = current_train_data['rrr'].values

        # 计算逆变换误差
        for col in numeric_columns:
            original_col = f"{col}_original"
            restored_col = f"{col}_restored"
            # 只比较非缺失值
            mask = ~restored_train_df[original_col].isna() & ~restored_train_df[restored_col].isna()
            if mask.sum() > 0:
                errors = restored_train_df.loc[mask, original_col] - restored_train_df.loc[mask, restored_col]
                mean_error = errors.mean()
                max_error = errors.abs().max()
                # logger.info(f"逆变换误差统计 - {col}: 平均误差={mean_error:.6f}, 最大绝对误差={max_error:.6f}")

        # 保存restored数据文件
        output_file_restored = os.path.join(output_dir, 'augmented_dataset_restored.xlsx')
        restored_train_df.to_excel(output_file_restored, index=False)
        # logger.info(f"已保存逆变换检查文件: {output_file_restored}")
    except Exception as e:
        logger.error(f"逆变换检查过程中出现错误: {e}")

    # --- 10. 保存拟合好的 Processor ---
    processor_output_file = os.path.join(output_dir, 'fitted_processor_for_train.joblib')
    joblib.dump(processor, processor_output_file)
    # logger.info(f"已保存拟合好的 processor 到: {processor_output_file}")

    # --- 11. 生成摘要 ---
    summary_report = [
        "=" * 60,
        "数据增强与处理摘要报告 (训练集增强版)",
        "=" * 60,
        f"原始数据量: {original_count}",
        f"原始训练集大小: {len(train_raw)} ({len(train_raw) / original_count * 100:.1f}%)",
        f"原始验证集大小: {len(val_raw)} ({len(val_raw) / original_count * 100:.1f}%)",
        f"最终增强后训练集大小: {len(current_train_data)}",
        f"最终验证集大小: {len(processed_val_df)}",
        f"训练集数据增强迭代次数: {iteration}",
        "-" * 60,
        "数据统计 (特征列, 增强后训练集 vs 原始验证集):",
    ]
    for col in numeric_columns:
        stats = {
            'train_augmented': {
                'mean': current_train_data[col].mean(),
                'std': current_train_data[col].std(),
                'min': current_train_data[col].min(),
                'max': current_train_data[col].max(),
                'missing_rate': current_train_data[col].isna().mean()
            },
            'val_original': {
                'mean': val_raw[col].mean(),
                'std': val_raw[col].std(),
                'min': val_raw[col].min(),
                'max': val_raw[col].max(),
                'missing_rate': val_raw[col].isna().mean()
            }
        }
        summary_report.append(f" {col}:")
        summary_report.append(
            f"  增强后训练集 - 均值: {stats['train_augmented']['mean']:.4f}, 标准差: {stats['train_augmented']['std']:.4f}")
        summary_report.append(f"  范围: [{stats['train_augmented']['min']:.4f}, {stats['train_augmented']['max']:.4f}]")
        summary_report.append(f"  缺失率: {stats['train_augmented']['missing_rate']:.1%}")
        summary_report.append(
            f"  原始验证集 - 均值: {stats['val_original']['mean']:.4f}, 标准差: {stats['val_original']['std']:.4f}")
        summary_report.append(f"  范围: [{stats['val_original']['min']:.4f}, {stats['val_original']['max']:.4f}]")
        summary_report.append(f"  缺失率: {stats['val_original']['missing_rate']:.1%}")
        summary_report.append("")

    summary_report.append("=" * 60)

    # 保存摘要报告
    summary_file = os.path.join(output_dir, 'augmentation_summary_report_train_val_split.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_report))
    # logger.info(f"已保存详细摘要报告到: {summary_file}")

    # logger.info("=" * 60)
    # logger.info("🎉🎉🎉🎉 数据增强、处理与分割（训练集增强版）成功完成!")
    # logger.info("=" * 60)

    # --- 12. 验证数据泄露 ---
    # logger.info("开始执行数据泄露验证...")
    leakage_check_passed = validate_no_leakage(train_raw, val_raw, current_train_data, processed_val_df)
    if leakage_check_passed:
        # logger.info("✅ 所有数据泄露检查通过！")
        pass
    else:
        logger.warning("⚠️ 发现潜在的数据泄露问题！")

    # 修改返回信息，包含输出文件夹路径
    return current_train_data, val_raw, summary_report, processed_train_df, processed_val_df, output_dir

if __name__ == "__main__":
    # 设置随机种子以确保结果可重现
    np.random.seed(42)

    # 执行主函数，接收额外的返回值
    final_train_data, val_data, summary, processed_train_data, processed_val_data, output_dir = main()

    # 如果成功执行，显示最终确认信息
    # if final_train_data is not None:
        # logger.info("=" * 60)
        # logger.info("🎉🎉 数据增强与处理（训练集增强版）成功完成!")
        # logger.info("=" * 60)
        # logger.info(f"📊 数据统计:")
        # logger.info(f" 最终增强训练集: {len(final_train_data)} 样本")
        # logger.info(f" 最终验证集: {len(val_data)} 样本")
        # logger.info(f"📁 生成文件已保存到文件夹: {output_dir}")
        # logger.info(f" 📄 augmented_dataset_processed_train.xlsx (增强后训练集, 已处理)")
        # logger.info(f" 📄 augmented_dataset_processed_val.xlsx (验证集, 已处理)")
        # logger.info(f" 📄 augmented_dataset_raw_train.xlsx (增强后训练集, 原始)")
        # logger.info(f" 📄 augmented_dataset_raw_val.xlsx (验证集, 原始)")
        # logger.info(f" 📄 augmented_dataset_restored.xlsx (逆变换检查文件)")
        # logger.info(f" 📄 fitted_processor_for_train.joblib (用于训练集的拟合处理器)")
        # logger.info(f" 📄 augmentation_summary_report_train_val_split.txt (详细的增强报告)")
