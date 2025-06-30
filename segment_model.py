import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
import os
from preprocessing import load_data, preprocess_data

class CustomerSegmentation:
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.scaler_socio = None
        self.scaler_products = None
        self.pca_socio = None
        self.socio_features = None
        self.product_features = None
        
    def find_optimal_clusters_elbow(self, X, max_clusters=10):
        """Find optimal number of clusters using Elbow Method (README requirement)"""
        distortions = []
        silhouette_scores = []
        K = range(2, max_clusters + 1)
        
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            distortions.append(kmeans.inertia_)
            score = silhouette_score(X, kmeans.labels_)
            silhouette_scores.append(score)
        
        # Calculate the "elbow" using the knee point detection
        # Find the point where the rate of decrease sharply changes
        elbow_k = self._find_elbow_point(K, distortions)
        
        return elbow_k, distortions, silhouette_scores, K
    
    def _find_elbow_point(self, K, distortions):
        """Find the elbow point in the distortion curve"""
        # Calculate the distance from each point to the line between first and last points
        npoints = len(distortions)
        allCoord = np.vstack((range(len(distortions)), distortions)).T
        np.array([range(npoints), distortions])
        firstPoint = allCoord[0]
        lineVec = allCoord[-1] - allCoord[0]
        lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
        
        vecFromFirst = allCoord - firstPoint
        scalarProduct = np.sum(vecFromFirst * lineVecNorm, axis=1)
        vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
        vecToLine = vecFromFirst - vecFromFirstParallel
        distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
        
        # Return the k value (adjusted for 0-indexing)
        return K[np.argmax(distToLine)]
    
    def fit(self, X):
        """Fit the clustering model"""
        self.kmeans.fit(X)
        return self
    
    def predict(self, X):
        """Predict cluster labels for new data"""
        return self.kmeans.predict(X)
    
    def get_cluster_centers(self):
        """Get cluster centers"""
        return self.kmeans.cluster_centers_
    
    def get_cluster_profiles(self, X, feature_names):
        """Get detailed cluster profiles with feature interpretations"""
        labels = self.predict(X)
        profiles = {}
        
        for cluster_id in range(self.n_clusters):
            cluster_mask = labels == cluster_id
            cluster_data = X[cluster_mask]
            
            if len(cluster_data) > 0:
                # Calculate mean values for each feature
                mean_values = np.mean(cluster_data, axis=0)
                profiles[cluster_id] = {
                    'size': len(cluster_data),
                    'percentage': len(cluster_data) / len(X) * 100,
                    'features': dict(zip(feature_names, mean_values))
                }
        
        return profiles
    
    def analyze_product_ownership(self, X_products, labels, product_features):
        """Analyze product ownership patterns across clusters"""
        product_analysis = {}
        
        for cluster_id in range(self.n_clusters):
            cluster_mask = labels == cluster_id
            cluster_products = X_products[cluster_mask]
            
            if len(cluster_products) > 0:
                # Calculate mean product ownership for this cluster
                mean_ownership = np.mean(cluster_products, axis=0)
                product_analysis[cluster_id] = dict(zip(product_features, mean_ownership))
        
        return product_analysis
    
    def save_model(self, path='models/segmentation_model.joblib'):
        """Save the model and related objects"""
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        model_data = {
            'kmeans': self.kmeans,
            'scaler_socio': self.scaler_socio,
            'scaler_products': self.scaler_products,
            'pca_socio': self.pca_socio,
            'socio_features': self.socio_features,
            'product_features': self.product_features
        }
        joblib.dump(model_data, path)
    
    def load_model(self, path='models/segmentation_model.joblib'):
        """Load a saved model"""
        try:
            model_data = joblib.load(path)
            self.kmeans = model_data['kmeans']
            self.n_clusters = len(self.kmeans.cluster_centers_)
            self.scaler_socio = model_data.get('scaler_socio')
            self.scaler_products = model_data.get('scaler_products')
            self.pca_socio = model_data.get('pca_socio')
            self.socio_features = model_data.get('socio_features', [])
            self.product_features = model_data.get('product_features', [])
            return self
        except Exception as e:
            print(f"Warning: Could not load model from {path}: {e}")
            print("The model will be retrained with current data.")
            raise

def train_segmentation_model():
    """Train and save the segmentation model according to README specifications"""
    # Load and preprocess data
    train_df, test_df = load_data()
    data = preprocess_data(train_df, test_df)
    
    # Initialize segmentation model
    segmentation = CustomerSegmentation()
    
    # Find optimal number of clusters using Elbow Method (README requirement)
    print("Finding optimal number of clusters using Elbow Method...")
    optimal_k, distortions, silhouette_scores, K = segmentation.find_optimal_clusters_elbow(
        data['X_train_socio'], max_clusters=10
    )
    print(f"Optimal number of clusters (Elbow Method): {optimal_k}")
    
    # Plot elbow curve for visualization
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(K, distortions, 'bo-')
    plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal K = {optimal_k}')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Distortion (Inertia)')
    plt.title('Elbow Method for Optimal K')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(K, silhouette_scores, 'go-')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs Number of Clusters')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('optimal_clusters_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Update number of clusters and fit model
    segmentation.n_clusters = optimal_k
    segmentation.kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    segmentation.fit(data['X_train_socio'])
    
    # Save preprocessing objects with the model
    segmentation.scaler_socio = data['scaler_socio']
    segmentation.scaler_products = data['scaler_products']
    segmentation.pca_socio = data['pca_socio']
    segmentation.socio_features = data['socio_features']
    segmentation.product_features = data['product_features']
    segmentation.save_model()
    
    # Analyze clusters
    print("\nCluster Analysis:")
    labels = segmentation.predict(data['X_train_socio'])
    
    # Get cluster profiles
    profiles = segmentation.get_cluster_profiles(
        data['X_train_socio'], 
        data['socio_features']
    )
    
    for cluster_id, profile in profiles.items():
        print(f"\nCluster {cluster_id}:")
        print(f"  Size: {profile['size']} ({profile['percentage']:.1f}%)")
    
    # Analyze product ownership patterns
    product_analysis = segmentation.analyze_product_ownership(
        data['X_train_products'],
        labels,
        data['product_features']
    )
    
    return segmentation, data

if __name__ == "__main__":
    model, data = train_segmentation_model()
    print("Segmentation model trained and saved successfully!") 