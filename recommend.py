import numpy as np
import pandas as pd
from segment_model import CustomerSegmentation

class InsuranceRecommender:
    def __init__(self):
        self.segmentation_model = None
        self.product_weights = None
        self.customer_profiles = None
        
    def load_model(self, model_path='models/segmentation_model.joblib'):
        """Load the segmentation model"""
        self.segmentation_model = CustomerSegmentation.load_model(model_path)
        
    def create_product_weights(self):
        """Create weights for different insurance products based on domain knowledge"""
        # Define product categories and their weights for each segment
        self.product_weights = {
            'caravan': {
                0: 0.8,  # High weight for segment 0
                1: 0.3,  # Low weight for segment 1
                2: 0.5,  # Medium weight for segment 2
                3: 0.2,  # Low weight for segment 3
                4: 0.6   # Medium-high weight for segment 4
            },
            'car': {
                0: 0.7,
                1: 0.8,
                2: 0.6,
                3: 0.9,
                4: 0.5
            },
            'home': {
                0: 0.6,
                1: 0.7,
                2: 0.8,
                3: 0.5,
                4: 0.9
            },
            'life': {
                0: 0.4,
                1: 0.6,
                2: 0.7,
                3: 0.8,
                4: 0.5
            }
        }
        
    def create_customer_profiles(self, X):
        """Create customer profiles based on their features"""
        # Get cluster assignments
        cluster_labels = self.segmentation_model.predict(X)
        
        # Create profiles based on cluster centers
        self.customer_profiles = self.segmentation_model.get_cluster_centers()
        
        return cluster_labels
    
    def get_recommendations(self, customer_data, top_n=3):
        """Get personalized insurance product recommendations"""
        if self.segmentation_model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        if self.product_weights is None:
            self.create_product_weights()
            
        # Get customer's segment
        customer_segment = self.segmentation_model.predict(customer_data)[0]
        
        # Get product weights for the customer's segment
        segment_weights = {product: weights[customer_segment] 
                         for product, weights in self.product_weights.items()}
        
        # Sort products by weight
        sorted_products = sorted(segment_weights.items(), 
                               key=lambda x: x[1], 
                               reverse=True)
        
        # Return top N recommendations
        return sorted_products[:top_n]
    
    def get_segment_insights(self, segment_id):
        """Get insights about a specific customer segment"""
        if self.customer_profiles is None:
            raise ValueError("Customer profiles not created. Call create_customer_profiles() first.")
            
        # Get the profile for the specified segment
        segment_profile = self.customer_profiles[segment_id]
        
        # Get product weights for the segment
        segment_weights = {product: weights[segment_id] 
                         for product, weights in self.product_weights.items()}
        
        return {
            'profile': segment_profile,
            'product_weights': segment_weights
        }

def train_recommender():
    """Train and initialize the recommendation system"""
    recommender = InsuranceRecommender()
    recommender.load_model()
    recommender.create_product_weights()
    return recommender

if __name__ == "__main__":
    # Initialize and train the recommender
    recommender = train_recommender()
    print("Recommendation system initialized successfully!") 