import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def plot_segment_distribution(labels, title="Customer Segment Distribution"):
    """Create an interactive pie chart of segment distribution"""
    segment_counts = pd.Series(labels).value_counts().sort_index()
    
    fig = px.pie(
        values=segment_counts.values,
        names=[f'Segment {i}' for i in segment_counts.index],
        title=title,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=True, height=400)
    return fig

def plot_segment_characteristics(segment_profiles, feature_names, title="Segment Characteristics"):
    """Create a radar chart showing segment characteristics"""
    # Create radar chart
    fig = go.Figure()
    
    # Ensure segment_profiles is iterable and handle single profile case
    if len(np.array(segment_profiles).shape) == 1:
        segment_profiles = [segment_profiles]
    
    for i, profile in enumerate(segment_profiles):
        # Normalize values to 0-1 range for better visualization
        normalized_profile = (np.array(profile) - np.min(profile)) / (np.max(profile) - np.min(profile)) if np.max(profile) != np.min(profile) else np.array(profile)
        
        fig.add_trace(go.Scatterpolar(
            r=normalized_profile,
            theta=feature_names,
            fill='toself',
            name=f'Segment {i}',
            line=dict(width=2)
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title=title,
        height=500
    )
    
    return fig

def plot_product_recommendations(recommendations, title="Product Recommendations by Segment"):
    """Create a heatmap of product recommendations by segment"""
    # Convert recommendations to a matrix
    segments = list(recommendations.keys())
    
    if not segments:
        # Return empty figure if no recommendations
        fig = go.Figure()
        fig.update_layout(title="No recommendations available")
        return fig
    
    products = list(recommendations[segments[0]].keys())
    
    values = np.array([[recommendations[seg][prod] for prod in products] 
                      for seg in segments])
    
    fig = go.Figure(data=go.Heatmap(
        z=values,
        x=products,
        y=[f'Segment {i}' for i in segments],
        colorscale='Viridis',
        text=values.round(3),
        texttemplate='%{text}',
        textfont={"size": 8}
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Products",
        yaxis_title="Segments",
        height=400
    )
    
    return fig

def plot_pca_visualization(X, labels, title="PCA Visualization of Customer Segments"):
    """Create a 2D PCA visualization of customer segments"""
    # Apply PCA if X has more than 2 dimensions
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        var_explained = pca.explained_variance_ratio_
    else:
        X_pca = X
        var_explained = [1.0, 0.0]  # Default values if already 2D
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Cluster': [f'Segment {label}' for label in labels]
    })
    
    # Create scatter plot
    fig = px.scatter(
        df,
        x='PC1',
        y='PC2',
        color='Cluster',
        title=title,
        labels={
            'PC1': f'PC1 ({var_explained[0]:.2%} variance)',
            'PC2': f'PC2 ({var_explained[1]:.2%} variance)',
        },
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_traces(marker=dict(size=6, opacity=0.7))
    fig.update_layout(height=500)
    
    return fig

def plot_feature_importance(importance_scores, feature_names, title="Feature Importance"):
    """Create a bar chart of feature importance"""
    # Sort features by importance
    sorted_idx = np.argsort(importance_scores)[-20:]  # Top 20 features
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=importance_scores[sorted_idx],
        y=[feature_names[i] for i in sorted_idx],
        orientation='h',
        marker=dict(color='lightblue')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Importance Score",
        yaxis_title="Features",
        showlegend=False,
        height=600
    )
    
    return fig

def plot_customer_journey(customer_data, recommendations, title="Customer Journey Analysis"):
    """Create a visualization of customer journey and recommendations"""
    # Create a timeline-like visualization
    fig = go.Figure()
    
    # Add customer data points
    fig.add_trace(go.Scatter(
        x=list(range(len(customer_data))),
        y=customer_data,
        mode='lines+markers',
        name='Customer Profile',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))
    
    # Add recommendation points
    for i, (product, score) in enumerate(recommendations[:min(10, len(recommendations))]):  # Limit to 10 recommendations
        fig.add_trace(go.Scatter(
            x=[i],
            y=[score],
            mode='markers',
            name=product,
            marker=dict(size=12, symbol='star')
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Features/Products",
        yaxis_title="Score/Weight",
        showlegend=True,
        height=400
    )
    
    return fig

def plot_elbow_curve(K, distortions, optimal_k, title="Elbow Method for Optimal K"):
    """Create an elbow curve plot for K-means optimization"""
    fig = go.Figure()
    
    # Add elbow curve
    fig.add_trace(go.Scatter(
        x=K,
        y=distortions,
        mode='lines+markers',
        name='Distortion',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))
    
    # Add optimal K marker
    optimal_idx = list(K).index(optimal_k)
    fig.add_trace(go.Scatter(
        x=[optimal_k],
        y=[distortions[optimal_idx]],
        mode='markers',
        name=f'Optimal K = {optimal_k}',
        marker=dict(size=15, color='red', symbol='star')
    ))
    
    # Add vertical line at optimal K
    fig.add_vline(x=optimal_k, line_dash="dash", line_color="red")
    
    fig.update_layout(
        title=title,
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Distortion (Inertia)",
        showlegend=True,
        height=400
    )
    
    return fig

def plot_silhouette_analysis(K, silhouette_scores, title="Silhouette Analysis"):
    """Create a silhouette score plot"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=K,
        y=silhouette_scores,
        mode='lines+markers',
        name='Silhouette Score',
        line=dict(color='green', width=2),
        marker=dict(size=8)
    ))
    
    # Highlight best silhouette score
    best_idx = np.argmax(silhouette_scores)
    fig.add_trace(go.Scatter(
        x=[K[best_idx]],
        y=[silhouette_scores[best_idx]],
        mode='markers',
        name=f'Best Score = {silhouette_scores[best_idx]:.3f}',
        marker=dict(size=15, color='red', symbol='star')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Silhouette Score",
        showlegend=True,
        height=400
    )
    
    return fig

def plot_cluster_heatmap(data, labels, feature_names, title="Cluster Feature Heatmap"):
    """Create a heatmap showing average feature values per cluster"""
    # Calculate mean values per cluster
    unique_labels = np.unique(labels)
    cluster_means = []
    
    for label in unique_labels:
        cluster_data = data[labels == label]
        cluster_means.append(np.mean(cluster_data, axis=0))
    
    cluster_means = np.array(cluster_means)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cluster_means,
        x=feature_names,
        y=[f'Segment {i}' for i in unique_labels],
        colorscale='RdBu',
        text=cluster_means.round(2),
        texttemplate='%{text}',
        textfont={"size": 8}
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Features",
        yaxis_title="Clusters",
        height=500
    )
    
    return fig 