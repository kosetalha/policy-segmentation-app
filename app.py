import streamlit as st
import pandas as pd
import numpy as np
import os
import io
from preprocessing import load_data, preprocess_data, get_feature_groups, load_column_names
from segment_model import CustomerSegmentation, train_segmentation_model
from recommend import InsuranceRecommender
import visualizations as viz
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Insurance Customer Segmentation",
    page_icon="📊",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("📊 Insurance Customer Segmentation & Product Recommendation")
st.markdown("""
    This application analyzes insurance customer data to identify distinct customer segments
    and understand product ownership patterns. It implements K-Means clustering on socio-demographic
    features and provides insights for targeted marketing strategies.
""")

# Sidebar navigation - exact tabs as specified in README
st.sidebar.title("📍 Navigation")
page = st.sidebar.radio(
    "Choose a page",
    ["🏠 Overview", "📈 Clustering Visualization", "👥 Cluster Profiles", 
     "🛡️ Product Ownership per Segment", "⬇️ Download"]
)

# Add cache clearing functionality
if st.sidebar.button("🔄 Clear Cache", help="Clear cached data and reload"):
    st.cache_data.clear()
    st.rerun()

# Load data and models
@st.cache_data
def load_models_and_data():
    """Load and process all data and models"""
    try:
        # Load raw data with debug info
        st.write("🔄 Loading data...")
        train_df, test_df = load_data()
        st.write(f"✅ Data loaded - Train: {train_df.shape}, Test: {test_df.shape}")
        
        # Preprocess data
        st.write("🔄 Preprocessing data...")
        data = preprocess_data(train_df, test_df)
        
        # Load or train segmentation model
        model_path = 'models/segmentation_model.joblib'
        
        if not os.path.exists(model_path):
            st.info("🤖 Training segmentation model... This may take a few minutes.")
            segmentation, _ = train_segmentation_model()
        else:
            try:
                st.write("🔄 Loading existing model...")
                segmentation = CustomerSegmentation()
                segmentation.load_model()
                st.write("✅ Model loaded successfully")
            except Exception as e:
                st.warning(f"⚠️ Could not load existing model: {str(e)}")
                st.info("🤖 Retraining model with current data... This may take a few minutes.")
                segmentation, _ = train_segmentation_model()
        
        # Get cluster predictions
        labels = segmentation.predict(data['X_train_socio'])
        
        # Get cluster profiles
        profiles = segmentation.get_cluster_profiles(
            data['X_train_socio'], 
            data['socio_features']
        )
        
        # Analyze product ownership
        product_analysis = segmentation.analyze_product_ownership(
            data['X_train_products'],
            labels,
            data['product_features']
        )
        
        # Create combined dataset with cluster labels
        combined_df = data['train_df'].copy()
        combined_df['Cluster'] = labels
        
        return {
            'data': data,
            'segmentation': segmentation,
            'labels': labels,
            'profiles': profiles,
            'product_analysis': product_analysis,
            'combined_df': combined_df,
            'train_df': train_df,
            'test_df': test_df
        }
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load everything
with st.spinner("Loading data and models..."):
    models_data = load_models_and_data()

if models_data is None:
    st.stop()

# PAGE 1: OVERVIEW (as specified in README)
if page == "🏠 Overview":
    st.header("📋 Project Overview")
    
    # Project description section
    st.subheader("🎯 Project Description")
    st.markdown("""
    This project uses real-world insurance customer data from a Dutch insurance company to:
    - **Cluster customers** based on socio-demographic features using K-Means clustering
    - **Analyze product ownership patterns** across different customer segments  
    - **Visualize insights** to support targeted marketing strategies
    - **Provide recommendations** for product offerings by segment
    """)
    
    # Dataset summary
    st.subheader("📊 Dataset Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Training Records", f"{len(models_data['train_df']):,}")
    with col2:
        st.metric("Test Records", f"{len(models_data['test_df']):,}")
    with col3:
        st.metric("Socio-Demographic Features", len(models_data['data']['socio_features']))
    with col4:
        st.metric("Product Features", len(models_data['data']['product_features']))
    
    # Cluster explanation
    st.subheader("🔍 Cluster Explanation")
    st.markdown(f"""
    Using the **Elbow Method**, we identified **{models_data['segmentation'].n_clusters} optimal customer segments** 
    based on socio-demographic characteristics. Each segment represents customers with similar:
    - Demographics (age, income, education)
    - Lifestyle characteristics (family status, housing)
    - Social and economic factors
    """)
    
    # Show basic data preview
    st.subheader("📋 Data Preview")
    tab1, tab2 = st.tabs(["Training Data", "Feature Groups"])
    
    with tab1:
        st.dataframe(models_data['train_df'].head(), use_container_width=True)
    
    with tab2:
        feature_groups = get_feature_groups(load_column_names())
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Socio-Demographic Features (1-43):**")
            st.write(feature_groups['socio_demographic'][:10])  # Show first 10
            st.caption(f"... and {len(feature_groups['socio_demographic'])-10} more")
        
        with col2:
            st.write("**Product Features (44-85):**")
            st.write(feature_groups['product_contributions'][:10])  # Show first 10
            st.caption(f"... and {len(feature_groups['product_contributions'] + feature_groups['product_counts'])-10} more")

# PAGE 2: CLUSTERING VISUALIZATION (as specified in README)
elif page == "📈 Clustering Visualization":
    st.header("📈 Clustering Visualization")
    
    # Interactive controls
    st.subheader("🎛️ Interactive Controls")
    col1, col2 = st.columns(2)
    
    with col1:
        # Option to retrain with different number of clusters
        new_k = st.slider(
            "Adjust Number of Clusters", 
            min_value=2, 
            max_value=10, 
            value=models_data['segmentation'].n_clusters,
            help="Change the number of clusters and see how it affects the segmentation"
        )
    
    with col2:
        if st.button("Retrain with New K"):
            with st.spinner("Retraining model..."):
                # Create new model with different K
                new_segmentation = CustomerSegmentation(n_clusters=new_k)
                new_segmentation.fit(models_data['data']['X_train_socio'])
                new_labels = new_segmentation.predict(models_data['data']['X_train_socio'])
                
                # Update session state
                st.session_state.temp_labels = new_labels
                st.session_state.temp_k = new_k
    
    # Use temporary labels if available
    display_labels = getattr(st.session_state, 'temp_labels', models_data['labels'])
    display_k = getattr(st.session_state, 'temp_k', models_data['segmentation'].n_clusters)
    
    # Visualization tabs
    viz_tab1, viz_tab2 = st.tabs(["PCA Projection", "Cluster Distribution"])
    
    with viz_tab1:
        st.subheader("🎯 PCA Visualization of Customer Segments")
        # Create PCA visualization
        fig = viz.plot_pca_visualization(
            models_data['data']['X_train_socio'], 
            display_labels,
            "Customer Segments in 2D PCA Space"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Interpretation:** This 2D projection shows how customers cluster in the reduced feature space.
        Points of the same color represent customers in the same segment.
        """)
    
    with viz_tab2:
        st.subheader("📊 Segment Distribution")
        fig = viz.plot_segment_distribution(
            display_labels,
            f"Distribution of {display_k} Customer Segments"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show segment sizes
        segment_counts = pd.Series(display_labels).value_counts().sort_index()
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Segment Sizes:**")
            for i, count in enumerate(segment_counts):
                percentage = count / len(display_labels) * 100
                st.write(f"Segment {i}: {count:,} customers ({percentage:.1f}%)")

# PAGE 3: CLUSTER PROFILES (as specified in README)
elif page == "👥 Cluster Profiles":
    st.header("👥 Cluster Profiles")
    
    # Segment selector
    selected_segment = st.selectbox(
        "Select Segment to Analyze",
        range(models_data['segmentation'].n_clusters),
        format_func=lambda x: f"Segment {x}"
    )
    
    # Show profile for selected segment
    if selected_segment in models_data['profiles']:
        profile = models_data['profiles'][selected_segment]
        
        # Segment overview
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Segment Size", f"{profile['size']:,}")
        with col2:
            st.metric("Percentage of Total", f"{profile['percentage']:.1f}%")
        with col3:
            st.metric("Segment ID", f"#{selected_segment}")
        
        # Detailed characteristics
        st.subheader(f"🔍 Socio-Demographic Profile - Segment {selected_segment}")
        
        # Create tabs for different feature categories
        demo_tab, lifestyle_tab, economic_tab = st.tabs(["Demographics", "Lifestyle", "Economic"])
        
        features_df = pd.DataFrame([
            {'Feature': feat, 'Value': val} 
            for feat, val in profile['features'].items()
        ])
        
        with demo_tab:
            # Show demographic features
            demo_features = features_df[features_df['Feature'].str.contains('MGEMLEEF|MGEMOMV|MAANTHUI|MFGEKIND|MFWEKIND')]
            if not demo_features.empty:
                st.dataframe(demo_features, use_container_width=True)
            else:
                st.write("No specific demographic features to display")
        
        with lifestyle_tab:
            # Show lifestyle features
            lifestyle_features = features_df[features_df['Feature'].str.contains('MREL|MFALLEEN|MHHUUR|MHKOOP|MAUT')]
            if not lifestyle_features.empty:
                st.dataframe(lifestyle_features, use_container_width=True)
            else:
                st.write("No specific lifestyle features to display")
        
        with economic_tab:
            # Show economic features
            economic_features = features_df[features_df['Feature'].str.contains('MINK|MKOOPKLA|MBERH|MBER|MOPL')]
            if not economic_features.empty:
                st.dataframe(economic_features, use_container_width=True)
            else:
                st.write("No specific economic features to display")
        
        # Radar chart for top features
        st.subheader("📊 Feature Comparison (Top 10)")
        top_features = list(profile['features'].keys())[:10]
        top_values = [profile['features'][feat] for feat in top_features]
        
        fig = viz.plot_segment_characteristics(
            [top_values],
            top_features,
            f"Segment {selected_segment} Characteristics"
        )
        st.plotly_chart(fig, use_container_width=True)

# PAGE 4: PRODUCT OWNERSHIP PER SEGMENT (as specified in README)
elif page == "🛡️ Product Ownership per Segment":
    st.header("🛡️ Product Ownership Analysis per Segment")
    
    # Heatmap of product ownership across segments
    st.subheader("🔥 Product Ownership Heatmap")
    
    # Prepare data for heatmap
    segments = list(models_data['product_analysis'].keys())
    products = list(models_data['product_analysis'][segments[0]].keys())
    
    # Create heatmap data
    heatmap_data = []
    for segment in segments:
        row = [models_data['product_analysis'][segment][product] for product in products]
        heatmap_data.append(row)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=products,
        y=[f'Segment {i}' for i in segments],
        colorscale='Viridis',
        text=np.round(heatmap_data, 3),
        texttemplate='%{text}',
        textfont={"size": 8}
    ))
    
    fig.update_layout(
        title="Product Ownership Intensity by Segment",
        xaxis_title="Insurance Products",
        yaxis_title="Customer Segments",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Segment comparison
    st.subheader("📊 Segment Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Select segments to compare
        compare_segments = st.multiselect(
            "Select segments to compare",
            segments,
            default=segments[:2]
        )
    
    with col2:
        # Select product categories
        product_categories = st.multiselect(
            "Select product categories",
            products[:10],  # Show first 10 products
            default=products[:5]
        )
    
    if compare_segments and product_categories:
        # Create comparison chart
        comparison_data = []
        for segment in compare_segments:
            for product in product_categories:
                comparison_data.append({
                    'Segment': f'Segment {segment}',
                    'Product': product,
                    'Ownership': models_data['product_analysis'][segment][product]
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        fig = px.bar(
            comparison_df,
            x='Product',
            y='Ownership',
            color='Segment',
            title="Product Ownership Comparison",
            barmode='group'
        )
        
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Top products per segment
    st.subheader("🏆 Top Products by Segment")
    
    for segment in segments:
        with st.expander(f"Segment {segment} - Top Products"):
            segment_products = models_data['product_analysis'][segment]
            sorted_products = sorted(segment_products.items(), key=lambda x: x[1], reverse=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Top 5 Products:**")
                for i, (product, value) in enumerate(sorted_products[:5]):
                    st.write(f"{i+1}. {product}: {value:.3f}")
            
            with col2:
                st.write("**Lowest 5 Products:**")
                for i, (product, value) in enumerate(sorted_products[-5:]):
                    st.write(f"{i+1}. {product}: {value:.3f}")

# PAGE 5: DOWNLOAD (as specified in README)
elif page == "⬇️ Download":
    st.header("⬇️ Download Clustered Data")
    
    st.markdown("""
    Download the clustered dataset for further analysis. The downloaded file includes
    all original features plus the assigned cluster labels.
    """)
    
    # Data preview
    st.subheader("📋 Data Preview")
    st.dataframe(models_data['combined_df'].head(), use_container_width=True)
    
    # Download options
    st.subheader("💾 Download Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV download
        csv_buffer = io.StringIO()
        models_data['combined_df'].to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="📄 Download as CSV",
            data=csv_data,
            file_name="insurance_customer_segments.csv",
            mime="text/csv",
            help="Download the complete dataset with cluster assignments"
        )
    
    with col2:
        # Excel download
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            models_data['combined_df'].to_excel(writer, sheet_name='Customer_Segments', index=False)
            
            # Add cluster summary sheet
            cluster_summary = []
            for cluster_id, profile in models_data['profiles'].items():
                cluster_summary.append({
                    'Cluster': cluster_id,
                    'Size': profile['size'],
                    'Percentage': profile['percentage']
                })
            
            pd.DataFrame(cluster_summary).to_excel(writer, sheet_name='Cluster_Summary', index=False)
        
        excel_data = excel_buffer.getvalue()
        
        st.download_button(
            label="📊 Download as Excel",
            data=excel_data,
            file_name="insurance_customer_segments.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Download as Excel with multiple sheets including cluster summary"
        )
    
    # Dataset statistics
    st.subheader("📈 Dataset Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(models_data['combined_df']):,}")
    with col2:
        st.metric("Features", f"{len(models_data['combined_df'].columns)-1}")  # -1 for cluster column
    with col3:
        st.metric("Clusters", models_data['segmentation'].n_clusters)
    with col4:
        st.metric("File Size (CSV)", f"{len(csv_data)/1024:.1f} KB")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Insurance Customer Segmentation Application | Built with Streamlit</p>
    <p>Educational and demonstration purposes only</p>
</div>
""", unsafe_allow_html=True)
