# Customer Segmentation and Product Recommendation in Insurance

This project uses real-world insurance customer data to build a customer segmentation model and analyze product ownership patterns. The goal is to provide insights that can help insurance companies personalize their marketing strategies and improve customer understanding.

<a href="https://policy-segmentation-app.streamlit.app/" target="_blank">Streamlit App</a>

---

## 🎯 Objectives

- **Cluster customers** based on their socio-demographic features using unsupervised learning (K-Means).
- **Analyze each segment** in terms of insurance product ownership.
- **Visualize clusters** and product distributions using interactive Streamlit components.
- **Provide a baseline for future extensions**, such as product recommendation or supervised policy purchase prediction.

---

## 🧾 Dataset Description

This dataset originates from a real-world business problem and was provided by the Dutch data mining company Sentient Machine Research. It includes:

- **TICDATA2000.txt**: Training set (5822 records) with 86 features (attributes 1–43: socio-demographic data, 44–85: product ownership, 86: caravan policy ownership).
- **TICEVAL2000.txt**: Evaluation set (4000 records), same format but without the target.
- **TICTGTS2000.txt**: Target labels for the evaluation set.
- **dictionary.txt**: Attribute definitions and data dictionary.

All data files are tab-separated.

---

## 📁 Project Structure

```
policy-segmentation-app/
│
├── data/
│   ├── ticdata2000.txt          # Training data
│   ├── ticeval2000.txt          # Test data
│   ├── tictgts2000.txt          # Test targets
│   └── dictionary.txt           # Feature definitions
│
├── models/
│   └── segmentation_model.joblib # Trained clustering model
│
├── app.py                       # Main Streamlit application
├── preprocessing.py             # Data loading and preprocessing
├── segment_model.py             # Customer segmentation model
├── visualizations.py            # Plotting and visualization functions
├── recommend.py                 # Product recommendation system
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 🚀 Quick Start

### 1. **Installation**

Clone the repository and navigate to the project directory:

```bash
git clone <repository-url>
cd policy-segmentation-app
```

Create and activate a virtual environment:

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. **Run the Application**

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

---

## 📊 Application Features

### 5-Tab Interface

#### 1. **🏠 Overview**
- Project description and objectives
- Dataset summary with key statistics
- Cluster explanation and methodology
- Data preview and feature group visualization

#### 2. **📈 Clustering Visualization**
- Interactive PCA visualization of customer segments
- Cluster distribution pie charts
- Interactive controls to adjust number of clusters
- Real-time model retraining capabilities

#### 3. **👥 Cluster Profiles**
- Detailed socio-demographic profiles for each segment
- Feature analysis by category (Demographics, Lifestyle, Economic)
- Interactive radar charts for segment characteristics
- Segment comparison tools

#### 4. **🛡️ Product Ownership per Segment**
- Heatmap visualization of product ownership across segments
- Interactive segment and product comparison
- Top products analysis for each segment
- Product recommendation insights

#### 5. **⬇️ Download**
- Download clustered data in CSV or Excel format
- Multiple sheets in Excel with cluster summaries
- Dataset statistics and file information

---

## 🔬 Technical Implementation

### Data Processing Pipeline

1. **Data Loading**: Column names extracted from `dictionary.txt`
2. **Feature Separation**: 
   - Socio-demographic features (columns 1-43) for clustering
   - Product ownership features (columns 44-85) for analysis
3. **Preprocessing**: StandardScaler normalization and optional PCA
4. **Clustering**: K-Means with Elbow Method for optimal K selection

### Key Features Implemented

- ✅ **Elbow Method**: Primary method for determining optimal number of clusters
- ✅ **Socio-demographic Focus**: Clustering based on customer characteristics (columns 1-43)
- ✅ **Product Analysis**: Separate analysis of insurance product patterns (columns 44-85)
- ✅ **Interactive Visualization**: PCA plots, heatmaps, and radar charts
- ✅ **Real-time Model Training**: Ability to retrain with different K values
- ✅ **Data Export**: CSV and Excel download functionality

### Model Performance

The application automatically:
- Determines optimal K using Elbow Method
- Validates results with Silhouette Score analysis
- Saves trained models for consistent results
- Provides detailed cluster profiling and analysis

---

## 📈 Understanding the Results

### Cluster Interpretation

Each customer segment represents a distinct group with similar:
- **Demographics**: Age, household composition, income levels
- **Lifestyle**: Housing type, car ownership, family status
- **Economic Status**: Income brackets, education levels, social class

### Product Ownership Analysis

The application analyzes 42 different insurance products:
- **P-features**: Contribution amounts to various insurance types
- **A-features**: Number of policies held for each insurance type
- **Patterns**: Which segments prefer which products

### Business Applications

1. **Targeted Marketing**: Customize campaigns for each segment
2. **Product Development**: Identify underserved market segments
3. **Risk Assessment**: Understand customer risk profiles
4. **Cross-selling**: Recommend products based on segment patterns

---

## 🛠️ Customization and Extension

### Adding New Features

1. **New Visualizations**: Add functions to `visualizations.py`
2. **Advanced Clustering**: Implement other algorithms in `segment_model.py`
3. **Enhanced Preprocessing**: Modify feature engineering in `preprocessing.py`
4. **Additional Tabs**: Extend the Streamlit interface in `app.py`

### Configuration Options

- Adjust PCA components in preprocessing
- Modify clustering parameters (K range, random state)
- Customize visualization colors and themes
- Add new product recommendation algorithms

---

## 📋 Dependencies

- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning algorithms
- **plotly**: Interactive visualizations
- **matplotlib**: Static plotting
- **joblib**: Model serialization
- **openpyxl**: Excel file handling

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

## 📝 License and Usage

This project is intended for educational and demonstration purposes only. Original dataset is provided by UCI Machine Learning Repository under their usage guidelines.

---

## 🆘 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed via `pip install -r requirements.txt`
2. **Data Loading Issues**: Verify data files are in the `data/` directory
3. **Model Training Slow**: PCA is enabled by default - disable for faster processing
4. **Visualization Not Loading**: Check browser compatibility with Plotly

### Performance Tips

- Use PCA for large datasets to reduce dimensionality
- Limit K range for faster elbow method computation
- Cache results using Streamlit's `@st.cache_data` decorator

---

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the code documentation
3. Open an issue in the repository

---

*Built with ❤️ using Streamlit, scikit-learn, and Plotly*
