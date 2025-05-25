# AI Tools - Machine Learning Explorer

A responsive and interactive web application for exploring various machine learning algorithms using built-in datasets. This tool helps users understand different ML concepts through beautiful visualizations and real-time model training.

## Features

- Multiple built-in datasets (Iris, Breast Cancer, Wine, Digits)
- Various ML algorithms (Random Forest, KNN, SVM)
- Interactive data visualizations:
  - Feature Distribution Analysis
  - Correlation Matrix
  - PCA Plot
- Real-time model training and evaluation
- Performance metrics visualization
- Responsive design for all screen sizes

## Requirements

- Python 3.8+
- Required packages are listed in `requirements.txt`

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Use the sidebar to:
   - Select a dataset
   - Choose an ML algorithm
   - Adjust the test size

4. Explore different visualizations and train models with the interactive interface

## Features Explanation

### Datasets
- **Iris**: Classic dataset for flower classification
- **Breast Cancer**: Binary classification dataset for tumor diagnosis
- **Wine**: Multi-class classification for wine origin
- **Digits**: Handwritten digits classification

### Algorithms
- **Random Forest**: Ensemble learning method using multiple decision trees
- **KNN**: K-Nearest Neighbors classification
- **SVM**: Support Vector Machine with RBF kernel

### Visualizations
- **Feature Distribution**: Analyze the distribution of individual features
- **Correlation Matrix**: Understand relationships between features
- **PCA Plot**: Visualize high-dimensional data in 2D space

## License

This project is licensed under the MIT License - see the LICENSE file for details. 