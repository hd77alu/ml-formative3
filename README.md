# ML Formative 3 - Probability Distributions, Bayesian Probability, and Gradient Descent

This repository contains the implementation of Machine Learning Formative Assignment 3, covering:
- **Part 1**: Bivariate Normal Distribution with custom PDF implementation
- **Part 2**: Bayesian Probability with sentiment analysis
- **Part 3**: Manual Gradient Descent calculations
- **Part 4**: Gradient Descent implementation in Python

## Table of Contents
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Running the Notebook](#running-the-notebook)
- [Project Structure](#project-structure)
- [Assignment Overview](#assignment-overview)

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/hd77alu/ml-formative3.git
cd ml-formative3
```

### Step 2: Install Pipenv (if not already installed)
```bash
pip install pipenv
```

### Step 3: Create the Virtual Environment and Install Packages
This will create a virtual environment and install all dependencies declared in the `Pipfile`:
```bash
pipenv install
```

### Step 4: Activate the Virtual Environment
```bash
pipenv shell
```
Your prompt will change to `(ml-formative3) ...`, indicating the environment is active.

### Step 5: Register the Kernel with Jupyter
Install the virtual environment as a named Jupyter kernel so it appears in Jupyter Notebook, JupyterLab, and VS Code:
```bash
python3 -m ipykernel install --user --name ml-formative3 --display-name "Python (ML Formative 3)"
```
This registers the kernel at `~/.local/share/jupyter/kernels/ml-formative3`.

### Step 6: Select the Kernel
When opening `ml_formative3_g6.ipynb`, select **"Python (ML Formative 3)"** as the kernel:
- **Jupyter Notebook / JupyterLab**: use the *Kernel → Change Kernel* menu.
- **VS Code**: click the kernel selector in the top-right corner of the notebook and choose **"Python (ML Formative 3)"**.

## Dataset Setup

### Part 1: IMDb Movies Dataset
1. The IMDb Movies dataset in the `data/imdb_movies_p1.csv` file  contains information about movies, including their names, release dates, user ratings, genres, overviews, cast and crew members, original titles, production status, original languages, budgets, revenues, and countries of origin. We used this data to explore the relationship between budget and revenue, and to predict the success of future movies.
2. Data source: [Kaggle: IMDb Movies Dataset](https://www.kaggle.com/datasets/ashpalsingh1525/imdb-movies-dataset?resource=download)

### Part 2: IMDb Reviews Dataset
1. The IMDB Movie Reviews dataset in the `data/imdb_movie_reviews.csv` file contains about 50K movie reviews that we used to apply Bayesian probability for text analytics.
2. Data source: [IMDB Movie Reviews Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

## Running the Notebook (remember to select the correct kernel)

### Option 1: Using Jupyter Notebook (Local)
```bash
jupyter notebook
```
- Navigate to `ml_formative3_g6.ipynb` in your browser
- Run all cells

### Option 2: Using JupyterLab (Local)
```bash
jupyter lab
```
- Open `ml_formative3_g6.ipynb` from the file browser
- Execute all cells

### Option 3: Using VS Code
1. Open the project folder in VS Code
2. Install the "Jupyter" extension (if not already installed)
3. Open `ml_formative3_g6.ipynb`
4. Select your Python interpreter
5. Run cells using the play button or `Shift+Enter`

### Option 4: Using Google Colab
1. Click the "Open in Colab" badge at the top of the notebook
2. Upload the required datasets
3. Run all cells

## Project Structure
```
ml-formative3/
│
├── data/
│   ├── imdb_movies_p1.csv        # Dataset for Part 1 (Budget vs Revenue)
│   └── imdb_movie_reviews.csv    # Dataset for Part 2 (Sentiment Analysis)
├── ml_formative3_g6.ipynb        # Main Jupyter notebook with all implementations
├── Pipfile                        # Pipenv dependencies specification
├── Pipfile.lock                   # Locked versions of dependencies
└── README.md                      # This file
```

## Assignment Overview

### Part 1: Probability Distributions
- Implements bivariate normal distribution from scratch (no scipy.stats or sklearn)
- Calculates PDF values for 2,668 US movie data points after rigorous filtering
- Creates contour plot (2D) and surface plot (3D) visualizations using 100×100 meshgrid
- Analyzes correlation between movie budget and revenue (ρ = 0.6903)
- Key statistics:
  - Sample: 2,668 US movies (USD currency standardized)
  - Mean Budget: $68.1M | Mean Revenue: $278.1M
- Data quality: Applied US-only filter, missing value removal, and bottom 1% percentile filtering

### Part 2: Bayesian Probability
- Uses IMDb Movie Reviews dataset (50K movie reviews)
- Selected keywords through statistical analysis:
  - Positive sentiment: "great", "very", "best", "love"
  - Negative sentiment: "worst", "nothing", "why", "waste"
- Implements Bayes' Theorem from scratch using only basic Python operations
- Calculates posterior probabilities P(Positive|keyword) for each selected keyword
- Key findings:
  - Reviews containing "great": 70%+ are positive
  - Reviews containing "worst": 92.3%+ are negative

### Part 3: Gradient Descent Manual Calculation
- Manual calculation of gradient descent updates
- Uses linear regression: y = mx + b
- Initial parameters: m=-1, b=1, learning rate=0.1
- Data points: (1,3) and (3,6)
- m moved from -1 → 1.7 → 1.26 → 1.34 → 1.334.
- b moved from 1 → 2.1 → 1.9 → 1.916 → 1.897.
- MSE moved from 36.5 → 1.04 → 1.9 → 0.064 → 0.0348 → 0.0318.
- Across the four iterations, (m,b) steadily adjusted toward values that better fit the data points.

### Part 4: Gradient Descent in Code
- Python implementation of Part 3 calculations
- Visualizes parameter updates over iterations
- Plots m, b, and Error changes using Matplotlib
