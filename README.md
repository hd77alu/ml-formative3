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
├── ml_formative3_g6.ipynb    # Main Jupyter notebook with all implementations
├── imdb_movies_p1.csv         # Dataset for Part 1 (Budget vs Revenue)
├── IMDB-Dataset.csv           # Dataset for Part 2 (Sentiment Analysis)
├── README.md                  # This file

```

## Assignment Overview

### Part 1: Probability Distributions
- Implements bivariate normal distribution from scratch
- Calculates PDF values for 9,909 movie data points
- Creates contour plot (2D) and surface plot (3D) visualizations
- Analyzes correlation between movie budget and revenue (ρ = 0.6709)

### Part 2: Bayesian Probability (In Progress)
- Uses IMDb Movie Reviews dataset
- Selects keywords indicating positive/negative sentiment
- Implements Bayes' Theorem from scratch
- Calculates posterior probabilities P(Positive|keyword)

### Part 3: Gradient Descent Manual Calculation (Pending)
- Manual calculation of gradient descent updates
- Uses linear regression: y = mx + b
- Initial parameters: m=-1, b=1, learning rate=0.1
- Data points: (1,3) and (3,6)

### Part 4: Gradient Descent in Code (Pending)
- Python implementation of Part 3 calculations
- Visualizes parameter updates over iterations
- Plots m, b, and Error changes using Matplotlib
