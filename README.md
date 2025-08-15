# Clothing Recommendation Pipeline Project

This project predicts whether a customer recommends a clothing item based on their review. The pipeline handles numerical, categorical, and text features, and uses models like Logistic Regression and Random Forest.

## Getting Started

To run this project locally, make sure you have Python 3.8 or newer. Clone the repo and install the required packages.

### Dependencies

All dependencies are listed in `requirements.txt`. Key packages include:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- spacy
- joblib
- plotly
- Flask

You will also need the spaCy English model:
```
python -m spacy download en_core_web_sm
```

### Installation

1. **Clone the repository:**
    ```
    git clone https://github.com/reytla/dsnd-pipelines-project.git
    cd dsnd-pipelines-project
    ```
2. **(Optional) Set up a virtual environment:**
    ```
    python -m venv venv
    venv\Scripts\activate
    ```
3. **Install dependencies:**
    ```
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    ```
## Web Application

This project includes a simple web application built with Flask, located in `starter/app.py`. The app allows users to interact with the trained models and view predictions or visualizations through a browser interface.

### Running the App

1. Make sure you have installed all dependencies as described above.
2. From the project root, run the following command:
    ```
    python starter/app.py
    ```
3. Open your web browser and go to [http://localhost:5000](http://localhost:5000) to access the app.

The app uses the trained models and outputs from the pipeline (such as saved figures in the `static/` directory) to provide an interactive experience.
---

## Project Structure

- `starter.ipynb`: Main Jupyter notebook for data exploration, pipeline building, model training, and evaluation.
- `requirements.txt`: List of required Python packages and versions.
- `custom_transformers.py`: Contains custom scikit-learn transformers for text feature engineering (e.g., `CharCountTransformer`, `SpacyLemmatizer`).  
  **Note:** These are essential for the pipeline to run. See comments in the file for usage details.
- `data/`: Folder containing the dataset (`reviews.csv`).
- `static/`: Folder for saving generated plots and figures.
- `tests/`: Contains test scripts for data loading, pipeline, and metrics.

## Custom Transformers

This project uses custom transformers implemented in `custom_transformers.py`:
- **CharCountTransformer**: Counts specific characters (spaces, exclamations, question marks) in text fields.
- **SpacyLemmatizer**: Lemmatizes text using spaCy for improved text vectorization.

These are imported and used in the pipeline within `starter.ipynb`.

## Testing

To run the tests:

```
pytest
```

### Test Breakdown

- **test_data_loading.py**: Checks that the data loads and cleans correctly.
- **test_pipeline.py**: Ensures the pipeline transforms and predicts as expected.
- **test_metrics.py**: Verifies that evaluation metrics are calculated properly.

## Project Overview

- Loaded and explored the data in `starter.ipynb`.
- Built feature engineering pipelines for numerical, categorical, and text data.
- Trained baseline models (Logistic Regression, Random Forest).
- Evaluated models using accuracy, classification report, and confusion matrix.
- Fine-tuned Random Forest using `RandomizedSearchCV`.
- Summarized results and findings.

## Built With

* [scikit-learn](https://scikit-learn.org/) - Machine learning
* [pandas](https://pandas.pydata.org/) - Data wrangling
* [spacy](https://spacy.io/) - Text preprocessing
* [matplotlib](https://matplotlib.org/) - Plotting
* [seaborn](https://seaborn.pydata.org/) - Visualization

## License

See [License](LICENSE.txt) for details.