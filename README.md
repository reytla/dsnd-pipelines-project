# Clothing Recommendation Pipeline Project

This project predicts whether a customer recommends a clothing item based on their review. The pipeline handles numerical, categorical, and text features, and uses models like Logistic Regression and Random Forest.

## Getting Started

To run this project locally, make sure you have Python 3.8 or newer. Clone the repo and install the required packages.

### Dependencies

```
pandas
numpy
scikit-learn
matplotlib
seaborn
spacy
```

### Installation

1. Clone the repository:
    ```
    git clone https://github.com/yourusername/dsnd-pipelines-project.git
    cd dsnd-pipelines-project
    ```
2. (Optional) Set up a virtual environment:
    ```
    python -m venv venv
    venv\Scripts\activate
    ```
3. Install dependencies:
    ```
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    ```

## Testing

To run the tests (if any are included):

```
pytest
```

### Break Down Tests

- **test_data_loading.py**: Checks that the data loads and cleans correctly.
- **test_pipeline.py**: Makes sure the pipeline transforms and predicts as expected.
- **test_metrics.py**: Verifies that evaluation metrics are calculated properly.

## Project Instructions

- Load and explore the data in `starter.ipynb`.
- Build feature engineering pipelines for numerical, categorical, and text data.
- Train baseline models (Logistic Regression, Random Forest).
- Evaluate models using accuracy, classification report, and confusion matrix.
- Fine-tune Random Forest using `RandomizedSearchCV`.
- Summarize your results and findings.

## Built With

* [scikit-learn](https://scikit-learn.org/) - Machine learning
* [pandas](https://pandas.pydata.org/) - Data wrangling
* [spacy](https://spacy.io/) - Text preprocessing
* [matplotlib](https://matplotlib.org/) - Plotting
* [seaborn](https://seaborn.pydata.org/) - Visualization

## License

See [License](LICENSE.txt) for details.

-----

# README Template

Below is a template provided for use when building your README file for students.

# Project Title

Project description goes here.

## Getting Started

Instructions for how to get a copy of the project running on your local machine.

### Dependencies

```
Examples here
```

### Installation

Step by step explanation of how to get a dev environment running.

List out the steps

```
Give an example here
```

## Testing

Explain the steps needed to run any automated tests

### Break Down Tests

Explain what each test does and why

```
Examples here
```

## Project Instructions

This section should contain all the student deliverables for this project.

## Built With

* [Item1](www.item1.com) - Description of item
* [Item2](www.item2.com) - Description of item
* [Item3](www.item3.com) - Description of item

Include all items used to build project.

## License

[License](LICENSE.txt)
