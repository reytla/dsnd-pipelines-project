# from fasthtml import fast_app, serve, Request
# from fasthtml.tags import Div, H1, Form, Textarea, Button, P
# import joblib

# # Load trained model pipeline
# model = joblib.load("model_logreg.pkl")

# app = fast_app()

# @app.route("/", methods=["GET", "POST"])
# async def classify_review(req: Request):
#     prediction = ""
#     input_text = ""

#     if req.method == "POST":
#         form_data = await req.form()
#         input_text = form_data.get("review", "")
#         if input_text.strip():
#             prediction = model.predict([input_text])[0]

#     return Div(
#         H1("Review Classifier"),
#         Form(
#             Textarea(input_text, name="review", rows="6", cols="60", placeholder="Enter review text here"),
#             Button("Classify", type="submit"),
#             method="post"
#         ),
#         P(f"Prediction: {prediction}" if prediction else "")
#     )

# serve(app)

# from flask import Flask, render_template, request
# import joblib
# import numpy as np
# import plotly.graph_objs as go
# from plotly.offline import plot
# import pandas as pd

# # Load model
# model_pipeline = joblib.load("model_logreg.pkl")

# # Access the logistic regression model
# logreg = model_pipeline.named_steps["logisticregression"]

# # Access the vectorizer from the column transformer
# column_transformer = model_pipeline.named_steps["columntransformer"]
# text_pipeline = column_transformer.named_transformers_["text"]
# vectorizer = text_pipeline.named_steps["countvectorizer"]

# app = Flask(__name__)

# # Utility: top features
# def get_top_features(n=10):
#     feature_names = vectorizer.get_feature_names_out()
#     coefs = logreg.coef_[0]
#     top_pos = sorted(zip(feature_names, coefs), key=lambda x: x[1], reverse=True)[:n]
#     top_neg = sorted(zip(feature_names, coefs), key=lambda x: x[1])[:n]
#     return top_pos, top_neg

# # Utility: plot class distribution
# def plot_class_distribution():
#     # NOTE: Replace with your actual dataset class counts if dynamic
#     class_counts = {'Negative': 707, 'Positive': 293}
#     df = pd.DataFrame({'Class': list(class_counts.keys()), 'Count': list(class_counts.values())})

#     fig = go.Figure(data=[go.Bar(x=df['Class'], y=df['Count'])])
#     fig.update_layout(title="Class Distribution", xaxis_title="Class", yaxis_title="Count")
#     return plot(fig, output_type='div', include_plotlyjs=False)

# @app.route("/", methods=["GET"])
# def index():
#     top_pos, top_neg = get_top_features()
#     class_plot = plot_class_distribution()
#     return render_template("index.html", top_pos=top_pos, top_neg=top_neg, class_plot=class_plot)

# @app.route("/predict", methods=["POST"])
# def predict():
#     text = request.form.get("text", "")
#     if not text:
#         return "Please enter text.", 400

#     proba = model_pipeline.predict_proba([text])[0]
#     pred = model_pipeline.predict([text])[0]
#     conf = np.max(proba)

#     return render_template("prediction_result.html", text=text, prediction=pred, confidence=conf)

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, render_template_string
import joblib
import numpy as np

# Load model pipeline
model_pipeline = joblib.load("model_logreg.pkl")

# Extract the column transformer
column_transformer = model_pipeline.named_steps["columntransformer"]

# Access the 'review_tfidf' sub-pipeline
review_tfidf_pipeline = column_transformer.named_transformers_["review_tfidf"]

# Get the TfidfVectorizer
tfidf_vectorizer = review_tfidf_pipeline.named_steps["tfidf_vectorizer"]

# Extract feature names
feature_names = tfidf_vectorizer.get_feature_names_out()

# Get coefficients from Logistic Regression
logreg = model_pipeline.named_steps["logisticregression"]
coefs = logreg.coef_[0]

# Get top 10 features by absolute weight
feature_importances = list(zip(feature_names, coefs))
sorted_features = sorted(feature_importances, key=lambda x: abs(x[1]), reverse=True)
top_features = sorted_features[:10]

# Flask app setup
app = Flask(__name__)

@app.route("/")
def home():
    html_template = """
    <html>
        <head><title>Top Features</title></head>
        <body>
            <h2>Top 10 Most Influential Review Words</h2>
            <table border="1">
                <tr><th>Feature</th><th>Coefficient</th></tr>
                {% for feature, coef in features %}
                <tr><td>{{ feature }}</td><td>{{ '{0:.4f}'.format(coef) }}</td></tr>
                {% endfor %}
            </table>
        </body>
    </html>
    """
    return render_template_string(html_template, features=top_features)

if __name__ == "__main__":
    app.run(debug=True)
