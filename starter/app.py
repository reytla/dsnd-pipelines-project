from flask import Flask, render_template_string, request
import joblib

app = Flask(__name__)

# Load precomputed data
models = {
    "Logistic Regression": "logreg",
    "Random Forest": "rf"
}
results = joblib.load("model_metrics.pkl")

@app.route("/", methods=["GET"])
def index():
    selected_model = request.args.get("model", "Logistic Regression")
    metrics = results[selected_model]

    model_key = models[selected_model]
    feature_img = f"/static/top_features_{model_key}.png"
    cm_img = f"/static/confusion_matrix_{model_key}.png"

    html_template = """
    <html>
        <head><title>Model Comparison Dashboard</title></head>
        <body>
            <h1>Model Results</h1>
            <form method="get" action="/">
                <label for="model">Choose a model:</label>
                <select name="model" onchange="this.form.submit()">
                    {% for m in models %}
                        <option value="{{ m }}" {% if m == selected_model %}selected{% endif %}>{{ m }}</option>
                    {% endfor %}
                </select>
            </form>

            <h2>Evaluation Metrics</h2>
            <ul>
                <li><strong>Accuracy:</strong> {{ '{:.4f}'.format(metrics['accuracy']) }}</li>
                <li><strong>Precision:</strong> {{ '{:.4f}'.format(metrics['precision']) }}</li>
                <li><strong>Recall:</strong> {{ '{:.4f}'.format(metrics['recall']) }}</li>
                <li><strong>F1 Score:</strong> {{ '{:.4f}'.format(metrics['f1']) }}</li>
            </ul>

            <h2>Top 10 Important Features</h2>
            <img src="{{ feature_img }}" width="600"/>

            <h2>Confusion Matrix</h2>
            <img src="{{ cm_img }}" width="400"/>
        </body>
    </html>
    """
    return render_template_string(
        html_template,
        selected_model=selected_model,
        models=models.keys(),
        metrics=metrics,
        feature_img=feature_img,
        cm_img=cm_img
    )

if __name__ == "__main__":
    app.run(debug=True)
