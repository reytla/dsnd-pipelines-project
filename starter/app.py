from fasthtml import fast_app, serve, Request
from fasthtml.tags import Div, H1, Form, Textarea, Button, P
import joblib

# Load trained model pipeline
model = joblib.load("model_logreg.pkl")

app = fast_app()

@app.route("/", methods=["GET", "POST"])
async def classify_review(req: Request):
    prediction = ""
    input_text = ""

    if req.method == "POST":
        form_data = await req.form()
        input_text = form_data.get("review", "")
        if input_text.strip():
            prediction = model.predict([input_text])[0]
    
    return Div(
        H1("Review Classifier"),
        Form(
            Textarea(input_text, name="review", rows="6", cols="60", placeholder="Enter review text here"),
            Button("Classify", type="submit"),
            method="post"
        ),
        P(f"Prediction: {prediction}" if prediction else "")
    )

serve(app)

