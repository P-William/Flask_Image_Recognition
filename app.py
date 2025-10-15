"""Flask app for digit image prediction using a pre-trained Keras model."""

from flask import Flask, render_template, request
from model import preprocess_img, predict_result

app = Flask(__name__)


@app.route("/")
def main():
    """Render the home page."""
    return render_template("index.html")


@app.route("/prediction", methods=["POST"])
def predict_image_file():
    """Process uploaded image and return prediction results."""
    result_html = "result.html"
    img_file = request.files.get("file")
    if not img_file:
        # Make error message consistent with tests
        return render_template(result_html, err="File cannot be processed.")

    try:
        img = preprocess_img(img_file.stream)
        pred = predict_result(img)
        return render_template(result_html, predictions=str(pred))
    except (ValueError, IOError):
        # Same error message for exceptions
        return render_template(result_html, err="File cannot be processed.")


if __name__ == "__main__":
    app.run(port=5000, debug=True)
