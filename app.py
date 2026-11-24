"""Flask demo: upload a content image and a style image, get the AdaIN
stylized result back. Runs on CPU."""
import os

import torch
from flask import Flask, render_template, request, send_from_directory
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from werkzeug.utils import secure_filename
from wtforms import FileField, FloatField, HiddenField, SubmitField
from PIL import Image

from src.style_transfer import load_models, save_tensor_image, stylize

app = Flask(__name__)
app.config["SECRET_KEY"] = "supersecretkey"
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}
Bootstrap(app)

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder, decoder = load_models("weights/vgg_normalised.pth", "weights/decoder_final.pth", device)


class UploadForm(FlaskForm):
    content = FileField("Content Image")
    style = FileField("Style Image")
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField("Alpha", default=1.0)
    submit = SubmitField("Transfer Style")


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]


@app.route("/", methods=["GET", "POST"])
def index():
    form = UploadForm()
    result_image = content_filename = style_filename = error = None

    if form.validate_on_submit():
        if form.content.data and form.content.data.filename:
            if allowed_file(form.content.data.filename):
                content_filename = secure_filename(form.content.data.filename)
                form.content.data.save(os.path.join(app.config["UPLOAD_FOLDER"], content_filename))
                form.content_path.data = content_filename
        else:
            content_filename = form.content_path.data

        if form.style.data and form.style.data.filename:
            if allowed_file(form.style.data.filename):
                style_filename = secure_filename(form.style.data.filename)
                form.style.data.save(os.path.join(app.config["UPLOAD_FOLDER"], style_filename))
                form.style_path.data = style_filename
        else:
            style_filename = form.style_path.data

        if content_filename and style_filename:
            content_path = os.path.join(app.config["UPLOAD_FOLDER"], content_filename)
            style_path = os.path.join(app.config["UPLOAD_FOLDER"], style_filename)
            try:
                content_image = Image.open(content_path).convert("RGB")
                style_image = Image.open(style_path).convert("RGB")
                alpha = float(form.alpha.data)

                output = stylize(content_image, style_image, encoder, decoder, device, alpha=alpha)

                result_filename = "stylized_" + content_filename
                save_tensor_image(output, os.path.join(app.config["UPLOAD_FOLDER"], result_filename))
                result_image = result_filename
            except Exception as e:  # noqa: BLE001 - surfaced to the user in the form
                error = str(e)
    else:
        if not content_filename:
            error = "Please upload content image"
        if not style_filename:
            error = "Please upload style image"

    return render_template(
        "index.html", form=form, result_image=result_image,
        content_image=content_filename, style_image=style_filename, error=error,
    )


@app.route("/uploads/<filename>")
def send_image(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/examples/<path:filename>")
def send_example(filename):
    return send_from_directory("examples", filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
