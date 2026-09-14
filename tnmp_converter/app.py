import os
import uuid
from datetime import datetime

import torch
import torchvision.transforms.functional as tf
from flask import Flask, redirect, render_template, request, url_for
from flask_sqlalchemy import SQLAlchemy
from PIL import Image
from torch import nn
from werkzeug.utils import secure_filename

# ============================================================
# Flask
# ============================================================

app = Flask(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

GENERATED_FOLDER = os.path.join(BASE_DIR, "static", "generated")

MODEL_PATH = os.path.join(BASE_DIR, "checkpoints", "tnmp_generator.pt")


# ============================================================
# Database
# ============================================================

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(
    BASE_DIR, "normal_maps.db"
)

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


# ============================================================
# Create directories
# ============================================================

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_FOLDER, exist_ok=True)


# ============================================================
# Database Model
# ============================================================


class NormalMap(db.Model):
    __tablename__ = "normal_maps"

    # UUID / image name
    id = db.Column(db.String(36), primary_key=True)

    # URL/path of uploaded texture
    texture_url = db.Column(db.String(500), nullable=False)

    # URL/path of generated normal map
    normal_map_url = db.Column(db.String(500), nullable=False)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<NormalMap {self.id}>"


# ============================================================
# Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Generator
# ============================================================


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        def down(in_c, out_c, normalize=True):

            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1)]

            if normalize:
                layers.append(nn.BatchNorm2d(out_c))

            layers.append(nn.LeakyReLU(0.2))

            return nn.Sequential(*layers)

        def up(in_c, out_c, dropout=False):

            layers = [
                nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
            ]

            if dropout:
                layers.append(nn.Dropout(0.2))

            return nn.Sequential(*layers)

        # Encoder

        self.d1 = down(3, 64, normalize=False)

        self.d2 = down(64, 128)
        self.d3 = down(128, 256)
        self.d4 = down(256, 512)
        self.d5 = down(512, 512)
        self.d6 = down(512, 512)

        self.d7 = down(512, 512, normalize=False)

        self.d8 = down(512, 512, normalize=False)

        # Decoder

        self.u1 = up(512, 512, dropout=True)

        self.u2 = up(1024, 512, dropout=True)

        self.u3 = up(1024, 512, dropout=True)

        self.u4 = up(1024, 512)

        self.u5 = up(1024, 256)

        self.u6 = up(512, 128)

        self.u7 = up(256, 64)

        self.final = nn.ConvTranspose2d(128, 3, 4, 2, 1)

        self.tanh = nn.Tanh()

    def forward(self, x):

        # Encoder

        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)
        d8 = self.d8(d7)

        # Decoder

        u1 = self.u1(d8)

        u2 = self.u2(torch.cat([u1, d7], 1))

        u3 = self.u3(torch.cat([u2, d6], 1))

        u4 = self.u4(torch.cat([u3, d5], 1))

        u5 = self.u5(torch.cat([u4, d4], 1))

        u6 = self.u6(torch.cat([u5, d3], 1))

        u7 = self.u7(torch.cat([u6, d2], 1))

        output = self.final(torch.cat([u7, d1], 1))

        return self.tanh(output)


# ============================================================
# Load Generator
# ============================================================

G = Generator().to(device)

checkpoint = torch.load(MODEL_PATH, map_location=device)

G.load_state_dict(checkpoint)

G.eval()

print("Generator loaded successfully.")
print("Device:", device)


# ============================================================
# Image preprocessing
# ============================================================


def preprocess_image(image):

    image = image.convert("RGB")

    # Same size used during training
    image = image.resize((512, 512))

    image = tf.to_tensor(image)

    image = tf.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    image = image.unsqueeze(0)

    return image.to(device)


# ============================================================
# Convert model output to image
# ============================================================


def tensor_to_image(tensor):

    tensor = tensor.squeeze(0)

    # [-1, 1] -> [0, 1]

    tensor = (tensor * 0.5) + 0.5

    tensor = tensor.clamp(0, 1)

    # CHW -> HWC

    tensor = tensor.permute(1, 2, 0)

    tensor = tensor.cpu()

    # [0, 1] -> [0, 255]

    tensor = (tensor * 255).byte()

    return Image.fromarray(tensor.numpy())


# ============================================================
# Allowed image extensions
# ============================================================

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


def allowed_file(filename):

    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ============================================================
# Home
# ============================================================


@app.route("/")
def index():

    # Newest first

    generated_maps = NormalMap.query.order_by(NormalMap.created_at.desc()).all()

    result = request.args.get("result")

    return render_template("index.html", generated_maps=generated_maps, result=result)


# ============================================================
# Generate Normal Map
# ============================================================


@app.route("/generate", methods=["POST"])
def generate():

    # Check file

    if "texture" not in request.files:
        return "No image uploaded", 400

    file = request.files["texture"]

    if file.filename == "":
        return "No file selected", 400

    if not allowed_file(file.filename):
        return ("Invalid image format. Allowed: PNG, JPG, JPEG, WEBP", 400)

    # ========================================================
    # Generate unique UUID
    # ========================================================

    image_id = str(uuid.uuid4())

    # Keep original extension

    original_filename = secure_filename(file.filename)

    extension = original_filename.rsplit(".", 1)[1].lower()

    # ========================================================
    # File names
    # ========================================================

    texture_filename = f"{image_id}.{extension}"

    normal_filename = f"{image_id}.png"

    texture_path = os.path.join(UPLOAD_FOLDER, texture_filename)

    normal_path = os.path.join(GENERATED_FOLDER, normal_filename)

    # ========================================================
    # Save uploaded texture
    # ========================================================

    file.save(texture_path)

    # ========================================================
    # Open image
    # ========================================================

    image = Image.open(texture_path)

    # ========================================================
    # Preprocess
    # ========================================================

    x = preprocess_image(image)

    # ========================================================
    # Generate normal map
    # ========================================================

    with torch.no_grad():
        fake_normal = G(x)

    # ========================================================
    # Convert tensor -> image
    # ========================================================

    normal_image = tensor_to_image(fake_normal)

    # ========================================================
    # Save generated normal map
    # ========================================================

    normal_image.save(normal_path)

    # ========================================================
    # URLs
    # ========================================================

    texture_url = f"/static/uploads/{texture_filename}"

    normal_map_url = f"/static/generated/{normal_filename}"

    # ========================================================
    # Save information in database
    # ========================================================

    record = NormalMap(
        id=image_id, texture_url=texture_url, normal_map_url=normal_map_url
    )

    db.session.add(record)

    db.session.commit()

    # ========================================================
    # Return homepage
    # ========================================================

    # Return homepage
    return redirect(url_for("index", result=normal_map_url))


# ============================================================
# Initialize database
# ============================================================

with app.app_context():
    db.create_all()


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)
