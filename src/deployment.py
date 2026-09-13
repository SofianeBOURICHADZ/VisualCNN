"""
Model Export & Deployment Toolkit
=================================
Bundles trained models into a self-contained deployment package:
- config.json: metadata, input specs, normalization parameters, and class names
- weights.pth: PyTorch model weights (state_dict)
- model.py: standalone nn.Module class code ready for production
- predict.py: plug-and-play CLI prediction script
Also provides ModelDeployer for in-app live testing and LocalAPIServer runner.
"""

from __future__ import annotations
from email.parser import BytesParser
from email.policy import default
from http.server import HTTPServer, BaseHTTPRequestHandler
import io
import json
import os
import socketserver
import sys
import threading
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from src.models import ModelDesign, generate_model_code


# ---------------------------------------------------------------------------
# Prediction & Server script templates
# ---------------------------------------------------------------------------

PREDICT_SCRIPT_TEMPLATE = '''"""
Standalone Image Inference Script
=================================
Usage:
    python predict.py --image path/to/sample.jpg
"""

import argparse
import json
import os
import sys
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Import model architecture from local model.py
from model import AutomateCNNModel

def load_deployed_model(model_dir: str = BASE_DIR):
    config_path = os.path.join(model_dir, "config.json")
    weights_path = os.path.join(model_dir, "weights.pth")
    
    with open(config_path, "r") as f:
        config = json.load(f)
        
    model = AutomateCNNModel(
        in_channels=config["in_channels"],
        num_classes=config["num_classes"]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model, config, device

def predict(image_path: str, model_dir: str = BASE_DIR):
    model, config, device = load_deployed_model(model_dir)
    
    in_ch = config["in_channels"]
    in_h = config["input_height"]
    in_w = config["input_width"]
    norm = config.get("normalization", {})
    mean = norm.get("mean", [0.485, 0.456, 0.406] if in_ch == 3 else [0.5])
    std = norm.get("std", [0.229, 0.224, 0.225] if in_ch == 3 else [0.5])
    
    transform = transforms.Compose([
        transforms.Resize((in_h, in_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    img = Image.open(image_path)
    img = img.convert("L" if in_ch == 1 else "RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        
    classes = config["classes"]
    pred_idx = int(probs.argmax())
    pred_class = classes[pred_idx]
    confidence = float(probs[pred_idx])
    
    ranked = sorted([(classes[i], float(probs[i])) for i in range(len(classes))], key=lambda x: x[1], reverse=True)
    return pred_class, confidence, ranked

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify image using deployed model.")
    parser.add_argument("image_pos", nargs="?", default=None, help="Path to input image (positional)")
    parser.add_argument("--image", "-i", default=None, help="Path to input image")
    args = parser.parse_args()
    img_path = args.image or args.image_pos
    if not img_path:
        parser.error("Please specify an image path: python predict.py <image_path> or --image <image_path>")
    
    pred_cls, conf, ranked = predict(img_path)
    print(f"\\nResult: {pred_cls} ({conf*100:.2f}% confidence)")
    print("-" * 35)
    for cls_name, p in ranked[:5]:
        bar = "█" * int(p * 20)
        print(f"{cls_name:15s}: {p*100:5.1f}% {bar}")
'''

API_SERVER_SCRIPT_TEMPLATE = '''"""
Standalone REST API Inference Server
====================================
Zero-dependency HTTP server using Python standard library.

Usage:
    python api_server.py --port 8000

Endpoints:
    GET  /health
    GET  /classes
    POST /predict (Upload image file with form-data field 'image' or 'file')
"""

import argparse
from email.parser import BytesParser
from email.policy import default
import io
import json
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from model import AutomateCNNModel

CONFIG = None
MODEL = None
DEVICE = None
TRANSFORM = None

def init_server(model_dir: str = BASE_DIR):
    global CONFIG, MODEL, DEVICE, TRANSFORM
    config_path = os.path.join(model_dir, "config.json")
    weights_path = os.path.join(model_dir, "weights.pth")
    with open(config_path, "r") as f:
        CONFIG = json.load(f)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL = AutomateCNNModel(
        in_channels=CONFIG["in_channels"],
        num_classes=CONFIG["num_classes"]
    )
    MODEL.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    MODEL.to(DEVICE)
    MODEL.eval()
    
    in_ch = CONFIG["in_channels"]
    in_h = CONFIG["input_height"]
    in_w = CONFIG["input_width"]
    norm = CONFIG.get("normalization", {})
    mean = norm.get("mean", [0.485, 0.456, 0.406] if in_ch == 3 else [0.5])
    std = norm.get("std", [0.229, 0.224, 0.225] if in_ch == 3 else [0.5])
    
    TRANSFORM = transforms.Compose([
        transforms.Resize((in_h, in_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

class InferenceHandler(BaseHTTPRequestHandler):
    def _send_json(self, data, status=200):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/health"):
            self._send_json({"status": "healthy", "model": CONFIG.get("model_name", "AutomateCNNModel")})
        elif self.path == "/classes":
            self._send_json({"num_classes": CONFIG["num_classes"], "classes": CONFIG["classes"]})
        else:
            self._send_json({"error": "Not found"}, status=404)

    def do_POST(self):
        if self.path == "/predict":
            try:
                ctype = self.headers.get("Content-Type", "")
                content_len = int(self.headers.get("Content-Length", 0))
                raw_data = self.rfile.read(content_len) if content_len > 0 else b""
                
                file_data = None
                if "multipart/form-data" in ctype:
                    msg = BytesParser(policy=default).parsebytes(
                        f"Content-Type: {ctype}\\r\\n\\r\\n".encode("utf-8") + raw_data
                    )
                    for part in msg.iter_parts():
                        cd = part.get("Content-Disposition", "")
                        if "filename=" in cd or part.get_content_type().startswith("image/"):
                            file_data = part.get_payload(decode=True)
                            break
                    if not file_data:
                        for part in msg.iter_parts():
                            payload = part.get_payload(decode=True)
                            if payload:
                                file_data = payload
                                break
                else:
                    file_data = raw_data

                if not file_data:
                    self._send_json({"error": "No image uploaded"}, status=400)
                    return

                img = Image.open(io.BytesIO(file_data))

                in_ch = CONFIG["in_channels"]
                img = img.convert("L" if in_ch == 1 else "RGB")
                tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    logits = MODEL(tensor)
                    probs = F.softmax(logits, dim=1)[0].cpu().numpy()

                classes = CONFIG["classes"]
                pred_idx = int(probs.argmax())
                pred_class = classes[pred_idx]
                conf = float(probs[pred_idx])

                resp = {
                    "prediction": pred_class,
                    "confidence": round(conf, 4),
                    "confidence_pct": f"{conf*100:.2f}%",
                    "probabilities": {classes[i]: round(float(probs[i]), 4) for i in range(len(classes))}
                }
                self._send_json(resp)
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
        else:
            self._send_json({"error": "Unknown endpoint. Use POST /predict"}, status=404)

def run(port: int = 8000):
    init_server()
    server = HTTPServer(("0.0.0.0", port), InferenceHandler)
    print(f"Server started on http://0.0.0.0:{port}")
    print(f"API Endpoint: POST http://localhost:{port}/predict")
    server.serve_forever()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start REST inference server.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    args = parser.parse_args()
    run(args.port)
'''


# ---------------------------------------------------------------------------
# Package Exporter Function
# ---------------------------------------------------------------------------

def export_model_package(
    design: ModelDesign,
    model_state_dict: Dict[str, Any],
    classes: List[str],
    export_dir: str,
    metric_info: Optional[Dict[str, Any]] = None,
    model_name: str = "AutomateCNNModel"
) -> Dict[str, str]:
    """
    Creates a standalone, self-contained deployment folder containing:
    1. config.json
    2. weights.pth
    3. model.py
    4. predict.py
    5. api_server.py
    """
    export_dir = os.path.abspath(export_dir)
    os.makedirs(export_dir, exist_ok=True)

    in_ch = design.in_channels
    mean = [0.485, 0.456, 0.406] if in_ch == 3 else [0.5]
    std = [0.229, 0.224, 0.225] if in_ch == 3 else [0.5]

    config_data = {
        "model_name": model_name,
        "in_channels": design.in_channels,
        "input_height": design.input_height,
        "input_width": design.input_width,
        "num_classes": len(classes),
        "classes": classes,
        "class_to_idx": {c: i for i, c in enumerate(classes)},
        "normalization": {
            "mean": mean,
            "std": std
        },
        "flatten_mode": design.flatten_mode,
        "metric_info": metric_info or {},
        "export_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "architecture_summary": {
            "num_backbone_layers": len(design.backbone),
            "num_head_layers": len(design.head),
        }
    }

    # 1. config.json
    config_path = os.path.join(export_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)

    # 2. class_map.json — human-readable class name <-> integer index table
    class_map = {cls: idx for idx, cls in enumerate(classes)}
    class_map_path = os.path.join(export_dir, "class_map.json")
    with open(class_map_path, "w") as f:
        json.dump(
            {
                "description": "Class name to integer index mapping used during training.",
                "class_to_idx": class_map,
                "idx_to_class": {str(idx): cls for cls, idx in class_map.items()},
            },
            f,
            indent=2,
        )

    # 3. best_model.pth — weights (state_dict) saved on CPU for portability
    best_model_path = os.path.join(export_dir, "best_model.pth")
    cpu_state_dict = {k: v.cpu() for k, v in model_state_dict.items()}
    torch.save(cpu_state_dict, best_model_path)
    weights_path = best_model_path

    # 4. model.py
    model_py_path = os.path.join(export_dir, "model.py")
    model_code = generate_model_code(design, class_name=model_name)
    with open(model_py_path, "w") as f:
        f.write(model_code)

    # 5. predict.py
    predict_py_path = os.path.join(export_dir, "predict.py")
    with open(predict_py_path, "w") as f:
        f.write(PREDICT_SCRIPT_TEMPLATE)

    # 6. api_server.py
    api_server_path = os.path.join(export_dir, "api_server.py")
    with open(api_server_path, "w") as f:
        f.write(API_SERVER_SCRIPT_TEMPLATE)

    # Make scripts executable
    try:
        os.chmod(predict_py_path, 0o755)
        os.chmod(api_server_path, 0o755)
    except Exception:
        pass

    return {
        "export_dir": export_dir,
        "config_path": config_path,
        "class_map_path": class_map_path,
        "weights_path": weights_path,
        "model_py_path": model_py_path,
        "predict_py_path": predict_py_path,
        "api_server_path": api_server_path,
    }


# ---------------------------------------------------------------------------
# In-App Live Model Deployer / Predictor
# ---------------------------------------------------------------------------

class ModelDeployer:
    """
    Loads an exported model package to perform live predictions and testing.
    """
    def __init__(self, export_dir: str, device: str = "cpu"):
        self.export_dir = os.path.abspath(export_dir)
        self.config_path = os.path.join(self.export_dir, "config.json")
        # Support both new (best_model.pth) and legacy (weights.pth) names
        best_path = os.path.join(self.export_dir, "best_model.pth")
        legacy_path = os.path.join(self.export_dir, "weights.pth")
        if os.path.isfile(best_path):
            self.weights_path = best_path
        else:
            self.weights_path = legacy_path
        self.model_path = os.path.join(self.export_dir, "model.py")

        if not os.path.isfile(self.config_path):
            raise FileNotFoundError(f"Missing config.json in {self.export_dir}")
        if not os.path.isfile(self.weights_path):
            raise FileNotFoundError(f"Missing best_model.pth / weights.pth in {self.export_dir}")
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Missing model.py in {self.export_dir}")

        with open(self.config_path, "r") as f:
            self.config = json.load(f)

        self.device = torch.device(device)
        self.classes = self.config["classes"]
        self.in_channels = self.config["in_channels"]
        self.input_height = self.config["input_height"]
        self.input_width = self.config["input_width"]

        # Dynamically load model class from model.py
        with open(self.model_path, "r") as f:
            code = f.read()
        namespace: Dict[str, Any] = {}
        exec(compile(code, "<deployed_model>", "exec"), namespace)
        model_cls_name = self.config.get("model_name", "AutomateCNNModel")
        ModelClass = namespace[model_cls_name]

        self.model = ModelClass(
            in_channels=self.in_channels,
            num_classes=len(self.classes)
        )
        self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        norm = self.config.get("normalization", {})
        mean = norm.get("mean", [0.485, 0.456, 0.406] if self.in_channels == 3 else [0.5])
        std = norm.get("std", [0.229, 0.224, 0.225] if self.in_channels == 3 else [0.5])

        self.transform = transforms.Compose([
            transforms.Resize((self.input_height, self.input_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def predict(self, image_input: Union[str, Image.Image]) -> Dict[str, Any]:
        """
        Run inference on image path or PIL Image.
        Returns:
            {
                'predicted_class': str,
                'confidence': float,
                'confidence_pct': str,
                'ranked_probabilities': [(class_name, prob), ...],
                'probabilities': {class_name: prob, ...}
            }
        """
        if isinstance(image_input, str):
            img = Image.open(image_input)
        else:
            img = image_input

        img = img.convert("L" if self.in_channels == 1 else "RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()

        pred_idx = int(probs.argmax())
        pred_cls = self.classes[pred_idx]
        conf = float(probs[pred_idx])

        ranked = sorted(
            [(self.classes[i], float(probs[i])) for i in range(len(self.classes))],
            key=lambda x: x[1],
            reverse=True
        )

        return {
            "prediction": pred_cls,
            "predicted_class": pred_cls,
            "confidence": conf,
            "confidence_pct": f"{conf*100:.2f}%",
            "ranked_probabilities": ranked,
            "probabilities": {self.classes[i]: float(probs[i]) for i in range(len(self.classes))},
        }


# ---------------------------------------------------------------------------
# Background API Server Runner
# ---------------------------------------------------------------------------

class LocalAPIServer:
    """Controller to start/stop the API server in a background thread."""
    def __init__(self, export_dir: str, port: int = 8000):
        self.export_dir = export_dir
        self.port = port
        self.server: Optional[HTTPServer] = None
        self.thread: Optional[threading.Thread] = None
        self.is_running = False

    def start(self):
        if self.is_running:
            return

        deployer = ModelDeployer(self.export_dir)

        class EmbeddedHandler(BaseHTTPRequestHandler):
            def _send_json(s, data, status=200):
                body = json.dumps(data, indent=2).encode("utf-8")
                s.send_response(status)
                s.send_header("Content-Type", "application/json")
                s.send_header("Content-Length", str(len(body)))
                s.send_header("Access-Control-Allow-Origin", "*")
                s.end_headers()
                s.wfile.write(body)

            def do_GET(s):
                if s.path in ("/", "/health"):
                    s._send_json({"status": "running", "classes": deployer.classes})
                elif s.path == "/classes":
                    s._send_json({"classes": deployer.classes})
                else:
                    s._send_json({"error": "Not found"}, status=404)

            def do_POST(s):
                if s.path == "/predict":
                    try:
                        ctype = s.headers.get("Content-Type", "")
                        content_len = int(s.headers.get("Content-Length", 0))
                        raw_data = s.rfile.read(content_len) if content_len > 0 else b""
                        
                        file_data = None
                        if "multipart/form-data" in ctype:
                            msg = BytesParser(policy=default).parsebytes(
                                f"Content-Type: {ctype}\r\n\r\n".encode("utf-8") + raw_data
                            )
                            for part in msg.iter_parts():
                                cd = part.get("Content-Disposition", "")
                                if "filename=" in cd or part.get_content_type().startswith("image/"):
                                    file_data = part.get_payload(decode=True)
                                    break
                            if not file_data:
                                for part in msg.iter_parts():
                                    payload = part.get_payload(decode=True)
                                    if payload:
                                        file_data = payload
                                        break
                        else:
                            file_data = raw_data

                        if not file_data:
                            s._send_json({"error": "No image provided"}, status=400)
                            return

                        img = Image.open(io.BytesIO(file_data))
                        result = deployer.predict(img)
                        s._send_json(result)
                    except Exception as e:
                        s._send_json({"error": str(e)}, status=500)
                else:
                    s._send_json({"error": "Use POST /predict"}, status=404)

            def log_message(s, format, *args):
                pass  # Suppress default noisy console logs

        import io
        self.server = HTTPServer(("0.0.0.0", self.port), EmbeddedHandler)
        self.is_running = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self):
        if self.is_running and self.server:
            self.server.shutdown()
            self.server.server_close()
            self.is_running = False
            self.server = None
