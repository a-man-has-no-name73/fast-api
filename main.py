# main.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import io
from PIL import Image
import torchvision.transforms as T

# --- 1) Model definitions matching your training script exactly ---

class Involution2d(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, reduction_ratio=4):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.o = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1),
            nn.BatchNorm2d(in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, kernel_size * kernel_size, 1)
        )
        self.unfold = nn.Unfold(kernel_size, padding=(kernel_size - 1)//2, stride=stride)

    def forward(self, x):
        kernel = self.o(x)
        b, c, h, w = x.shape
        x_unf = self.unfold(x).view(b, self.in_channels, self.kernel_size*self.kernel_size, h, w)
        kernel = kernel.view(b, 1, self.kernel_size*self.kernel_size, h, w)
        return (kernel * x_unf).sum(dim=2)

class ASDNet(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.involution_block = nn.Sequential(
            Involution2d(input_channels, 3, reduction_ratio=2),
            Involution2d(input_channels, 3, reduction_ratio=2),
            Involution2d(input_channels, 3, reduction_ratio=2),
            nn.ReLU()
        )
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.flatten = nn.Flatten()
        # Renamed to match checkpoint keys
        self.dense_block = nn.Sequential(
            nn.Linear(128*4*4, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.involution_block(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.flatten(x)
        return self.dense_block(x)


# --- 2) FastAPI setup and model loading ---

app = FastAPI(title="ASDNet Inference API")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ASDNet().to(device)
model.load_state_dict(torch.load("asdnet_v2_best_Dataset2.pth", map_location=device))
model.eval()

# --- 3) Preprocessing pipeline ---

preprocess = T.Compose([
    T.Resize((48, 48)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# --- 4) Inference endpoint ---

@app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are accepted")

    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    input_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
        pred_class = int(torch.argmax(logits, dim=1).item())

    return JSONResponse({
        "filename": image.filename,
        "prediction": pred_class,
        "probabilities": {
            "class_0": probs[0],
            "class_1": probs[1]
        }
    })
