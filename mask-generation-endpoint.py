import base64
import io
from pathlib import Path
from pydantic import BaseModel

import modal
from fastapi import FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles


model_repo_id = "facebook/sam2??"


app = modal.App("sam2-class-test-example")
image = (
    modal.Image.debian_slim()
    .apt_install("git")  # Install git before pip installs
    .pip_install(
        "huggingface-hub",
        "Pillow",
        "opencv-python",
        "timm",
        "transformers",
        "torchvision",
        "torch",
        "git+https://github.com/facebookresearch/segment-anything-2.git"
    )
    .apt_install("fonts-freefont-ttf","libxext6","libsm6","ffmpeg")
)

with image.imports():
    import torch
    from huggingface_hub import snapshot_download
    from PIL import Image, ImageColor, ImageDraw, ImageFont
    from transformers import DetrForObjectDetection, DetrImageProcessor
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    import numpy as np
    import base64
    from io import BytesIO
    


@app.cls(
    cpu=1,
    image=image,
    gpu="any",
)
class ObjectDetection:
    @modal.build()
    def download_model(self):
        pass
        # self.predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")


    @modal.enter()
    def load_model(self):
        self.predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
        
    def generate_mask_image(self, mask, borders=True):
        import cv2
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        
        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        if borders:
            contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
        return mask_image
    
    @modal.method()
    def detect(self, img_data, positive_coordinates):
        print(self.predictor)
        print(len(img_data))
       

        header, encoded = img_data.split(",", 1)  # Split the header from the base64 data
        image_data = base64.b64decode(encoded)  # Decode the base64 data
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image = np.array(image)
        self.predictor.set_image(image)
        input_point = np.array(positive_coordinates)
        input_label = np.array([1])
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]
        print("----Masks----")
        print(masks)
        print("----Scores----")
        print(scores)
        print("----Logits----")
        print(logits)
        image = self.generate_mask_image(masks[0])
        
        buffered = BytesIO()
        Image.fromarray((image * 255).astype(np.uint8)).save(buffered, format="PNG")
        b64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")        
        return {
            "mask": b64_image,
            # "mask": masks[0].tolist(),  # Convert numpy array to list
            # "score": scores[0].item()    # Convert numpy scalar to Python float
        }
     
    @modal.method()
    def generate_3d_video(self, email, title, description, frames, selected_frame_idx, selected_point):
        print(email)
        print(title)
        print(description)
        print(frames)
        print(selected_frame_idx)
        print(selected_point)
        return "HELLO"
    
class ItemToMask(BaseModel):
    image: str
    positive_coordinates: list[tuple[int, int]] = [(0, 0)]
    dimensions: tuple[int, int] = (0, 0)

    
@app.function()
@modal.web_endpoint(
    docs=True,
    method="POST"
)
def fetch_image_mask(body: ItemToMask) -> str:
    image = body.image
    positive_coords = body.positive_coordinates
    dimensions = body.dimensions
    # Run SAM 2 Mask Segmentation
    result = ObjectDetection().detect.remote(
        image,
        positive_coords
        )
    return result




class Generate3DVideoInput(BaseModel):
    email: str
    title: str
    description: str
    frames: list[str]
    selected_frame_idx: int
    selected_point: tuple[int, int]
    
    
@app.function()
@modal.web_endpoint(
    docs=True,
    method="POST"
)
def generate_3d_video(body: Generate3DVideoInput) -> str:
    email = body.email
    title = body.title
    description = body.description
    frames = body.frames
    selected_frame_idx = body.selected_frame_idx
    selected_point = body.selected_point
    
    return "HELLO"