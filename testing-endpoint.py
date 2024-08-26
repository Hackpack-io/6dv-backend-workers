import modal
from pydantic import BaseModel




app = modal.App(name="sam2-mask-segmentation-endpoint")


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
    name = body.image
    positive_coords = body.positive_coordinates
    dimensions = body.dimensions
    # Run SAM 2 Mask Segmentation
    
    return len(name)