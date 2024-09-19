import base64
from dataclasses import dataclass
from io import BytesIO

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Polygon
from matplotlib.path import Path
from openai import OpenAI
from PIL import Image


@dataclass
class Constraint:
    points: np.array
    direction_str: str

    @property
    def direction(self):
        direction_dir = {
            "left": [1, 0],
            "right": [1, 0],
            "up": [1, 0],
            "down": [1, 0],
        }
        return np.array(direction_dir[self.direction_str])


def downscale_img(image):
    return image.resize((256, 256)), image.size


def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def make_request(image, condition):
    client = OpenAI()
    text = "You are given an image of an robotic manipulation task. \
            We already have an multimodal policy which can execute the task. \
            The task has one additional constraints which you should define based on the image and some condition. \
            Only use the information given. \
            The constraint consist of a region of the image and a direction. \
            The robot is pushed away from this region in the given direction. \
            You can use four pixel points of the image between (0,0) and (255,255) to define the region. \
            Try to make the region as big as possible. \
            Do not cover up parts of the image where the robots should go. \
            Only output the points and the direction without additional text. \
            Split the points and direction with ; and different constraints with |. \
            Example output: (0,0);(0,255);(255,0);(255,255);left \
            The task and condition is: "

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text + condition},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(image)}",
                        },
                    },
                ],
            }
        ],
        max_tokens=100,
    )

    return response.choices[0].message.content


def extract_data(str):
    ans_str = str.replace(" ", "")
    constraint_str = ans_str.split("|")

    constraints = []
    for c in constraint_str:
        d = c.split(";")

        points = [[int(i) for i in s.replace("(", "").replace(")", "").split(",")] for s in d[:-1]]

        # Sort points based on their angle with respect to the centroid
        centroid = np.mean(points, axis=0)

        def angle_from_centroid(point):
            return np.arctan2(point[1] - centroid[1], point[0] - centroid[0])

        constraints.append(Constraint(np.array(sorted(points, key=angle_from_centroid)), d[-1]))

    return constraints


def create_mask(rect_points, size, direction):
    rect_path = Path(rect_points)

    x, y = np.meshgrid(np.arange(256), np.arange(256))
    points = np.vstack((x.ravel(), y.ravel())).T

    # Boolean mask for 256,256
    bool_mask = rect_path.contains_points(points).reshape(256, 256)

    dir_mask = torch.where(
        torch.tensor(bool_mask).unsqueeze(-1),
        torch.tensor(direction),
        torch.tensor([0, 0]),
    ).to(torch.float32)

    small_mask = einops.rearrange(torch.tensor(dir_mask), "w h d -> 1 d w h")

    # Upscale mask
    interp_mask = torch.nn.Upsample(size=size, mode="nearest")(small_mask)
    interp_mask = einops.rearrange(interp_mask, "1 d w h -> w h d")

    mask = torch.zeros((*size, 3))
    mask[:, :, :2] = interp_mask
    return mask


def show_result(constaints, image):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)

    ax.imshow(image, extent=[0, 255, 0, 255], zorder=0)

    for c in constaints:
        polygon = Polygon(
            c.points, closed=True, edgecolor="r", facecolor="red", alpha=0.5, zorder=1
        )
        ax.add_patch(polygon)

    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()


def make_llm_mask(image, condition, show_mask=False):
    resized_image, size = downscale_img(image)

    # response = "(0,0);(0,128);(128,0);(128,255);right | (128,0);(128,128);(255,0);(255,255);left"
    response = "(0,0);(0,255);(128,0);(128,255);right"
    # response = make_request(resized_image, condition)
    print(response)

    constraints = extract_data(response)

    if show_mask:
        show_result(constraints, resized_image)

    return create_mask(constraints[0].points, size, constraints[0].direction)


def main():
    image_path = "test.png"
    condition = "The robot can pick up either box. Make it pick up the left box."

    print(make_llm_mask(Image.open(image_path), condition))


if __name__ == "__main__":
    main()
