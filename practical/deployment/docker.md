# Docker

Docker provides an easy way to package and deploy applications with all dependencies included. This section includes steps for creating a Docker repository, building and pushing images, and using others’ Docker images.

## Project Setup

1. **Create Project Directory**:
    ```bash
    mkdir docker_app
    cd docker_app
    ```
2. **Create Files**:
   * `main.py`: Script to run your model.
   * `Dockerfile`: Instructions for building the Docker image.

## Code for Dockerized Application

### main.py
```python
import argparse
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import requests

def run_inference(image_url, captions):
    image = Image.open(requests.get(image_url, stream=True).raw)
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    inputs = processor(text=captions.split(','), images=image, return_tensors="pt")
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    print(probs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_url", type=str, help="URL of the image")
    parser.add_argument("captions", type=str, help="Comma-separated captions")
    args = parser.parse_args()
    run_inference(args.image_url, args.captions)
```

### Dockerfile

```Dockerfile
FROM python:3.9
WORKDIR /app
COPY main.py .
RUN pip install transformers pillow requests torch torchvision
ENTRYPOINT ["python", "main.py"]
Build and Run the Docker Image Locally
```
## Build and Run the Docker Image Locally

1. **Build the Docker image**:
    ```bash
    docker build -t clip_demo .
    ```

2. **Run the Docker container**:
   ```bash
    docker run clip_demo <image_url> "<comma_separated_captions>"
    ```
    Replace <image_url> with the URL of an image and <comma_separated_captions> with a comma-separated list of captions to compare.

3. **Create a Docker Repository and Push Image**

    Log in to Docker Hub:
    ```bash
    docker login
    ```
   Enter your Docker Hub username and password.

4. **Tag Your Image**:
    Before pushing, tag the image with your Docker Hub username and repository name:
    ```bash
    docker tag clip_demo your_dockerhub_username/clip-demo:v1
    ```
    Push the Image to Docker Hub:
    ```bash
    docker push your_dockerhub_username/clip-demo:v1
    ```
5. **Verify the Image**:
   * Go to your Docker Hub account and confirm that the image appears in the repository.

## Use Other People’s Docker Images

Docker Hub hosts pre-built images from other developers. To use one:

1. **Search for the Image**:
   Visit Docker Hub and search for the desired image.

2. **Pull the Image**:
   Use the `docker pull` command to download the image:
   ```bash
    docker pull username/image_name:tag
    ```
    Replace username/image_name with the full image name and tag with the version tag (e.g., latest).

3. **Run the Image: Once pulled, you can run the container**:
    ```bash
    docker run username/image_name:tag
    ```
4. **Inspect the Image (Optional)**:
    To learn more about the image:
    ```bash
    docker inspect username/image_name:tag
    ```
    Example: Using an Existing Docker Image


**To run a pre-built Hugging Face image:**
```bash
docker pull huggingface/transformers-pytorch-gpu:latest
docker run -it --gpus all huggingface/transformers-pytorch-gpu:latest
```