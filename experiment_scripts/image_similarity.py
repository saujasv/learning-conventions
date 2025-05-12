import torch
from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image
from pathlib import Path
from tqdm import tqdm
from accelerate import find_executable_batch_size
import itertools
import numpy as np
import json


@find_executable_batch_size(starting_batch_size=1024)
def embed_images(batch_size, model_name_or_path, images_path, save_path):
    model = AutoModel.from_pretrained(model_name_or_path, device_map="auto").eval()
    processor = AutoProcessor.from_pretrained(model_name_or_path)

    image_names = list()
    embeddings = list()
    with torch.no_grad():
        for i, batch in tqdm(
            enumerate(itertools.batched(Path(images_path).glob("*.jpg"), n=batch_size))
        ):
            images = [load_image(str(img)) for img in batch]
            inputs = processor(images=images, return_tensors="pt").to(model.device)
            image_embedding = model.get_image_features(**inputs)

            embeddings.extend(image_embedding.cpu())
            image_names.extend([img.stem for img in batch])

    # Save the embeddings
    embeddings_array = torch.stack(embeddings).numpy()

    Path(save_path).mkdir(parents=True, exist_ok=True)
    np.save(Path(save_path) / "embeddings.npy", embeddings_array)
    with open(Path(save_path) / "image_names.json", "w") as f:
        json.dump(image_names, f)


@find_executable_batch_size(starting_batch_size=2048)
def find_similarity(batch_size, embeddings_path):
    embeddings = torch.from_numpy(np.load(Path(embeddings_path) / "embeddings.npy")).to(
        "cuda"
    )

    similarity_matrix = torch.zeros(
        (embeddings.shape[0], embeddings.shape[0]), device="cpu"
    )
    for start1 in tqdm(range(0, embeddings.shape[0], batch_size)):
        end1 = min(start1 + batch_size, embeddings.shape[0])
        batch1 = embeddings[start1:end1]
        for start2 in tqdm(range(0, embeddings.shape[0], batch_size)):
            end2 = min(start2 + batch_size, embeddings.shape[0])
            batch2 = embeddings[start2:end2]
            similarity = torch.nn.functional.cosine_similarity(
                batch1[:, None, :], batch2[None, :, :], dim=-1
            )
            similarity_matrix[start1:end1, start2:end2] = similarity.cpu()

    np.save(Path(embeddings_path) / "similarity_matrix.npy", similarity_matrix.numpy())
