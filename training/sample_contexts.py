from typing import List
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import find_executable_batch_size
from transformers import AutoModel, AutoProcessor
from tqdm import tqdm
import itertools
from PIL import Image


class ImageSimilarityIndex:
    embeddings: torch.Tensor
    image2idx: dict[str, int]
    idx2image: dict[int, str]

    def __init__(
        self,
        embeddings: torch.Tensor,
        image2idx: dict[str, int],
        idx2image: dict[int, str],
    ):
        if not isinstance(embeddings, torch.Tensor):
            raise TypeError("Embeddings must be a PyTorch tensor.")
        if embeddings.dtype != torch.float32:
            # Ensure embeddings are float32 for consistency and downstream torch operations
            self.embeddings = embeddings.float()
        else:
            self.embeddings = embeddings

        if not isinstance(image2idx, dict) or not isinstance(idx2image, dict):
            raise TypeError("image2idx and idx2image must be dictionaries.")

        self.image2idx = image2idx
        self.idx2image = idx2image

        if len(self.image2idx) != self.embeddings.shape[0]:
            raise ValueError(
                "Number of image names in image2idx does not match number of embeddings. "
                f"({len(self.image2idx)} names, {self.embeddings.shape[0]} embeddings)"
            )
        if len(self.idx2image) != self.embeddings.shape[0]:
            raise ValueError(
                "Number of image names in idx2image does not match number of embeddings. "
                f"({len(self.idx2image)} names, {self.embeddings.shape[0]} embeddings)"
            )

    @staticmethod
    @find_executable_batch_size(starting_batch_size=1024)
    def _generate_image_embeddings_and_names(
        batch_size, model_name_or_path, images_path, save_path=None, device=None
    ) -> tuple[np.ndarray, list[str]] | None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model = AutoModel.from_pretrained(model_name_or_path, device_map=device).eval()
        processor = AutoProcessor.from_pretrained(model_name_or_path)

        image_names = list()
        embeddings_list = (
            list()
        )  # Changed from 'embeddings' to 'embeddings_list' to avoid confusion
        with torch.no_grad():
            image_file_iterator = Path(images_path).glob("*")
            for i, batch_of_paths in tqdm(
                enumerate(itertools.batched(image_file_iterator, n=batch_size))
            ):
                loaded_images_data = [
                    Image.open(str(img_path)).convert("RGB")
                    for img_path in batch_of_paths
                ]
                if not loaded_images_data:
                    continue

                inputs = processor(images=loaded_images_data, return_tensors="pt").to(
                    model.device
                )
                image_embedding_batch = model.get_image_features(**inputs)

                embeddings_list.extend(image_embedding_batch.cpu())
                image_names.extend([img_path.name for img_path in batch_of_paths])

        if not embeddings_list:
            final_embeddings_array = np.array([], dtype=np.float32)
            final_image_names_list = image_names  # could be empty
        else:
            final_embeddings_array = torch.stack(embeddings_list).cpu().numpy()
            final_image_names_list = image_names

        if save_path is not None:
            output_file_path = Path(save_path)
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output_file_path,
                embeddings=final_embeddings_array,
                image_names=np.array(final_image_names_list),
            )

        return final_embeddings_array, final_image_names_list

    @classmethod
    def from_images(
        cls,
        model_name_or_path: str,
        images_path: str,
        device: str | None = None,
        save_path: str | None = None,
    ) -> "ImageSimilarityIndex":
        # Call the static method (batch_size is handled by its decorator)
        result = cls._generate_image_embeddings_and_names(
            model_name_or_path=model_name_or_path,
            images_path=images_path,
            save_path=save_path,
            device=device,
        )

        if result is None:
            raise RuntimeError(
                f"Failed to generate embeddings from images at {images_path}. _generate_image_embeddings_and_names returned None."
            )

        embeddings_np, image_names_list = result

        if not isinstance(embeddings_np, np.ndarray) or not isinstance(
            image_names_list, list
        ):
            raise TypeError("Invalid data returned from embedding generation.")

        # Convert numpy array to PyTorch tensor and ensure it's float32
        embeddings_tensor = torch.from_numpy(embeddings_np)
        if embeddings_tensor.dtype != torch.float32:
            embeddings_tensor = embeddings_tensor.float()

        image2idx = {name: i for i, name in enumerate(image_names_list)}
        idx2image = {i: name for i, name in enumerate(image_names_list)}

        return cls(embeddings_tensor, image2idx, idx2image)

    @classmethod
    def from_file(cls, embeddings_path: str) -> "ImageSimilarityIndex":
        """
        Creates an ImageSimilarityIndex instance from a .npz file.

        Args:
            embeddings_path (str): Path to the .npz file containing
                                   'embeddings' and 'image_names'.

        Returns:
            ImageSimilarityIndex: An instance of the class.
        """
        image_data_file = Path(embeddings_path)

        if not image_data_file.is_file():  # Check if it's a file
            raise FileNotFoundError(
                f"Embeddings file '{image_data_file}' not found or is not a file."
            )

        try:
            with np.load(image_data_file, allow_pickle=True) as data:
                if "embeddings" not in data or "image_names" not in data:
                    raise ValueError(
                        f"'{image_data_file}' must contain 'embeddings' and 'image_names' keys."
                    )

                embeddings_np = data["embeddings"]
                # Ensure image_names are loaded as a list of strings
                image_names_raw = data["image_names"]
                if isinstance(image_names_raw, np.ndarray):
                    image_names_list = image_names_raw.tolist()
                elif isinstance(image_names_raw, list):
                    image_names_list = list(
                        image_names_raw
                    )  # ensure it's a proper list
                else:
                    raise TypeError(
                        f"Unsupported type for image_names in {image_data_file}: {type(image_names_raw)}"
                    )

                if not all(isinstance(name, str) for name in image_names_list):
                    raise ValueError("All image names must be strings.")

        except Exception as e:
            raise IOError(
                f"Error loading or processing data from '{image_data_file}': {e}"
            )

        # Ensure embeddings are float and convert to PyTorch tensor
        if not np.issubdtype(embeddings_np.dtype, np.floating):
            embeddings_np = embeddings_np.astype(np.float32)

        embeddings_tensor = torch.from_numpy(embeddings_np)
        # Ensure it's float32, __init__ will also check but good to be consistent
        if embeddings_tensor.dtype != torch.float32:
            embeddings_tensor = embeddings_tensor.float()

        image2idx = {img: i for i, img in enumerate(image_names_list)}
        idx2image = {i: img for i, img in enumerate(image_names_list)}

        return cls(embeddings_tensor, image2idx, idx2image)

    def _calculate_all_similarities_to_image(self, image_name: str) -> np.ndarray:
        """
        Computes cosine similarities from a given image to all images in the dataset
        using torch.nn.functional.cosine_similarity.
        Returns a NumPy array.
        """
        if image_name not in self.image2idx:
            raise ValueError(f"Image '{image_name}' not found in index.")

        query_idx = self.image2idx[image_name]
        query_emb_tensor = self.embeddings[query_idx]  # This is a 1D tensor (D,)

        # F.cosine_similarity between a 1D tensor (query_emb_tensor) and a 2D tensor (self.embeddings)
        # query_emb_tensor needs to be unsqueezed to (1, D) to broadcast with self.embeddings (N, D)
        # dim=1 specifies that similarity is computed along the feature dimension.
        similarities_tensor = F.cosine_similarity(
            query_emb_tensor.unsqueeze(0), self.embeddings, dim=1
        )

        return similarities_tensor.cpu()

    def __getitem__(self, key: tuple[str, str]) -> float:
        """
        Returns the similarity between two images using torch.nn.functional.cosine_similarity.

        Args:
            key (tuple[str, str]): A tuple containing two image names.

        Returns:
            float: The cosine similarity between the two images.
        """
        img_name1, img_name2 = key
        if img_name1 not in self.image2idx:
            raise ValueError(f"Image '{img_name1}' not found in index.")
        if img_name2 not in self.image2idx:
            raise ValueError(f"Image '{img_name2}' not found in index.")

        idx1 = self.image2idx[img_name1]
        idx2 = self.image2idx[img_name2]

        emb1_tensor = self.embeddings[idx1]  # 1D PyTorch tensor
        emb2_tensor = self.embeddings[idx2]  # 1D PyTorch tensor

        # F.cosine_similarity for two 1D tensors (shape [D]), use dim=0.
        similarity_tensor = F.cosine_similarity(emb1_tensor, emb2_tensor, dim=0)
        return (
            similarity_tensor.item()
        )  # .item() converts a scalar tensor to a Python number

    def get_top_k_similar_images(self, img: str, k: int) -> list[str]:
        """
        Gets the top k most similar images to the given image, excluding itself.
        The returned list is ordered from (k+1)th most similar to 2nd most similar.
        """
        if k <= 0:
            return []
        if (
            img not in self.image2idx
        ):  # Check added for robustness, though _calculate_all handles it
            raise ValueError(f"Image '{img}' not found in index.")

        similarities = self._calculate_all_similarities_to_image(img)

        sorted_indices = np.argsort(similarities.numpy())

        num_images = self.embeddings.shape[0]
        num_other_images = num_images - 1

        if num_other_images <= 0:
            return []

        effective_k = min(k, num_other_images)
        if effective_k == 0:
            return []

        top_k_other_indices = sorted_indices[-(effective_k + 1) : -1]

        return [self.idx2image[i] for i in top_k_other_indices]

    def sample_similar_images(
        self, img: str, k: int, temperature: float = 0.05
    ) -> list[str]:
        """
        Samples k images based on their similarity to the given image.
        """
        if k <= 0:
            return []
        if temperature <= 0:
            raise ValueError("Temperature must be positive.")
        if img not in self.image2idx:  # Check added for robustness
            raise ValueError(f"Image '{img}' not found in index.")

        query_img_idx = self.image2idx[img]
        similarities = self._calculate_all_similarities_to_image(img)

        all_indices = np.arange(self.embeddings.shape[0])
        mask_not_self = all_indices != query_img_idx

        sampleable_indices = all_indices[mask_not_self]
        sampleable_similarities = similarities[mask_not_self]

        if sampleable_indices.size == 0:
            return []

        actual_k = min(k, sampleable_indices.size)
        if actual_k == 0:
            return []

        probabilities = F.softmax(sampleable_similarities / temperature, dim=-1).numpy()

        sampled_relative_indices = np.random.choice(
            sampleable_indices.size,
            p=probabilities,
            size=actual_k,
            replace=False,
        )
        sampled_chosen_indices = sampleable_indices[sampled_relative_indices]

        return [self.idx2image[i] for i in sampled_chosen_indices]

    def sample_similar_image_set(self, k: int, temperature: float = 0.05) -> list[str]:
        """
        Samples a random image and k-1 images similar to it.
        """
        if k <= 0:
            return []

        num_total_images = self.embeddings.shape[0]
        if num_total_images == 0:
            return []

        random_idx = np.random.choice(num_total_images)
        random_image_name = self.idx2image[random_idx]

        if k == 1:
            return [random_image_name]

        similar_images = self.sample_similar_images(
            random_image_name, k - 1, temperature
        )

        return [random_image_name, *similar_images]


def build_index(model_name_or_path: str, images_path: str, device: str, save_path: str):
    ImageSimilarityIndex.from_images(
        model_name_or_path=model_name_or_path,
        images_path=images_path,
        device=device,
        save_path=save_path,
    )


def sample_contexts(
    index_path: str,
    save_path: str,
    context_size: int,
    num_contexts: int,
    temperature: float = 0.05,
    prefix: str = "",
):
    import uuid
    import json

    index = ImageSimilarityIndex.from_file(index_path)
    sampled_contexts = dict()
    for i in range(num_contexts):
        context = index.sample_similar_image_set(
            k=context_size, temperature=temperature
        )
        sampled_contexts[str(uuid.uuid4())] = [f"{prefix}{img}" for img in context]

    with open(save_path, "w") as f:
        json.dump(sampled_contexts, f)
