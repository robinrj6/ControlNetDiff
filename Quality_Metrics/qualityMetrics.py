#!/usr/bin/env python3
"""
Compute FID (with pytorch-fid) and CLIP score for generated images.

Edit the path variables in the CONFIG section, then run:
	python Quality_Metrics/qualityMetrics.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from PIL import Image
from pytorch_fid.fid_score import calculate_fid_given_paths
from transformers import CLIPModel, CLIPProcessor


# =========================
# CONFIG: EDIT THESE PATHS
# =========================
REAL_IMAGES_DIR = Path("shared/datasets/coco/metricsDataset/images/")
CANNY_IMAGES_DIR = Path("shared/datasets/coco/metricsDataset/edges/")
CONTROLNET_IMAGES_DIR = Path("shared/datasets/coco/metricsDataset/generated_images_ControlNet/")
SD15_IMAGES_DIR = Path("shared/datasets/coco/metricsDataset/generated_images_SD15/")
METADATA_JSONL_PATH = Path("shared/datasets/coco/metricsDataset/metadata.jsonl")


# Optional settings
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
FID_BATCH_SIZE = 32
FID_DIMS = 2048
FID_NUM_WORKERS = 4
CLIP_BATCH_SIZE = 16


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder: Path) -> List[Path]:
	if not folder.exists() or not folder.is_dir():
		raise FileNotFoundError(f"Directory not found: {folder}")
	return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS])


def load_metadata_prompts(metadata_jsonl: Path) -> Dict[str, str]:
	"""
	Build a mapping from image stem -> prompt.
	Supports both:
	  - file_name: images/000000123456.jpg
	  - conditioning_file_name: edges/000000123456.png
	"""
	if not metadata_jsonl.exists():
		raise FileNotFoundError(f"Metadata file not found: {metadata_jsonl}")

	stem_to_prompt: Dict[str, str] = {}
	with metadata_jsonl.open("r", encoding="utf-8") as f:
		for line_idx, line in enumerate(f, start=1):
			line = line.strip()
			if not line:
				continue
			try:
				row = json.loads(line)
			except json.JSONDecodeError as exc:
				raise ValueError(f"Invalid JSON at line {line_idx} in {metadata_jsonl}: {exc}") from exc

			text = row.get("text", "")
			if not isinstance(text, str) or not text.strip():
				continue
			text = text.strip()

			for key in ("file_name", "conditioning_file_name"):
				value = row.get(key)
				if isinstance(value, str) and value.strip():
					stem = Path(value).stem
					stem_to_prompt[stem] = text

	return stem_to_prompt


def fid_score(real_dir: Path, generated_dir: Path, device: torch.device) -> float:
	# pytorch-fid expects folder paths and computes FID over all images in each folder.
	return float(
		calculate_fid_given_paths(
			[str(real_dir), str(generated_dir)],
			batch_size=FID_BATCH_SIZE,
			device=device,
			dims=FID_DIMS,
			num_workers=FID_NUM_WORKERS,
		)
	)


def _batched(items: List[Tuple[Path, str]], batch_size: int) -> Iterable[List[Tuple[Path, str]]]:
	for i in range(0, len(items), batch_size):
		yield items[i : i + batch_size]


@torch.inference_mode()
def clip_score_for_folder(
	image_dir: Path,
	stem_to_prompt: Dict[str, str],
	model: CLIPModel,
	processor: CLIPProcessor,
	device: torch.device,
) -> Tuple[float, int, int]:
	"""
	Returns (mean_clip_score, used_images, skipped_images).

	Score per pair is cosine similarity between normalized image/text embeddings.
	"""
	all_images = list_images(image_dir)

	pairs: List[Tuple[Path, str]] = []
	for img_path in all_images:
		prompt = stem_to_prompt.get(img_path.stem)
		if prompt:
			pairs.append((img_path, prompt))

	if not pairs:
		return 0.0, 0, len(all_images)

	total_score = 0.0
	total_count = 0

	for batch in _batched(pairs, CLIP_BATCH_SIZE):
		pil_images = [Image.open(img_path).convert("RGB") for img_path, _ in batch]
		texts = [prompt for _, prompt in batch]

		inputs = processor(text=texts, images=pil_images, return_tensors="pt", padding=True, truncation=True)
		inputs = {k: v.to(device) for k, v in inputs.items()}

		image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
		text_features = model.get_text_features(
			input_ids=inputs["input_ids"],
			attention_mask=inputs["attention_mask"],
		)

		image_features = image_features / image_features.norm(dim=-1, keepdim=True)
		text_features = text_features / text_features.norm(dim=-1, keepdim=True)

		# cosine per matched pair (diagonal)
		scores = (image_features * text_features).sum(dim=-1)

		total_score += scores.sum().item()
		total_count += scores.numel()

	mean_score = total_score / max(total_count, 1)
	skipped = len(all_images) - len(pairs)
	return float(mean_score), int(total_count), int(skipped)


def main() -> None:
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Validate required dirs/files
	for p in [REAL_IMAGES_DIR, CANNY_IMAGES_DIR, CONTROLNET_IMAGES_DIR, SD15_IMAGES_DIR]:
		if not p.exists() or not p.is_dir():
			raise FileNotFoundError(f"Missing directory: {p}")
	if not METADATA_JSONL_PATH.exists():
		raise FileNotFoundError(f"Missing metadata file: {METADATA_JSONL_PATH}")

	# Quick visibility on dataset sizes
	n_real = len(list_images(REAL_IMAGES_DIR))
	n_canny = len(list_images(CANNY_IMAGES_DIR))
	n_controlnet = len(list_images(CONTROLNET_IMAGES_DIR))
	n_sd15 = len(list_images(SD15_IMAGES_DIR))

	print("===== Input summary =====")
	print(f"Real images      : {n_real} -> {REAL_IMAGES_DIR}")
	print(f"Canny images     : {n_canny} -> {CANNY_IMAGES_DIR}")
	print(f"ControlNet images: {n_controlnet} -> {CONTROLNET_IMAGES_DIR}")
	print(f"SD1.5 images     : {n_sd15} -> {SD15_IMAGES_DIR}")
	print(f"Metadata         : {METADATA_JSONL_PATH}")
	print(f"Device           : {device}")
	print()

	# 1) FID with pytorch-fid
	print("Computing FID with pytorch-fid...")
	fid_controlnet = fid_score(REAL_IMAGES_DIR, CONTROLNET_IMAGES_DIR, device)
	fid_sd15 = fid_score(REAL_IMAGES_DIR, SD15_IMAGES_DIR, device)

	# 2) CLIP score (image-prompt alignment)
	print("Loading CLIP model...")
	processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
	model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device).eval()

	stem_to_prompt = load_metadata_prompts(METADATA_JSONL_PATH)

	print("Computing CLIP scores...")
	clip_controlnet, used_cn, skipped_cn = clip_score_for_folder(
		CONTROLNET_IMAGES_DIR,
		stem_to_prompt,
		model,
		processor,
		device,
	)
	clip_sd15, used_sd, skipped_sd = clip_score_for_folder(
		SD15_IMAGES_DIR,
		stem_to_prompt,
		model,
		processor,
		device,
	)

	print("\n===== Results =====")
	print(f"FID   (Real vs ControlNet): {fid_controlnet:.4f}")
	print(f"FID   (Real vs SD1.5)    : {fid_sd15:.4f}")
	print(f"CLIP  (ControlNet)       : {clip_controlnet:.4f}  [used={used_cn}, skipped={skipped_cn}]")
	print(f"CLIP  (SD1.5)            : {clip_sd15:.4f}  [used={used_sd}, skipped={skipped_sd}]")


if __name__ == "__main__":
	main()
