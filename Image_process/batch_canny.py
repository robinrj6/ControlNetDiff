import os
import cv2
import numpy as np
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def auto_canny(gray, sigma: float):
    # gray: uint8 HxW
    v = float(cv2.medianBlur(gray, 5).mean())  # slightly stabilized estimate
    # If you prefer true median (slower), use: 
    # v = float(np.median(gray))
    low = int(max(0, (1.0 - sigma) * v))
    high = int(min(255, (1.0 + sigma) * v))
    if high <= low:  # safeguard for flat images
        high = min(255, low + 1)
    edges = cv2.Canny(gray, low, high)
    return edges, low, high

def process_one(in_path: str, out_path: str, sigma: float, invert: bool):
    in_path = Path(in_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
    if img is None:
        return (str(in_path), False, "imread_failed")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges, low, high = auto_canny(gray, sigma=sigma)

    if invert:
        edges = 255 - edges  # ControlNet sometimes uses white background/black edges conventions

    ok = cv2.imwrite(str(out_path), edges, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    return (str(in_path), ok, f"low={low},high={high}")

def iter_images(input_dir: Path, recursive: bool):
    if recursive:
        for p in input_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                yield p
    else:
        for p in input_dir.iterdir():
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                yield p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input image directory")
    ap.add_argument("--output", required=True, help="Output directory for canny maps")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    ap.add_argument("--sigma", type=float, default=0.33, help="Auto-canny sigma (typical: 0.33)")
    ap.add_argument("--invert", action="store_true", help="Invert output (255-edges)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 8, help="Number of worker processes")
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    imgs = list(iter_images(in_dir, args.recursive))
    total = len(imgs)
    if total == 0:
        print("No images found.")
        return

    print(f"Found {total} images. Processing with {args.workers} workers...")

    futures = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for p in imgs:
            rel = p.relative_to(in_dir)
            out_path = out_dir / rel.with_suffix(".png")  # store edges as PNG
            futures.append(ex.submit(process_one, str(p), str(out_path), args.sigma, args.invert))

        done = 0
        failed = 0
        for f in as_completed(futures):
            src, ok, msg = f.result()
            done += 1
            if not ok:
                failed += 1
            if done % 1000 == 0 or done == total:
                print(f"{done}/{total} done (failed={failed})")

    print(f"Finished. total={total}, failed={failed}. Output: {out_dir}")

if __name__ == "__main__":
    main()