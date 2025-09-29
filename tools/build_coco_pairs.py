#!/usr/bin/env python3
import json, os, argparse
from collections import defaultdict

def main(images_dir, captions_json, pick="longest"):
    with open(captions_json, "r") as f:
        ann = json.load(f)
    id_to_name = {im["id"]: im["file_name"] for im in ann["images"]}
    captions = defaultdict(list)
    for a in ann["annotations"]:
        captions[a["image_id"]].append(a["caption"])
    os.makedirs(images_dir, exist_ok=True)
    wrote = 0
    for img_id, file_name in id_to_name.items():
        caps = captions.get(img_id, [])
        if not caps: continue
        if pick == "longest":
            cap = max(caps, key=len)
        elif pick == "first":
            cap = caps[0]
        else:
            cap = caps[0]
        stem, _ = os.path.splitext(file_name)
        with open(os.path.join(images_dir, f"{stem}.txt"), "w") as out:
            out.write(cap.strip())
        wrote += 1
    print(f"Wrote {wrote} caption files to {images_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--captions_json", required=True)
    ap.add_argument("--pick", default="longest", choices=["longest","first"])
    args = ap.parse_args()
    main(args.images_dir, args.captions_json, args.pick)

