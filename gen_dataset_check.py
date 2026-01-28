import json
import os
import re
from typing import List, Set

def sanitize_filename(text, max_len=256):
    text = re.sub(
        r'^\s*generate\s+an?\s+image\s+of\s+\d+x\d+\s+(according\s+to\s+the\s+following\s+prompt[:,]?\s*)?',
        '',
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(r'[\/:*?"<>|]', '', text)
    text = text.strip().replace(' ', '_')

    return text[:max_len]

def load_prompts(json_path: str, prompt_key: str = None) -> List[str]:
    """
    Load prompts from JSON.
    - If prompt_key is None, JSON is assumed to be a list of strings.
    - Otherwise, JSON is assumed to be a list of dicts with prompt_key.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if prompt_key is None:
        if not all(isinstance(x, str) for x in data):
            raise ValueError("JSON is not a list of strings; specify prompt_key.")
        return data

    return [item[prompt_key] for item in data]


from collections import defaultdict

def build_prefix_index(image_dir, extension=".png", prefix_len=10):
    index = defaultdict(list)

    for fname in os.listdir(image_dir):
        if fname.lower().endswith(extension):
            stem = os.path.splitext(fname)[0]
            prefix = stem[:prefix_len]
            index[prefix].append(stem)

    return index

def find_missing_prompt_ids(
    prompts,
    image_dir,
    extension=".png",
    prefix_len=10
):
    prefix_index = build_prefix_index(image_dir, extension, prefix_len)

    missing_ids = []

    for idx, prompt in enumerate(prompts):
        sanitized = sanitize_filename(prompt)
        prefix = sanitized[:prefix_len]

        # no image shares this prefix
        if prefix not in prefix_index:
            missing_ids.append(idx)
            continue

        # safety check: ensure at least one real prefix match
        candidates = prefix_index[prefix]
        if not any(c.startswith(prefix) for c in candidates):
            missing_ids.append(idx)

    return missing_ids



if __name__ == "__main__":
    JSON_PATH = "/work1/deming/seliny2/LANTERN/data/prompts/captions_val2017_longest.json"
    IMAGE_DIR = "/work1/deming/shared/lumina/fixed-results/0.9-kl1.0-0.625-kl1.0"
    PROMPT_KEY = None  # set to e.g. "prompt" if JSON is list of objects

    prompts = load_prompts(JSON_PATH, PROMPT_KEY)
    missing_ids = find_missing_prompt_ids(prompts, IMAGE_DIR)

    print(f"Missing image count: {len(missing_ids)}")
    print("Missing prompt IDs:")
    print(missing_ids)

