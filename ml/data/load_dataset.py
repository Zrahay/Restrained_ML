from datasets import load_dataset

ds = load_dataset("detection-datasets/fashionpedia", split="train")
subset = ds.shuffle(seed=42).select(range(5000))
subset.save_to_disk("ml/data/fashionpedia_5k")