# Restrained_ML

Natural-language photo search that runs **entirely on your phone**. Type "a red dress" and get the matching photos from your gallery — no cloud, no account, no photo ever leaves the device.

Under the hood: Apple's [MobileCLIP-S1](https://github.com/apple/ml-mobileclip) vision-language model, exported to [ExecuTorch](https://pytorch.org/executorch/) and driven by a pure-Kotlin Android app. The CLIP BPE tokenizer and image preprocessing are ported to Kotlin, so inference is fully native — the only Python in this repo is the desktop reference pipeline used to validate everything.

## How it works

```
Indexing (background):   gallery photo ─► decode + preprocess ─► image encoder (.pte)
                         ─► L2-normalized embedding ─► appended to binary store

Search (query time):     "a red dress" ─► Kotlin BPE tokenizer ─► text encoder (.pte)
                         ─► dot product against all stored embeddings ─► top-k photos
```

Because both towers output L2-normalized embeddings, the dot product equals cosine similarity. The on-device index is an append-only binary file (`photo ID → embedding`), so indexing survives app restarts and resumes incrementally.

## Repo layout

```
ml/
  data/load_dataset.py        # download Fashionpedia 5k subset, export JPGs
  models/Indexer/indexer.py   # batch-embed a folder of images -> embeddings.pt
  models/Retriever/retriever.py  # text query -> top-k images (desktop reference)
  export/
    export_executorch.py      # export image + text encoders to .pte (XNNPACK)
    export_tokenizer_assets.py# dump BPE vocab/merges for the Kotlin tokenizer
    make_golden.py            # golden text embeddings for on-device parity tests
    make_golden_phase_d.py    # golden tokenizer + image-preprocessing cases
    parity_check.py           # gate: .pte outputs vs PyTorch (fails < 0.999 cosine)
android/                      # the app (pure Kotlin, single activity)
  app/src/main/java/com/zrahay/restrainedml/
    ClipTokenizer.kt          # full CLIP BPE port (byte-to-unicode, ranked merges)
    ImagePreprocessor.kt      # resize-256 / center-crop / CHW float, matches torchvision
    GalleryIndexer.kt         # MediaStore enumeration + background embedding
    EmbeddingStore.kt         # persistent binary index + top-k search
    MainActivity.kt           # permission flow, indexing progress, search UI
```

## 1. Python setup (desktop pipeline)

Python 3.10+ with PyTorch ≥ 2.9:

```bash
pip install -r requirements.txt
pip install "mobileclip @ git+https://github.com/apple/ml-mobileclip.git"
```

Download the MobileCLIP-S1 checkpoint into `ml/models/`:

```bash
wget -P ml/models https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s1.pt
```

Then run the reference pipeline end to end:

```bash
python ml/data/load_dataset.py                 # 5k Fashionpedia images -> ml/data/images/
python ml/models/Indexer/indexer.py            # embed them -> ml/models/embeddings.pt
python ml/models/Retriever/retriever.py --query "a red dress"   # sanity-check retrieval
```

## 2. Export the models for Android

```bash
python ml/export/export_executorch.py          # image + text encoders -> .pte (XNNPACK)
python ml/export/parity_check.py               # verify .pte == PyTorch (cosine >= 0.999)
python ml/export/export_tokenizer_assets.py    # BPE vocab/merges -> clip_tokenizer.json
python ml/export/make_golden.py                # golden cases for on-device tests
python ml/export/make_golden_phase_d.py
```

Copy the resulting `.pte` files into the app (they are gitignored — ~86 MB image encoder, ~254 MB text encoder):

```bash
cp ml/export/mobileclip_image.pte ml/export/mobileclip_text.pte android/app/src/main/assets/
```

The JSON assets (`clip_tokenizer.json`, golden files) are written into `android/app/src/main/assets/` directly by the export scripts.

## 3. Build and install the app

Requirements: JDK 17, the Android SDK (set `sdk.dir` in `android/local.properties`), and Gradle 8.13 on your PATH — the project is built with the Gradle CLI and does not check in a wrapper. Toolchain: AGP 8.7.3, Kotlin 2.0.21, compileSdk 35, minSdk 29 (Android 10+).

```bash
cd android
gradle :app:installDebug    # with a device connected over adb
```

On first launch, grant the photos permission (`READ_MEDIA_IMAGES`); the app starts indexing your gallery in the background with a progress readout, and you can search as soon as the first photos are embedded. Results show in a 3-column grid; tap a thumbnail to open the photo.

## Verification

Every stage is gated against the PyTorch reference:

- **Desktop export:** `parity_check.py` compares `.pte` outputs to PyTorch on real images and queries — all cases at cosine 1.000000 (max diff ~1e-6).
- **On-device:** the app ships three parity suites (run from the UI, reported on screen and in logcat): tokenizer 50/50 exact token-ID matches; text pipeline end-to-end cosine 1.000000; image pipeline cosine 0.992–0.999 vs. desktop (bilinear resize filter differences).

Measured on a Samsung Galaxy A34 (Android 16, arm64): ~110 ms per text query and ~300–400 ms per image embedding after warm-up (XNNPACK, CPU), indexing ~3 photos/sec in the background.

## Roadmap

- Offline evaluation harness (Recall@k, nDCG) built from Fashionpedia annotations
- Query prompt-template ensembling
- Two-stage retrieve-and-rerank
- Hybrid (lexical + dense) retrieval with rank fusion
