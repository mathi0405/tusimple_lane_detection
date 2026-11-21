# TuSimple lane detection (UNet)

Binary lane segmentation on the [TuSimple lane dataset](https://github.com/TuSimple/tusimple-benchmark)
with a UNet in PyTorch, plus a video pipeline that fits polynomials to the predicted
lanes and draws a smoothed centre path.

## Pipeline

| Step | File | What it does |
|---|---|---|
| Labels | `merge_labels.py` | Merges the three TuSimple training label files into `data/train_label.json` |
| Dataset | `dataset/dataset.py` | Loads images, rescales lane x / `h_samples` y coordinates to the network size (352×640) |
| Targets | `utils/mask_utils.py` | Draws lane polylines into a 0/1 mask (points with x = -2 are skipped) |
| Model | `models/model.py` | 4-level UNet, sigmoid output, one channel |
| Training | `train.py` | BCE loss, Adam, saves a checkpoint per epoch to `outputs/` |
| Evaluation | `evaluate.py` | IoU, precision, recall, F1 at thresholds 0.1–0.5 on the test split, plus overlay images |
| Video | `video_inference.py` | Per-frame segmentation, polynomial lane fit, exponential smoothing, centre path, MP4 output |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download TuSimple and place it so that `raw_file` paths resolve under `data/tusimple/`.
Then build the merged training labels. The script expects the original label files under
`data/TUSimple/train_set/`:

```bash
python merge_labels.py          # writes data/train_label.json
```

`data/test_label_new.json` (test split labels) is included in the repository.

## Train and evaluate

```bash
python train.py --train-json data/train_label.json --epochs 10 --batch-size 2
python evaluate.py              # uses outputs/lane_model.pth and the test split
python video_inference.py       # edit VIDEO_PATH at the top of the file
```

`train.py` refuses label files whose name contains `test`, so the test split
cannot be used for training by mistake.

## Tests

```bash
pip install pytest
python -m pytest -q tests
```

The smoke tests run on CPU without the dataset. They check the UNet output shape and
range, mask drawing, coordinate scaling and the metrics. CI runs them on every push.

## Status and limits

- No trained checkpoint and no benchmark numbers are included in this repository, so
  none are claimed. Train and run `evaluate.py` to get them for your setup.
- The metrics are pixel-level IoU/precision/recall/F1 on rasterised lanes. They are
  not the official TuSimple accuracy/FP/FN lane metric.
- Lanes are segmented as one binary class: the model does not separate individual
  lane instances. `video_inference.py` assigns left/right lanes afterwards by polynomial fitting.

## Changelog
- 2026-11: training now uses the merged train split; earlier code pointed `train.py`
  at the test labels. The IoU in `evaluate.py` now uses the threshold being reported.
  Added CLI options, smoke tests and CI.
