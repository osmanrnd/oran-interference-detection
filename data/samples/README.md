# Sample data

A **small illustrative subset** of the dataset — about **6 spectrogram images per class**
— kept here so the repo shows what the model actually sees (similar to the dataset slide
in the presentation). This is *not* the full training set; the complete dataset (~6,000
images) is generated from the srsRAN O-RAN testbed and is not distributed with this repo.

## Layout

Drop your sample PNGs into the matching class folder:

```
data/samples/
├── SOI/    # signal of interest only (no interference)
├── CWI/    # SOI + continuous-wave interference
└── CI/     # SOI + chirped interference
```

Any image filenames work (`.png`, `.jpg`). The grid in the main README uses up to 6 images
per class, in sorted filename order.

## Regenerating the README figure

After adding/changing images, rebuild the contact sheet shown in the main README:

```bash
python scripts/make_sample_grid.py
```

This writes `docs/figures/sample_dataset.png`.
