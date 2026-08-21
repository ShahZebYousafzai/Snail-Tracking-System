# TODO

- **Relabel bead rotation.** `data/bead_crops/from_video_dedup/manual_labels.csv` has
  `rotation_deg == 0` for all 2,742 digit-labeled rows -- the rotation preview was never
  adjusted during labeling, so `TwoHeadBeadNet`'s rotation head (`src/bead_classifier.py`)
  is currently training against a constant target and learning nothing useful. Re-run
  `src/label_bead_crops.py` (rotation-only correction pass) against the existing digit
  labels before trusting/using that head. Also invalidates the synthetic rotation labels
  in `data/bead_crops/from_video_dedup/augmented/augmented_labels.csv`
  (`src/augment_minority_classes.py` derives them from the source crop's rotation_deg,
  which was always 0) -- regenerate augmented crops after the relabel.
- **Spot-check 6/9 digit labels.** Digits 6 and 9 look identical under 180° rotation;
  since rotation wasn't tracked during labeling (see above), some manual digit labels in
  the current white-bead video dataset may have picked the wrong one of the pair. Worth a
  manual review pass over the 6/9 crops specifically.
- **Future physical tag revision: color-code 6 and 9 differently.** Printing 6 and 9 in
  distinguishable colors on any future bead/tag hardware removes the rotation-ambiguity
  problem at the source. Only applies to tags manufactured after this PoC -- doesn't help
  the already-recorded video (C0021/C0023/C0025), which uses uniform white beads.
