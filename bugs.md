# Known Bugs

## XRF55 uses the test set for checkpoint selection by default

**Location:** `code/XRF55_HAR/config.py` and `code/XRF55_HAR/train.py`

The default configuration sets `val_ratio=0.0`, so no validation subset is
created from the training data. The test loader is then reused as the
checkpoint-selection loader, and the checkpoint with the highest test accuracy
is saved as the best model. Repeatedly consulting test accuracy to choose an
epoch turns the test set into a validation set and makes the final reported test
result optimistically biased.

Use a nonzero validation ratio drawn only from the training split, select the
checkpoint using validation performance, and evaluate the held-out test split
only after selection is complete.

## Shared polynomial and cosine learning-rate schedulers are incorrect

**Status:** Removed. The five completed COMPASS-on-XRF55 runs were unaffected.

**Location:** `code/shared/utils/schedulers.py`

The broken `polylr` and `warmupcosinelr` implementations and the unimplemented
`warmupsteplr` option were removed. The scheduler factory now accepts only
`warmuppolylr` and rejects every other name explicitly.

All five completed COMPASS-on-XRF55 logs show the expected warm-up followed by
polynomial decay, confirming that their reported results used the retained
`warmuppolylr` implementation. This scheduler was not introduced for our
reproduction: the paper and the authors' initial public configuration both
specify a five-epoch warmup-polynomial schedule with power 0.9.

## MMFi discards point-cloud padding masks

**Status:** Deferred while work is focused on XRF55.

**Location:** `code/MMFI_HAR/util.py`, `code/MMFI_HAR/new_train.py`, and
`code/MMFI_HAR/Encoders.py`

The MMFi collate function pads variable-length mmWave and LiDAR point clouds
and calculates masks identifying the padded positions. The training loop
discards this metadata and passes only the padded tensors to the model. The
point-cloud encoders also receive no mask or original-length information, so
zero-padding can be processed as if it contained real points. This can alter
sampling, attention, and the resulting features whenever examples in a batch
have different point counts.

Propagate the masks or original lengths through the training loop and make the
point-cloud encoders exclude padded positions.

## Standalone MMFi evaluation can reconstruct the wrong model variant

**Status:** Deferred while work is focused on XRF55.

**Location:** `code/MMFI_HAR/eval_all.py`

Training constructs `MMFi_CMPT_Net` using method-dependent options such as
`missing_fill`, `generator_mode`, `source_priority`, and imputation settings.
The standalone evaluator always constructs the default model and does not
restore those options. An `impute` checkpoint can therefore fail strict state
loading, while a `cmptstyle` checkpoint can load but be evaluated with behavior
different from the model that was trained.

Save the complete model configuration with each checkpoint and reconstruct the
same model variant during standalone evaluation before loading its state.

## MMFi training can use masked modalities as proxy sources

**Status:** Deferred while work is focused on XRF55.

**Location:** `code/MMFI_HAR/models/cmpt_model_mmfi.py`, `_encode_for_fill`

The default `compass` training configuration leaves
`mask_sources_during_training=False`. During simulated missing-modality
training, `_encode_for_fill` therefore encodes every supplied modality instead
of restricting proxy sources to modalities marked as available. When multiple
modalities are marked missing, one masked modality can be used to generate a
proxy for another masked modality. At inference, those source features are not
available, creating a train/evaluation mismatch and possible information
leakage.

Before changing the implementation, confirm whether this is an intentional
teacher-assisted training strategy. If it is not intentional, restrict
`available_feats` to `available_inputs` during training, equivalent to enabling
`mask_sources_during_training`.
