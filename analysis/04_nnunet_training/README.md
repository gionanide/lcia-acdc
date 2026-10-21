# 4. Official nnU-Net training and OOF qualification
Requires externally prepared `Dataset701_ACDCDevelopment`; no images are distributed.
Train one fold (replace 0 with the desired fold):
`python analysis/04_nnunet_training/train_fold.py --fold 0 --nnunet-raw nnunet/nnUNet_raw --nnunet-preprocessed nnunet/nnUNet_preprocessed --out-dir outputs/training`
The five folds use 1,000 official epochs; fold 0 adds a separate nonselectable 200-epoch continuation.
The batch/patch contract is 56 and 256x224; workers=8 and compilation=false.
`python analysis/04_nnunet_training/out_of_fold_qualification.py --training-dir outputs/training --nnunet-preprocessed nnunet/nnUNet_preprocessed --raw-dataset-dir nnunet/nnUNet_raw/Dataset701_ACDCDevelopment --out-dir outputs/oof`
Outputs: native development OOF masks, segmentation tables and one global checkpoint-rule JSON.
Selection uses segmentation only, never volumes, RVEF or clinical categories.
The packaged splits and plans must match; existing fold destinations are rejected.
