# Kaggle Notes

- CMeEE + GlobalPointer runs on Kaggle P100 with about 8.3 GB / 16 GB GPU memory used before batching validation.
- Validation now uses batched prediction with `GP_EVAL_BATCH_SIZE=16` by default.
- If validation is stable and more speed is needed, try `GP_EVAL_BATCH_SIZE=32`.
- If Kaggle reports GPU OOM during validation, lower it to `GP_EVAL_BATCH_SIZE=8`.
- Model outputs and prediction files are written only to `/kaggle/working/`.
