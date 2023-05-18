# Memory-Based Cross-Image Contexts for Weakly Supervised Semantic Segmentation
This is the implementation of the method described in paper: Memory-Based Cross-Image Contexts for Weakly Supervised Semantic Segmentation.

## Requirements
- Pyhton3.7+
- Pytorch1.0+
- Numpy, OpenCV
- Pydensecrf

## Usage
The script `main.py` is self-contained, which calls scripts in other `train_eval_xxx.py` files for training. By defualt, simply run:

```
python main.py --gpus 0,1,2,3
```

