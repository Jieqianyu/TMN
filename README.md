# Code for "Fast Real-time Video Object Segmentation wit Tangled Memory Network" (ACM TIST 2023)

## Required Package
- torch >= 1.6.0
- torchvison >= 0.7.0

## Data Organization

### Youtbe-VOS Organization
To run the training script on youtube-vos dataset, please ensure the data is organized as following format
```
YTBVOS
      |----train
      |     |-----JPEGImages
      |     |-----Annotations
      |     |-----meta.json
      |----valid
      |     |-----JPEGImages
      |     |-----Annotations
      |     |-----meta.json 
```
Where `JPEGImages` and `Annotations` contain the frames and annotation masks of each video.

### DAVIS Organization

To run the training script on davis16/17 dataset, please ensure the data is organized as following format
```
DAVIS
      |----JPEGImages
      |     |-----480p
      |----Annotations
      |     |-----480p (annotations for DAVIS 2017)
      |     |-----480p_16 (annotations for DAVIS 2016)
      |----ImageSets
      |     |-----2016
      |     |-----2017
      |----db_info.yaml
      |----DAVIS-test-dev (data for DAVIS 2017 test-dev)
```
The `db_info.yaml` contains the meta information of each video sequences and can be found at the davis evaluation [repository](https://github.com/fperazzi/davis-2017/blob/master/data/db_info.yaml).

## Training and Testing
Please change the data root in [`./libs/dataset/data.py`](./libs/dataset/data.py), i.e., `ROOT_YT` and `ROOT_DAVIS`, to the custom path.

To train the TMN network, run following command.
```python
python train_all.py --gpu ${GPU-IDS}
```

we provide the [weights](./checkpoints/davis_best.tar) of TMN without pretraining on COCO in the directory `checkpoints`.
To eval the TMN network on (DAVIS16/17), modify `OPTION.valset`, then run following command
```python
python test_all.py --checkpoint ./checkpoints/davis_best.tar --gpu 0
```

Additionally, you can modify some setting parameters in `options.py` to change configuration.

## Reproduce by Liming Xu (徐李明) on 2024.1.5
