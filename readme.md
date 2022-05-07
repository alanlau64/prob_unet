## A Probabilistic U-Net for Segmentation of Ambiguous Images

Hantian Liu, Jiacheng Gao, Zhicheng Dong

This is an implementation of [probabilistic U-Net](https://arxiv.org/pdf/1806.05034.pdf).

We implemented probabilistic U-Net model and baseline method U-Net.

Use ```./citysacpe/preprocess``` to preprocess the cityscape dataset.

Use ```./LIDC/load_process_LIDC``` to preprocess the LIDC dataset.

**Training Usage** 

```main.py [-h] -b BATCH_SIZE [--val-after VAL_AFTER] -e EPOCH [--gpu] [--lr LR] dataset```

positional arguments:
  dataset               Type of dataset, "city" for Cityscape, "lidc" for LIDC dataset

optional arguments:
  -h, --help            show this help message and exit \
  -b BATCH_SIZE, --batch-size BATCH_SIZE:
                        Batch size for train and val \
  --val-after VAL_AFTER:
                        Run validation after this No of iterations \
  -e EPOCH, --epoch EPOCH:
                        Number of epochs \
  --gpu:                 Use GPU if available \
  --lr LR:               Learning rate


After training, we could use ```produce_label_img.py``` to segment images with our trained model.
After getting the segmentations, use ```./citysacpe/covert_label2RGB.py``` to transfer label images to colorful images.