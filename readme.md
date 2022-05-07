## A Probabilistic U-Net for Segmentation of Ambiguous Images

Hantian Liu, Jiacheng Gao, Zhicheng Dong

This is an implementation of [probabilistic U-Net](https://arxiv.org/pdf/1806.05034.pdf).

We implemented probabilistic U-Net model and baseline method U-Net.   
Now we can train the model with images from LIDC dataset and cityscape dataset with ```main.py```
with default parameters. To train the model with different datasets, you should modify the dataset loading in the ```main.py```.

Use ```./citysacpe/preprocess``` to preprocess the cityscape dataset.

Use ```./LIDC/load_process_LIDC``` to preprocess the LIDC dataset.

After training, we could use ```produce_label_img.py``` to segment images with our trained model.
After getting the segmentations, use ```./citysacpe/covert_label2RGB.py``` to transfer label images to colorful images.