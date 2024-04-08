## Installation
1.I tried both methods via pip and via source -like mentioned in this link
https://github.com/openvinotoolkit/anomalib?tab=readme-ov-file#-installation

but in python 3.11 it not works,so i tried 3.10 and it works fine

## Dataset preparation
2.after that i dont know how to prepare custom dataset ,
so, i followed the docs -https://anomalib.readthedocs.io/en/latest/markdown/guides/how_to/data/custom_data.html
mentioned in that link

## Train
## When training i got os windows error -
i solved it by opening vs code in admin mode.

3.My first model is based on Patch core for classification-you can see in below file 
train_patch_core_architechture_classification_true_false.ipynb

## Inference
4.and i tried to inference using detect.py

## Multiple Engines
5.And i tried another engine called padim with segmentation,
Segmentation_ Only Normal Images_padim.ipynb

## Inference
4.and i tried to inference using 
anomalib\detect_padim.py


## i faced set epoch level , i solved it by like below code
we can pass model arguments in engine object instantiation ,
if you have any cache ,delete that.
# start training
engine = Engine(task=TaskType.SEGMENTATION,accelerator="auto",
    check_val_every_n_epoch=1,
    max_epochs=2,
    num_sanity_val_steps=0,
    val_check_interval=1.0
    )
engine.fit(model=model, datamodule=datamodule)

# To do:
1.Try with only true images
2.Need to try different datasets,engines,diff,epochs
3.Need to track experiments using ML flow
4.Need to run script in command line


