## Installation
1.steps:
Use only python 3.10
1.pip install anomalib
2.!anomalib install -h --for help
3. !anomalib install -v -- for install

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
1.Try with only true images -tried
2.Need to try different datasets,engines,diff,epochs -tried 
3.Need to track experiments using ML flow
4.Need to run script in command line

# Guidelines:
1.Try to add 30 to 50 dataset ,dont go beyond it makes cuda gpu memory error

# In gpu :
1.once your trained one model and that generated model will saved in results/latest
2.so if you run again train (fit) without changing the file name ,it will through error.
so,try to change the name of the folder or save the model in local

# if you wanto change epoch ,you have to change in 
You could check them from either model config
anomalib/configs/model/efficient_ad.yaml

Line 17 in 7a963e9

 max_epochs: 200 
or the model implementation
anomalib/src/anomalib/models/image/patchcore/lightning_model.py

Line 118 in 7a963e9

 return {"gradient_clip_val": 0, "max_epochs": 1, "num_sanity_val_steps": 0} 

 check this link:https://github.com/openvinotoolkit/anomalib/issues/1968
 

 In google colab Gpu:
 1. if you want to change epoch size ,you have to press folder dot icon,then you can get usr folder,
 like this # /usr/local/lib/python3.10/dist-packages/anomalib/cli/utils/installation.py:271


 2. If you want to change epoch you can go to their models and ligtening.py and /image/patchcore/lightning_model.py like this ,but you want change again and again ,each time you have to 
    1.Delete Cache,tmp
    2.Restart run time /run all

For inference :
i changed in https://vscode.dev/github/shanmugamani1023/anomalib_pattern_recognition/blob/maine-packages/anomalib/deploy/inferencers/openvino_inferencer.py#L278 

        if "image_threshold" in metadata:
            pred_idx = pred_score >= metadata["image_threshold"]
            # pred_label = LabelName.ABNORMAL if pred_idx else LabelName.NORMAL
            pred_label = "ABNORMAL" if pred_idx else "NORMAL"
i commented LabelName.ABNORMAL if pred_idx else LabelName.NORMAL and i changed it into string type

