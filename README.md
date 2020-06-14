This is the implementation of the proposed framework. 

To run this you need to place the raw data, which is the SFEW dataset v2 into the direction data/dataset/raw_data

Then use the provided scripts to crop faces, split train and test set, and do data augmentation. Please specify correct target and source path. 



The folder *models* are our networks. To run them, you need the below requirements:

 

You can also download the pre-processed dataset from here. 

https://drive.google.com/file/d/1E5NITEV_i0DHsz8KLvPr66n4gV6JRdTY/view?usp=sharing

The voting mechanism is not integrated in the network. We manually predict the classes and process the output by jupyter notebook. This is shown in the folder *notebooks*



## Requirements

```
tensorflow-gpu
cv2
pytorch
scikit-learn
```



