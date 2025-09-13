# CNN(ResNet18) for Film (Color Negatives) Detection 

## Background

Before digital cameras and image technology (2000s), film cameras and analog technology were the only choice for photography / movies, leaving films and negatives for file storage. But 
there are some limitations to use film as files, because of the following reasons.  

1. **Overexposure/Underexposure**. Overexposuing or underexposing makes the films have more noise than others. After converting negatives into normal images,
   noise would be enriched, making the pictures not readable.  
2. **Improper chemical processing** After filming, chemical processing is needed for people to see what they have shot. Chemical processing includes developing (use developer liquid),
   blix(use bleach + fixer), water washing and stabilizing (use stabilizer, optional). Improper chemical processing such as uneven development, fading liquid effect,
   and incomplete reaction can destroy the content of films, making them unable to read. Improper chemical processing directly leads to large blank areas on films, artifacts, residual blank spots,
   color patches, accidental imprints, et al.
3. **Improper storage**. Holes can be on films. Long term storage could lead to color fading. Re-coiling films could also lead to dye removal due to friction.

### Films I am trying to detect whether it is problematic:
_Too Much Noise_ / _Big Blank Area_ / _Color Patches_ / _Holes On Films_ / _Residual Blank Spots_ / _Dye Removal_

To detect whether a negative film is **readable**, I developed and train a ResNet 18 model. I marked the films 1 if the film is normal or 0 otherwise. The pipeline and the structure of 
the model will be discussed later.  

## Examples of the films from the dataset

Due to privacy issue, I am not able to provide all the pictures used for training. Here are some examples so you can understand how I define normal and abnormal pictures.  
Use cvt2pos function in Convert.ipynb to convert from negative films to positive (readable) images. 

here are the normal films:
| Tag | Negative | Positive |
|-----|----------|--------- |
|482.jpeg|![Otay Mesa Border](images/otay_mesa.jpeg) | ![Otay Mesa Border](images/positive_image_otay_mesa.jpeg) |
|21.jpeg|![Beach](images/beach.jpeg) | ![Beach](images/positive_image_beach.jpeg) |
|317.jpeg|![Train](images/train.jpeg) | ![Train](images/positive_image_train.jpeg) |
|336.jpeg|![Mountain](images/mountain.jpeg) | ![Street](images/positive_image_mountain.jpeg) |
|233.jpeg|![Wait](images/wait.jpeg) | ![Street](images/positive_image_wait.jpeg) |
|264.jpeg|![Flower](images/flower.jpeg) | ![Street](images/positive_image_flower.jpeg) |

here are the abnormal films:
| Tag | Negative | Positive | Issue | Reason |
|-----|----------|----------|-------|--------|
|399.jpeg|![noise](images/noise.jpeg) | ![noise](images/positive_image_noise.jpeg) | Noise | Underexposure |
|19.jpeg|![blank](images/blank.jpeg) | ![blank](images/positive_image_blank.jpeg) | Big Blank Area | Uneven Development |
|449.jpeg|![patches](images/patches.jpeg) | ![patches](images/positive_image_patches.jpeg) | Color Patches | Fade Developer |
|396.jpeg|![spots](images/residual_spots.jpeg) | ![residual spots](images/positive_image_residual_spots.jpeg) | Residual Spot | Fade Developer |
|8.jpeg|![holes](images/hole.jpeg) | ![holes](images/positive_image_hole.jpeg) | Holes | Folding Films / Improper storage |
|193.jpeg|![dye removal](images/dye_removal.jpeg) | ![holes](images/positive_image_dye_removal.jpeg) | Dye Removal | Re-coiling Films |

**Additional Info**

1. Normal:Abonormal ~ 7:3 
2. Some films can have more than 1 problem.
3. Different problems can lead to similar consequence. (e.g. dye removal & color patches)
4. Insufficient-detailed photos and slightly color-shifted photos are considered normal as they are readable.
5. All 494 film clips were taken by me between July and September in 2025. I used Nikon F2 and Nikon Nikkormat FT3 as my cameras, and Kodak Gold 200, Kodak UltraMax 400, Kodak ColorPlus 200, and Fujifim 400 for my color negatives. All color negatives were developed by me and retaken by phone. 

## Model Structure

The model I used is ResNet18, whose basic structure is a ResNet block. 

A Residual Network block consists of the following structures: (reference from https://d2l.ai/chapter_convolutional-modern/resnet.html)
1. 2 layers, each layer has a convolutional layer and a batch norm layer
2. a bypassing channel which adds input X with 1 or Identity matrix
3. adding the results coming from the 2 layers and the bypassing channel if there is a bypassing channel, otherwise adding X with the 2 layers
4. then the last layer with a conv layer and a batch norm layer

The resnet 18 sequential network consists of: (reference from https://d2l.ai/chapter_convolutional-modern/resnet.html)

Conv -> BatchNorm -> MaxPool -> Resnet Block w/o bypass -> Resnet Block w/o bypass -> Resnet Block w/ bypass -> Resnet Block w/o bypass -> Resnet Block w/ bypass -> Resnet Block w/o bypass - >Resnet Block w/ bypass -> Resnet Block w/o bypass -> AveragePool -> FC layer

## Training Pipeline
1. Collect photos of the negatives, and use cvt2pos function to obtain their positive images. Label the condition of films manually (eg, 1 for normal, 0 for abnormal) 
2. Load the same photos of film clips into two datasets. One is pandas.DataFrame, used for retrieving pictures given indices.  Another one is a Custumized data set used for data split into different batches for training and testing. (Customized Dataset reference: https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)
3. Transform the pictures in Custumized data set to be of the same shape (244 x 244), then split them into training and teating batches.
4. Create Resnet Block and ResNet18.
5. Train the model for 10 epoches. For each epoch, test the model on the test data batches, and record the accuracy rate, training loss, validation loss, and f1 score. Display the confusion matrix every 2 epoches. Use AdamW as the optimizer. Use ReduceLROnPlateau to regulate the learning rate in case the learning performance gets stuck. F1 score is an estimator when both classes (normal, abnormal) is not even.
6. Plot the changes of accuracy rate, training loss, validation loss, and F1 score in the 10 epoches. Plot the changes of learning rate as a reference.
7. Test the model on the test data batches. Compute the accuracy rate, f1 score, precision rate, and recall rate. Then print all the misclassified pictures for further correction.

## Results and Analysis
This is the plot about the changes of train loss, val loss, accuracy rate, and f1 score in the training 10 epoches.
<img width="538" height="390" alt="image" src="https://github.com/user-attachments/assets/e9c7908d-53fb-4f53-96b7-3446c52c06b7" />  
**train loss** keeps dreasing in the 10 epoches, meaning it keeps learning in the training process. 
**val loss, accuracy rate, and f1 score** flutuate at the beginning of the training process. Val loss increases then decreases; accuracy rate and f1 score decrease then increase. It explains the model might tend to believe most of the photos to be all 1 or all 0.

After 7th epoch, val loss,  
**train loss, val loss, accuracy rate, and f1 score** converge  
**accuracy rate** remains at ~ 80%. **f1 score** remains at ~ 90%. This can be explained by the unbalanced dataset, since normal:abnormal ~ 7:3, therefore, the accuracy rate might be underestimated. 
Val loss 


**Confusion Matrix Changes**
The confusion matrix shows that at the beginning of the training process, the model predict most of the photos to be all 0 (102 were predicted 0 and only 21 were predicted 1). During the remaining training epoches, more pictures were predicted to be 1. At 10th epoch, FP:FN ~ 1:1, becoming balanced.

precision rate: 86.957% -> 86.957% of the pictures classified as normal are actually normal. 
recall rate: 88.889% -> 88.889% of the actual normal pictures are classified correctly. 

**Some Wrong Result Analysis**
| Tag | Negative | Positive | Label | Predict | Explanation |
|-----|----------|----------|-------|---------|-------------|
|449.jpeg|![patches](/images/patches.jpeg)|![patches](/images/positive_image_patches.jpeg)|0|1|The model detected its edges but ignore its patches|
|273.jpeg|![273](/images/273.jpeg)|![273](/images/positive_image273.jpeg)|1|0|The picture is likely to be mislabeled. High Noise|
|403.jpeg|![403](/images/403.jpeg)|![403](/images/positive_image403.jpeg)|1|0|The picture has some noise but still readable, therefore it is labeled 1 but the model predicted 0|
|141.jpeg|![141](/images/141.jpeg)|![403](/images/positive_image141.jpeg)|1|0|The picture is underexposed, but noise is bearable.


# Limitation
1. Mislabeling can be a significant problem. Tere is no strict rubric to determine whether a film is readable or destroyed. Some worse pictures can be labeled as 'normal' condition because they could be read despite high noise, or because they are clean despite lack of details. In the dataset, the number of pictures labeled 'abnormal' is underestimated.
2. The way of labeling could cause a domain shift. I manually labeled the condition of the negatives based on seeing their inverted pictures. The method to invert the negatives is by taking their pictures with phone, then inverted by cvt2pos function. Strictly speaking it is not professional. A professional way is to use film scanner to scan negatives. Poor scanning adds in additional noise and color shift into the positive images, so some films might be determined 'abnormal' although their condition is actually 'normal'. However, the goal is to detect whether the films have normal condition, so the model should be trained on films, not inverted pictures, but humans find it hard to detect the film conditions by purely looking to the negatives. Therefore, I remain using the method: manually label pictures by seeing inverted images, then train models on negatives.
3. The dataset is unbalanced (normal:abnormal ~ 7:3). The model is more likely to classify a photo to be a normal picture. In real life, abnormal pictures might just be a minority. But when there comes a query, we should assume the probability of normal/abnormal is ~ 1:1.
4. The dataset is too small (494 pictures). Data argumentation should be considered to enlarge the dataset, to increase accuracy and prevent over-fitting.
5. The model is only trained 10 epoches. The model can be trained for more epoches and set an earlt stop to prevent it from over fitting. 

# Further Study and Use
For further study and improvement, rechecking all the labels and discussing whether they are all labeled correctly are needed. The model can also be trained for more epoches, and set an earlt stop to prevent it from over fitting. 

This model can be used for film photographers to filter the pictures that they are not satisfied with. This model can also help historical museums / archives to discover badly conditional negatives in a short period of time, and determine if it is actually damaged and if it is still remedable. 








