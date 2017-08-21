### Dataset:
32x32x3 images of 43 types of german road signs
Training on 34799, validation on 4410 images, testing on 12,630 images

### Model architecture

I preprocesed the data by dividing by 127.5 (half of 255) and subtracting 1. This left the data with mean pixel value of -.34 with standard deviation of -.5, and was certainly not perfect but got the job done.

My additions to the lenet architecture were in the form of two types of normalization. Dropout with keep_prob=0.5 after each of the first two fully connected layers, and batch norm after the second conv layer. Adding batch norm after the first conv layer hurt performance.
With just normalized data I was at 91%, with dropout I got to 93% and with batch norm I got to 94%. All results were 20 epochs, as soon as I started dropout the gap between valid and test data performance went to 0. Train was still considerably better, 98% or 99%. I always used batch size 156, because it didn't break.


My final model is very similar to the lenet architecture and consisted of the following layers:

normalize images, shape stays 32x32x3

5x5 conv layer with relu activation and depth 6 (relu activation)

max pooling with stride shape of 2 

5x5 conv layer with relu activation and depth 16

max pooling with stride shape of 2 

batch norm with default settings

Flatten

Fully connected layer with 120 outputs and .5 dropout

Fully connected layer with 84 outputs and .5 dropout

Fully connected layer with 43 outputs {logits}


### Differences between new images and old images
Old images and new images are both 32x32x3, but old images have a higher average pixel value in all three channels than new images, with an average value of 87 vs 73. Since the normalization is hard coded to be x/127.5 -1, this means that in normalized space, the average pixel value is -.46 on new images vs. -.35 on train images, which could make generalizing difficult.


### Poor Performance on New images
the model only gets images with numbers in them correct (speed limit), as these seem easy (at least for me) and are very frequent in the training data. I think if the signs from the web occupied more of the image performance might have been better. The outputted probabilities show that the model is very certain of its speed limit sign predictions, and less certain about other predicitons, which is reassuring. I think if I had trained on augmented data it could have helped my model's generalization ability. My performance on new images was 2/7. Accuracy on the test images was 93.8%, suggesting that we have overfit to the given data.

# Other things I didnt try
- taking out dropout and just doing batch norm
- changing dropout hyper parameter (.5 seems aggressive, but could go lower or higher)
- fine tuning with vgg16 weights (different size image scared me)
- adding more layers
- different optimizer like momentum or some such
- grayscale

### Issues/ Questions
Huge issues setting up AWS, specifically sshing into boxes after buying them.
 Also often, if I tried to re-run the model in my notebook the accuracy fell apart. Maybe I need to call tf.reset_default_graph() every new training or some other resetter? this hurt me a lot

How big can I make by batches?
Easy way to test different hyperparams without running 20 epochs?
possible to make the `evaluate` code quicker?

