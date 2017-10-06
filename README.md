Finetuning a ResNet50 model using Keras
=======================================

This very simple repository shows how to use a ResNet50 model (pretrained on the [ImageNet](http://image-net.org/) dataset) and finetune it for your own data. The script is just 50 lines of code and is written using Keras 2.0. It expects the data to be placed separate folders for each of your classes in the `train` and `valid` folders under the `data` directory.

I trained this model on a small dataset containing just 1,000 images spread across 5 classes. I modified the [ImageDataGenerator](https://keras.io/preprocessing/image/#imagedatagenerator) to augment my data and generate some more images based on my samples. Using a Tesla K80 GPU, the average epoch time was about 10 seconds, which is a about 6 times faster than a comparable VGG16 model set up for the same purpose.

To use this model for prediction call the `resnet50_predict.py` script with the following:

```
python3 resnet50_predict.py <path_to_h5_file> <path_to_image_file>
```

As an example:

```
python3 resnet50_predict.py resnet50_final.h5 my_test_image.png
```