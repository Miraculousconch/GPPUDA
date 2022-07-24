# GPPUDA

The source code of GAN-based Privacy-Preserving Unsupervised domain adaptation.
Tested on Python3.8 and Torch 1.11.0.

## Running Experiements

* For training the model, use: `python3 train.py`.
* For testing the model, use: `python3 accuracy_test.py`.
* For generating the images, use: `python3 generate_image.py`. 

## Experimental Parameters

* The source and target model parameters are in `Source_model_parameter` and `Target_model_parameter` respectively.
* The intermediate results are in `USPS_MNIST_results/Fixed_results_1`.
* The final classifier parameters are in `Final_model_parameter`.





