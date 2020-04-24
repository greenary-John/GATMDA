# GATMDA
**GATMDA：A novel Graph attention networks based framework for human microbe-disease association prediction.**

# Data description
* microbes: ID and names for microbes.
* diseases: ID and names for diseases.
* adj: interaction pairs between microbes and diseases.
* disease_features: pre-processing feature matrix for diseases.
* microbe_features: pre-processing feature matrix for microbes.
* interaction: known microbe-disease interaction matrix.

# Run steps
1. To generate training data and test data.
2. Run train.py to train the model and obtain the predicted scores for microbe-disease associations.

# Requirements
* GATMDA is implemented to work under Python 3.7.
* Tensorflow
* numpy
* scipy
* sklearn

