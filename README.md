# Insight AI Project -- ProG-train

## Motivation for this project:
- **training** : Progressively growing image resolution during training (ProG-train) an image classifier CNN model. ProG-train promises faster convergence and better numerical stability. The model thus trained is shown to generalize better than trained with fixed image size. ProG-train works with TensorFlow, Keras and Fastai.
- **adversarial attack** : Set up interface of ProG-train with cleverhans. ProG-train enables training surrogate for blackbox faster.

## Prerequisite
tensorflow
keras
fastai
cleverhans
joblib
scikit-image

Install them with conda:
```
conda install -c anaconda tensorflow-gpu 
conda install keras
pip install joblib
pip install scikit-image
git clone https://github.com/tensorflow/cleverhans
pip install -e ./cleverhans
pip install sklearn
```

## Install and train a blackbox surrogate
Clone repository
```
git clone git@github.com:XiaohanZhangCMU/insight.git
cd insight
python src/model/gtsrb_blackbox.py
```

## Configs
- We recommond using either .yaml or .txt for your config files, not .json
- If credentials are needed, use environment variables or HashiCorp's [Vault](https://www.vaultproject.io/)

## Test
- Include instructions for how to run all tests after the software is installed
```
# Example

# python src/model/pg_train_gtsrb.py trains a CNN model for GTSRB dataset
# python src/model/gtsrb_blackbox.py trains a CNN surrogate and generate attacks with FGSM. 

``
