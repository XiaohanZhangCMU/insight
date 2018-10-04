# Insight AI Project

## Motivation for this project:
- **training** : Progressively growing image resolution during training. Faster convergence? Better stability? Higher accuracy?
- **adversarial attack** : Training with coarse images help nn to stuck with local minimum. Thus prevent adversarial example attack?

## Prerequisite
tensorflow
keras
gpu
cleverhans

```
conda install -c anaconda tensorflow-gpu 
conda install keras
```

## Install
Clone repository and update python path
```
git clone git@github.com:XiaohanZhangCMU/insight.git
cd insight
pip install joblib
pip install scikit-image
git clone https://github.com/tensorflow/cleverhans
pip install -e ./cleverhans
pip install sklearn
```

## Configs
- We recommond using either .yaml or .txt for your config files, not .json
- If credentials are needed, use environment variables or HashiCorp's [Vault](https://www.vaultproject.io/)

## Test
- Include instructions for how to run all tests after the software is installed
```
# Example

# Step 1
# Step 2
```

## Run Inference
- Include instructions on how to run inference
- i.e. image classification on a single image for a CNN deep learning project
```
# Example

# Step 1
# Step 2
```

## Build Model
- Include instructions of how to build the model
- This can be done either locally or on the cloud
```
# Example

# Step 1
# Step 2
```


## Analysis
- Include some form of EDA (exploratory data analysis)
- And/or include benchmarking of the model and results
```
# Example

# Step 1
# Step 2
```
