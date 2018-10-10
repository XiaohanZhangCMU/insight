# Insight AI Project -- ProG-train

## Motivation for this project:
- **training** : Progressively growing image resolution during training (ProG-train) an image classifier CNN model. ProG-train promises faster convergence and better numerical stability. The model thus trained is shown to generalize better than trained with fixed image size. ProG-train works with TensorFlow, Keras and Fastai.
- **adversarial attack** : Set up interface of ProG-train with cleverhans. ProG-train enables training surrogate for blackbox faster.
- **Presentation to the work** : https://drive.google.com/open?id=1LM6A24zkyLYhKG9eqTpYQx0-XYGar6ievm4pEbC1XH8

## Prerequisite
* tensorflow 1.11  
* keras 2.2.4  
* fastai (https://github.com/fastai)  
* cleverhans (https://github.com/tensorflow/cleverhans)  
* joblib (https://github.com/joblib/joblib)  
* scikit-image 0.15  

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

## Train a image classifier on GTSRB dataset with and without ProG-train. 
```
cd src/models/ 

python pg_train_gtsrb.py --train_opt 1  (using ProG-train, by default)

and 

python pg_train_gtsrb.py --train_opt 0 (not using ProG-train)
```
### Plot results and check the convergence rate and validation accuracy.
```
cd tests/analysis

python plot_loss_vs_iterations.py
```
![](https://github.com/XiaohanZhangCMU/insight/blob/test_fastai/Xiaohan_Zhang_Demo_Final.png)

## Train a blackbox CNN surrogate and generate a series of attacks:
```
git clone git@github.com:XiaohanZhangCMU/insight.git
cd insight
python src/model/gtsrb_blackbox.py

```

