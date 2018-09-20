# https://savan77.github.io/blog/imagenet_adv_examples.html
import torch
import torch.nn
from torch.autograd.gradcheck import zero_gradients
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from torchvision import transforms
import numpy as np
import requests, io
import matplotlib.pyplot as plt
from torch.autograd import Variable
from iterative_fgsm_visualize import visualize, classify


inceptionv3 = models.inception_v3(pretrained=True) #download and load pretrained inceptionv3 model
inceptionv3.eval();

url = "https://savan77.github.io/blog/images/ex4.jpg"  #tiger cat #i have uploaded 4 images to try- ex/ex2/ex3.jpg
response = requests.get(url)
img = Image.open(io.BytesIO(response.content))

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

preprocess = transforms.Compose([
                transforms.Resize((299,299)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])


image_tensor = preprocess(img) #preprocess an i
image_tensor = image_tensor.unsqueeze(0) # add batch dimension.  C X H X W ==> B X C X H X W

img_variable = Variable(image_tensor, requires_grad=True) #convert tensor into a variable

output = inceptionv3.forward(img_variable)
label_idx = torch.max(output.data, 1)[1][0]   #get an index(class number) of a largest element
print("label_idx = {0}".format(label_idx))

labels_link = "https://savan77.github.io/blog/files/labels.json"
labels_json = requests.get(labels_link).json()
labels = {int(idx):label for idx, label in labels_json.items()}
#print("labels = {0}".format(labels))
x_pred = labels[int(label_idx)]
print(x_pred)

output_probs = F.softmax(output, dim=1)
x_pred_prob = np.round((torch.max(output_probs.data, 1)[0][0]) * 100,4)
print(x_pred_prob)


y_target = Variable(torch.LongTensor([9]), requires_grad=False)    #9= ostrich
epsilon = 0.25
num_steps = 5
alpha = 0.025

img_variable.data = image_tensor   #in previous method we assigned it to the adversarial img

for i in range(num_steps):
  zero_gradients(img_variable)
  output = inceptionv3.forward(img_variable)
  loss = torch.nn.CrossEntropyLoss()
  loss_cal = loss(output, y_target)
  loss_cal.backward()
  x_grad = alpha * torch.sign(img_variable.grad.data)
  adv_temp = img_variable.data - x_grad
  total_grad = adv_temp - image_tensor
  total_grad = torch.clamp(total_grad, -epsilon, epsilon)
  x_adv = image_tensor + total_grad
  img_variable.data = x_adv

output_adv = inceptionv3.forward(img_variable)
x_adv_pred = labels[int(torch.max(output_adv.data, 1)[1][0])]
output_adv_probs = F.softmax(output_adv, dim=1)
x_adv_pred_prob =  np.round((torch.max(output_adv_probs.data, 1)[0][0]) * 100,4)
visualize(image_tensor, img_variable.data, total_grad, epsilon, x_pred,x_adv_pred, x_pred_prob,  x_adv_pred_prob, std, mean)

classify(output_probs.data.numpy(),labels, 'clean_bar.png', correct_class =int(label_idx), target_class = 9)
classify(output_adv_probs.data.numpy(),labels, 'hack_bar.png', correct_class =int(label_idx), target_class = 9)



