import os
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from torch.utils.data import Dataset , random_split,DataLoader
from PIL import Image,ImageFilter
import torchvision.models as models
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from sklearn.metrics import f1_score
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid

device = torch.device('cpu')

#train=pd.read_csv('E:/Vehicles classification/new_train.csv')

def to_device(data,device):
    """Load data to device"""
    if isinstance(data,(list,tuple)):
        return [to_device(x,device) for x in data]
    return data.to(device,non_blocking=True)

class DeviceDataLoader():
    """Wraps a dataloader to move data into device"""
    def __init__(self,dl,device):
        self.dl=dl
        self.device=device
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b,self.device)
    def __len__(self):
        """Number of batches"""
        return len(self.dl)

class VehicleDataset(Dataset):
    
    def __init__(self,csv_name,folder,transform=None,label=False):
        self.label=label
        self.folder=folder
        self.dataframe=pd.read_csv(csv_name)
        self.tms=transform
    def __len__(self):
        return len(self.dataframe)
    def __getitem__(self, index):
        row=self.dataframe.iloc[index]
        img_index=row['image_names']
        image_file=os.path.join('/content/drive/My Drive/Vehicle classification/data/images',img_index)
        image=Image.open(image_file)
        if(self.label):
            target=row['emergency_or_not']
            if(target==0):
                encode=torch.FloatTensor([1,0])
            else:
                encode=torch.FloatTensor([0,1])
            return self.tms(image),encode
        return self.tms(image)
    
"""Code to find per-channel mean and std

nimages = 0
mean = 0.0
var = 0.0
for i_batch, batch_target in enumerate(trainloader):
    batch = batch_target[0]
    # Rearrange batch to be the shape of [B, C, W * H]
    batch = batch.view(batch.size(0), batch.size(1), -1)
    # Update total number of images
    nimages += batch.size(0)
    # Compute mean and std here
    mean += batch.mean(2).sum(0) 
    var += batch.var(2).sum(0)

mean /= nimages
var /= nimages
std = torch.sqrt(var)

print(mean)
print(std)
"""
transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406],
                                                   [0.229, 0.224, 0.225])])

train_dataset=VehicleDataset('/content/drive/My Drive/Vehicle classification/data/new_train.csv',
                             'images',
                             label=True,
                             transform=transform)

test_dataset=VehicleDataset('/content/drive/My Drive/Vehicle classification/new_test.csv',
                             'images',
                             label=False,
                             transform=transform)
torch.manual_seed(42)
batch_size=32
valid_split=int(0.2*len(train_dataset))
train_split=len(train_dataset)-valid_split
train_ds,val_ds=random_split(train_dataset,[train_split,valid_split])

#Dataloaders - To combine dataset and sampler , and provides an iterable over the given dataset
train_loader=DataLoader(train_ds,batch_size,shuffle=True,num_workers=4,pin_memory=True)
val_loader=DataLoader(val_ds,batch_size,num_workers=4,pin_memory=True)

#Device data loaders- To load data loaders onto device
train_dl=DeviceDataLoader(train_loader,device)
val_dl=DeviceDataLoader(val_loader,device)

"""
#Testing train_dl
for i,y in train_dl:
    print(i.shape,y.shape)
    break
"""

def accuracy(outputs,labels):
    _,preds=torch.max(outputs,axis=1)
    _,truths=torch.max(outputs,axis=1)
    return torch.tensor(torch.sum(preds==truths).item()/len(preds))

#Sigmoid activation used since it's Binary Cross entropy
class ImageClassificationBase(nn.Module):
    def training_step(self,batch):
        images,targets=batch
        out=self(images)
        loss=F.binary_cross_entropy(torch.sigmoid(out),targets)
        return loss
    def validation_step(self,batch):
        images,targets=batch
        out=self(images)
        loss=F.binary_cross_entropy(torch.sigmoid(out),targets)
        score=accuracy(out,targets)
        return {'val_loss':loss.detach(),'val_score':score.detach()}
    def validation_epoch_end(self,outputs):
        batch_losses=[x['val_loss'] for x in outputs]
        epoch_loss=torch.stack(batch_losses).mean()
        batch_scores=[x['val_score'] for x in outputs]
        epoch_score=torch.stack(batch_scores).mean()
        return {'val_loss':epoch_loss.item(),'val_score':epoch_score.item()}
    def epoch_end(self,epoch,result):
        print("Epoch [{}] , train_loss: {:.4f} , val_loss: {:.4f} ,val_score: {:.4f}".format(
            epoch,result['train_loss'],result['val_loss'],result['val_score']))
        
@torch.no_grad()
def evaluate(model,val_loader):
    model.eval()
    outputs=[model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(model,epochs,lr,train_loader,val_loader,weight_decay=0,
                  grad_clip=None,opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history=[]
    optimizer=opt_func(model.parameters(),lr,weight_decay=weight_decay)
    for epoch in range(epochs):
        model.train()
        train_losses=[]
        lrs=[]
        for batch in train_loader:
            loss=model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        result=evaluate(model,val_loader)
        result['train_loss']=torch.stack(train_losses).mean().item()
        model.epoch_end(epoch,result)
        history.append(result)
    return history

class Resnet50(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.pretrained_model=models.resnet50(pretrained=True)
        feature_in=self.pretrained_model.fc.in_features
        self.pretrained_model.fc=nn.Linear(feature_in,2)
    def forward(self,x):
        return self.pretrained_model(x)
    
def plot(history,epochs=10):
    train_loss=[]
    val_loss=[]
    val_score=[]
    for i in range(epochs):
        train_loss.append(history[i]['train_loss'])
        val_loss.append(history[i]['val_loss'])
        val_score.append(history[i]['val_score'])
    plt.plot(train_loss,label='train_loss')
    plt.plot(val_loss,label='val_loss')
    plt.legend()
    plt.title('loss')
    
    plt.figure()
    plt.plot(val_score,label='val_score')
    plt.legend()
    plt.title('accuracy')
    
lr=1e-4
epochs=5
opt_func=torch.optim.Adam
weight_decay=1e-4
gradient_clipping=0

model=to_device(Resnet50(),device)
torch.cuda.empty_cache()
history=fit(model,epochs,lr,train_dl,val_dl,weight_decay=weight_decay,grad_clip=gradient_clipping,opt_func=opt_func)

plot(history,epochs=5)

torch.save(model.state_dict(),'/content/drive/My Drive/Vehicle classification/model.pt')
"""
#LOADING MODEL
loaded_model=Resnet50()
loaded_model.load_state_dict(torch.load('model.pt'))
loaded_model.eval() 
"""

#Testing on unseen images

len(test_dataset)

#Visualising unseen test set result
import matplotlib.pyplot as plt
import requests
test_csv=pd.read_csv('/content/drive/My Drive/Vehicle classification/new_test.csv')
img_path='/content/drive/My Drive/Vehicle classification/data/images'
plt.figure(figsize=(10,10))
i=0
for test_image in test_dataset:
  if(i<16):
    dispimage=Image.open(os.path.join('/content/drive/My Drive/Vehicle classification/data/images',test_csv.iloc[i]['image_names']))
    img= transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225])(test_image).view(1,3,224,224)
    pred=model.forward(img.cuda())
    _,index=torch.max(pred,axis=1)
    index=index.cpu().numpy()[0]
    if (index==0):
      plt.title("Non-Emergency")
    else:
      plt.title("Emergency")
    plt.subplot(4,4,i+1)
    plt.axis('off')
    plt.imshow(dispimage)
    i+=1
  else:
    break

#Submission making
preds = []
for test_image in test_dataset:
    test_image = transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225])(test_image).view(1,3,224,224)
    pred  = model.forward(test_image.cuda())
    _,idx = torch.max(pred,dim = 1)
    idx = idx.cpu().numpy()[0]
    preds.append(idx)

pd.Series(preds).value_counts()

subs = pd.read_csv('/content/drive/My Drive/Vehicle classification/new_test.csv')
subs['emergency_or_not'] = preds
subs.to_csv('/content/drive/My Drive/Vehicle classification/Submission.csv',index = False)

#Cleaning test.csv
import pandas as pd
import os
train1=pd.read_csv('/content/drive/My Drive/Vehicle classification/test set.csv')
filenames=os.listdir('/content/drive/My Drive/Vehicle classification/data/images')

for i,v in enumerate(list(train1['image_names'])):
    if (v in filenames) is False:
        train1=train1.drop(axis=0,index=train1[train1['image_names']==v].index[0])
train1.to_csv('/content/drive/My Drive/Vehicle classification/new_test.csv',index=False)

"""**TESTING**"""

#LOADING MODEL
loaded_model=Resnet50()
loaded_model.load_state_dict(torch.load('E:/Vehicles classification/model.pt',map_location=torch.device('cpu')))
loaded_model.eval()
model=loaded_model

#Visualising single images
import requests

#testcsv=pd.read_csv('/content/drive/My Drive/Vehicle classification/new_test.csv')

def display_results(url):
  imag=Image.open(requests.get(url,stream=True).raw)
  imag=imag.resize((224,224),resample=2)
  #print(numpy.array(imag).shape)
  img=transform(imag)
  img=img.view(1,3,224,224)
  pred=model.forward(img)
  val,idx=torch.max(torch.sigmoid(pred),axis=1)
  idx=idx.cpu().numpy()[0]
  if (idx==0):
    plt.title('Non-Emergency')
  elif(idx==1):
    plt.title('Emergency')
  plt.axis('off')
  plt.imshow(imag)

display_results('https://upload.wikimedia.org/wikipedia/commons/4/4f/LFB_Pump_Ladder.jpg')

display_results('https://www.extremetech.com/wp-content/uploads/2019/12/SONATA-hero-option1-764A5360-edit-640x354.jpg')

display_results('https://i.pinimg.com/originals/93/81/1c/93811cdd8e5fa660b7f8e359baa37d4a.jpg')

display_results('https://m.economictimes.com/thumb/msid-61078735,width-1200,height-900,resizemode-4,imgsize-55292/wealth/spend/best-5-suvs-to-buy-this-diwali/st3-thumb.jpg')
