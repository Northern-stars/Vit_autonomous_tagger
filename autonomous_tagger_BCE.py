import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from conv import load_image,resize_img,check_dataset_exist
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
import numpy as np
from Vit import VisionTransformer

DEVICE="cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE=80
LR=1e-3
MODEL_NAME="model/autonomous_tagger.pth"

class Env(Dataset):
    def __init__(self):
        super().__init__()
        self.tag=[]
        f=open("images/word_counts.txt","r",encoding="UTF-8")
        tagtext=f.readlines()
        f.close()
        for i in range(len(tagtext)):
            self.tag.append(tagtext[i].split(":")[0])
        
        self.choice_list=check_dataset_exist()
        self.img_id=-1
    
    def __len__(self):
        return len(self.choice_list)

    def get_image(self,id):
        img=resize_img(load_image(self.choice_list[id])).squeeze()
        return img.to(DEVICE)

    def __getitem__(self,idx):
        index=idx
        ftag=open("images/tags/"+str(self.choice_list[index])+".txt","r",encoding="UTF-8")
        tag_list=ftag.readlines()
        ftag.close()
        tag_token=[0 for _ in range(len(self.tag))]
        tag_list=tag_list[:15]
        for i in range(len(tag_list)):
            # tag_list[i]=tag_list[i].split("\n")
            tag_list[i]=tag_list[i][:len(tag_list[i])-1]
            tag_index=self.tag2token(tag_list[i])
            if tag_index is not None:
                tag_token[self.tag2token(tag_list[i])]=1
        img=self.get_image(index)
        tag_tensor=torch.tensor(tag_token,dtype=torch.float32).to(DEVICE)
        return img,tag_tensor
    
    def token2tag(self,token):
        return self.tag[token]

    def tag2token(self,tag_):
        for i in range(len(self.tag)):
            if self.tag[i]==tag_:
                return i
        print(tag_)

    def tagTokenDecoding(self,tags):
        string=''
        for i in range(len(tags)):
            if tags[i]==0:
                string=string[:len(string)-1]
                return string
            else:
                string+='#'+self.token2tag(tags[i])+','
        return string

def train(model,env:Env,epoch_num=100,load=False):
    if load:
        model.load_state_dict(torch.load(MODEL_NAME))
    loss_fn=nn.BCELoss()
    dataloader=DataLoader(env,BATCH_SIZE,shuffle=True,drop_last=True)
    optimizer=torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.9)
    
    for epoch in range(epoch_num):
        loss_list=[]
        for img,label in tqdm(dataloader):
            pred=model(img)
            loss=loss_fn(pred,label)
            loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
        print(f"Epoch id: {epoch+1}, avg_loss: {np.mean(loss_list)}")
        torch.save(model.state_dict(),MODEL_NAME)



if __name__=="__main__":
    env=Env()
    model=VisionTransformer(picture_size=[1,3,224,224]
                            ,patch_size=14
                            ,encoder_layer_num=6
                            ,n_head=4
                            ,encoder_hidden=512
                            ,out_size=len(env.tag)
                            ,output_channel=3).to(DEVICE)
    train(model=model,env=env,epoch_num=100)
    


    
