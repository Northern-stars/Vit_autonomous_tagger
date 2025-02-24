import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from conv import load_image,resize_img,check_dataset_exist


#Embedding
tag=['<start>','<end>','<pad>']
def token2tag(token):
    return tag[token]

def tag2token(tag_):
    for i in range(len(tag)):
        if tag[i]==tag_:
            return i

def tagTokenDecoding(tags):
    string='#'
    for i in range(len(tags)):
        if tags[i]==0 or tags[i]==2:
            continue
        elif tags[i]==2:
            return string
        else:
            string+=token2tag(tags[i])+'#,'
    return string


class PositionEmbedding(nn.Module):
    def __init__(self,d_model,max_len,device):
        super(PositionEmbedding,self).__init__()
        self.encoding=torch.zeros(max_len,d_model,device=device)
        self.encoding.requires_grad=False
        #position encoding do not need gradient
        pos=torch.arange(0,max_len,device=device,requires_grad=False)
        pos=pos.float().unsqueeze(dim=1)
        #transform the pos matrix into a 2-dimentional float matrix
        _2i=torch.arange(0,d_model,step=2,device=device,requires_grad=False).float()
        self.encoding[:,0::2]=torch.sin(pos/(10000**(_2i/d_model)))#start from 0, step=2
        self.encoding[:,1::2]=torch.cos(pos/(10000**((_2i+1)/d_model)))#start from 1, step=2
        #Based on code, the 0 dimension of self.encoding is the position encode parameter and the 1 dimension is the length of str

    
    def forward(self,x):
        with torch.no_grad():
            seq_len=x.size(0)
            # print("seq:",seq_len)
        return self.encoding[:seq_len,:]


class PictureEmbedding(nn.Module):
    def __init__(self,picture_size,patch_size,device):
        super(PictureEmbedding,self).__init__()
        # self.pos=nn.Linear(picture_size[0]*picture_size[1]*3,picture_size[0]*picture_size[1]*3) #Can't do that, too large linear layer
        self.width=picture_size[1]
        self.height=picture_size[2]
        self.channel_num=picture_size[3]
        self.batch_size=picture_size[0]
        self.patch_size=patch_size
        # self.positional_embedding=PositionEmbedding(d_model,max_len=self.length*self.height*3,device=device)

    def forward(self,x):
        with torch.no_grad():
            # print(x.size())
            patched_image=x.unfold(0,self.patch_size,self.patch_size)
            # print(patched_image.size())
            patched_image=patched_image.unfold(1,self.patch_size,self.patch_size)
            # print(patched_image.size())
            patched_image=patched_image.reshape(-1,self.patch_size**2,self.channel_num)
            # print(patched_image.size())
            patched_image=patched_image.permute(0,2,1)
            patched_image=patched_image.reshape(int(self.width*self.height*self.channel_num/(self.patch_size**2)),self.patch_size**2)
        # patched_image.requires_grade=True
        # print(patched_image.requires_grad)
            return patched_image




class PictureEncoding(nn.Module):
    def __init__(self,picture_size,patch_size,drop_prob,device):
        super(PictureEncoding,self).__init__()
        self.tok_emb=PictureEmbedding(picture_size,patch_size,device=Device)
        self.pos_emb=PositionEmbedding(patch_size**2,picture_size[1]*picture_size[2]*picture_size[3],device)
        self.drop_out=nn.Dropout(p=drop_prob)# regular way of normalization, ramdomly set some of the output into 0
        self.patch_size=patch_size
    def forward(self,x):
        with torch.no_grad():#changed
            batch_size,width,height,channel=x.size()
            
            out=torch.zeros(batch_size,int(width*height*channel/(self.patch_size**2)),self.patch_size**2).to(Device)
            for batch in range(batch_size):
                
                tok_emb=self.tok_emb(x[batch])
                pos_emb=self.pos_emb(tok_emb)
                out[batch]=self.drop_out(tok_emb+pos_emb)
                # out[batch]=tok_emb+pos_emb
        # out.requires_grad=True
        # print(out.requires_grad)
        return out
    

Device=("cuda" if torch.cuda.is_available() else "cpu")
# print(Device)
# x=torch.randn([10,480,480,3]).to(Device)
# pictureencoder=TransformerEmbedding(x.size(),d_model=500,drop_prob=0.1,device=Device)
# x=pictureencoder(x)
# print("x: ",x.size())

class TokenEmbedding(nn.Module):
    def __init__(self,d_model):
        super(TokenEmbedding,self).__init__()
        self.d_model=d_model
    
    def forward(self,x):
        with torch.no_grad():
            len=x.size(0)
            x=x.repeat(1,self.d_model).reshape(len,self.d_model)
        return x

class TransformerEmbedding(nn.Module):
    def __init__(self,d_model,max_len,drop_prob,device):
        super(TransformerEmbedding,self).__init__()
        self.tok_emb=TokenEmbedding(d_model)
        # print("d_model",d_model)
        # print("max_len",max_len)
        self.pos_emb=PositionEmbedding(d_model,max_len,device)
        self.drop_out=nn.Dropout(p=drop_prob)# regular way of normalization, ramdomly set some of the output into 0
        self.d_model=d_model
    def forward(self,x):
        with torch.no_grad():#changed
            batches=x.size(0)
            time=x.size(1)
            dimension=self.d_model
            out=torch.zeros(batches,time,dimension).to(Device)
            for batch in range(batches):
                tok_emb=self.tok_emb(x[batch])
                # print("tok_emb:",tok_emb.size())
                pos_emb=self.pos_emb(x[batch])
                # print('pos_emb:',pos_emb.size())
                out[batch]=self.drop_out(tok_emb+pos_emb)
                # out[batch]=tok_emb+pos_emb
        # out.requires_grad=True
        # print(out.requires_grad)
        return out





#Multi_head_attention

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,n_head):
        super(MultiHeadAttention,self).__init__()
        self.n_head=n_head# the number of head
        self.d_model=d_model
        self.w_q=nn.Linear(d_model,d_model)
        self.w_k=nn.Linear(d_model,d_model)
        self.w_v=nn.Linear(d_model,d_model)
        self.w_combine=nn.Linear(d_model,d_model)
        self.softmax=nn.Softmax(dim=-1)
    
    def forward(self,q,k,v,mask=None):
        # print("q:",q.size())
        batch,time,dimension=q.size()
        n_d=self.d_model//self.n_head
        q,k,v=self.w_q(q),self.w_k(k),self.w_v(v)
        # print("q=",q.size())
        q=q.view(batch,time,self.n_head,n_d).permute(0,2,1,3)
        k=k.view(batch,time,self.n_head,n_d).permute(0,2,1,3)
        v=v.view(batch,time,self.n_head,n_d).permute(0,2,1,3)
        score=torch.matmul(q,k.transpose(2,3))/math.sqrt(n_d)
        if mask is not None:
            score=score.masked_fill(mask==0,-10000)#mask code
        score=self.softmax(score)@v# @: matrix multiplication
        score=score.permute(0,2,1,3).contiguous().view(batch,time,dimension)
        output=self.w_combine(score)
        return output

#Encoding
class LayerNorm(nn.Module):
    def __init__(self,d_model,eps=1e-12):
        super(LayerNorm,self).__init__()
        self.gamma=nn.Parameter(torch.ones(d_model))
        self.beta=nn.Parameter(torch.zeros(d_model))
        self.eps=eps#stability,extremely small number
    def forward(self,x):
        mean=x.mean(-1,keepdim=True)
        # var=x.var(-1,unbiased=False,keepdim=True)
        var=x.var(-1,unbiased=True,keepdim=True)
        #calculate the average number and variance of the last dimension of input x
        out=(x-mean)/torch.sqrt(var+self.eps)
        out=self.gamma*out+self.beta
        return out
    
class PositionwiseForward(nn.Module):
    def __init__(self,d_model,hidden,dropout=0.1):
        super(PositionwiseForward,self).__init__()
        self.fc1=nn.Linear(d_model,hidden)
        self.fc2=nn.Linear(hidden,d_model)
        self.dropout=nn.Dropout(dropout)
    
    def forward(self,x):
        x=self.fc1(x)
        x=F.relu(x)
        x=self.dropout(x)
        x=self.fc2(x)
        return x

class EncoderLayer(nn.Module):#single Encoder layer
    def __init__(self,d_model,hidden,n_head,dropout=0.1):
        super(EncoderLayer,self).__init__()
        self.attention=MultiHeadAttention(d_model,n_head)
        self.layernorm1=LayerNorm(d_model)
        self.dropout1=nn.Dropout(dropout)
        self.ffn=PositionwiseForward(d_model,hidden,dropout)
        self.layernorm2=LayerNorm(d_model)
        self.dropout2=nn.Dropout(dropout)
    def forward(self,x,mask=None):
        _x=x #save the origin input into _x,for calculating residual shrinkage
        # print('Attention:',x.size())
        x=self.attention(x,x,x,mask)#let the x be the q,k,v and calculate the attention score
        x=self.dropout1(x)
        x=self.layernorm1(x+_x) #residual shrinkage link
        _x=x
        x=self.ffn(x)
        x=self.dropout2(x)
        x=self.layernorm2(x+_x)
        return x

class Encoder(nn.Module):
    def __init__(self,picture_size,patch_size,hidden,n_head,n_layer,dropout=0.1,device="cpu"):
        super(Encoder,self).__init__()
        self.embedding=PictureEncoding(picture_size,patch_size,dropout,device)
        self.layers=nn.ModuleList( 
            [
                EncoderLayer(patch_size*patch_size,hidden,n_head)
                for _ in range(n_layer)#create n_layer Encoderlayers
            ]
        )

    def forward(self,x,s_mask=None):
        x=self.embedding(x)
        for layer in self.layers:
            x=layer(x,s_mask) #pass
        return x

#Decoding
class DecoderLayer(nn.Module):
    def __init__(self,d_model,ffn_hidden,n_head,dropout):
        super(DecoderLayer,self).__init__()
        self.attention1=MultiHeadAttention(d_model,n_head)
        self.norm1=LayerNorm(d_model)
        self.Dropout=nn.Dropout(dropout)
        self.cross_attention=MultiHeadAttention(d_model,n_head)
        self.norm2=LayerNorm(d_model)
        self.Dropout2=nn.Dropout(dropout)
        self.ffn=PositionwiseForward(d_model,ffn_hidden,dropout)
        self.norm3=LayerNorm(d_model)
        self.Dropout3=nn.Dropout(dropout)
    
    def forward(self,dec,enc,t_mask=None,s_mask=None):
        #dec is the output of the decoder,enc is the output of the encoder
        #t_mask, s_mask is the mask of the target, s_mask is the source mask used in encoder
        
        #first layer block, feeding the output of the last decoder
        _x=dec#saving the origin input
        x=self.attention1(dec,dec,dec,t_mask)
        x=self.Dropout(x)
        x=self.norm1(x+_x)
        _x=x

        #second layer block: feeding the output of encoder into the multi_head_attention
        x=self.cross_attention(x,enc,enc,s_mask)
        x=self.Dropout2(x)
        x=self.norm2(x+_x)
        _x=x#?

        #third layer block: feeding the output of the decoder corss-multi-attention
        x=self.ffn(x)
        x=self.Dropout3(x)
        x=self.norm3(x+_x)
        return x

class Decoder(nn.Module):
    def __init__(self,dec_voc_size,max_len,d_model,ffn_hidden,n_head,n_layer,dropout,device):
        super(Decoder,self).__init__()
        self.embedding=TransformerEmbedding(d_model,max_len,dropout,device)
        self.layers=nn.ModuleList(
            [
                DecoderLayer(d_model,ffn_hidden,n_head,dropout)
                for _ in range(n_layer)
            ]
        )
        # self.fc=nn.Linear(d_model,dec_voc_size)

    def forward(self,dec,enc,t_mask=None,s_mask=None):#need two input: dec for the last output of the decoder, enc for the output of the encoder

        dec=self.embedding(dec)
        for layer in self.layers:
            dec = layer(dec,enc,t_mask,s_mask)
        # dec=self.fc(dec)
        return dec


#Transformer final model

class Transformer(nn.Module):
    def __init__(self,
                 src_pad_idx,#source padding index
                 trg_pad_idx,#target padding index
                 picture_size,#encoder vocab size
                 dec_voc_size,#decoder vocab size
                 patch_size,#dimension of the feature
                 n_heads,
                 ffn_hidden,
                 max_len,
                 n_layers,
                 drop_prob,
                 device
                 ):
        super(Transformer,self).__init__()
        d_model=patch_size**2
        self.encoder=Encoder(picture_size,patch_size=patch_size,hidden=ffn_hidden,n_head=n_heads,n_layer=n_layers,dropout=drop_prob,device=device)
        self.decoder=Decoder(dec_voc_size,max_len,d_model,ffn_hidden,n_heads,n_layers,drop_prob,device)
        self.src_pad_idx=src_pad_idx
        self.trg_pad_idx=trg_pad_idx
        self.device=device
        self.softmax=nn.Softmax(dim=-1)
        self.fc=nn.Linear(max_len*d_model,dec_voc_size)

    def make_pad_mask(self,q,k,pad_idx_q,pad_idx_k):
        len_q,len_k=q.size(1),k.size(1)
        q=q.ne(pad_idx_q).unsqueeze(1).unsqueeze(3)
        q=q.repeat(1,1,1,len_k)
        k=k.ne(pad_idx_k).unsqueeze(1).unsqueeze(2)
        k=k.repeat(1,1,len_q,1)
        mask=q&k
        return mask

    def make_casual_mask(self,q,k):
        len_q,len_k=q.size(1),k.size(1)
        mask=torch.tril(torch.ones(len_q,len_k)).type(torch.BoolTensor).to(self.device)
        return mask

    def forward(self,src,trg):
        # print(src.size())
        # src_mask=self.make_pad_mask(src,src,self.src_pad_idx,self.src_pad_idx)
        
        # enc=self.encoder(src,src_mask)
        # out=self.decoder(trg,enc,trg_mask,src_mask)
        enc=self.encoder(src)
        # print('enc',enc.size())
        trg_mask=self.make_pad_mask(trg,trg,self.trg_pad_idx,self.trg_pad_idx).to(Device)*self.make_casual_mask(trg,trg).to(Device)
        out=self.decoder(trg,enc,trg_mask)
        out=out.view(-1,out.size(-1)*out.size(-2))
        # print("out",out.size())
        out=self.fc(out)
        # out=self.softmax(out)

        return out



#DataLoader
f=open("images/word_counts.txt","r",encoding="UTF-8")
tagtext=f.readlines()
f.close()
for i in range(len(tagtext)):
    tag.append(tagtext[i].split(":")[0])



def DataLoader(choice_list=[],index=None):
    # print("choice",choice_list)
    if not choice_list:
        choice_list=check_dataset_exist()#one epoch finished, reload the epoch

    if index==None:
        index=random.randint(0,len(choice_list)-1)
    # print(index)
    # print("loading image:",choice_list[index],"    remaining image:",len(choice_list))
    img=load_image(choice_list[index])

    img=resize_img(img).permute(0,2,3,1)
    ftag=open("images/tags/"+str(choice_list[index])+".txt","r",encoding="UTF-8")
    tag_list=ftag.readlines()
    ftag.close()
    tag_list=tag_list[:15]

    choice_list.pop(index)#this object has been taken

    tag_token=[0]#<start>
    for i in range(len(tag_list)):
        tag_list[i]=tag_list[i][:len(tag_list[i])-1]
        # print(tag_list[i])
        tag_token.append(tag2token(tag_list[i]))
    tag_token = [x for x in tag_token if x is not None or x == 0]
    tag_tensor=torch.tensor(tag_token).unsqueeze(0).repeat(len(tag_token),1)
    # print(tag_tensor.size())
    img=img.repeat(len(tag_token),1,1,1)
    tag_token.append(1)#<end>
    tag_token.pop(0)
    tag_token=torch.tensor(tag_token)

    return img,tag_tensor,tag_token,choice_list


# imgtest,tag_tensor_test,tag_token_test,choice_list_test=DataLoader()
# print(imgtest.size())
# print(tag_tensor_test.size())
# print(len(tag_token_test))



#Train



model=Transformer(
    src_pad_idx=0,
    trg_pad_idx=2,
    picture_size=[20,224,224,3],
    dec_voc_size=len(tag),
    patch_size=16,
    n_heads=8,
    n_layers=6,
    ffn_hidden=6,
    max_len=588,
    drop_prob=0.1,
    device=Device,
).to(Device)

loss_fn=torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
# optimizer=torch.optim.SGD(model.parameters(),lr=1e-3)

def train(epoch_num=20):
    model.train()
    choice_list=[]
    epoch=0
    loss_sum=0
    while epoch<epoch_num:
        
        img,tag_tensor,tag_token,choice_list=DataLoader(choice_list=choice_list)
        img=img.to(Device)
        padding_matrix=torch.ones(img.size(0),588-(len(tag_tensor)))*2
        tag_tensor=torch.cat((tag_tensor,padding_matrix),dim=1)
        tag_tensor=tag_tensor.to(Device)
        pred_label=model(img,tag_tensor)
        tag_token=tag_token.to(Device)
        # pred_token=torch.argmax(pred_label,1)
        # print("pred_token",pred_token.size())
        # print("tag_token",tag_token.size())
        # pred_token=pred_token.to(torch.float64)
        # tag_token=tag_token.to(torch.float64)
        loss=loss_fn(pred_label,tag_token)
        if not math.isnan(loss.item()):
            loss_sum+=loss.item()
        
            optimizer.zero_grad()
            # loss.requires_grad_(True)
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
            optimizer.step()
        if choice_list==[]:
            epoch=epoch+1
            print(epoch,":",loss_sum)
            torch.save(model.state_dict(),"Vit.pth")
            loss_sum=0
        
   
def following_train(epoch_num=20):
    print("loading model")
    model.load_state_dict(torch.load("Vit.pth"))
    model.train()
    print("start training")
    choice_list=[]
    epoch=0
    loss_sum=0
    while epoch<epoch_num:
        
        img,tag_tensor,tag_token,choice_list=DataLoader(choice_list=choice_list)
        img=img.to(Device)
        padding_matrix=torch.ones(img.size(0),588-(len(tag_tensor)))*2
        tag_tensor=torch.cat((tag_tensor,padding_matrix),dim=1)
        tag_tensor=tag_tensor.to(Device)
        pred_label=model(img,tag_tensor)
        tag_token=tag_token.to(Device)
        # pred_token=torch.argmax(pred_label,1)
        # print("pred_token",pred_token.size())
        # print("tag_token",tag_token.size())
        # pred_token=pred_token.to(torch.float64)
        # tag_token=tag_token.to(torch.float64)
        loss=loss_fn(pred_label,tag_token)

        if not math.isnan(loss.item()):
            loss_sum+=float(loss.item())
            optimizer.zero_grad()
            loss.requires_grad_(True)
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
            optimizer.step()
   
        else:
            print("pred_label:",pred_label.size())
            print("tag_token",tag_tensor.size())
            print(img.size())      

            # print(pred_label[0])

        print("epoch",epoch,"loss",loss.item())
        
        
        if choice_list==[]:
            epoch=epoch+1
            print(epoch,":",loss_sum)
            torch.save(model.state_dict(),"Vit.pth")
            loss_sum=0
    



# train()
following_train(30)








#Test



# x=torch.randn([50,224,224,3]).to(Device)
# trg=torch.zeros(50,1).to(Device)
# padding_matrix=torch.ones(50,588-1).to(Device)*2
# trg=torch.concat((trg,padding_matrix),dim=1)
# y=model(x,trg)
# print(y.size())
