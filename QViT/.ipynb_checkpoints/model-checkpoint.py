import torch
import torch.nn as nn
from math import sqrt
import math
from .parametrizations import convert_array
from .circuits import *
from torch.utils.data import Dataset
import warnings
device='cuda'
#################### 1st Hybrid Approach
class EncoderLayer_hybrid1(nn.Module):
    def __init__(self,Token_Dim,Embed_Dim,head_dimension,ff_dim=None):
        super(EncoderLayer_hybrid1,self).__init__()
        self.MultiHead_Embed_Dim = Embed_Dim//head_dimension
        self.heads =  nn.ModuleList([AttentionHead_Hybrid2(Token_Dim,self.MultiHead_Embed_Dim) for i in range(head_dimension)]) 
        # self.merger = construct_FNN([ff_dim,Embed_Dim],activation=nn.GELU)
        self.merger = QLayer(measure_value,[3*Embed_Dim],int(Embed_Dim))
        self.norm1 = nn.LayerNorm([Embed_Dim],elementwise_affine=False)
    def forward(self,input1):
        
        input1_norm = self.norm1(input1)
        head_result = torch.cat(  [m(input1_norm[...,(i*self.MultiHead_Embed_Dim):( (i+1)*self.MultiHead_Embed_Dim)]) for i,m in enumerate(self.heads)],dim=-1)
        res = self.merger(head_result.flatten(0,1)).reshape(head_result.shape)+input1
        return res

    

# has a quantum circuit based attention head then we construct a neural netwoek 
class EncoderLayer_hybrid2(nn.Module):
    def __init__(self,Token_Dim,Embed_Dim,head_dimension,ff_dim=None):
        super(EncoderLayer_hybrid2,self).__init__()
        self.MultiHead_Embed_Dim = Embed_Dim//head_dimension
        self.heads =  nn.ModuleList([AttentionHead_Hybrid2(Token_Dim,self.MultiHead_Embed_Dim) for i in range(head_dimension)]) 
        self.merger = construct_FNN([ff_dim,Embed_Dim],activation=nn.GELU)
        self.norm1 = nn.LayerNorm([Embed_Dim],elementwise_affine=False)
    def forward(self,input1):
        
        input1_norm = self.norm1(input1)
        head_result = torch.cat(  [m(input1_norm[...,(i*self.MultiHead_Embed_Dim):( (i+1)*self.MultiHead_Embed_Dim)]) for i,m in enumerate(self.heads)],dim=-1)
        res = self.merger(head_result)+input1
        return res


# This class defines a single hybrid attention head using both classical and quantum computations.
class AttentionHead_Hybrid2(nn.Module):
    def __init__(self,Token_Dim,MultiHead_Embed_Dim):
        super(AttentionHead_Hybrid2,self).__init__()

        self.MultiHead_Embed_Dim = MultiHead_Embed_Dim
        
        
        self.norm = nn.LayerNorm(MultiHead_Embed_Dim,elementwise_affine=False)
        # the v, q and k is the same as the previous classcial attention head, but whith q layers simulating a quantum circuit
        self.V = QLayer(measure_value,[3*MultiHead_Embed_Dim],int(MultiHead_Embed_Dim))
        self.Q = QLayer(measure_query_key,[3*MultiHead_Embed_Dim+1],int(MultiHead_Embed_Dim))
        self.K = QLayer(measure_query_key,[3*MultiHead_Embed_Dim+1],int(MultiHead_Embed_Dim))
        print("Quantum Circuit for V layer:")
        self.V.print_circuit()
        print("Quantum Circuit for Q layer:")
        self.Q.print_circuit()
        print("Quantum Circuit for K layer:")
        self.K.print_circuit()
        
        self.attention = lambda A,V : torch.bmm(nn.Softmax(dim=-1)(A/MultiHead_Embed_Dim**.5),V)
        self.flattener = lambda A: A.flatten(0,1)

    def forward(self,input1):

        flat_input  = self.flattener(input1)

        V = self.V(flat_input).reshape(input1.shape)
        # V = self.V(flat_input).reshape(input1.shape[0],input1.shape[2],input1.shape[1]).permute(0,2,1)
        
        Q = self.Q(flat_input).reshape(*input1.shape[:2])
        K = self.K(flat_input).reshape(*input1.shape[:2])
        A = torch.empty((*input1.shape[:-1],input1.shape[-2]),device=input1.device)
        for j in range(input1.shape[-2]):
            A[...,j] = -(Q-K[...,j][...,None])**2
        return self.attention(A,V)
        
        

#################### Classical Approach ######################
# we are now defining encoder layers
class EncoderLayer(nn.Module):
    def __init__(self,Token_Dim,Embed_Dim,head_dimension,ff_dim):
        super(EncoderLayer,self).__init__()
        # layerNorm is a function in pytorch This is a type of normalization layer used in neural networks
        self.norm1 = nn.LayerNorm([Embed_Dim],elementwise_affine=False)
        self.norm2 = nn.LayerNorm([Embed_Dim],elementwise_affine=False)
        # now we define the multi head attention layer
        self.MHA = MultiHead(Token_Dim,Embed_Dim,head_dimension)
        self.merger = construct_FNN([ff_dim,Embed_Dim],activation=nn.GELU)
    def forward(self,input1):
      
        input1_norm = self.norm1(input1)
        res = self.MHA(input1_norm)+input1

        return self.merger(self.norm2(res))+res

# this defines all the attention heads. we define the number of attention head and call a function called attention head with 
# the dimention. Then we concatinate all that and return a multiHEad attention layer
class MultiHead(nn.Module):
    def __init__(self,Token_Dim,Embed_Dim,head_dimension):
        super(MultiHead,self).__init__()
        self.MultiHead_Embed_Dim = Embed_Dim//head_dimension
        
        self.heads =  nn.ModuleList([AttentionHead(Token_Dim,self.MultiHead_Embed_Dim) for i in range(head_dimension)])
    def forward(self,input1):

        
        return torch.cat(  [m(input1[...,(i*self.MultiHead_Embed_Dim):( (i+1)*self.MultiHead_Embed_Dim)]) for i,m in enumerate(self.heads)],dim=-1)
    
class AttentionHead(nn.Module):
    def __init__(self,Token_Dim,embed_per_head_dim):
        super(AttentionHead,self).__init__()
        self.Q = nn.Linear(embed_per_head_dim,embed_per_head_dim,bias=False)
        # self.V: A linear transformation for the Value vectors. This layer projects the input embeddings into a new space. The Value holds the information we will extract using the attention mechanism.
        self.V = nn.Linear(embed_per_head_dim,embed_per_head_dim,bias=False)
        # self.K: A linear transformation for the Key vectors. The Key is used to match with the Query to determine how much attention should be given to each input token.
        self.K = nn.Linear(embed_per_head_dim,embed_per_head_dim,bias=False)
        # self.Q: A linear transformation for the Query vectors. This layer projects the input embeddings (of size embed_per_head_dim) into a new space of the same dimensionality. The Query represents what we want to focus on in the input.
        self.soft = nn.Softmax(dim=-1)

    def attention(self,Q,K,V):
        # This method implements the scaled dot-product attention mechanism
        return torch.bmm(self.soft(torch.bmm(Q,K.permute(0,2,1) )/math.sqrt(Q.shape[-1])),V)
    def forward(self,input1):

        Q = self.Q(input1)
        K = self.K(input1)
        V = self.V(input1)
        # The attention method is called with the Query, Key, and Value vectors to compute the final output of the attention head. 
        # This output is a combination of all tokens' Value vectors weighted by their attention scores
        return self.attention(Q,K,V)




#
############################### Shared Functions for all transformer architectures used here ####################
# THis transofmer consists of Input Representation, Self-Attention Mechanism
class Transformer(nn.Module):
    def __init__(self,Token_Dim,Image_Dim,head_dimension,n_layers,Embed_Dim,ff_dim,pos_embedding,classifying_type,attention_type):
        super(Transformer,self).__init__()
        self.cls_type = classifying_type
        self.embedding = pos_embedding
        # this is positional embedding added to the embeddings to provide information about the order of the sequence
        self.pos_embedding = nn.parameter.Parameter(torch.tensor( [ math.sin(1/10000**((i-1)/Embed_Dim))  if i%2==1 else math.cos(i/10000**((i-1)/Embed_Dim)) for i in range(Embed_Dim) ]))
        self.pos_embedding.requires_grad = False
        # this is the attention layers defined
        attention_dict={'hybrid2':EncoderLayer_hybrid2,'classic':EncoderLayer,'hybrid1':EncoderLayer_hybrid1}
        if self.cls_type=='cls_token':Token_Dim+=1
        
        self.encoder_layers = nn.ModuleList([ attention_dict[attention_type](Token_Dim,Embed_Dim,head_dimension,ff_dim) for i in range(n_layers)])
        
        if self.cls_type == "cls_token":self.class_token = nn.parameter.Parameter(torch.rand(Embed_Dim,dtype=torch.float32).abs().to('cuda')/math.sqrt(Embed_Dim))
        
        self.embedder = nn.Linear(Image_Dim,Embed_Dim)
        
        
        if self.cls_type == 'max':
            self.final_act =  lambda temp: temp[-1].max(axis=1).values
        if self.cls_type == 'mean':
            self.final_act =  lambda temp: temp[-1].mean(axis=1)
        if self.cls_type == 'sum':
            self.final_act =  lambda temp:temp[-1].sum(axis=1)
        if self.cls_type == 'cls_token':
            self.final_act =  lambda temp:temp[-1][:,0]

    def forward(self,input1):
        if self.cls_type == "cls_token":
            cls_token = self.class_token.expand(input1.shape[0],1,-1)
            input1_ = torch.cat( (cls_token, self.embedder(input1)),axis=1)
        else: 
            input1_ = self.embedder(input1)
        if self.embedding: temp =[input1_+self.pos_embedding[None,None,:]]
        else: temp = [input1_]
        
        
        for i,m in enumerate(self.encoder_layers):temp.append( m(temp[i]))

        return self.final_act(temp)

# this takes the params token dimension - embedding size of each token, Image_Dim - size of the input images 14x14 pixels
# head_dimension - dimention of each attention head 8, n_layers - Number of transformer layers 2, FC_layers - architecture of the feedforward neural network (classifier), attention_type - classical attention or hybrid, pos_embedding - Positional embedding encodes spatial relationships between image patches, True
# classifying_type - how the model classifies the output 0 to 10, Embed_Dim - dimention of each token 16, ff_dim - Feedforward dimension inside the transformer 32
# This line initializes a Transformer module - processing the tokenized image and applying self-attention to learn important relationships between different parts of the image
# The classifier takes the output from the transformer and applies a series of transformations (defined by FC_layers and activations) to produce the final classification
class HViT(nn.Module):
    def __init__(self,Token_Dim,Image_Dim,head_dimension,n_layers,FC_layers,attention_type,pos_embedding,classifying_type,Embed_Dim,ff_dim):
        super(HViT,self).__init__()
        self.transformer = Transformer(Token_Dim,Image_Dim,head_dimension,n_layers,Embed_Dim,ff_dim,pos_embedding,classifying_type,attention_type)
        self.classifier = construct_FNN(FC_layers,activation=nn.LeakyReLU)
    def forward(self,input1):
        return self.classifier(self.transformer(input1))

# constructs a fully connected neural network (FNN) using the FC_layers provided by the user
def construct_FNN(layers,activation=nn.GELU,output_activation=None,Dropout = None):
    layer = [j for i in layers for j in [nn.LazyLinear(i),activation()] ][:-1]
    if Dropout:
        layer.insert(len(layer)-2,nn.Dropout(Dropout))
    if output_activation is not None:
        layer.append(output_activation)
    return nn.Sequential(*layer)
