"""

lightly modified quantization  from 
https://github.com/CompVis/taming-transformers/blob/master/taming/modules/vqvae/quantize.py

"""

import torch
import torch.nn as nn
from torch import Tensor


class VectorQuantizer(nn.Module):
    def __init__(self,codebook_size:int,
                    token_size:int,
                    beta
                 ):
        self.codebook_size = codebook_size
        self.token_size  = token_size
        self.beta= beta
        self.codebook = nn.Embedding(codebook_size,token_size)
        self.codebook.weight.data.uniform_(-1.0 / self.codebook_size , 1.0/ self.codebook_size)
        
    def forward(self,x:Tensor)->Tensor:
        x = torch.rearrange(x,'b c h w -> b h w c').contiguous()
        x_flattened = x.view(-1,self.token_size)
        
        # d.shape = [ */token_size, token_size] 
        d = torch.sum(x_flattened**2 , dim=1,keepdim=True) + \
            torch.sum(self.codebook.weight**2,dim=1) - 2  * \
            torch.einsum('bd,dn->bn',x_flattened, torch.rearrange(self.codebook.weight,'n d -> d n'))
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        
        min_encodings = torch.zeros( min_encoding_indices.shape[0] , self.codebook_size).to(x)
        min_encodings.scatter_(1,min_encoding_indices,1)
        
        # get quantize embeddings        
        x_quantized  = self.codebook(min_encoding_indices).view(x.shape)
        
        # compute loss for embedding        
        loss =  self.beta * torch.mean( (x.detach() - x_quantized  )) +\
                torch.mean( (x - x_quantized.detach()) ** 2)
        # preserve gradiensts
        x_quantized = x + (x_quantized - x).detach()
        
        e_mean = torch.mean( min_encodings, dim=0)
        perplexity = torch.exp( -torch.sum( e_mean * torch.log(e_mean + 1e-10 )))
        # reshape to match original input shape
        x_quantized = torch.rearrange(x_quantized, 'b h w c  -> b c h w').contiguous()
        
        return x_quantized,loss,(perplexity,min_encodings,min_encoding_indices)
    def get_codebook_entry(self,indices,shape):
        x_quantized = self.codebook(indices)
        if shape is not None:
            x_quantized = x_quantized.view(shape)
            # reshape to match original input shape
            x_quantized = torch.rearrange(x_quantized, 'b h w c -> b c h w').contiguous()
        return x_quantized