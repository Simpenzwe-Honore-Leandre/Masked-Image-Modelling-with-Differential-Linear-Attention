"""

lightly modified quantization  from 
https://github.com/CompVis/taming-transformers/blob/master/taming/modules/vqvae/quantize.py

"""


from Difflinear import DIFFLINattn
from torch.nn import Embedding, Module
from torch import  Tensor, einsum , sum as torchsum, rearrange, argmin , mean ,zeros, exp,log

class VectorQuantizer(Module):
    def __init__(self,codebook_size:int,
                    token_size:int,
                    beta
                 ):
        self.codebook_size = codebook_size
        self.token_size  = token_size
        self.beta= beta
        self.codebook = Embedding(codebook_size,token_size)
        self.codebook.weight.data.uniform_(-1.0 / self.codebook_size , 1.0/ self.codebook_size)
        
    def forward(self,x:Tensor)->Tensor:
        x = rearrange(x,'b c h w -> b h w c').contiguous()
        x_flattened = x.view(-1,self.token_size)
        
        # d.shape = [ */token_size, token_size] 
        d = torchsum(x_flattened**2 , dim=1,keepdim=True) + \
            torchsum(self.codebook.weight**2,dim=1) - 2  * \
            einsum('bd,dn->bn',x_flattened, rearrange(self.codebook.weight,'n d -> d n'))
        min_encoding_indices = argmin(d, dim=1).unsqueeze(1)
        
        min_encodings = zeros( min_encoding_indices.shape[0] , self.codebook_size).to(x)
        min_encodings.scatter_(1,min_encoding_indices,1)
        
        # get quantize embeddings        
        x_quantized  = self.codebook(min_encoding_indices).view(x.shape)
        
        # compute loss for embedding        
        loss =  self.beta * mean( (x.detach() - x_quantized  )) +\
                mean( (x - x_quantized.detach()) ** 2)
        # preserve gradiensts
        x_quantized = x + (x_quantized - x).detach()
        
        e_mean = mean( min_encodings, dim=0)
        perplexity = exp( -torchsum( e_mean * log(e_mean + 1e-10 )))
        # reshape to match original input shape
        x_quantized = rearrange(x_quantized, 'b h w c  -> b c h w').contiguous()
        
        return x_quantized,loss,(perplexity,min_encodings,min_encoding_indices)
    def get_codebook_entry(self,indices,shape):
        x_quantized = self.codebook(indices)
        if shape is not None:
            x_quantized = x_quantized.view(shape)
            # reshape to match original input shape
            x_quantized = rearrange(x_quantized, 'b h w c -> b c h w').contiguous()
        return x_quantized