import torch
from torch.nn import functional as F, Linear, Module, Parameter, Sequential , Dropout , RMSNorm , GELU ,init
from einops.layers.torch import Rearrange


@staticmethod
def init_weights(m:Module)->None:
    if isinstance(m,(Conv2d,Linear,Parameter)):
        init.orthogonal_(m.weight)
    if m.bias is not None:
        init.constant_(m.bias, 0)
 

class Decoder(Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.width = config.out_features
        self.num_heads= config.num_heads
        scale = self.width ** - 0.5
        self.mask_tokens =Parameter(scale * torch.randn( 1, 1, self.width))
        self.grid_size = config.image_size//config.patch_size
        self.token_size = config.token_size
        self.class_embedding = Parameter(scale * torch.randn(1,self.width))
        self.positional_embeddings = Parameter(scale * torch.randn( self.grid_size ** 2+1 ,self.width))
        self.latent_token_positional_embeddings = Parameter(
            scale * torch.randn(self.num_latent_tokens ,  self.width))
        self.decoder_embed = Linear(config.token_size,
                                  	                   config.width,
                                    	                   bias = config.bias)
        self.prenorm = RMSNorm(config.in_features)
        self.postnorm= RMSNorm(config.out_features)
        self.transformer = Sequential(
          *[DIFFLINattn(config=config,
                        layer=i+1,
                        kernel=F.elu) for i in range(config.num_layers)]
        )
        self.sequence_pooler = SeqPooler(config)#Pooling over all sequences
        #directly predicting rgb pixels as in 1d tokenzier/modelling/modules/blocks
        self.ffn  = Sequential(
          Conv2d(self.width,self.patch_size ** 2 * 3,1,padding = 0, bias = config.bias),
          Rearrange( 'b (p1 p2 c) h w -> b c (h p1) (w p2)',
                     p1 = self.patch_size , p2 = self.patch_size),)
        )
        # self.shuffle =  lambda x: x.reshape(-1,3,self.patch_size*self.image_size,
        #                                         self.patch_size * self.image_size)
        self.conv_out = Conv2d( nn.Conv2d(3, 3, 3, padding=1, bias=True))
        self.apply(self.init_weights)

    def forward(self,z_quantized:torch.Tensor , latent_tokens:torch.Tensor)->torch.Tensor:
      N,C,H,W = z_quantized.shape
      assert H == 1 and W == self.num_latent_tokens, f"{H} , {W} ,{ self.num_latent_tokens}"
      x  = z_quantized.reshape(N,C*H,W).permute(0,2,1) # N, num_latent_tokens,token_size
      x  = self.decoder_embed(x) # N, num_latent_tokens, width
      batch_size , seq_len ,_ = x.shape
      mask_tokens = self.mask_tokens.repeat( batch_size , self.grid_size ** 2 , 1).to(x.dtype)
      mask_tokens = torch.cat( [ _expand_token(self.class_embedding , mask_tokens.shape[0]).to(mask_tokens.dtype),
                                 mask_tokens ], dim = 1)
      mask_tokens += self.positional_embeddings.to(mask_tokens.dtype)
      x += self.latent_token_positional_embeddings[:seq_len]
      x += torch.cat([mask_tokens,x],dim=1)
      x  = self.prenorm(x) # N,  1+grid**2+num_latent_tokens,width
      x = x.permute(1,0,2) #  1+grid**2+ num_latent_tokens, N, width

      x = self.transformer( x )
      x = x.permute(1,0,2) # N, 1+grid**2+num_latent_tokens,width
      x = x[:,1:1+self.grid_size**2] #remove class embeddging
      x = self.postnorm( x )
      
      # N, grid_size **2 , width -> N , width, grid_size,grid_size
      x = x.permute(0,2,1).reshape(batch_size,self.width, self.grid_size,self.grid_size)
      x= self.ffn( x.contiguous() )
      x= self.conv_out( x )
      return x
      
