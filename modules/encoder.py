from Difflinear import DIFFLINattn,SDPA
from torch.nn import functional as F, Module,Parameter,RMSNorm,Sequential,Conv2d,Linear,Dropout , init
from torch import randn , Tensor,matmul , concat

class Encoder(Module):
  def __init__(self, in_features:int,
               out_features:int,
               patch_size:int,
               layers:int,
               token_size:int,
               num_latent_tokens:int,
               image_size:int,
               num_heads:int,
               bias:bool=False):
    super().__init__()
    self.width = out_features
    self.num_heads= num_heads
    scale = self.width ** - 0.5
    self.grid_size = image_size//patch_size
    self.num_latent_tokens = num_latent_tokens
    self.token_size = token_size
    self.class_embedding = Parameter(scale * randn(1,self.width))
    self.positional_embeddings = Parameter(scale * randn( self.grid_size ** 2 ,self.width))
    self.latent_token_positional_embeddings = Parameter(scale * \
                                              randn(self.num_latent_tokens ,\
                                              self.width))
    self.patch_embed = Conv2d(in_features,
                              out_features,
                              kernel_size=patch_size,
                              stride=patch_size)
    self.prenorm = RMSNorm(in_features)
    self.postnorm= RMSNorm(out_features)
    self.transformer = Sequential(
          *[DIFFLINattn(in_features=in_features if i == 0 else out_features,
                    embed_dim= out_features,
                    num_heads=self.num_heads,
                    layer=i+1
                       ) for i in range(layers)]
    )
    self.sequence_pooler = SeqPooler(in_features=out_features,
                                     bias=bias)#Pooling over all sequences
    self.conv_out = Conv2d(self.width,
                           self.token_size,
                           kernel_size=1,
                           bias=bias)
    self.apply(self.init_weights)

  def forward(self,x:Tensor , latent_tokens:Tensor)->Tensor:
    batch_size = x.shape[0]
    x = self.patch_embed(x).reshape(x.shape[0],x.shape[1],-1)
    x = x.permute(0,2,1) # B,seq_len ,width
    x = concat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)

    x = x + self.positional_embeddings.to(x.dtype)
    latent_tokens = _expand_token(latent_tokens,x.shape[0])
    x = concat([x,latent_tokens],dim=1)
    x = self.prenorm(x).permute(1,0,2)# B,Seqlen,D -> Seqlen,B,D
    x = self.transformer(x).permute(1,0,2)# Seqlen,B,D -> B,Seqlen,D 
    latent_tokens = self.postnorm(x[:,1+self.grid_size**2:])
    latent_tokens = latent_tokens.reshape(batch_size,self.width,self.num_latent_tokens,1)
    latent_tokens = self.conv_out(latent_tokens)
    latent_tokens = latent_tokens.reshape(batch_size,self.token_size,1,self.num_latent_tokens)
    return latent_tokens


  @staticmethod
  def init_weights(m:Module)->None:
    if isinstance(m,(Conv2d,Linear,Parameter)):
      init.orthogonal_(m.weight)

class SeqPooler(Module):
  """
  Sequence Pooling as in https://arxiv.org/pdf/2104.05704
  """
  def __init__(self,
               in_features:Tensor,
               bias:bool=False):
    super().__init__()
    self.attn_pooler = Linear(in_features,out_features=1,bias=bias)
    self.dropout = Dropout(0.3)
  def forward(self,x:Tensor)->Tensor:
    return self.dropout(
            matmul( F.softmax(self.attn_pooler(x).transpose(-2,-1)) ,x )
    )

def _expand_token(token, batch_size: int):
    return token.unsqueeze(0).expand(batch_size, -1, -1)
  
  
class TiTok(Encoder):
  def __init__(self, in_features: int, out_features: int,num_heads:int, patch_size: int, layers: int, token_size: int, num_latent_tokens: int, image_size: int, bias: bool = False):
    super().__init__(in_features, out_features, patch_size, layers, token_size, num_latent_tokens, image_size, bias)
    self.transformer = Sequential(
      *[SDPA(in_features=in_features,
             num_heads= num_heads,
             embed_dim=out_features,
             num_latent_tokens=num_latent_tokens,
             bias=bias) for _ in range(layers)]
    )
  def forward(self, x: Tensor, latent_tokens: Tensor) -> Tensor:
    return super().forward(x, latent_tokens)