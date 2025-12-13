from omegaconf  import OmegaConf
from dataclasses import dataclass

@dataclass
class Config:
    num_channels: int = 3 #  default CIFAR100
    image_size: int = 32  # default CIFAR100
    num_layers: int = 6
    in_features: int = 512
    patch_size: int = 3
    conv_stride: int = 3
    vq_beta:float = 0.25
    out_features: int = 512 
    num_heads: int = 16 
    num_latent_tokens: int = 128 
    codebook_size: int = 4096 
    token_size: int = 512 
    dropout: float = 0.3
    enable_reverse_difflin:bool = False 
    bias:bool = False 
    enable_cudnn_benchmark: bool = True
    enable_cudnn_deterministic: bool = False
    config_save_path: str ="config.yaml"


def create_config_(num_layers: int ,
                   num_channels: int ,
                   image_size: int ,
                   in_features: int ,
                   patch_size: int ,
                   conv_stride: int ,
                   vq_beta: float,
                   dropout: float,
                   out_features: int =512,
                   num_heads: int =16,
                   num_latent_tokens: int =128,
                   codebook_size: int =4096,
                   enable_reverse_difflin: bool =False,
                   token_size: int = 512,
                   bias: bool = False,
                   enable_cudnn_benchmark: bool = True,
                   enable_cudnn_deterministic: bool = False,
                   config_save_path="config.yaml"):
    conf = Config(
        num_channels=num_channels,
        num_layers=num_layers,
        image_size=image_size,
        in_features=in_features,
        out_features=out_features,
        patch_size=patch_size,
        conv_stride=conv_stride,
        vq_beta=vq_beta,
        dropout=dropout,
        num_heads=num_heads,
        token_size=token_size,
        codebook_size=codebook_size,
        num_latent_tokens=num_latent_tokens,
        enable_reverse_difflin=enable_reverse_difflin,
        bias=bias,
        enable_cudnn_benchmark=enable_cudnn_benchmark,
        enable_cudnn_deterministic= enable_cudnn_deterministic,
        config_save_path=config_save_path
    )
    conf = OmegaConf.structured(conf)
    OmegaConf.save(config=conf,f=config_save_path)
        
    return conf