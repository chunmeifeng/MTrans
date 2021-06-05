from .sr_single import _C as SRS
from .sr_multi_early import _C as SRME
from .sr_multi_cross import _C as SRMC
from .mcsr import _C as MCSR
from .edsr import _C as EDSR


from .reconstruction_multi_cross import _C as RMC
from .reconstruction_multi_early import _C as RME
from .reconstruction_single import _C as RS
from .unet_reconstruction_single import _C as URS
from .unet_reconstruction_multi import _C as URM


config_factory = {
    'sr_single': SRS,
    'sr_multi_early': SRME,
    'sr_multi_cross': SRMC,
    'mcsr': MCSR,
    'edsr': EDSR,


    'reconstruction_multi_cross': RMC,
    'reconstruction_multi_early' : RME,
    'reconstruction_single': RS,
    'unet_reconstruction_single': URS,
    'unet_reconstruction_multi': URM,

}


def build_config(mode):

    assert mode in config_factory.keys(), 'unknown config'

    return config_factory[mode]
