from .cmmt_sr_single import build_model as SRS
from .cmmt_sr_multi_early import build_model as SRME
from .cmmt_sr_multi_cross import build_model as SRMC

from .cmmt_reconstruction_multi_cross import build_model as RMC
from .cmmt_reconstruction_multi_early import build_model as RME
from .cmmt_reconstruction_single import build_model as RS

from .contrast_model.unet import build_model as UNET
from .contrast_model.unet_multi import build_model as UNETMULTI

from .contrast_model.mcsr import build_model as MCSR
from .contrast_model.edsr import build_model as EDSR


model_factory = {
    'sr_single': SRS,
    'sr_multi_early': SRME,
    'sr_multi_cross': SRMC,

    'reconstruction_multi_cross': RMC,
    'reconstruction_multi_early': RME,
    'reconstruction_single': RS,

    'mcsr': MCSR,
    'edsr': EDSR,
    'unet_reconstruction_single': UNET,
    'unet_reconstruction_multi': UNETMULTI,

}

def build_model_from_name(args, model_name):
    assert model_name in model_factory.keys(), 'unknown model name'
    return model_factory[model_name](args)