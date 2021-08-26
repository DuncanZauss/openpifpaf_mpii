import openpifpaf

from .mpii import MPII


def register():
    openpifpaf.DATAMODULES['mpii'] = MPII
    openpifpaf.CHECKPOINT_URLS['shufflenetv2k16-mpii'] = 'https://github.com/DuncanZauss/' \
        'openpifpaf_mpii/releases/download/v.0.1.0-alpha/mpii_sk16.pkl.epoch350'

