import openpifpaf

from .mpii import MPII


def register():
    openpifpaf.DATAMODULES['mpii'] = MPII
