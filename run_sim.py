# Running delenisng code

# from external module
import numpy as np

# from this directory
import local
import tools_cmb

kwargs_ov   = {\
    'overwrite':True, \
    'verbose':True \
}

kwargs_glob = {\
    'snmin':1, \
    'snmax':100, \
}

kwargs_cmb  = {\
    't':'id', \
}


# //// Main calculation ////#
glob = local.analysis(**kwargs_glob)
cobj = tools_cmb.cmb_anisotropies(**kwargs_cmb)
cobj.create_freq_map(glob,**kwargs_ov)
cobj.create_white_noise_map(glob,**kwargs_ov)

