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

#run = ['alm','aps']
run = ['aps']

# //// Main calculation ////#
tools_cmb.interface(kwargs_glob=kwargs_glob,kwargs_ov=kwargs_ov,kwargs_cmb=kwargs_cmb,run=run)
