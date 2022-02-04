# Running delenisng code

# from external module
import numpy as np

# from this directory
import tools_cmb
import tools_lens
import tools_multitracer
import tools_delens


#run_qrec = ['norm','qrec','n0','mean','aps']
#run_qrec = ['norm','qrec','mean']
#run_qrec = ['n0','mean','aps']
#run_qrec = ['aps']
#run_qrec = ['norm']
run_qrec = []

#run_mass = ['gen_alm','comb']
run_mass = ['comb']
#run_mass = []

run_del = ['alm','aps']
#run_del = ['rho']
#run_del = []


kwargs_ov   = {\
    'overwrite':True, \
    'verbose':True \
}

kwargs_glob  = {\
    'snmin':1, \
    'snmax':100, \
}

kwargs_cmb  = {\
    't':'id', \
    'fltr':'none', \
}

kwargs_qrec = {\
    'qlist':['EB'], \
    #'qMV':['TT','TE','EE','EB'], \
    'rlmin':200, \
    'rlmax':1024, \
    'olmax':1024, \
    'nside':512, \
    'n0min':1, \
    'n0max':int(kwargs_glob['snmax']/2), \
    'mfmin':1, \
    'mfmax':kwargs_glob['snmax'], \
    'rdmin':1, \
    'rdmax':kwargs_glob['snmax'] \
}

kwargs_mass = {\
    #//// mass tracers to be combined before lensing template construction ////#
    #//// cmb kappa ////#
    #'add_cmb':['TT','TE','EE','EB'], \
    #'add_cmb':['EB'], \
    'add_cmb':[], \
    #'add_gal':np.arange(3), \
    'add_gal':[], \
    #//// CIB ////#
    'add_cib':True, \
    #'add_cib':False, \
    'lmax':2048, \
}


kwargs_del = {\
    #//// minimum/maximum multipole of E modes ////#
    'elmin':10, \
    'elmax':1024, \
    #//// minimum/maximum multipole of mass tracers ////#
    'klmin':10, \
    #//// kappa cinv filter (this does not work now) ////#
    'kfltr':'none', \
    #//// output template maximum multipole ////#
    'olmax':1024, \
}


# //// Main calculation ////#
qobj = tools_lens.interface( run=run_qrec, kwargs_glob=kwargs_glob, kwargs_ov=kwargs_ov, kwargs_cmb=kwargs_cmb, kwargs_qrec=kwargs_qrec )

mobj = tools_multitracer.interface( qobj, run=run_mass, kwargs_glob=kwargs_glob, kwargs_ov=kwargs_ov, kwargs_cmb=kwargs_cmb, kwargs_mass=kwargs_mass )

kwargs_emode = kwargs_cmb.copy()
dobj = tools_delens.interface( mobj, run_del=run_del, kwargs_glob=kwargs_glob, kwargs_ov=kwargs_ov, kwargs_emode=kwargs_emode, kwargs_del=kwargs_del )

