# Module for Multitracers
import numpy as np

# from cmblensplus/utils

# from local module
import local
import tools_cmb

# //// Fixed values //// #
masks = tools_cmb.masks

class filename():
    # define object which has parameters and filenames for multitracer analysis
    
    def __init__( self ):

        #set directory
        d = local.data_directory()
 
        # Lensing B-mode template
        self.fLTlm = {m: [ d['del'] + 'alm/LT_' + m + '_' + str(rlz) + '.pkl' for rlz in local.ids ] for m in masks }

        # BB spectra
        self.cl = {m: [ [ d['del'] + 'aps/rlz/LT_' + m + '_' + str(mi) + '_' + str(rlz) + '.pkl' for rlz in local.ids ] for mi in range(5) ] for m in masks }

