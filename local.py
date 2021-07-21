#///////////////////////////////////////////////////////////////////////////////////////////////////#
# This file is intended to provide parametes, functions, etc, affecting the delensing code globally #
# Set up analysis parameters, filenames, arrays, functions                                          #
#///////////////////////////////////////////////////////////////////////////////////////////////////#

import constants
import numpy as np
import healpy as hp
import sys
import pickle

# from cmblensplus/utils/
import curvedsky as cs
import cmb


#////////// Define Fixed Values //////////#
# CMB temperauter in uK
Tcmb = cmb.Tcmb

# Index for realizations e.g. 0001
ids = [str(i).zfill(4) for i in range(-1,1000)]
ids[0] = 'real'


# directory
def data_directory(root_pub='../data/PTEP_20200915_compsep/',root_lens='../data/lensing/'):
    direct = {}
    direct['pub']  = root_pub + '/'
    direct['inp']  = '../data/'
    direct['loc']  = root_lens
    direct['cmb']  = root_lens + 'cmb/'
    direct['win']  = root_lens + 'window/'
    direct['del']  = root_lens + 'delens/'
    direct['dlm']  = direct['del']+'alm/'
    direct['dps']  = direct['del']+'aps/'
    direct['mas']  = root_lens + 'mass/'
    return direct


# Define parameters, filename and array
class analysis:

    def __init__(self,doreal=False,snmin=1,snmax=10,ilmax=5100):

        #//// set parameters ////#
        # minimum/maximum of realization index to be analyzed
        # the 1st index (0000) is used for real (or mock) data
        self.snmin  = snmin
        self.snmax  = snmax

        # use real data or not for index = 0000
        self.doreal = doreal
        
        # total number of realizations and array of realization index
        self.snum   = snmax - snmin + 1
        self.rlz    = np.linspace(snmin,snmax,self.snum,dtype=np.int)
        
        # input cmb maximum multipole
        self.ilmax  = ilmax

        #//// filename for input fixed data ////#
        # set directory
        d = data_directory()
        #d_alm = d['cmb'] + 'alm/'
        #d_map = d['cmb'] + 'map/'
        
        # input kappa alm
        #self.fkalm = [d_alm+'/iklm_nside'+str(nside)+'_'+x+'.pkl' for x in ids]

        # input unlensed cmb alm
        #self.fualm = [d_alm+'/iulm_nside'+str(nside)+'_'+x+'.pkl' for x in ids]
        
        # input lensed cmb alm
        #self.flalm = [d_alm+'/illm_nside'+str(nside)+'_'+x+'.pkl' for x in ids]

        #input SO lensed CMB alm
        self.ficmb = ['/project/projectdirs/sobs/v4_sims/mbs/cmb/fullskyLensedUnabberatedCMB_alm_set00_0'+x+'.fits' for x in ids]
        
        #input SO phi alm
        self.fiplm = ['/global/project/projectdirs/sobs/v4_sims/mbs/cmb/input_phi/fullskyPhi_alm_0'+x+'.fits' for x in ids]

        # input cmb cls
        self.fucl = d['inp']+'cosmo2017_10K_acc3_scalCls.dat'
        self.flcl = d['inp']+'cosmo2017_10K_acc3_lensedCls.dat'

        # loading theoretical cl
        self.ucl = cmb.read_camb_cls(self.fucl,output='array')[:,:ilmax+1]
        self.lcl = cmb.read_camb_cls(self.flcl,ftype='lens',output='array')[:,:ilmax+1]

        #//// set arrays ////#
        #multipole
        self.l  = np.linspace(0,ilmax,ilmax+1)

        #conversion factor from phi to kappa
        self.kL = self.l*(self.l+1)*.5

        #rename cls
        self.uTT = self.ucl[0]
        self.uEE = self.ucl[1]
        self.uTE = self.ucl[2]
        self.lTT = self.lcl[0]
        self.lEE = self.lcl[1]
        self.lBB = self.lcl[2]
        self.lTE = self.lcl[3]

        #kappa cl
        self.pp = self.ucl[3]
        self.kk = self.ucl[3]*self.kL**2


    '''
    def sim_ucmb(self,**kwargs_ov): # generate unlensed CMB alms

        for i in glob.rlz:
            if misctools.check_path(glob.fualm[i],**kwargs_ov): continue
            if kwargs_ov['verbose']: print('generate Gaussian T/E alms', i)
            Talm, Ealm = cs.utils.gauss2alm(glob.lmax,glob.uTT,glob.uEE,glob.uTE) 
            pickle.dump((Talm,Ealm),open(glob.fualm[i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)


    def get_klm(self,**kwargs_ov):

        for i in glob.rlz:
            if misctools.check_path(glob.fkalm[i],**kwargs_ov): continue
            if kwargs_ov['verbose']:  print('generate plm, rlz = '+str(i))
            klm = cs.utils.gauss1alm(glob.lmax,glob.kk)
            pickle.dump((klm),open(glob.fkalm[i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)
        
        
    def remap_cmb(self,**kwargs_ov):  # Remapping CMB in full sky

        for i in glob.rlz:
            if misctools.check_path(glob.fsalm['T'][i],overwrite=overwrite): continue

        # load cmb and phi
        Talm, Ealm = pickle.load(open(p.fcmb.ualm[i],"rb"))
        plm = pickle.load(open(p.fpalm[i],"rb"))

        # remap
        grad = curvedsky.delens.phi2grad(p.npixr,p.lmax,plm)
        Talm, Ealm, Balm = curvedsky.delens.remap_tp(p.npixr,p.lmax,grad,np.array((Talm,Ealm,0*Ealm)))

        # save
        pickle.dump((Talm),open(p.fcmb.salm['T'][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump((Ealm),open(p.fcmb.salm['E'][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump((Balm),open(p.fcmb.salm['B'][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)
    '''

    
def load_input_kappa(rlz_index,glob,lmax,to_healpix=True):
    # load input phi alm and then convert it to kappa alm
    iplm = hp.read_alm( glob.fiplm[rlz_index] )
    #if to_healpix:
    iplm = cs.utils.lm_healpy2healpix( iplm, 5100 ) [:lmax+1,:lmax+1]
    iklm = iplm * glob.kL[:lmax+1,None]
    #else:
    #    iklm = hp.almxfl( iplm, glob.kL[:lmax+1] )
    return iklm


