# map -> alm
import numpy as np
import healpy as hp
import pickle
import os
import sys
import tqdm

# from cmblensplus/wrap/
import curvedsky as cs

# from cmblensplus/utils/
import constants
import cmb
import misctools

# local
import local


# LiteBIRD instrumental parameters
INST = {}
INST['LFT'] = { \
        '40': (70.5,37.42), \
        '50': (58.5,33.46), \
        '60': (51.1,21.31), \
       '68a': (41.6,19.91), \
       '68b': (47.1,31.77), \
       '78a': (36.9,15.55), \
       '78b': (43.8,19.13), \
       '89a': (33.0,12.28), \
       '89b': (41.5,28.77), \
       '100': (30.2,10.34), \
       '119': (26.3, 7.69), \
       '140': (23.7, 7.25) 
    }
INST['MFT'] = { \
       '100': (37.8,8.48), \
       '119': (33.6,5.70), \
       '140': (30.8,6.38), \
       '166': (28.9,5.57), \
       '195': (28.0,7.05)
    }
INST['HFT'] = { \
       '195': (28.6,10.50), \
       '235': (24.7,10.79), \
       '280': (22.5,13.80), \
       '337': (20.9,21.95), \
       '402': (17.9,47.45)
    }
INST['id']  = {'com': (30.,3.)}
INST['ALL'] = {'com': ('','')}


# Define parameters and filename for CMB map
class cmb_anisotropies:

    def __init__(self,t='id',ntype='white',wind='G70',ascale=5.,fltr='none',lmin=2,lmax=1024,nside=512):

        #//// get parameters ////#
        # specify telescope (t)
        # LFT/MFT/HFT --- individual telescope
        # comb --- component separated map
        # id --- idealistic fullsky cmb, isotropic noise
        self.telescope = t
        self.freqs = list(INST[t].keys())
        if self.telescope not in ['id','ALL']:  self.freqs.append('com')

        # CMB alms filtering
        self.fltr   = fltr
        if self.telescope == 'id':  self.fltr = 'none'
            
        # window
        self.wind   = wind
        if self.telescope == 'id':  self.wind = 'fullsky'

        # apodization scale
        self.ascale = ascale
        if self.telescope == 'id':  self.ascale = 0.

        # CMB map noise type
        self.ntype = ntype
        
        # minimum/maximum multipoles of CMB alms
        self.lmin  = lmin
        self.lmax  = lmax
        
        # map resolution
        self.nside = nside
        self.npix  = 12*self.nside**2

        #//// tags for filename ////#
        # type of window
        if self.fltr == 'cinv':
            wftag = '_'+self.fltr
        else:
            wftag = '_' + self.wind + '_a'+str(self.ascale)+'deg'
        
        # specify CMB map
        self.stag = {freq: self.telescope + freq + '_' + self.ntype + wftag for freq in self.freqs}

        #//// Filenames ////#
        ids = local.ids.copy()
        
        # cmb map, alm and aps
        d = local.data_directory()
        d_alm = d['cmb'] + 'alm/'
        d_aps = d['cmb'] + 'aps/'
        d_map = d['cmb'] + 'map/'
        
        #cmb signal map
        self.fscmb = {freq: [d_map+'/cmb_uKCMB_'+t+freq+'_nside'+str(nside)+'_'+x+'.fits' for x in ids] for freq in self.freqs}

        #cmb noise map
        self.fnois = {freq: [d_map+'/noise_uKCMB_'+t+freq+'_'+ntype+'_nside'+str(nside)+'_'+x+'.fits' for x in ids] for freq in self.freqs}
        
        #cmb alm/aps
        self.falms, self.fcl, self.fmcl = {}, {}, {}
        for freq in self.freqs:
            self.falms[freq], self.fcl[freq], self.fmcl[freq] = {}, {}, {}
            for s in ['s','n','o']:
                self.falms[freq][s] = {}
                for m in ['T','E','B']:
                    if s=='s': # remove noise type for signal
                        Stag = self.stag[freq].replace('_'+ntype,'')
                    else:
                        Stag = self.stag[freq]
                    self.falms[freq][s][m] = [d_alm+'/'+s+'_'+m+'_'+Stag+'_'+x+'.pkl' for x in ids]
                self.fcl[freq][s]  = [d_aps+'/rlz/cl_'+Stag+'_'+s+'_'+x+'.dat' for x in ids]
                self.fmcl[freq][s] = d_aps+'/mcl_'+Stag+'_'+s+'.dat'


    #-------------------------
    # LiteBIRD beam
    #-------------------------

    def get_beam(self): # Return Gaussian beam function
        # get fwhm
        self.theta = np.array( [ x[0] for x in list(INST[self.telescope].values()) ] )
        # compute 1D Gaussian beam function from cmblensplus/utils/cmb.py
        self.bl = {freq: cmb.beam(self.theta[n],self.lmax,inv=False) for n, freq in enumerate(self.freqs)}

    #-------------------------
    # LiteBIRD white nosie level
    #-------------------------

    def get_noise_spec(self):
        self.sigma = np.array( [ x[1] for x in list(INST[self.telescope].values()) ] )
        # white noise spectrum    
        self.nl = {}
        for n, freq in enumerate(self.freqs):
            self.nl[freq] = np.zeros((4,self.lmax+1))
            self.nl[freq][0,:] = .5*(self.sigma[n]*cmb.ac2rad/cmb.Tcmb)**2
            self.nl[freq][1,:] = 2*self.nl[freq][0,:]
            self.nl[freq][2,:] = 2*self.nl[freq][0,:]
            

    def create_freq_map(self,glob,overwrite=True,verbose=False):
        # transform SO alms to a frequency map which is convolved by a Gaussian beam
        
        self.get_beam()
        
        for freq in self.freqs:
            
            for i in tqdm.tqdm(glob.rlz):

                alms = hp.read_alm(glob.ficmb[i],hdu=(1,2,3))
            
                Talm = cs.utils.lm_healpy2healpix( alms[0], 5100 ) [:self.lmax+1,:self.lmax+1] / cmb.Tcmb * self.bl[freq][:,None]
                Ealm = cs.utils.lm_healpy2healpix( alms[1], 5100 ) [:self.lmax+1,:self.lmax+1] / cmb.Tcmb * self.bl[freq][:,None]
                Balm = cs.utils.lm_healpy2healpix( alms[2], 5100 ) [:self.lmax+1,:self.lmax+1] / cmb.Tcmb * self.bl[freq][:,None]
            
                T = cs.utils.hp_alm2map(self.nside, self.lmax, self.lmax, Talm)
                Q, U = cs.utils.hp_alm2map_spin(self.nside, self.lmax, self.lmax, 2, Ealm, Balm)
            
                hp.fitsfunc.write_map(self.fscmb[freq][i],np.array((T,Q,U)),overwrite=overwrite)

    
    def create_white_noise_map(self,glob,overwrite=True,verbose=False):

        self.get_noise_spec()
        
        for freq in self.freqs:
            
            for i in tqdm.tqdm(glob.rlz):

                Talm = cs.utils.gauss1alm(self.lmax,self.nl[freq][0,:])
                Ealm = cs.utils.gauss1alm(self.lmax,self.nl[freq][1,:])
                Balm = cs.utils.gauss1alm(self.lmax,self.nl[freq][2,:])
            
                T = cs.utils.hp_alm2map(self.nside, self.lmax, self.lmax, Talm)
                Q, U = cs.utils.hp_alm2map_spin(self.nside, self.lmax, self.lmax, 2, Ealm, Balm)
            
                hp.fitsfunc.write_map(self.fnois[freq][i],np.array((T,Q,U)),overwrite=overwrite)


            
#---------------------------
# Window function operation
#---------------------------

def window(wind,nside=None,ascale=0.):

    # load window
    if wind=='fullsky':
        w = 1.
    else:
        fmask = local.data_directory()['win'] + wind + '_binary.fits'
        if ascale!=0: fmask = fmask.replace('binary','a'+str(ascale)+'deg')
        w = hp.fitsfunc.read_map(fmask,verbose=False)
        if nside is not None:  
            w = hp.pixelfunc.ud_grade(w,nside)
    
    # normalization correction due to window
    wn = cmb.wfactor(w)

    return w, wn


#-------------------------
# map -> alm -> aps
#-------------------------

def map2alm(glob,cobj,freq,w,mtype=['T','E','B'],kwargs_ov={}):

    # beam
    cobj.get_beam()

    # map -> alm
    for i in tqdm.tqdm(glob.rlz,ncols=100,desc='map2alm (freq='+freq+')'):

        if misctools.check_path([cobj.falms[freq]['o']['T'][i],cobj.falms[freq]['o']['E'][i],cobj.falms[freq]['o']['B'][i]],**kwargs_ov): continue

        salm = cmb.map2alm_curvedsky(cobj.lmax,cobj.fscmb[freq][i],w=w,bl=cobj.bl[freq])
        nalm = cmb.map2alm_curvedsky(cobj.lmax,cobj.fnois[freq][i],w=w,bl=cobj.bl[freq])

        oalm = {}
        for m in mtype:
            oalm[m] = salm[m] + nalm[m]

        # save to files
        for m in mtype:
            pickle.dump((oalm[m]),open(cobj.falms[freq]['o'][m][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump((nalm[m]),open(cobj.falms[freq]['n'][m][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump((salm[m]),open(cobj.falms[freq]['s'][m][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)


def alm_comb_freq(rlz,fcmbfreq,fcmbcomb,verbose=True,overwrite=False,freqs=['93','145','225'],mtype=[(0,'T'),(1,'E'),(2,'B')],roll=2):
    
    for i in tqdm.tqdm(rlz,ncols=100,desc='alm combine'):

        for (mi, m) in mtype:

            if misctools.check_path(fcmbcomb.alms['o'][m][i],overwrite=overwrite,verbose=verbose): continue

            salm, nalm, Wl = 0., 0., 0.
            for freq in freqs:
                Nl = np.loadtxt(fcmbfreq[freq].scl['n'],unpack=True)[mi+1]
                Nl[0:2] = 1.
                Il = 1./Nl
                salm += pickle.load(open(fcmbfreq[freq].alms['s'][m][i],"rb"))*Il[:,None]
                nalm += pickle.load(open(fcmbfreq[freq].alms['n'][m][i],"rb"))*Il[:,None]
                Wl   += Il
            salm /= Wl[:,None]
            nalm /= Wl[:,None]
            oalm = salm + nalm

            # remove low-ell for roll-off effect
            if roll > 2:
                oalm[:roll,:] = 0.

            pickle.dump((salm),open(fcmbcomb.alms['s'][m][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump((nalm),open(fcmbcomb.alms['n'][m][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump((oalm),open(fcmbcomb.alms['o'][m][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)



def aps(glob,cobj,freq,w2,stype=['o','s','n'],mtype=['T','E','B'],**kwargs_ov):

    # compute aps for each rlz
    L = np.linspace(0,cobj.lmax,cobj.lmax+1)
    
    for s in stype:
        
        if misctools.check_path(cobj.fcl[freq][s],**kwargs_ov): continue
        
        if kwargs_ov['verbose']: print('stype =',s)
        
        cl = cmb.aps(glob.rlz,cobj.lmax,cobj.falms[freq][s],odd=False,mtype=mtype,**kwargs_ov,w2=w2,fname=cobj.fcl[freq][s])

        # save average to files
        mcl = np.mean(cl,axis=0)
        vcl = np.std(cl,axis=0)
        np.savetxt(cobj.fmcl[freq][s],np.concatenate((L[None,:],mcl,vcl)).T)


#////////////////////////////////////////////////////////////////////////////////
# Wiener filter
#////////////////////////////////////////////////////////////////////////////////

class wiener_objects(cmb_anisotropies):

    def __init__(self,tqu):

        self.tqu  = tqu
        self.maps = np.zeros((tqu,len(self.freqs),self.npix))
        self.invN = np.zeros((tqu,len(self.freqs),self.npix))
        self.calm = self.falm['com']['o'] # output alm

        # load hit count map
        self.W = hitmap(self.telescope,self.nside)
        
        # load survey boundary
        self.M, __ = window(self.wind,nside=self.nside,ascale=0.)
        

    def load_maps(self,rlz,verbose=False): # T or Q/U maps are loaded 

        for ki, freq in enumerate(self.freqs):
        
            if self.tqu == 1:
            
                Ts = hp.fitsfunc.read_map(self.fscmb[freq][rlz],field=0,verbose=verbose)
                Tn = hp.fitsfunc.read_map(self.fnois[freq][rlz],field=0,verbose=verbose)
                self.maps[0,ki,:] = self.M * hp.pixelfunc.ud_grade(Ts+Tn,self.nside)/cmb.Tcmb
        
            if self.tqu == 2:
        
                Qs = hp.fitsfunc.read_map(self.fscmb[freq][rlz],field=1,verbose=verbose)
                Us = hp.fitsfunc.read_map(self.fscmb[freq][rlz],field=2,verbose=verbose)
                Qn = hp.fitsfunc.read_map(self.fnois[freq][rlz],field=1,verbose=verbose)
                Un = hp.fitsfunc.read_map(self.fnois[freq][rlz],field=2,verbose=verbose)

                self.maps[0,ki,:] = self.M * hp.pixelfunc.ud_grade(Qs+Qn,self.nside)/cmb.Tcmb
                self.maps[1,ki,:] = self.M * hp.pixelfunc.ud_grade(Us+Un,self.nside)/cmb.Tcmb


    def load_invN(self):  # inv noise covariance

        for ki, sigma in enumerate(self.sigma):

            self.invN[0,ki,:] = self.W * (sigma*(np.pi/10800.)/cmb.Tcmb)**(-2)

            if self.tqu == 2:
                self.invN[:,ki,:] /= 2.
                self.invN[1,ki,:] = self.invN[0,ki,:]



def cinv_core(i,wobj,cl,lTmax=1000,lTcut=100,**kwargs):

    # number of frequencies
    mn  = len(wobj.freqs) - 1
    
    # load maps
    wobj.load_maps(i)

    if wobj.tqu==1: # temperature only case
        cl[0,:lTcut+1] = 0.
        Tlm = cs.cninv.cnfilter_freq(1,mn,wobj.nside,wobj.lmax,cl[0:1,:],wobj.bl,wobj.invN,wobj.maps,**kwargs)
        
        # output alms
        pickle.dump((Tlm),open(wobj.falm['T'][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)

    if wobj.tqu==2: # polarization only case
        Elm, Blm = cs.cninv.cnfilter_freq(2,mn,wobj.nside,wobj.lmax,cl[1:3,:],wobj.bl,wobj.invN,wobj.maps,**kwargs)

        # output alms
        pickle.dump((Elm),open(wobj.falm['E'][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump((Blm),open(wobj.falm['B'][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)



def cinv(tqu,glob,cobj,overwrite=False,verbose=False,**kwargs):

    # prepare objects for wiener filtering
    wobj = wiener_objects(cobj)
    wobj.load_invN()

    # start loop for realizations
    for i in tqdm.tqdm(glob.rlz,ncols=100,desc='cinv'):

        if misctools.check_path([wobj.falm['T'][i],wobj.falm['E'][i],wobj.falm['B'][i]],overwrite=overwrite): continue

        cinv_core(i,wobj,glob.lcl[:4,:cobj.lmax+1],verbose=verbose,**kwargs)


def load_cl(filename,lTmin=None,lTmax=None):

    print('loading TT/EE/BB/TE from pre-computed spectrum:',filename)
    
    cls = np.loadtxt(filename,unpack=True,usecols=(1,2,3,4))
    
    if lTmin is not None:  cls[0,:lTmin] = 1e30
    if lTmax is not None:  cls[0,lTmax+1:] = 1e30
    
    return cls


def interface(kwargs_glob={},kwargs_ov={},kwargs_cmb={},run=[],mtype=['T','E','B']):

    glob = local.analysis(**kwargs_glob)
    cobj = cmb_anisotropies(**kwargs_cmb)                    

    if cobj.fltr == 'none':

        if cobj.telescope == 'id': # map -> alm for fullsky case
            w, wn = 1., np.ones(5)
        else:
            # load survey window
            w, wn = local.window(wind,ascale=cobj.ascale) 

        # map -> alm for each freq
        for freq in cobj.freqs:
            # define map object
            # map -> alm
            if 'alm' in run:
                map2alm(glob,cobj,freq,w,mtype=mtype,kwargs_ov=kwargs_ov)
            # alm -> aps
            if 'aps' in run:
                aps(glob,cobj,freq,wn[2],mtype=mtype,**kwargs_ov)

        # combine alm over freqs
        if cobj.telescope != 'id':
            # define map object for coadd
            # map -> alm
            if 'alm' in run:
                alm_comb_freq(glob,cobj,**kwargs_ov)
            # alm -> aps
            if 'aps' in run: # compute for freq=com
                aps(glob,cobj,'com',wn[2],mtype=mtype,**kwargs_ov)


    elif kwargs_cmb['fltr'] == 'cinv':  # full wiener filtering

        __, wn = local.window(cobj.wind,ascale=cobj.ascale) 
        
        # Polarization
        cinv_params = {\
            'chn' : 1, \
            'eps' : [1e-4], \
            'lmaxs' : [cobj.lmax], \
            'nsides' : [cobj.nside], \
            'itns' : [1000], \
            'ro' : 1, \
            'filter' : 'W' \
        }
        cinv(2,glob,cobj,**cinv_params,**kwargs_ov)
