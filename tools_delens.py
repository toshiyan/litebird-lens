# Linear template delensing
import numpy as np
import healpy as hp
import pickle
import tqdm
    
# from cmblensplus
import curvedsky as cs
import misctools
import cmb

# local
import local
import tools_cmb


class lensing_template:

    # Define objects containing E and kappa to be combined to form a lensing template

    def __init__(self,Eobj,mobj,olmax=2048,elmin=20,elmax=1024,klmin=20,nside=2048,kfltr='none'):
        # E mode
        self.etag  = Eobj.stag['com']
        self.fElm  = Eobj.falms['com']['o']['E']

        # kappa
        self.mtag  = mobj.tag
        self.klist = mobj.klist
        self.fklm  = mobj.fcklm
        self.kfltr = kfltr # this does not work now

        # minimum/maximum multipole of E and kappa in lensing template construction
        self.elmin = elmin
        self.elmax = elmax
        self.klmin = klmin
        self.klmax = mobj.lmax

        # E-mode ell filter
        self.filter_E = np.zeros((self.elmax+1))
        self.filter_E[self.elmin:] = 1.
        if Eobj.telescope == 'id':
            EE = (np.loadtxt(Eobj.fmcl['com']['s'])).T[2]
            EO = (np.loadtxt(Eobj.fmcl['com']['o'])).T[2]
            self.filter_E[self.elmin:self.elmax+1] = EE[self.elmin:self.elmax+1]/EO[self.elmin:self.elmax+1]

        # kappa ell filter
        self.filter_kap = np.zeros((self.klmax+1))
        self.filter_kap[self.klmin:] = 1.

        # output template
        self.olmax = olmax
        self.l     = np.linspace(0,self.olmax,self.olmax+1)

        # filename tag for lensing template
        self.tag  = 'le'+str(self.elmin)+'-'+str(self.elmax)+'_lk'+str(self.klmin)+'-'+str(self.klmax)+'_'+self.etag+'_'+self.mtag

        # alm of lensing template B-modes
        d   = local.data_directory()
        self.falm = [d['del']+'alm/alm_'+self.tag+'_'+x+'.pkl' for x in local.ids]
        self.cl   = [d['del']+'aps/rlz/cl_'+self.tag+'_'+x+'.dat' for x in local.ids]


    # operation to this object

    def template_alm(self,glob,**kwargs_ov):

        for i in tqdm.tqdm(glob.rlz,ncols=100,desc='template alm'):
        
            if misctools.check_path(self.falm[i],**kwargs_ov): continue
            
            # load E mode
            wElm = pickle.load(open(self.fElm[i],"rb"))[:self.elmax+1,:self.elmax+1] * self.filter_E[:,None]
            
            # load Wiener-filtered kappa
            wklm = pickle.load(open(self.fklm[i],"rb"))[:self.klmax+1,:self.klmax+1] * self.filter_kap[:,None]

            # construct lensing B-mode template
            dalm = cs.delens.lensingb( self.olmax, self.elmin, self.elmax, self.klmin, self.klmax, wElm, wklm, gtype='k')

            # save to file
            pickle.dump((dalm),open(self.falm[i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)

            
    def template_aps(self,glob,W,**kwargs_ov):
    
        for i in tqdm.tqdm(glob.rlz,ncols=100,desc='delens aps'):
        
            if misctools.check_path(self.cl[i],**kwargs_ov): continue
            
            wdlm = pickle.load(open(self.falm[i],"rb"))[0:self.olmax+1,0:self.olmax+1]
            #wdlm = curvedsky.utils.mulwin_spin(0*dalm,dalm,W)[1]
            
            alms = hp.read_alm(glob.ficmb[i],hdu=(3))
            Balm = cs.utils.lm_healpy2healpix( alms, glob.ilmax ) [:self.olmax+1,:self.olmax+1] / cmb.Tcmb
            if not np.isscalar(W): Balm = cs.utils.mulwin_spin(0*Balm,Balm,W)[1]
            
            clbb = cs.utils.alm2cl(self.olmax,Balm)
            cldd = cs.utils.alm2cl(self.olmax,wdlm)
            clbd = cs.utils.alm2cl(self.olmax,wdlm,Balm)
            np.savetxt(self.cl[i],np.array((clbb,cldd,clbd)).T)



def interface(mobj,run_del=[],kwargs_ov={},kwargs_glob={},kwargs_emode={},kwargs_del={}):

    # //// prepare E modes //// #
    Eobj = tools_cmb.cmb_anisotropies(**kwargs_emode)
    
    # //// prepare phi //// #
    # define object
    glob = local.analysis( **kwargs_glob )
    dobj = lensing_template( Eobj, mobj, **kwargs_del )
    
    # window function for input B-mode to be correlated with template B-mode
    W, __ = tools_cmb.window(Eobj.wind,ascale=Eobj.ascale)

    # //// compute lensing template alm //// #
    if 'alm' in run_del:
        dobj.template_alm( glob, **kwargs_ov )

    if 'aps' in run_del:
        # compute lensing template spectrum
        dobj.template_aps( glob, W, **kwargs_ov )

    return dobj

