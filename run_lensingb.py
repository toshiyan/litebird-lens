#!/usr/bin/env python

import numpy as np
import healpy as hp
import pickle
import tqdm
# from cmblensplus/wrap/
import curvedsky as cs
# from cmblensplus/utils/
import cmb as CMB
import misctools
import constant as c
# others
import local
import tools_cmb
import tools_multitracer as mass
import tools_delens
import warnings
warnings.filterwarnings("ignore")

# define parameters
simn = 300
kwargs_ov = {'overwrite':False,'verbose':True}
#kwargs_ov = {'overwrite':True,'verbose':True}

# fixed/derived parameters
rlzs = range(1,simn+1)
lmax = 2*512
elmin, elmax, klmin, klmax = 2, lmax, 2, lmax
lbmax  = 190
pobj = local.analysis()
#mask = tools_cmb.prepare_masks(nside=64)
cobj = tools_cmb.cmb_map()
mobj  = mass.mass_tracer(lmin=2,lmax=lmax,gal_zbn={'euc':5,'lss':5})
dobj = tools_delens.filename()


# Read pre-computed coadded tracers
for rlz in tqdm.tqdm(rlzs,ncols=100,desc='each rlz'):

    # read coadd kappa map
    klm = {}
    klm[0], klm[1], klm[2], klm[3], klm[4] = pickle.load(open(mobj.fwklm[rlz],"rb"))

    #sBmap, nBmap, oBmap, sBlm = tools_cmb.prepare_obs_Bmap(pobj,rlz)
    #ibb = cs.utils.alm2cl(lbmax,sBlm)

    for m in tools_cmb.masks:
        
        if misctools.check_path(dobj.fLTlm[m][rlz],**kwargs_ov):
            
            blm = pickle.load(open(dobj.fLTlm[m][rlz],"rb"))
    
        else:

            # read Wiener-filtered polarization
            wElm = pickle.load(open(cobj.fwalm[m][rlz],"rb"))[0]

            # compute lensing B-mode template
            blm = {}
            for i in range(4):
                blm[i] = cs.delens.lensingb( lmax, elmin, elmax, klmin, klmax, wElm, klm[i], gtype='k')

            pickle.dump( (blm), open(dobj.fLTlm[m][rlz],"wb"), protocol=pickle.HIGHEST_PROTOCOL )

        if misctools.check_path(dobj.cl[m][0][rlz],**kwargs_ov): continue
        
        # wiener-filtered observed B-mode map
        wBlm, rBlm = pickle.load(open(cobj.foblm[m][rlz],"rb"))
        #NSIDE = hp.get_nside(oBmap)
        #oBlm = cs.utils.hp_map2alm(NSIDE,lbmax,lbmax,mask[m]*oBmap)
        #nBlm = cs.utils.hp_map2alm(NSIDE,lbmax,lbmax,mask[m]*nBmap)

        # aps
        obb = cs.utils.alm2cl(lbmax,wBlm)
        rbb = cs.utils.alm2cl(lbmax,rBlm)
        
        for i in range(5):

            lbb = cs.utils.alm2cl(lbmax,blm[i][:lbmax+1,:lbmax+1])
            xbb = cs.utils.alm2cl(lbmax,blm[i][:lbmax+1,:lbmax+1],wBlm)
            #Xbb = cs.utils.alm2cl(lbmax,blm[i][:lbmax+1,:lbmax+1],sBlm)

            #np.savetxt(dobj.cl[m][i][rlz],np.array((ibb,obb,nbb,lbb,xbb,Xbb)).T)
            np.savetxt(dobj.cl[m][i][rlz],np.array((obb,rbb,lbb,xbb)).T)

