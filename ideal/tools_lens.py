# from external
import numpy as np
import healpy as hp
import sys
import pickle
import tqdm

# from cmblensplus/wrap/
import curvedsky as cs

# from cmblensplus/utils/
import misctools
import quad_func

# local
import local
import tools_cmb


def aps(glob,qobj,wn,verbose=True,mean_sub=True):
    # Compute aps of reconstructed lensing map
    # This code can be used for checking reconstructed map

    for q in tqdm.tqdm(qobj.qlist,ncols=100,desc='aps'):

        cl = np.zeros((len(glob.rlz),4,qobj.olmax+1))

        W2, W4 = wn[2], wn[4]

        for ii, i in enumerate(tqdm.tqdm(glob.rlz,ncols=100,desc='each rlz ('+q+'):')):

            # load reconstructed kappa and curl alms
            glm, clm = quad_func.load_rec_alm(qobj,q,i,mean_sub=mean_sub)

            # load kappa
            klm = local.load_input_kappa(i,glob,qobj.olmax)

            # compute cls
            cl[ii,0,:] = cs.utils.alm2cl(qobj.olmax,glm)/W4
            cl[ii,1,:] = cs.utils.alm2cl(qobj.olmax,clm)/W4
            cl[ii,2,:] = cs.utils.alm2cl(qobj.olmax,glm,klm)/W2
            cl[ii,3,:] = cs.utils.alm2cl(qobj.olmax,klm)
            np.savetxt(qobj.f[q].cl[i],np.concatenate((qobj.l[None,:],cl[ii,:,:])).T)

        # save sim mean
        if glob.rlz[0]>=1 and len(glob.rlz)>1:
            np.savetxt(qobj.f[q].mcls,np.concatenate((qobj.l[None,:],np.mean(cl[1:,:,:],axis=0),np.std(cl[1:,:,:],axis=0))).T)


def init_qobj(stag,**kwargs):
    # setup parameters for lensing reconstruction (see cmblensplus/utils/quad_func.py)
    return quad_func.reconstruction(local.data_directory()['loc'],local.ids,stag=stag,run=[],**kwargs)


def interface(run=[],kwargs_glob={},kwargs_ov={},kwargs_cmb={},kwargs_qrec={}):

    glob = local.analysis(**kwargs_glob)
    cobj = tools_cmb.cmb_anisotropies(**kwargs_cmb)
    __, wn = tools_cmb.window(cobj.wind,ascale=cobj.ascale)

    # Compute filtering
    if cobj.fltr == 'none': # for none-filtered alm
        # Load "observed" aps containing signal, noise, and some residual. 
        # This aps will be used for normalization calculation
        ocl = tools_cmb.load_cl(cobj.fmcl['com']['o'])
        # CMB alm will be multiplied by 1/ifl before reconstruction process
        ifl = ocl.copy()

    elif cobj.fltr == 'cinv': # for C^-1 wiener-filtered alm
        sys.exit('cinv is not supported')

    else:
        sys.exit('unknown filtering')

    qobj = quad_func.reconstruction(local.data_directory()['loc'],local.ids,rlz=glob.rlz,stag=cobj.stag['com'],run=run,wn=wn,lcl=glob.lcl,ocl=ocl,ifl=ifl,falm=cobj.falms['com']['o'],**kwargs_ov,**kwargs_qrec)

    # Aps of reconstructed phi
    if 'aps' in run: 
        aps(glob,qobj,wn)
        
    return qobj

