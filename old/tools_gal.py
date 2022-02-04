# prepare gal alm
import numpy as np
import pickle
import curvedsky
import liblocal as local
import misctools



root = liblocal.root

direct = {}
direct['gal']  = root + 'gal/map/'
direct['glm']  = root + 'gal/alm/'
direct['gps']  = root + 'gal/aps/'


class galaxy: 

    def __init__(self,gtype='n',nres=12,ngal=20.,zn=1,zm=1.0,gbias='sqrtz',a=2.,b=1.,b0=1.,lmin=2,lmax=4096):

        # output multipole range
        self.lmin = lmin
        self.lmax = lmax

        # map resolution
        self.nres = nres
        
        # gaussian or nonlinear
        self.gtype = gtype

        # galaxy
        self.zn    = zn
        self.ngal  = ngal  #per arcmin^2

        # bias
        self.gbias = gbias #galaxy bias
        self.b0    = b0    #b0

        # galaxy parameters
        self.a  = a
        self.b  = b
        self.zm = zm

        self.zbin = basic.galaxy.zbin(zn,a,b,zm)
        self.frac = basic.galaxy.frac(zn,self.zbin,a,b,zm)
        self.ngzi = 4.*np.pi*(180.*60./np.pi)**2*self.frac*self.ngal  # total galaxy number x fraction

        self.gltag = '_l'+str(lmin)+'-'+str(lmax)
        self.gtag  = self.gtype+self.nres+'_zn'+str(zn)+'_zm'+str(zm)+'_'+gbias+'_ngal'+str(ngal)[:4]


    def filename(self,ids):

        d = data_directory()
        
        self.inp = [d['inp']+'delta/allskymap_nres'+str(self.nres)+'r'+x+'.delta_shell.dat' for x in ids]
        self.alm = [d['glm']+'galm_'+self.gtag+'_'+x+'.pkl' for x in ids]
        self.cl  = [d['gps']+'rlz/cgg_'+self.gtag+'_'+x+'.dat' for x in ids]
        self.mcl = d['gps'] + 'aps_' + self.gtag + '.dat' # used for wiener filter

        # gal map mean values
        self.mkap  = d['glm'] + 'mkap_' + self.gtag + '.dat'

        # cl for generating a Gaussian galaxy alm
        self.fgfix = d['gps'] + 'aps_1d_' + self.gtag.replace('g'+str(self.nres),'n'+str(self.nres)) + '.dat'



def gmap2galm(rlz,g,overwrite=False,verbose=True):

    for i in rlz:
    
        if misctools.check_path(g.alm[i],overwrite=overwrite): continue

        gmap = curvedsky.utils.mock_galaxy_takahashi(g.inp[i],g.zn,g.ngzi,g.zbin,b0=g.b0,btype=g.gbias,a=g.a,b=g.b,zm=g.zm)

        # map to alm
        mgmap = np.average(gmap)
        gmap = (gmap-mgmap)/mgmap
        glm = curvedsky.utils.hp_map2alm(4096,g.lmax,g.lmax,gmap)

        if verbose:  print('start save to file')
        pickle.dump((glm),open(g.alm[i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)


def galm2gaps(rlz,g,fpalm,lmax,overwrite=False):
    # compute power spectrum

    eL = np.linspace(0,lmax,lmax+1)
    cl = np.zeros((len(rlz),3,lmax+1))

    for i in rlz:
    
        ii = i - min(rlz)

        if misctools.check_path(g.cl[i],overwrite=overwrite): 
            
            cl[ii,:,:] = np.loadtxt(g.cl[i],unpack=True,usecols=(1,2,3))
        
        else:

            galm = pickle.load(open(g.alm[i],"rb"))[:lmax+1,:lmax+1]
            palm = pickle.load(open(fpalm[i],"rb"))[:lmax+1,:lmax+1]

            cl[ii,0,:] = curvedsky.utils.alm2cl(lmax,galm)
            cl[ii,1,:] = curvedsky.utils.alm2cl(lmax,galm,palm)
            cl[ii,2,:] = curvedsky.utils.alm2cl(lmax,palm)
            np.savetxt(g.cl[i],np.concatenate((eL[None,:],cl[ii,:,:])).T)

    np.savetxt(g.mcl,np.concatenate((eL[None,:],np.mean(cl,axis=0),np.std(cl,axis=0))).T)


def interface(run=[],ow=False,**kwargs):

    pobj = local.init_analysis(snmax=100)
    gobj = galaxy(**kwargs)
    gobj.filename(pobj.ids)

    if 'alm' in run:
        gmap2galm(pobj.rlz,gobj,overwrite=ow)
    if 'aps' in run:
        galm2gaps(pobj.rlz,gobj,pobj.fpalm,gobj.lmax,overwrite=ow)



