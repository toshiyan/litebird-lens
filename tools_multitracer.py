# Module for Multitracers
import numpy as np
import healpy as hp
import pickle
import tqdm
from astropy import units as u

# from cmblensplus/wrap
import curvedsky as cs

# from cmblensplus/utils
import misctools
import cmb
import quad_func
import delens_func

# from local module
import local
import tools_cmb
import tools_lens


class mass_tracer():
    # define object which has parameters and filenames for multitracer analysis
    
    def __init__( self, qobj, lmin=5, lmax=2048, add_cmb=['EB'], add_gal=np.arange(3), add_cib=True ):
        
        # construct list of mass tracers to be combined
        self.klist_cmb = {}
        self.klist_gal = {}
        self.klist_cib = {}

        # store id for cmb lensing maps
        kid = 0
        for k in add_cmb:
            self.klist_cmb[k] = kid
            kid += 1

        # store id for galaxy maps
        for z in add_gal:
            self.klist_gal['g'+str(z)] = kid 
            kid += 1

        # store id for cib maps
        if add_cib: 
            self.klist_cib['cib'] = kid
        
        # define list of all mass tracers
        self.klist = { **self.klist_cmb, **self.klist_gal, **self.klist_cib }

        # define list of non-CMB mass tracers
        self.klist_ext = { **self.klist_gal, **self.klist_cib }

        # total number of mass tracer maps
        self.nkap = len(self.klist)
        
        # multipole range of the mass tracer
        self.lmin = lmin
        self.lmax = lmax

        # noise curve for cmb lensing map
        self.nlkk = {}
        for k, n in self.klist_cmb.items():
            self.nlkk[n] = np.zeros(lmax+1)
            self.nlkk[n][:qobj.olmax+1] = np.loadtxt( qobj.f[k].al, unpack=True )[1]

        #set directory
        d = local.data_directory()
 
        # cls
        self.fspec = {spec: d['mas'] + 'spec/cl'+spec+'.dat' for spec in ['kk','II','gg','kI','kg','Ig']}

        # kappa alm of each mass tracer
        self.fklm = {}
        for k in self.klist:
            self.fklm[k] = [ d['mas'] + 'alm/' + k + '_' + str(i) + '.pkl' for i in local.ids ]
        
        # kappa alm of combined mass tracer
        self.tag   = qobj.cmbtag + qobj.bhe_tag + qobj.ltag + '_' + '-'.join(self.klist.keys())
        self.fcklm = [ d['mas'] + 'alm/comb_' + self.tag + '_' + str(i) + '.pkl' for i in local.ids ]
        self.fcovs = d['mas'] + 'cov/' + self.tag + '.pkl'


    def load_mass_tracer_spectra(self,mass_clid):
        data  = np.loadtxt(self.fspec[mass_clid])
        cln   = len(data[1:,0])
        cl    = np.zeros((cln,self.lmax+1))
        ilmin = np.int(data[0,0])
        ilmax = np.int(data[0,-1])
        for l in range(self.lmin,self.lmax+1):
            if l<ilmin or l>ilmax: continue
            cl[:,l] = data[1:,l-ilmin]
        return cl

    
    def get_spectra_matrix(self):
        # currently correlations between galaxies of different z-bins are ignored
        cl = {}
        for clid in ['gg','kg','Ig','kk','II','kI']:
            cl[clid] = self.load_mass_tracer_spectra(clid)
        
        # used for generating sim
        cl_matrix   = np.zeros( ( self.nkap, self.nkap, self.lmax+1) ) #Theory auto and cross spectra

        # //// auto spectra //// #
        for n in self.klist_cmb.values():
            cl_matrix[n,n,:] = cl['kk'][0]
    
        for k, n in self.klist_gal.items():
            z = int(k[1])
            cl_matrix[n,n,:] = cl['gg'][z,:]
    
        for n in self.klist_cib.values():
            cl_matrix[n,n,:] = cl['II'][0]

        # //// cross spectra //// #
        for n0 in self.klist_cmb.values():
            for n1 in self.klist_cmb.values():
                if n1 > n0: 
                    continue
                cl_matrix[n0,n1,:] = cl_matrix[n1,n0,:] = cl['kk'][0]
    
        for n0 in self.klist_cmb.values():
            for j, n1 in self.klist_gal.items():
                z = int(j[1])
                cl_matrix[n0,n1,:] = cl_matrix[n1,n0,:] = cl['kg'][z,:]

        for n0 in self.klist_cmb.values():
            for n1 in self.klist_cib.values():
                cl_matrix[n0,n1,:] = cl_matrix[n1,n0,:] = cl['kI']

        for n0 in self.klist_cib.values():
            for j, n1 in self.klist_gal.items():
                z = int(j[1])
                cl_matrix[n0,n1,:] = cl_matrix[n1,n0,:] = cl['Ig'][z,:]

        self.signal_covariance = cl_matrix.copy()

        # used for weights for coadding
        clnl_matrix = cl_matrix.copy()
        for n in self.klist_cmb.values():
            clnl_matrix[n,n,:] += self.nlkk[n]

        self.clnl_matrix = clnl_matrix.copy()


#//// Load mass tracers ////#

def load_mass_tracers( rlz_index, qobj, mobj, mmask=None, kmask=None ):
    
    alms = np.zeros( ( mobj.nkap, mobj.lmax+1, mobj.lmax+1 ), dtype=np.complex )
    
    # get tracers from CMB lensing
    for k, n in mobj.klist_cmb.items():
        alms[n,:qobj.olmax+1,:qobj.olmax+1] = quad_func.load_rec_alm(qobj,k,rlz_index,mean_sub=True)[0]
        if kmask is not None and not np.isscalar(kmask): 
            alms[n,:,:] = cs.utils.mulwin( alms[n,:,:], kmask )

    # get tracers from LSS
    for k, n in mobj.klist_ext.items():
        alms[n,:,:] = pickle.load( open(mobj.fklm[k][rlz_index],"rb") )
        if mmask is not None and not np.isscalar(mmask): 
            alms[n,:,:] = cs.utils.mulwin( alms[n,:,:], mmask )
    
    return alms


#//// Generate alms from covariance matrix ////#

def generate_tracer_alms( signal_covariance, iklm, num_of_kcmb, lmin):

    # Calculate the weights and auxiliary spectra needed to generate Gaussian sims of individual tracers
    aux_cl, A = calculate_sim_weights( signal_covariance, lmin, num_of_kcmb )

    # Draw harmonic coefficients from Gaussian distributions with the calculated auxiliary spectra
    a_alms = draw_gaussian_a_p( iklm, aux_cl, num_of_kcmb )

    # Combine weights and coefficients to generate sims of individual tracers
    tracer_alms = generate_individual_gaussian_tracers( a_alms, A, num_of_kcmb )

    return tracer_alms


def calculate_sim_weights( cl, lmin, num_of_kcmb ):
    '''
    Calculate the weights A_l^{ij} and the auxiliary spectra C_l^{ij}={C_l^{uu},C_l^{ee},...} from which the to draw the alm coefficients a_p={u_{lm},e_{lm},...}
    The simulated alm has the form, alm = sum_{p=0}^i A^{ip} a^p, where a^p is the auxiliary alm. To abvoid completely degenerate case for CMB estimators,
    we set A^{ip} = 0 for p within p > 0 and p < num of kcmb.
    '''
    num_of_tracers = len(cl[:,0,0]) 
    num_of_multipoles = len(cl[0,0,:])
    aux_cl = np.zeros( (num_of_tracers, num_of_multipoles) ) #Auxiliary spectra
    A = np.zeros( (num_of_tracers,num_of_tracers,num_of_multipoles) ) #Weights for the alms

    for j in range(num_of_tracers):

        if 0<j<num_of_kcmb: continue
        
        for i in range(j,num_of_tracers):
        
            if 0<i<num_of_kcmb : continue

            aux_cl[j,:] = np.nan_to_num(cl[j,j,:])
            for p in range(j):
                aux_cl[j] -= np.nan_to_num(A[j,p,:]**2 * aux_cl[p,:])

            A[i,j,lmin:] = np.nan_to_num((1./aux_cl[j,lmin:])*cl[i,j,lmin:])
            for p in range(j):
                A[i,j,lmin:] -= np.nan_to_num((1./aux_cl[j,lmin:])*A[j,p,lmin:]*A[i,p,lmin:]*aux_cl[p,lmin:])
    
    return aux_cl, A


def draw_gaussian_a_p(input_kappa_alm, aux_cl, num_of_kcmb):
    '''
    Draw a_p alms from distributions with the right auxiliary spectra.
    '''
    lmax = len(aux_cl[0,:]) - 1
    num_of_tracers = len(aux_cl[:,0])
    a_alms = np.zeros((num_of_tracers, lmax+1, lmax+1), dtype='complex128') #Unweighted alm components

    a_alms[0:num_of_kcmb,:,:] = input_kappa_alm
    for j in range(num_of_kcmb, num_of_tracers):
        a_alms[j,:,:] = cs.utils.gauss1alm(lmax,aux_cl[j,:])

    return a_alms


def generate_individual_gaussian_tracers(a_alms, A, num_of_kcmb):
    '''
    Put all the weights and alm components together to give appropriately correlated tracers
    '''
    #num_of_tracers = len(a_alms[:,0])
    num_of_tracers = len(a_alms[:,0,0])
    #tracer_alms = np.zeros((num_of_tracers, len(a_alms[0,:])), dtype='complex128') #Appropriately correlated final tracers
    tracer_alms = 0.*a_alms

    for i in range(num_of_kcmb,num_of_tracers):
        for j in range(i+1):
            #tracer_alms[i,:] += hp.almxfl(a_alms[j,:], A[i,j,:])
            tracer_alms[i,:,:] += a_alms[j,:,:] * A[i,j,:,None]

    return tracer_alms


#//// Combining multitracer alms ////#

def coadd_kappa_alms(tracer_alms, weights):
    # summing up mass tracer alms with appropriate weights
    combined_kappa_alms = 0.*tracer_alms[0,:,:]
    for index, individual_alms in enumerate(tracer_alms):
        combined_kappa_alms +=  weights[index,:,None] * individual_alms
    
    return combined_kappa_alms



def calculate_multitracer_weights_sim(glob,qobj,mobj,mmask=None,kmask=None,**kwargs_ov):
    '''
    Get covariance and weights from simulated alms
    '''

    lmin = mobj.lmin
    lmax = mobj.lmax

    if misctools.check_path(mobj.fcovs,**kwargs_ov): 
        
        vec, cov = pickle.load(open(mobj.fcovs,"rb"))
    
    else:

        vec = np.zeros( ( len(glob.rlz), mobj.nkap, lmax+1 ) )
        cov = np.zeros( ( len(glob.rlz), mobj.nkap, mobj.nkap, lmax+1 ) )
    
        for ii, i in enumerate(tqdm.tqdm(glob.rlz,ncols=100,desc='compute coeff')):

            # load mass tracer alms
            kalm = load_mass_tracers( i, qobj, mobj, mmask=mmask, kmask=kmask )

            # load input kappa and multiply lens window
            kilm = local.load_input_kappa( i, glob, lmax )

            # compute auto and cross
            vec[ii,:,:] = np.array([ cs.utils.alm2cl(lmax,kalm[ki,:,:],kilm) for ki in range(mobj.nkap) ])            
            cov[ii,:,:,:] = cs.utils.alm2cov(kalm)

        # compute weights as w = C^-1 V
        pickle.dump( (vec,cov), open(mobj.fcovs,"wb"), protocol=pickle.HIGHEST_PROTOCOL )


    weight = np.zeros( ( len(glob.rlz), mobj.nkap, lmax+1 ) )
    
    for ii, i in enumerate(tqdm.tqdm(glob.rlz,ncols=100,desc='compute weights')):
    
        mvec, mcov = np.mean(np.delete(vec,ii,0),axis=0),  np.mean(np.delete(cov,ii,0),axis=0)

        for l in range(lmin,lmax+1):
            for n in mobj.klist_cmb.values():
                if mvec[n,l] == 0.: mcov[n,n,l] = 1. # for reconstructed lensing kappa above rlmax
            weight[ii,:,l] = np.dot( np.linalg.inv(mcov[:,:,l]), mvec[:,l] )
    
    return weight


#//// interface function ////#

def interface( qobj, run=['gen_alm','comb'], kwargs_glob={}, kwargs_ov={}, kwargs_cmb={}, kwargs_mass={} ):

    # load parameters and filenames
    glob = local.analysis( **kwargs_glob )
    cobj = tools_cmb.cmb_anisotropies( **kwargs_cmb )
    mobj = mass_tracer( qobj, **kwargs_mass )

    # setup window function
    if len(run) != 0: 

        W, __ = tools_cmb.window(cobj.wind,ascale=cobj.ascale)
        if not np.isscalar(W):  W[W!=0.] = 1.

        if 'white' in cobj.ntype:
            mmask, kmask = W, W
        else:
            mmask, kmask = W, None

    # generate random gaussian alms of tracers
    if 'gen_alm' in run:
        
        # load cl-matrix and covariance of alms
        mobj.get_spectra_matrix()

        # loop over realizations
        for i in tqdm.tqdm(glob.rlz,ncols=100,desc='generating multitracer klms'):
        
            # load input phi alm and then convert it to kappa alm
            iklm = local.load_input_kappa( i, glob, mobj.lmax )

            # generate tracer alms
            alms = generate_tracer_alms( mobj.signal_covariance, iklm, len(mobj.klist_cmb), mobj.lmin )

            # save to files for external mass tracers
            for k, n in mobj.klist_ext.items():
                
                # check if file exist
                if misctools.check_path( mobj.fklm[k][i], **kwargs_ov ): continue
            
                # save
                pickle.dump( (alms[n,:,:]), open(mobj.fklm[k][i],"wb"), protocol=pickle.HIGHEST_PROTOCOL )
            
    # Co-add the individual tracers using the weights we just calculated
    if 'comb' in run:
        
        # Calculate the optimal weights to form a multitracer map for delensing
        #mobj.get_spectra_matrix()
        #weight = delens_func.multitracer_weights( mobj.clnl_matrix, mobj.signal_covariance[0,:,:], mobj.lmin ) # for analytic filter
        weight = calculate_multitracer_weights_sim( glob, qobj, mobj, mmask=mmask, kmask=kmask, **kwargs_ov )
        
        # loop over realizations to combine mass tracers with the above weight
        for ii, i in enumerate(tqdm.tqdm(glob.rlz,ncols=100,desc='coadding multitracer')):
            
            if misctools.check_path(mobj.fcklm[i],**kwargs_ov): continue
                
            # prepare alm array
            alms = load_mass_tracers( i, qobj, mobj, mmask=mmask, kmask=kmask )
            
            # coadd tracers
            #cklms = coadd_kappa_alms( alms, weight ) # for analytic filter
            cklms = coadd_kappa_alms( alms, weight[ii,:,:] )
            
            # save
            pickle.dump( (cklms), open(mobj.fcklm[i],"wb"), protocol=pickle.HIGHEST_PROTOCOL )

    return mobj
