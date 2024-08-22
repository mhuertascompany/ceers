import sys
import numpy as np
import astropy.io.fits as fits
from astropy.io import ascii
from astropy.table import Table, join, hstack, vstack
from scipy.optimize import curve_fit
from scipy import interpolate


from CosmicVariance import cvc

area_fullsky = 41253

from astropy.cosmology import FlatLambdaCDM as FLCDM             # we import the package that we need
from astropy.cosmology import z_at_value
cosmo_ap = FLCDM(H0=70, Om0=0.3, Tcmb0=2.725)  


#cvdat = np.array(ascii.read('/n08data/shuntov/COSMOS2020_works/conservative_8to13_005.csv'))
#cols= [f'col{i}' for i in range(2,101)]
#cvdat_zidx  = {'z1': 11, 
 #             'z2': 10, 
 #             'z3': 9, 
 ##             'z4': 8, 
 #             'z5': 7, 
 #             'z6': 6, 
 #             'z7': 5, 
 #             'z8': 4, 
 #             'z9': 3, 
 #             'z10': 2,
 #             'z11': 1,
 #             'z12': 1,
 #             'z13': 1,
 #             'z14': 1}

def get_CosmicVariance_cvc(N, z1, z2, area_survey):
    ''' Uses the Cosmic Variance Calculator of Trenti & Stiavelli
        Computes the cosmic variance error

        ARGUMENTS:
        N: float 
            number of galaxies
        z1,z2: float
            lower and upper redshift range
        area_survey: float
            the area of the survey in deg^2
        RETURNS:
        v_r: float
            the relative cosmic variance + poisson error: dFi = Fi*v_r
        

    '''
    sizeXarcmin = np.sqrt(area_survey*60*60)
    var = cvc.CosmicVariance({ 'SurveyAreaX':sizeXarcmin, 
                               'SurveyAreaY':sizeXarcmin, 
                               'MeanZ': (z1+z2)/2, 
                               'Dz': (z2-z1),
                               'IntNum': N,
                               'HaloOff': 1.0,
                               'Completeness': 0.7,
                               'Bias': 1,},
                                # cosmo={'OmegaL': (1-cosmo['Omega_c']+cosmo['Omega_b']), 
                                #        'OmegaM': (cosmo['Omega_c']+cosmo['Omega_b']), 
                                #        'Omegab': cosmo['Omega_b'], 
                                #        'h': cosmo['h'], 
                                #        'sigma8': cosmo['sigma8'], 
                                #        'ns': cosmo['n_s']}
                                cosmo={'OmegaL': 0.7, 
                                       'OmegaM': 0.3, 
                                       'Omegab': 0.0469, 
                                       'h': 0.7, 
                                       'sigma8': 0.8159, 
                                       'ns': 0.97}
                                       )
    
    var.clean_sigmaDM()
    var.compute_sigmaDM()
    var.compute_meanbias()
    
#     ngal = var.beamdata.density*var.indata.HaloOff * (cosmo['h']**(3)) #in units of Mpc-3
    # v_r = np.sqrt(1.0/(var.indata.IntNum*var.indata.Completeness) + var.meanbias*var.meanbias*var.sigma2DM)
    v_r = np.sqrt(var.meanbias*var.meanbias*var.sigma2DM)
    
    return v_r[0]



def get_weight_fit(mass_lo, mass_up, mstar, mpdf):
    ''' For a sample of sources passed directly to this function
    '''
    dm = mass_up-mass_lo
    logM_forpdf = np.arange(8., 13.01, 0.05) # the stellar masses for which the pdf is defined

    cond_pdfmbin = (logM_forpdf>=mass_lo) & (logM_forpdf<mass_up)
    wgh=[]
    if len(mstar)>0:
        for j in range(len(mstar)):
            mpdf_mbin = mpdf[j][cond_pdfmbin]
            # compute the weight for each source
            pdf_integr = np.trapz(mpdf_mbin, x=logM_forpdf[cond_pdfmbin])#dx=0.05)
            pdf_wgh = np.trapz(mpdf[j],dx=0.05)
            wgh.append(pdf_integr / pdf_wgh)
    return np.array(wgh)



def fitfunc(Ms, A1,s1):
    ''' Fitting function (exponential) for the low mass end of the cosmic variance error
    '''
    return A1*np.exp(s1*Ms)#+A2*np.exp(s2*Ms)


def compute_z_faint(z, mS, mS_faint, z_max=99): 
    """ Compute the maximum redshift for galaxy to be included in sample
    used for 1/Vmax luminosity function, see Ilbert+2005

    COMMENT: only include flux variation with redshift
    in principle should include size dependence as well !

    z: array of floats
        redshift
    mS: array of floats
        apparent magnitude in the selected band (observed frame, unredshifted filter curve)
    mS_faint: float
        apparent magnitude limit in the selected band
    z_max: float
        redshift interval upper bound
    """
    assert(np.shape(z) == np.shape(mS))
    if (mS_faint < mS).any(): print('WARNING: objects fainter than detection limit')
    assert((z <= z_max).all())

    dL = cosmo_ap.luminosity_distance(z)
    f = 10**(0.2*(mS_faint-mS)) # luminosity distance inversely propto the sqrt of flux
    f = np.maximum(f, 1.) # correction for faint objects, mS>mS_faint
    dL_faint = dL*f
    dL_faint = np.minimum(dL_faint, cosmo_ap.luminosity_distance(z_max)) # avoid too large distances for z_at_value
    z_faint = np.array([z_at_value(cosmo_ap.luminosity_distance, d) for d in dL_faint])
    z_faint = np.clip(z_faint, z, z_max) # security, z_at_value not precise enough
    return z_faint



def v_max(maglim, mag, z, z1, z2, area_survey):
    ''' Computes the maximum volume in which a galaxy of a given magnitude can be seen.
    
    ARGUMENTS:
    maglim: float
        the magnitude limit of the band
    mag: array of floats
        the magnitudes of the sources
    z: array of floats
        the redshits of the sources
    z1,z2: float
        lower and upper redshift range
    area_survey: float
        the area of the survey in deg^2
    RETURNS:
    vmax: array of floats
        the maximum comoving volume a galaxy can be seen in

    '''
    
    zlim = compute_z_faint(z, mag, maglim, z_max=99)
    
    zmin = z1
    zmax = np.amin([np.ones_like(zlim)*z2, zlim], axis=0)
    
    vmax = 4*np.pi/3 * area_survey/area_fullsky * (cosmo_ap.comoving_distance(zmax)**3 - cosmo_ap.comoving_distance(zmin)**3)
    return vmax.value


def get_smf_vmax_OLD(logMass, dLogMass, mag, maglim, z, zmin, zmax, Szbin, survey_area):
    ''' Computes the SMF

    ARGUMENTS:
    logMass: array float
        Log stellar mass of the sample
    dLogMass: float
        the width of the mass bins
    mag: array of floats
        magnitudes of the sources
    maglim: floats
        magnitude limit of the band
    z: array of floats
        the redshits of the sources
    zmin,zmax: float
        lower and upper redshift range
    area_survey: float
        the area of the survey in deg^2
    RETURNS:
    LogMassbin: array of floats
        The centers of the mass bins
    Fi: array of floats
        the SMF 
    dFi: array of floats
        the errorbars of the SMF including poisson noise and cosmic variance

    '''

    mass_min = np.amin(logMass) 
    mass_max = np.amax(logMass) + dLogMass
    bin_edges = np.arange(mass_min, mass_max, dLogMass)
    LogMassbin = np.zeros(len(bin_edges)-1)
    magbin = np.zeros(len(bin_edges)-1)
    for j in range(len(bin_edges)-1):
        LogMassbin[j] = (bin_edges[j]+bin_edges[j+1])/2

    ### CV errors  will be added from external file computed with mocks. Use the fit at Ms < np.min(sigma_cv_perZ[zbin][2,:]); interpolate from a linear spline at higher masses
    ik = 5
    if zmax>5:
        ik = 4
    valid = ~(np.isnan(sigma_cv_perZ[Szbin][0,:ik]))
    npopt, npcov = curve_fit(fitfunc, sigma_cv_perZ[Szbin][2,:ik][valid], sigma_cv_perZ[Szbin][0,:ik][valid])
    # interp = interpolate.interp1d(sigma_cv_perZ[Szbin][2,:], sigma_cv_perZ[Szbin][0,:], kind='slinear')

        
    Fi = np.zeros_like(LogMassbin)
    dFi = np.zeros_like(LogMassbin)

    for k in range(len(LogMassbin)):
        zbin = z[(logMass >= bin_edges[k]) & (logMass < bin_edges[k+1])]
        magbin = mag[(logMass >= bin_edges[k]) & (logMass < bin_edges[k+1])]
        vmax = np.ones_like(zbin)
        
        if len(zbin)<1:
            fi = 0
            fi_err = np.nan
        else:
            vmax = v_max(maglim, magbin, zbin, zmin, zmax, survey_area)     
            fi = np.sum(1/vmax)/dLogMass
            poiss = np.sqrt(np.sum(1/vmax**2))/dLogMass
            cv_rel = get_CosmicVariance_cvc(len(logMass), zmin, zmax, survey_area)
            ## if LogMassbin[k] <= np.min(sigma_cv_perZ[Szbin][2,:]):
            # cv_rel = fitfunc(LogMassbin[k], *npopt)
            ## elif LogMassbin[k] >= np.max(sigma_cv_perZ[Szbin][2,:]):
            ##     cv_rel = interp(np.max(sigma_cv_perZ[Szbin][2,:]))
            ## else:
            ##     cv_rel = interp(LogMassbin[k])
            fi_err = np.hypot(fi*cv_rel, poiss)
        Fi[k] = fi 
        dFi[k] = fi_err
    
    return LogMassbin, np.asarray(Fi), np.asarray(dFi)



def get_smf_vmax(lensID, logMass, dLogMass, mag, maglim, z, zmin, zmax, Szbin, survey_area, FtabMpdf, cvdat, cvdat_zidx):
    ''' Computes the SMF

    ARGUMENTS:
    logMass: array float
        Log stellar mass of the sample
    dLogMass: float
        the width of the mass bins
    mag: array of floats
        magnitudes of the sources
    maglim: floats
        magnitude limit of the band
    z: array of floats
        the redshits of the sources
    zmin,zmax: float
        lower and upper redshift range
    area_survey: float
        the area of the survey in deg^2
    RETURNS:
    LogMassbin: array of floats
        The centers of the mass bins
    Fi: array of floats
        the SMF 
    dFi: array of floats
        the errorbars of the SMF including poisson noise and cosmic variance

    '''

    mass_min = np.amin(logMass) 
    mass_max = np.amax(logMass) + dLogMass
    bin_edges = np.arange(mass_min, mass_max, dLogMass)
    LogMassbin = np.zeros(len(bin_edges)-1)
    magbin = np.zeros(len(bin_edges)-1)
    for j in range(len(bin_edges)-1):
        LogMassbin[j] = (bin_edges[j]+bin_edges[j+1])/2

    ### CV errors  will be added from external file computed from the modified Moster code
    masscv = np.array([cvdat[c][0] for c in cols])
    cv = np.array([cvdat[c][cvdat_zidx] for c in cols])
    # do interpolation
    cv_interp = interpolate.interp1d(masscv, cv, kind='slinear')
    
        
    Fi = np.zeros_like(LogMassbin)
    dFi = np.zeros_like(LogMassbin)
    fit_wghts = []
    
    for k in range(len(LogMassbin)):
        zbin = z[(logMass >= bin_edges[k]) & (logMass < bin_edges[k+1])]
        magbin = mag[(logMass >= bin_edges[k]) & (logMass < bin_edges[k+1])]
        vmax = np.ones_like(zbin)

    
        if len(zbin)<1:
            fi = 0
            fi_err = np.nan
        else:

            lTid = Table()
            lTid['ID'] = lensID[(logMass >= bin_edges[k]) & (logMass < bin_edges[k+1])]

            tabMpdf = join(lTid, FtabMpdf, keys = 'ID', join_type='inner')
            mstar = tabMpdf['MASS_MED'].data  # column names may be subject to change
            mpdf = tabMpdf['PDM'].data
            
            vmax = v_max(maglim, magbin, zbin, zmin, zmax, survey_area)   
            weight_fit = get_weight_fit(bin_edges[k], bin_edges[k+1], mstar, mpdf)
            # 'remove' outliers that have very low probability to be in the mass bin
            mr = np.where(weight_fit>0.004)
            # THE SMF
            fi = np.sum(1/vmax)/dLogMass
            # THE ERRORS
            poiss_fit = np.sqrt(np.sum(1/(weight_fit[mr])**2/vmax[mr]**2))/dLogMass
            cv_rel = cv_interp(np.mean([bin_edges[k], bin_edges[k+1]])) 
            
            fi_err = np.hypot(cv_rel*fi, poiss_fit)

        Fi[k] = fi
        dFi[k] = fi_err
    
    return LogMassbin, Fi, dFi



def get_smf_cv_poiss_err(lensID, logMass, dLogMass, mag, maglim, z, zmin, zmax, survey_area):
    ''' Computes the SMF
    !!!!!!!! HAS ONLY POISSON AND CV ERRORS !!!!!!!!!!!!
    
    ARGUMENTS:
    logMass: array float
        Log stellar mass of the sample
    dLogMass: float
        the width of the mass bins
    mag: array of floats
        magnitudes of the sources
    maglim: floats
        magnitude limit of the band
    z: array of floats
        the redshits of the sources
    zmin,zmax: float
        lower and upper redshift range
    area_survey: float
        the area of the survey in deg^2
    RETURNS:
    LogMassbin: array of floats
        The centers of the mass bins
    Fi: array of floats
        the SMF 
    dFi: array of floats
        the errorbars of the SMF including poisson noise and cosmic variance

    '''

    mass_min = np.amin(logMass) 
    mass_max = np.amax(logMass) + dLogMass
    bin_edges = np.arange(mass_min, mass_max, dLogMass)
    LogMassbin = np.zeros(len(bin_edges)-1)
    magbin = np.zeros(len(bin_edges)-1)
    for j in range(len(bin_edges)-1):
        LogMassbin[j] = (bin_edges[j]+bin_edges[j+1])/2
        
    Fi = np.zeros_like(LogMassbin)
    dFi = np.zeros_like(LogMassbin)
    dFi_cv = np.zeros_like(LogMassbin)
    dFi_pois = np.zeros_like(LogMassbin)
    
    volume = (4*np.pi/3 * survey_area/area_fullsky * (cosmo_ap.comoving_distance(zmax)**3 - cosmo_ap.comoving_distance(zmin)**3)).value
    
    for k in range(len(LogMassbin)):
        zbin = z[(logMass >= bin_edges[k]) & (logMass < bin_edges[k+1])]
        magbin = mag[(logMass >= bin_edges[k]) & (logMass < bin_edges[k+1])]
        vmax = np.ones_like(zbin) 
        
        if len(zbin)>0:
            vmax = v_max(maglim, magbin, zbin, zmin, zmax, survey_area)   
            fi = np.sum(1/vmax)/dLogMass
            # THE ERRORS
#             poiss_fit = poiss = np.sqrt((len(zbin)/volume**2))/dLogMass
            poiss_fit = poiss = np.sqrt(np.sum(1/vmax**2))/dLogMass
            cv_rel = get_CosmicVariance_cvc(len(zbin), zmin, zmax, survey_area)
            fi_err = np.hypot(cv_rel*fi, poiss_fit)
#### no vmax, fast:
# #             vmax = v_max(maglim, magbin, zbin, zmin, zmax, survey_area)   
#             fi = len(zbin)/volume/dLogMass
#             # THE ERRORS
#             poiss_fit = poiss = np.sqrt((len(zbin)/volume**2))/dLogMass
#             cv_rel = get_CosmicVariance_cvc(len(zbin), zmin, zmax, survey_area)
#             fi_err = np.hypot(cv_rel*fi, poiss_fit)
        elif len(zbin)<1:
            fi = 0
            fi_err = 1.841/volume#get_CosmicVariance_cvc(1, zmin, zmax, survey_area)

        Fi[k] = fi
        dFi[k] = fi_err
        dFi_cv[k] = cv_rel*fi
        dFi_pois[k] = poiss_fit
    
    return LogMassbin, Fi, dFi, dFi_cv, dFi_pois


def get_smf_only(logMass, dLogMass, z, zmin, zmax, mass_min, mass_max, survey_area):
    ''' Computes the SMF
    !!!!!!!! FAST, NO ERRORS !!!!!!!!!!!!
    
    ARGUMENTS:
    logMass: array float
        Log stellar mass of the sample
    dLogMass: float
        the width of the mass bins
    z: array of floats
        the redshits of the sources
    zmin,zmax: float
        lower and upper redshift range
    area_survey: float
        the area of the survey in deg^2
    RETURNS:
    LogMassbin: array of floats
        The centers of the mass bins
    Fi: array of floats
        the SMF 
    dFi: array of floats
        the errorbars of the SMF including poisson noise and cosmic variance

    '''
#     mass_min = np.amin(logMass) 
#     mass_max = np.amax(logMass) + dLogMass
    
    bin_edges = np.arange(mass_min, mass_max, dLogMass)
    LogMassbin = np.zeros(len(bin_edges)-1)
    magbin = np.zeros(len(bin_edges)-1)
    for j in range(len(bin_edges)-1):
        LogMassbin[j] = (bin_edges[j]+bin_edges[j+1])/2
        
    Fi = np.zeros_like(LogMassbin)
    dFi = np.zeros_like(LogMassbin)
    dFi_cv = np.zeros_like(LogMassbin)
    dFi_pois = np.zeros_like(LogMassbin)
    dFi_sed = np.zeros_like(LogMassbin)
    
    volume = (4*np.pi/3 * survey_area/area_fullsky * (cosmo_ap.comoving_distance(zmax)**3 - cosmo_ap.comoving_distance(zmin)**3)).value
    print('mass array', logMass)
    for k in range(len(LogMassbin)):
        zbin = z[(logMass >= bin_edges[k]) & (logMass < bin_edges[k+1])]
        
        if len(zbin)>0:
#             vmax = v_max(maglim, magbin, zbin, zmin, zmax, survey_area)   
            fi = len(zbin)/volume/dLogMass
            poiss_fit = poiss = np.sqrt((len(zbin)/volume**2))/dLogMass
        elif len(zbin)<1:
            fi = 0
            poiss_fit = 1.841/volume/dLogMass

        Fi[k] = fi
        dFi[k] = poiss_fit
        dFi_cv[k] = poiss_fit
        dFi_pois[k] = poiss_fit
        dFi_sed[k] = poiss_fit
        
    return LogMassbin, Fi #dFi, #dFi_cv, dFi_pois, dFi_sed


def get_sigma_SED(path, zmin, zmax, sample='all'):
    ''' Gets the SED fitting M* uncertainties on the SMF. The std on the SMF is computed from 1000 samples and is saved in a folder. 
    This reads the saved files and does interpolations
    sample can be 'all', 'Quies' or 'SF'
    '''
    # read the SMF
    pth_zbin = f'{path}/z-'+str(zmin)+'-'+str(zmax)
    fname = pth_zbin+f'/SMF-{sample}_z-'+str(zmin)+'-'+str(zmax)+'.out'
    get_smf_data = ascii.read(fname, header_start=0, data_start=1)
    log10Ms_data = get_smf_data['logM']
    smf_data = get_smf_data['Fi']
    smf_err_data = get_smf_data['dFi']
    sigma_rel = smf_err_data/smf_data

    # interpolate
    sigma_rel = np.nan_to_num(sigma_rel, nan=1)
    f = interpolate.interp1d(np.append(log10Ms_data,13), np.append(sigma_rel,sigma_rel[-1]))

    return f, np.amax(log10Ms_data)

#ver = 'v1.7-SED-Uncertainties'
#folderout_ver = f'/n08data/shuntov/COSMOS-Web/SMF_measurements/{ver}'


def get_smf_CW(lensID, logMass, dLogMass, mag, maglim, z, zmin, zmax, zbin, path_sigma_sed, sample, survey_area):
    ''' Computes the SMF for COSMOS-Web
    
    ARGUMENTS:
    logMass: array float
        Log stellar mass of the sample
    dLogMass: float
        the width of the mass bins
    mag: array of floats
        magnitudes of the sources
    maglim: floats
        magnitude limit of the band
    z: array of floats
        the redshits of the sources
    zmin,zmax: float
        lower and upper redshift range
    area_survey: float
        the area of the survey in deg^2
    RETURNS:
    LogMassbin: array of floats
        The centers of the mass bins
    Fi: array of floats
        the SMF 
    dFi: array of floats
        the errorbars of the SMF including poisson noise and cosmic variance

    '''
    sigma_sed_rel, maxmassinterp = get_sigma_SED(path_sigma_sed, zmin, zmax, sample=sample)
    
    mass_min = np.amin(logMass) 
    mass_max = np.amax(logMass) + dLogMass
    print('mass_max', mass_max, 'maxmassinterp', maxmassinterp)
    bin_edges = np.arange(mass_min, mass_max, dLogMass)
    LogMassbin = np.zeros(len(bin_edges)-1)
    magbin = np.zeros(len(bin_edges)-1)
    for j in range(len(bin_edges)-1):
        LogMassbin[j] = (bin_edges[j]+bin_edges[j+1])/2
        
    Fi = np.zeros_like(LogMassbin)
    dFi = np.zeros_like(LogMassbin)
    dFi_cv = np.zeros_like(LogMassbin)
    dFi_pois = np.zeros_like(LogMassbin)
    dFi_sed = np.zeros_like(LogMassbin)
    
    ### CV errors  will be added from external file computed from the modified Moster code
    # do interpolation
    #cv_interp = interpolate.interp1d(cv_mass, cv_sig[zbin], kind='slinear')
    
    volume = (4*np.pi/3 * survey_area/area_fullsky * (cosmo_ap.comoving_distance(zmax)**3 - cosmo_ap.comoving_distance(zmin)**3)).value
    
    
    for k in range(len(LogMassbin)):
        zbin = z[(logMass >= bin_edges[k]) & (logMass < bin_edges[k+1])]
        magbin = mag[(logMass >= bin_edges[k]) & (logMass < bin_edges[k+1])]
        vmax = np.ones_like(zbin) 
        
        if len(zbin)>0:
            vmax = v_max(maglim, magbin, zbin, zmin, zmax, survey_area) 
            fi = np.sum(1/vmax)/dLogMass
            # THE ERRORS
            poiss_fit = poiss = np.sqrt(np.sum(1/vmax**2))/dLogMass
            if True:
                cv_rel = get_CosmicVariance_cvc(len(zbin), zmin, zmax, survey_area)
            if False:
                cv_rel = cv_interp(np.mean([bin_edges[k], bin_edges[k+1]])) 
                
        elif len(zbin)<1:
            fi = 0
            fi_err = 1.841/volume/dLogMass #get_CosmicVariance_cvc(1, zmin, zmax, survey_area)

        Fi[k] = fi
        dFi_cv[k] = cv_rel*fi
        dFi_pois[k] = poiss_fit
    dFi_sed = sigma_sed_rel(LogMassbin)*Fi
    dFi = np.sqrt((dFi_cv)**2 + (dFi_pois)**2 + (dFi_sed)**2)
    
    return LogMassbin, Fi, dFi, dFi_cv, dFi_pois, dFi_sed




def matching_catalogs(ra0, dec0, ra1, dec1, matchdist):
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    refcat = SkyCoord(ra=ra0*u.degree, dec=dec0*u.degree)
    matchcat =  SkyCoord(ra=ra1*u.degree, dec=dec1*u.degree)
    idx, d2d, d3d = matchcat.match_to_catalog_sky(refcat)
    print('max separ', np.amax(d2d.arcsec), 'min separ', np.amin(d2d.arcsec), 'mean separ', np.mean(d2d.arcsec), 'stdev', np.std(d2d.arcsec))
    print('len idx', len(idx[np.where(d2d.arcsec<matchdist)]), 'from initial', len(idx), 'no. discarded:', len(idx)-len(idx[np.where(d2d.arcsec<matchdist)]))
    refcat_best = refcat[idx[np.where(d2d.arcsec<matchdist)]]
    idx2, d2d2, d3d2 = refcat_best.match_to_catalog_sky(matchcat)
    print('len idx2 of matched catalog', len(idx2))
    print('----end of matching----')
    return idx[np.where(d2d.arcsec<matchdist)], idx2
