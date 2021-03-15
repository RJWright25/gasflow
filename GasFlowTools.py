import os
import sys
import time
import logging
import pickle
import h5py
import numpy as np
import pandas as pd

from read_eagle import EagleSnapshot
from scipy.spatial import cKDTree
from astropy.cosmology import FlatLambdaCDM
from astropy import units

#high level

def submit_function(function,arguments,memory,time):
    filename=sys.argv[0]
    cwd=os.getcwd()
    run=cwd.split('/')[-1]

    if not os.path.exists('jobs'):
        os.mkdir('jobs')
    if not os.path.exists('logs'):
        os.mkdir('logs')

    if function=='analyse_gasflow':
        jobname=function+'_'+run+f"_snapidx_{str(arguments['snapidx']).zfill(3)}_n_{str(arguments['nvol']).zfill(2)}_volume_{str(arguments['ivol']).zfill(3)}"
    else:
        jobname=function+'_'+run
    
    runscriptfilepath=f'{cwd}/jobs/{jobname}-run.py'
    jobscriptfilepath=f'{cwd}/jobs/{jobname}-submit.slurm'
    if os.path.exists(jobscriptfilepath):
        os.remove(runscriptfilepath)
    if os.path.exists(jobscriptfilepath):
        os.remove(jobscriptfilepath)

    argumentstring=''
    for arg in arguments:
        if type(arguments[arg])==str:
            argumentstring+=f"{arg}='{arguments[arg]}',"
        else:
            argumentstring+=f"{arg}={arguments[arg]},"


    with open(runscriptfilepath,"w") as runfile:
        runfile.writelines(f"import warnings\n")
        runfile.writelines(f"warnings.filterwarnings('ignore')\n")
        runfile.writelines(f"import sys\n")
        runfile.writelines(f"sys.path.append('/home/rwright/Software/gasflow')\n")
        runfile.writelines(f"from GasFlowTools import *\n")
        runfile.writelines(f"{function}({argumentstring})")
    runfile.close()

    with open(jobscriptfilepath,"w") as jobfile:
        jobfile.writelines(f"#!/bin/sh\n")
        jobfile.writelines(f"#SBATCH --job-name={jobname}\n")
        jobfile.writelines(f"#SBATCH --nodes=1\n")
        jobfile.writelines(f"#SBATCH --ntasks-per-node={1}\n")
        jobfile.writelines(f"#SBATCH --mem={memory}GB\n")
        jobfile.writelines(f"#SBATCH --time={time}\n")
        jobfile.writelines(f"#SBATCH --output=jobs/{jobname}.out\n")
        jobfile.writelines(f" \n")
        jobfile.writelines(f"OMPI_MCA_mpi_warn_on_fork=0\n")
        jobfile.writelines(f"export OMPI_MCA_mpi_warn_on_fork\n")
        jobfile.writelines(f"echo JOB START TIME\n")
        jobfile.writelines(f"date\n")
        jobfile.writelines(f"echo CPU DETAILS\n")
        jobfile.writelines(f"lscpu\n")
        jobfile.writelines(f"python {runscriptfilepath} \n")
        jobfile.writelines(f"echo JOB END TIME\n")
        jobfile.writelines(f"date\n")
    jobfile.close()
    os.system(f"sbatch {jobscriptfilepath}")

def extract_tree(path,mcut,snapidxmin=0):

    outname='catalogues/catalogue_tree.hdf5'
    fields=['snapshotNumber',
            'nodeIndex',
            'fofIndex',
            'hostIndex',
            'descendantIndex',
            'mainProgenitorIndex',
            'enclosingIndex',
            'isFoFCentre',
            'positionInCatalogue']

    mcut=10**mcut/10**10 

    if not os.path.exists('logs'):
        os.mkdir('logs')
    
    if not os.path.exists('catalogues'):
        os.mkdir('catalogues')
    
    if os.path.exists('logs/extract_tree.log'):
        os.remove('logs/extract_tree.log')

    logging.basicConfig(filename='logs/extract_tree.log', level=logging.INFO)
    logging.info(f'Running tree extraction for haloes with mass above {mcut*10**10:.1e} after (and including) snapidx {snapidxmin} ...')

    # get file names
    tree_fnames=os.listdir(path)
    tree_fnames=[tree_fname for tree_fname in tree_fnames if 'tree' in tree_fname]
    nfiles=len(tree_fnames)

    # iterate through all tree files
    t0=time.time()
    for ifile,tree_fname in enumerate(tree_fnames):
        logging.info(f'Processing file {ifile+1} of {nfiles}')
        treefile=h5py.File(f'{path}/{tree_fname}')

        #mass mask
        masses=treefile['/haloTrees/nodeMass'][:];snipshotidx=treefile['/haloTrees/snapshotNumber'][:]
        mask=np.logical_and(masses>mcut,snipshotidx>=snapidxmin)

        #initialise new data
        logging.info(f'Extracting position for {np.sum(mask):.0f} nodes [runtime {time.time()-t0:.2f} sec]')
        newdata=pd.DataFrame(treefile['/haloTrees/position'][mask,:],columns=['position_x','position_y','position_z'])

        #grab all fields
        logging.info(f'Extracting data for {np.sum(mask):.0f} nodes [runtime {time.time()-t0:.2f} sec]')
        newdata.loc[:,fields]=np.column_stack([treefile['/haloTrees/'+field][mask] for field in fields])

        #append to data frame
        if ifile==0:
            data=newdata
        else:
            data=data.append(newdata,ignore_index=True)


        #close file, move to next
        treefile.close()

    if os.path.exists(outname):
        os.remove(outname)
        
    data.to_hdf(f'{outname}',key='Tree')

def extract_fof(path,mcut,snapidxmin=0):
    outname='catalogues/catalogue_fof.hdf5'
    fields=['/FOF/GroupMass',
            '/FOF/Group_M_Crit200',
            '/FOF/Group_R_Crit200',
            '/FOF/NumOfSubhalos',
            '/FOF/GroupCentreOfPotential']

    mcut=10**mcut/10**10 
    redshift_table=pd.read_hdf('snapshot_redshifts.hdf5',key='snapshots')
    dims='xyz'

    groupdirs=os.listdir(path)
    groupdirs=sorted([path+'/'+groupdir for groupdir in groupdirs if ('groups_snip' in groupdir and 'tar' not in groupdir)])

    if os.path.exists('logs/extract_fof.log'):
        os.remove('logs/extract_fof.log')

    logging.basicConfig(filename='logs/extract_fof.log', level=logging.INFO)
    logging.info(f'Running FOF extraction for FOFs with mass above {mcut*10**10:.1e} after (and including) snapidx {snapidxmin} ...')

    t0=time.time()
    ifile=0
    isnap=-1
    for groupdir in groupdirs:
        snap=int(groupdir.split('snip_')[-1][:3])
        try:
            snapidx=redshift_table.loc[snap==redshift_table['snapshot'],'snapshotidx'].values[0]
        except:
            logging.info(f'Skipping snap {snapidx} (not in trees) [runtime {time.time()-t0:.2f} sec]')
            continue

        if snapidx>=snapidxmin:
            isnap+=1
            logging.info(f'Processing snap {snapidx} ({isnap+1}/{len(groupdir)} total) [runtime {time.time()-t0:.2f} sec]')
            groupdirfnames=os.listdir(groupdir)
            groupdirfnames=sorted([groupdir+'/'+groupdirfname for groupdirfname in groupdirfnames if groupdirfname.startswith('eagle_subfind')])
            groupdirfnames_n=len(groupdirfnames)

            for ifile_snap, groupdirfname in enumerate(groupdirfnames):
                groupdirifile=h5py.File(groupdirfname,'r')

                ifile_fofmasses=groupdirifile['/FOF/GroupMass'].value
                ifile_mask=ifile_fofmasses>mcut
                ifile_nfof=np.sum(ifile_mask)
    
                logging.info(f'Snap {snapidx} ({isnap+1}/{len(groupdir)} total), file {ifile_snap+1}/{groupdirfnames_n}: extracting data for {ifile_nfof:.0f} FOFs [runtime {time.time()-t0:.2f} sec]')
                
                if ifile_nfof:
                    newdata=pd.DataFrame(groupdirifile['/FOF/GroupMass'][ifile_mask],columns=['GroupMass'])
                    newdata.loc[:,'snapshotidx']=snapidx

                    for field in fields:
                        dset_shape=groupdirifile[field].shape
                        if len(dset_shape)==2:
                            for icol in range(dset_shape[-1]):
                                if dset_shape[-1]==3:
                                    newdata.loc[:,field.split('FOF/')[-1]+f'_{dims[icol]}']=groupdirifile[field][ifile_mask,icol]
                                else:
                                    if icol in [0,1,4,5]:
                                        newdata.loc[:,field.split('FOF/')[-1]+f'_{icol}']=groupdirifile[field][ifile_mask,icol]
                        else:
                            newdata.loc[:,field.split('FOF/')[-1]]=groupdirifile[field][ifile_mask]

                    if ifile==0:
                        data=newdata
                    else:
                        data=data.append(newdata,ignore_index=True)

                    ifile+=1
                    groupdirifile.close()
        
    if os.path.exists(f'{outname}'):
        os.remove(f'{outname}')

    data.to_hdf(f'{outname}',key='FOF')

def extract_subhalo(path,mcut,snapidxmin=0,overwrite=True):
    outname='catalogues/catalogue_subhalo.hdf5'
    fields=['/Subhalo/GroupNumber',
            '/Subhalo/SubGroupNumber',
            '/Subhalo/Mass',
            '/Subhalo/MassType',
            '/Subhalo/ApertureMeasurements/Mass/030kpc',
            '/Subhalo/ApertureMeasurements/SFR/030kpc',
            '/Subhalo/Vmax',
            '/Subhalo/CentreOfPotential',
            '/Subhalo/Velocity',
            '/Subhalo/CentreOfMass',
            '/Subhalo/HalfMassRad']

    mcut=10**mcut/10**10 
    redshift_table=pd.read_hdf('snapshot_redshifts.hdf5',key='snapshots')
    dims='xyz'

    groupdirs=os.listdir(path)
    groupdirs=sorted([path+'/'+groupdir for groupdir in groupdirs if ('groups_snip' in groupdir and 'tar' not in groupdir)])

    if os.path.exists('logs/extract_subhalo.log'):
        os.remove('logs/extract_subhalo.log')

    logging.basicConfig(filename='logs/extract_subhalo.log', level=logging.INFO)
    logging.info(f'Running subhalo extraction for subhaloes with mass above {mcut*10**10:.1e} after (and including) snapidx {snapidxmin} ...')

    t0=time.time()
    ifile=0
    isnap=-1
    for groupdir in groupdirs:
        snap=int(groupdir.split('snip_')[-1][:3])
        try:
            snapidx=redshift_table.loc[snap==redshift_table['snapshot'],'snapshotidx'].values[0]
        except:
            logging.info(f'Skipping snap {snapidx} (not in trees) [runtime {time.time()-t0:.2f} sec]')
            continue

        if snapidx>=snapidxmin:
            isnap+=1
            logging.info(f'Processing snap {snapidx} ({isnap+1}/{len(groupdirs)} total) [runtime {time.time()-t0:.2f} sec]')
            groupdirfnames=os.listdir(groupdir)
            groupdirfnames=sorted([groupdir+'/'+groupdirfname for groupdirfname in groupdirfnames if groupdirfname.startswith('eagle_subfind')])
            groupdirfnames_n=len(groupdirfnames)

            for ifile_snap, groupdirfname in enumerate(groupdirfnames):
                groupdirifile=h5py.File(groupdirfname,'r')

                ifile_submasses=groupdirifile['/Subhalo/Mass'].value
                ifile_mask=ifile_submasses>mcut
                ifile_nfof=np.sum(ifile_mask)
    
                logging.info(f'Snap {snapidx} ({isnap+1}/{len(groupdirs)} total), file {ifile_snap+1}/{groupdirfnames_n}: extracting data for {ifile_nfof:.0f} subhaloes [runtime {time.time()-t0:.2f} sec]')
                
                if ifile_nfof:
                    newdata=pd.DataFrame(groupdirifile['/Subhalo/Mass'][ifile_mask],columns=['Mass'])
                    newdata.loc[:,'snapshotidx']=snapidx

                    for field in fields:
                        dset_shape=groupdirifile[field].shape
                        if len(dset_shape)==2:
                            for icol in range(dset_shape[-1]):
                                if dset_shape[-1]==3:
                                    newdata.loc[:,field.split('Subhalo/')[-1]+f'_{dims[icol]}']=groupdirifile[field][ifile_mask,icol]
                                else:
                                    if icol in [0,1,4,5]:
                                        newdata.loc[:,field.split('Subhalo/')[-1]+f'_{icol}']=groupdirifile[field][ifile_mask,icol]
                        else:
                            newdata.loc[:,field.split('Subhalo/')[-1]]=groupdirifile[field][ifile_mask]

                    if ifile==0:
                        data=newdata
                    else:
                        data=data.append(newdata,ignore_index=True)

                    ifile+=1
                    groupdirifile.close()
    try:
        if overwrite:
            
            if os.path.exists(f'{outname}'):
                os.remove(f'{outname}')
            data.to_hdf(f'{outname}',key='Subhalo')

        else:
            logging.info(f'Loading old catalogue ...')
            data_old=pd.read_hdf(f'{outname}',key='Subhalo')
            fields_new=list(data)
            fields_old=list(data_old)
            fields_new_mask=np.isin(fields_new,fields_old,invert=True)
            fields_to_add=fields_new[np.where(fields_new_mask)]
            for field_new in fields_to_add:
                logging.info(f'Adding new field to old catalogue: {field_new}')
                data_old.loc[:,field_new]=data[field_new].values

            data_old.to_hdf(f'{outname}',key='Subhalo')
    except:
        data.to_hdf(f'catalogues/catalogue_subhalo-BACKUP.hdf5',key='Subhalo')

    pass

def match_tree(mcut,snapidxs=[]):

    outname='catalogues/catalogue_subhalo.hdf5'
    catalogue_subhalo=pd.read_hdf('catalogues/catalogue_subhalo.hdf5',key='Subhalo',mode='r')
    catalogue_tree=pd.read_hdf('catalogues/catalogue_tree.hdf5',key='Tree',mode='r')
    fields_tree=['nodeIndex',
                 'fofIndex',
                 'hostIndex',
                 'descendantIndex',
                 'mainProgenitorIndex']


    mcut=10**mcut/10**10

    if os.path.exists('logs/match_tree.log'):
        os.remove('logs/match_tree.log')

    logging.basicConfig(filename='logs/match_tree.log', level=logging.INFO)
    logging.info(f'Running tree matching for subhaloes with mass above {mcut*10**10:.1e} for {len(snapidxs)} snaps ')

    nsub_tot=catalogue_subhalo.shape[0]

    t0=time.time()
    for isnap,snapidx in enumerate(snapidxs):
        logging.info(f'Processing snap {snapidx} ({isnap+1}/{len(snapidxs)}) [runtime {time.time()-t0:.2f} sec]')
        snap_mass_mask=np.logical_and(catalogue_subhalo['snapshotidx']==snapidx,catalogue_subhalo['Mass']>mcut)
        snap_catalogue_subhalo=catalogue_subhalo.loc[snap_mass_mask,:]
        for field in fields_tree:
            snap_catalogue_subhalo.loc[:,field]=-1
        snap_tree_catalogue=catalogue_tree.loc[catalogue_tree['snapshotNumber']==snapidx,:]
        snap_tree_coms=snap_tree_catalogue.loc[:,[f'position_{x}' for x in 'xyz']].values

        iisub=0;nsub_snap=snap_catalogue_subhalo.shape[0]
        t0halo=time.time()
        for isub,sub in snap_catalogue_subhalo.iterrows():
            isub_com=[sub[f'CentreOfPotential_{x}'] for x in 'xyz']
            isub_match=np.sqrt(np.sum(np.square(snap_tree_coms-isub_com),axis=1))==0
            isnap_match=snap_catalogue_subhalo.index==isub
            if np.sum(isub_match):
                isub_treedata=snap_tree_catalogue.loc[isub_match,fields_tree].values
                snap_catalogue_subhalo.loc[isnap_match,fields_tree]=isub_treedata
            else:
                logging.info(f'Warning: could not match subhalo {iisub} at ({isub_com[0]:.2f},{isub_com[1]:.2f},{isub_com[2]:.2f}) cMpc')
                pass

            if not iisub%100:
                logging.info(f'Done matching {(iisub+1)/nsub_snap*100:.1f}% of subhaloes at snap {snapidx} ({isnap+1}/{len(snapidxs)}) [runtime {time.time()-t0:.2f} sec]')

            iisub+=1
        
        catalogue_subhalo.loc[snap_mass_mask,:]=snap_catalogue_subhalo
        print(catalogue_subhalo)

    os.remove(outname)
    catalogue_subhalo.to_hdf(outname,key='Subhalo')

def match_fof(mcut,snapidxs=[]):

    outname='catalogues/catalogue_subhalo.hdf5'
    catalogue_subhalo=pd.read_hdf('catalogues/catalogue_subhalo.hdf5',key='Subhalo',mode='r')
    catalogue_fof=pd.read_hdf('catalogues/catalogue_fof.hdf5',key='FOF',mode='r')
    fields_fof=['GroupMass',
                'Group_M_Crit200',
                'Group_R_Crit200',
                'NumOfSubhalos']

    mcut=10**mcut/10**10

    if os.path.exists('logs/match_fof.log'):
        os.remove('logs/match_fof.log')

    logging.basicConfig(filename='logs/match_fof.log', level=logging.INFO)
    logging.info(f'Running FOF matching for subhaloes with mass above {mcut*10**10:.1e} for {len(snapidxs)} snaps ...')

    
    t0=time.time()
    for isnap,snapidx in enumerate(snapidxs):
        logging.info(f'Processing snap {snapidx} ({isnap+1}/{len(snapidxs)}) [runtime {time.time()-t0:.2f} sec]')
        snap_mass_mask=np.logical_and(catalogue_subhalo['snapshotidx']==snapidx,catalogue_subhalo['Mass']>mcut)
        central_mask=np.logical_and.reduce([snap_mass_mask,catalogue_subhalo['SubGroupNumber']==0])
        snap_catalogue_subhalo=catalogue_subhalo.loc[snap_mass_mask,:]
        for field in fields_fof:
            snap_catalogue_subhalo.loc[:,field]=-1
        snap_central_catalogue=catalogue_subhalo.loc[central_mask,:]

        logging.info(f'Matching for {np.sum(central_mask)} groups with centrals above {mcut*10**10:.1e}msun at snipshot {snapidx} [runtime {time.time()-t0:.2f} sec]')
        central_coms=snap_central_catalogue.loc[:,[f"CentreOfPotential_{x}" for x in 'xyz']].values
        central_groupnums=snap_central_catalogue.loc[:,f"GroupNumber"].values

        fofcat_snap=catalogue_fof.loc[catalogue_fof['snapshotidx']==snapidx,:]
        fofcat_coms=catalogue_fof.loc[catalogue_fof['snapshotidx']==snapidx,[f"GroupCentreOfPotential_{x}" for x in 'xyz']].values
        for icentral,(central_com,central_groupnum) in enumerate(zip(central_coms,central_groupnums)):
            if icentral%1000==0:
                logging.info(f'Processing group {icentral+1} of {np.sum(central_mask)} at snipshot {snapidx} ({icentral/np.sum(central_mask)*100:.1f}%) [runtime {time.time()-t0:.2f} sec]')
            fofmatch=np.sum(np.square(fofcat_coms-central_com),axis=1)<=(0.001)**2
            ifofmatch_data=fofcat_snap.loc[fofmatch,fields_fof].values
            ifofsubhaloes=snap_catalogue_subhalo['GroupNumber']==int(central_groupnum)
            if np.sum(ifofsubhaloes):
                snap_catalogue_subhalo.loc[ifofsubhaloes,fields_fof]=ifofmatch_data
            else:
                logging.info(f'Warning: no matching group for central {icentral}')
        
        catalogue_subhalo.loc[snap_mass_mask,:]=snap_catalogue_subhalo

    os.remove(outname)
    catalogue_subhalo.to_hdf(outname,key='Subhalo')

def analyse_subhalo(path,mcut,snapidx,nvol,ivol):
    ivol=int(ivol)
    ivol=str(ivol).zfill(3)
    ix,iy,iz=ivol_idx(ivol,nvol=nvol)


    t0=time.time()
    logfile=f'logs/subhalo/subhalo_snapidx_{snapidx}_n_{str(nvol).zfill(2)}_volume_{ivol}.log'
    if os.path.exists(logfile):
        os.remove(logfile)
    logging.basicConfig(filename=logfile, level=logging.INFO)
    
    output_fname=f'catalogues/subhalo/subhalo_snapidx_{snapidx}_n_{str(nvol).zfill(2)}_volume_{str(ivol).zfill(3)}.hdf5'
    if os.path.exists(output_fname):
        os.remove(output_fname)

    redshift_table=pd.read_hdf('snapshot_redshifts.hdf5',key='snapshots')

    snapidx_tag=str(redshift_table.loc[redshift_table['snapshotidx']==snapidx,'tag'].values[0])
    snapidx_particledatapath=f'{path}/particledata_{snapidx_tag}/eagle_subfind_snip_particles_{snapidx_tag[5:]}.0.hdf5'
    
    #read data
    boxsize=h5py.File(snapidx_particledatapath,'r')['Header'].attrs['BoxSize']
    redshift=h5py.File(snapidx_particledatapath,'r')['Header'].attrs['Redshift']
    subvol_edgelength=boxsize/nvol
    buffer=subvol_edgelength/10
    cosmology=FlatLambdaCDM(H0=h5py.File(snapidx_particledatapath,'r')['Header'].attrs['HubbleParam']*100,
                            Om0=h5py.File(snapidx_particledatapath,'r')['Header'].attrs['Omega0'])

    rhocrit=cosmology.critical_density(redshift)
    rhocrit=rhocrit.to(units.Msun/units.Mpc**3)
    rhocrit=rhocrit.value
    xmin=ix*subvol_edgelength;xmax=(ix+1)*subvol_edgelength
    ymin=iy*subvol_edgelength;ymax=(iy+1)*subvol_edgelength
    zmin=iz*subvol_edgelength;zmax=(iz+1)*subvol_edgelength

    logging.info(f'Considering region: (1/{nvol**3} of full box) [runtime = {time.time()-t0:.2f}s]')
    logging.info(f'ix: {ix} - x in [{xmin},{xmax}]')
    logging.info(f'iy: {iy} - y in [{ymin},{ymax}]')
    logging.info(f'iz: {iz} - z in [{zmin},{zmax}]')

    snapidx_eagledata = EagleSnapshot(snapidx_particledatapath)
    snapidx_eagledata.select_region(xmin-buffer, xmax+buffer, ymin-buffer, ymax+buffer, zmin-buffer, zmax+buffer)

    logging.info(f'Initialising particle data with IDs [runtime = {time.time()-t0:.2f}s]')
    particledata_snap=pd.DataFrame(snapidx_eagledata.read_dataset(0,'ParticleIDs'),columns=['ParticleIDs'])
    particledata_snap.loc[:,"ParticleTypes"]=0
    particledata_snap_star=pd.DataFrame(snapidx_eagledata.read_dataset(4,'ParticleIDs'),columns=['ParticleIDs'])
    particledata_snap_star.loc[:,"ParticleTypes"]=4;particledata_snap_star.loc[:,"Temperature"]=-1.;particledata_snap_star.loc[:,"Density"]=10**10

    logging.info(f'Reading gas datasets [runtime = {time.time()-t0:.2f}s]')
    for dset in ['Coordinates','Velocity','Mass','Density','Temperature','Metallicity','SubGroupNumber']:
        dset_snap=snapidx_eagledata.read_dataset(0,dset)
        if dset_snap.shape[-1]==3:
                particledata_snap.loc[:,[f'{dset}_x',f'{dset}_y',f'{dset}_z']]=dset_snap
        else:
            if dset=='Mass':
                particledata_snap[dset]=dset_snap
            else:
                particledata_snap[dset]=dset_snap

    logging.info(f'Reading star datasets [runtime = {time.time()-t0:.2f}s]')
    for dset in ['Coordinates','Velocity','Mass','SubGroupNumber']:
        dset_snap=snapidx_eagledata.read_dataset(4,dset)
        if dset_snap.shape[-1]==3:
            particledata_snap_star.loc[:,[f'{dset}_x',f'{dset}_y',f'{dset}_z']]=dset_snap
        else:
            if dset=='Mass':
                particledata_snap_star[dset]=dset_snap
            else:
                particledata_snap_star[dset]=dset_snap

    logging.info(f'Done reading datasets - concatenating gas and star data [runtime = {time.time()-t0:.2f}s]')
    particledata_snap=particledata_snap.append(particledata_snap_star,ignore_index=True)

    logging.info(f'Sorting by IDs [runtime = {time.time()-t0:.2f}s]')
    particledata_snap.sort_values(by="ParticleIDs",inplace=True);particledata_snap.reset_index(inplace=True,drop=True)
    size1=np.sum(particledata_snap.memory_usage().values)/10**9
    
    logging.info(f'Particle data snap 1 memory usage: {size1:.2f} GB')

    #particle KD trees
    logging.info(f'Searching for existing KDTree [runtime = {time.time()-t0:.2f}s]')
    treefname1=f'catalogues/kdtrees/kdtree_snapidx_{snapidx}_n_{str(nvol).zfill(2)}_volume_{ivol}.dat'
    if os.path.exists(treefname1):
        logging.info(f'Loading existing KDTree for snap 1 [runtime = {time.time()-t0:.2f}s]')
        treefile1=open(treefname1,'rb')
        try:
            kdtree_snap1_periodic=pickle.load(treefile1)
            treefile1.close()
            gen1=False
        except:
            logging.info(f'Could not load snap 1 KD tree - generating [runtime = {time.time()-t0:.2f}s]')
            treefile1.close()
            gen1=True
            pass
    else:
        gen1=True

    if gen1:
        logging.info(f'Generating KDTree for snap 1 [runtime = {time.time()-t0:.2f}s]')
        kdtree_snap1_periodic= cKDTree(np.column_stack([particledata_snap[f'Coordinates_{x}'] for x in 'xyz']),boxsize=boxsize)
        treefile1=open(treefname1,'wb')
        pickle.dump(kdtree_snap1_periodic,treefile1)
        treefile1.close()

    
    #load catalogues into dataframes
    catalogue_subhalo=pd.read_hdf('catalogues/catalogue_subhalo.hdf5',key='Subhalo')
    catalogue_subhalo=catalogue_subhalo.loc[catalogue_subhalo['snapshotidx']==snapidx,:]

    #select relevant subhaloes
    snap_mask=catalogue_subhalo[f'snapshotidx']==snapidx
    snap_mass_mask=catalogue_subhalo[f'ApertureMeasurements/Mass/030kpc_4']>=10**mcut/10**10
    snap_com_mask_1=np.logical_and.reduce([catalogue_subhalo[f'CentreOfPotential_{x}']>=ixmin for x,ixmin in zip('xyz',[xmin,ymin,zmin])])
    snap_com_mask_2=np.logical_and.reduce([catalogue_subhalo[f'CentreOfPotential_{x}']<=ixmax for x,ixmax in zip('xyz',[xmax,ymax,zmax])])
    snap_com_mask=np.logical_and.reduce([snap_com_mask_1,snap_com_mask_2,snap_mask,snap_mass_mask])
    numgal_subvolume=np.sum(snap_com_mask);numgal_total=np.sum(np.logical_and(snap_mask,snap_mass_mask))
    logging.info(f'Using {numgal_subvolume} of {numgal_total} valid galaxies from box [runtime = {time.time()-t0:.2f}s]')

    #initialise output
    initfields=['nodeIndex','GroupNumber','SubGroupNumber']
    output_df=catalogue_subhalo.loc[snap_com_mask,initfields]
    output_df.loc[:,'BaryMP-radius']=np.nan

    for iigalaxy,(igalaxy,galaxy) in enumerate(catalogue_subhalo.loc[snap_com_mask,:].iterrows()):
        
        nodeidx=galaxy['nodeIndex']
        subgroupnumber=galaxy['SubGroupNumber']

        if subgroupnumber==0:
            icen=True
        else:
            icen=False

        com=[galaxy[f"CentreOfPotential_{x}"] for x in 'xyz']

        #select particles in halo-size sphere
        if icen:
            r200_eff=galaxy['Group_R_Crit200']
        else:
            r200_eff=r200(m200=galaxy['Mass']*10**10,rhocrit=rhocrit)
        
        part_idx_within_radius=kdtree_snap1_periodic.query_ball_point(com,r200_eff)
        npart_galaxy=len(part_idx_within_radius)

        part_IDs_within_radius=(particledata_snap.loc[part_idx_within_radius,"ParticleIDs"].values).astype(np.int64)
        part_idx_candidates=particledata_snap['ParticleIDs'].searchsorted(part_IDs_within_radius)
        part_data_candidates=particledata_snap.loc[part_idx_candidates,:]
        part_data_candidates.loc[:,"rrel_com"]=np.sqrt(np.sum(np.square(np.column_stack([part_data_candidates[f'Coordinates_{x}']-com[ix] for ix,x in enumerate('xyz')])),axis=1))/r200_eff #Mpc


        totbaryonmass=np.nansum(part_data_candidates.loc[:,"Mass"])
        
        #fit baryon mass profile
        r200_bins=np.linspace(0,1,200)
        r200_bins_mid=r200_bins[1:]
        masks=[np.logical_and(part_data_candidates["rrel_com"]>bin_lo,part_data_candidates["rrel_com"]<bin_hi) for bin_lo, bin_hi in zip(r200_bins[:-1],r200_bins[1:])]
        mass_binned=[np.nansum(part_data_candidates.loc[mask,"Mass"]) for mask in masks]
        mass_binned_cumulative=np.cumsum(mass_binned)/totbaryonmass

        barymp,nfit=BaryMP(r200_bins_mid,mass_binned_cumulative)
        print(f'barymp = {barymp:.3f}R200 = {barymp*r200_eff*1000:.1f} kpc (nfit = {nfit})')




        # """
        # Find the radius for a galaxy from the BaryMP method
        # x = r/r_200
        # y = cumulative baryonic mass profile
        # eps = epsilon, if data 
        # """
        # print(icen,f"{galaxy['Mass']*10**10:.1e}",f" | eff radius: {r200_eff*1000:.2f} kpc | host radius {r200_host*1000:.2f} kpc |")


def analyse_gasflow(path,mcut,snapidx,nvol,ivol,snapidx_delta=1):

    ivol=int(ivol)
    ivol=str(ivol).zfill(3)
    ix,iy,iz=ivol_idx(ivol,nvol=nvol)

    t0=time.time()
    logfile=f'logs/gasflow/gasflow_snapidx_{snapidx}_delta_{str(snapidx_delta).zfill(3)}_n_{str(nvol).zfill(2)}_volume_{ivol}.log'
    if os.path.exists(logfile):
        os.remove(logfile)
    logging.basicConfig(filename=logfile, level=logging.INFO)

    output_fname=f'catalogues/gasflow/gasflow_snapidx_{snapidx}_delta_{str(snapidx_delta).zfill(3)}_n_{str(nvol).zfill(2)}_volume_{str(ivol).zfill(3)}.hdf5'
    if os.path.exists(output_fname):
        os.remove(output_fname)

    #background data for calc
    redshift_table=pd.read_hdf('snapshot_redshifts.hdf5',key='snapshots')

    snapidx2=snapidx;snapidx2_tag=redshift_table.loc[redshift_table['snapshotidx']==snapidx2,'tag'].values[0]
    snapidx1=snapidx2-snapidx_delta;snapidx1_tag=redshift_table.loc[redshift_table['snapshotidx']==snapidx1,'tag'].values[0]

    snapidx1_particledatapath=f'{path}/particledata_{snapidx1_tag}/eagle_subfind_snip_particles_{snapidx1_tag[5:]}.0.hdf5'
    snapidx2_particledatapath=f'{path}/particledata_{snapidx2_tag}/eagle_subfind_snip_particles_{snapidx2_tag[5:]}.0.hdf5'

    cosmology=FlatLambdaCDM(H0=h5py.File(snapidx2_particledatapath,'r')['Header'].attrs['HubbleParam']*100,
                            Om0=h5py.File(snapidx2_particledatapath,'r')['Header'].attrs['Omega0'])

    snapidx1_z=h5py.File(snapidx1_particledatapath,'r')['Header'].attrs['Redshift'];snapidx1_lt=cosmology.lookback_time(snapidx1_z)
    snapidx2_z=h5py.File(snapidx2_particledatapath,'r')['Header'].attrs['Redshift'];snapidx2_lt=cosmology.lookback_time(snapidx2_z)
    delta_lt=snapidx1_lt-snapidx2_lt
    boxsize=h5py.File(snapidx2_particledatapath,'r')['Header'].attrs['BoxSize']
    nh_conversion=6.76991e-31/(1.6726219e-24)

    #read data
    snapidx1_eagledata = EagleSnapshot(snapidx1_particledatapath)
    snapidx2_eagledata = EagleSnapshot(snapidx2_particledatapath)

    subvol_edgelength=boxsize/nvol
    buffer=subvol_edgelength/10

    xmin=ix*subvol_edgelength;xmax=(ix+1)*subvol_edgelength
    ymin=iy*subvol_edgelength;ymax=(iy+1)*subvol_edgelength
    zmin=iz*subvol_edgelength;zmax=(iz+1)*subvol_edgelength


    logging.info(f'Considering region: (1/{nvol**3} of full box) [runtime = {time.time()-t0:.2f}s]')
    logging.info(f'ix: {ix} - x in [{xmin},{xmax}]')
    logging.info(f'iy: {iy} - y in [{ymin},{ymax}]')
    logging.info(f'iz: {iz} - z in [{zmin},{zmax}]')

    snapidx1_eagledata.select_region(xmin-buffer, xmax+buffer, ymin-buffer, ymax+buffer, zmin-buffer, zmax+buffer)
    snapidx2_eagledata.select_region(xmin-buffer, xmax+buffer, ymin-buffer, ymax+buffer, zmin-buffer, zmax+buffer)

    logging.info(f'Initialising particle data with IDs [runtime = {time.time()-t0:.2f}s]')
    particledata_snap1=pd.DataFrame(snapidx1_eagledata.read_dataset(0,'ParticleIDs'),columns=['ParticleIDs']);particledata_snap1.loc[:,"ParticleTypes"]=0
    particledata_snap2=pd.DataFrame(snapidx2_eagledata.read_dataset(0,'ParticleIDs'),columns=['ParticleIDs']);particledata_snap2.loc[:,"ParticleTypes"]=0
    particledata_snap1_star=pd.DataFrame(snapidx1_eagledata.read_dataset(4,'ParticleIDs'),columns=['ParticleIDs']);particledata_snap1_star.loc[:,"ParticleTypes"]=4;particledata_snap1_star.loc[:,"Temperature"]=-1.;particledata_snap1_star.loc[:,"Density"]=10**10
    particledata_snap2_star=pd.DataFrame(snapidx2_eagledata.read_dataset(4,'ParticleIDs'),columns=['ParticleIDs']);particledata_snap2_star.loc[:,"ParticleTypes"]=4;particledata_snap2_star.loc[:,"Temperature"]=-1.;particledata_snap2_star.loc[:,"Density"]=10**10

    logging.info(f'Reading gas datasets [runtime = {time.time()-t0:.2f}s]')
    for dset in ['Coordinates','Velocity','Mass','Density','Temperature','Metallicity']:
        dset_snap1=snapidx1_eagledata.read_dataset(0,dset)
        dset_snap2=snapidx2_eagledata.read_dataset(0,dset)
        if dset_snap2.shape[-1]==3:
                particledata_snap1.loc[:,[f'{dset}_x',f'{dset}_y',f'{dset}_z']]=dset_snap1
                particledata_snap2.loc[:,[f'{dset}_x',f'{dset}_y',f'{dset}_z']]=dset_snap2
        else:
            if dset=='Mass':
                particledata_snap1[dset]=dset_snap1
                particledata_snap2[dset]=dset_snap2
            else:
                particledata_snap1[dset]=dset_snap1
                particledata_snap2[dset]=dset_snap2

    logging.info(f'Reading star datasets [runtime = {time.time()-t0:.2f}s]')
    for dset in ['Coordinates','Velocity','Mass']:
        dset_snap1=snapidx1_eagledata.read_dataset(4,dset)
        dset_snap2=snapidx2_eagledata.read_dataset(4,dset)
        if dset_snap2.shape[-1]==3:
            particledata_snap1_star.loc[:,[f'{dset}_x',f'{dset}_y',f'{dset}_z']]=dset_snap1
            particledata_snap2_star.loc[:,[f'{dset}_x',f'{dset}_y',f'{dset}_z']]=dset_snap2
        else:
            if dset=='Mass':
                particledata_snap1_star[dset]=dset_snap1
                particledata_snap2_star[dset]=dset_snap2
            else:
                particledata_snap1_star[dset]=dset_snap1
                particledata_snap2_star[dset]=dset_snap2

    logging.info(f'Done reading datasets - concatenating gas and star data [runtime = {time.time()-t0:.2f}s]')
    particledata_snap1=particledata_snap1.append(particledata_snap1_star,ignore_index=True)
    particledata_snap2=particledata_snap2.append(particledata_snap2_star,ignore_index=True)

    logging.info(f'Sorting by IDs [runtime = {time.time()-t0:.2f}s]')
    particledata_snap1.sort_values(by="ParticleIDs",inplace=True);particledata_snap1.reset_index(inplace=True,drop=True)
    particledata_snap2.sort_values(by="ParticleIDs",inplace=True);particledata_snap2.reset_index(inplace=True,drop=True)
    size1=np.sum(particledata_snap1.memory_usage().values)/10**9;size2=np.sum(particledata_snap2.memory_usage().values)/10**9
    
    logging.info(f'Particle data snap 1 memory usage: {size1:.2f} GB')
    logging.info(f'Particle data snap 2 memory usage: {size2:.2f} GB')

    #particle KD trees
    logging.info(f'Searching for existing KDTrees [runtime = {time.time()-t0:.2f}s]')

    treefname1=f'catalogues/kdtrees/kdtree_snapidx_{snapidx1}_n_{str(nvol).zfill(2)}_volume_{ivol}.dat'
    treefname2=f'catalogues/kdtrees/kdtree_snapidx_{snapidx2}_n_{str(nvol).zfill(2)}_volume_{ivol}.dat'

    if os.path.exists(treefname1):
        logging.info(f'Loading existing KDTree for snap 1 [runtime = {time.time()-t0:.2f}s]')
        treefile1=open(treefname1,'rb')
        try:
            kdtree_snap1_periodic=pickle.load(treefile1)
            treefile1.close()
            gen1=False
        except:
            logging.info(f'Could not load snap 1 KD tree - generating [runtime = {time.time()-t0:.2f}s]')
            treefile1.close()
            gen1=True
            pass
    else:
        gen1=True

    if os.path.exists(treefname2):
        logging.info(f'Loading existing KDTree for snap 2 [runtime = {time.time()-t0:.2f}s]')
        treefile2=open(treefname2,'rb')
        try:
            kdtree_snap2_periodic=pickle.load(treefile2)
            treefile2.close()
            gen2=False
        except:
            logging.info(f'Could not load snap 2 KD tree - generating [runtime = {time.time()-t0:.2f}s]')
            treefile2.close()
            gen2=True
            pass
    else:
        gen2=True
    
    if gen1:
        logging.info(f'Generating KDTree for snap 1 [runtime = {time.time()-t0:.2f}s]')
        kdtree_snap1_periodic= cKDTree(np.column_stack([particledata_snap1[f'Coordinates_{x}'] for x in 'xyz']),boxsize=boxsize)
        treefile1=open(treefname1,'wb')
        pickle.dump(kdtree_snap1_periodic,treefile1)
        treefile1.close()
    if gen2:
        logging.info(f'Generating KDTree for snap 2 [runtime = {time.time()-t0:.2f}s]')
        kdtree_snap2_periodic= cKDTree(np.column_stack([particledata_snap2[f'Coordinates_{x}'] for x in 'xyz']),boxsize=boxsize)
        treefile2=open(treefname2,'wb')
        pickle.dump(kdtree_snap2_periodic,treefile2)
        treefile2.close()

    
    #load catalogues into dataframes
    catalogue_subhalo=pd.read_hdf('catalogues/catalogue_subhalo.hdf5',key='Subhalo')
    catalogue_subhalo=catalogue_subhalo.loc[np.logical_and(catalogue_subhalo['snapshotidx']<=snapidx2,catalogue_subhalo['snapshotidx']>=snapidx1),:]

    #select relevant subhaloes
    snap2_mask=catalogue_subhalo[f'snapshotidx']==snapidx2
    snap2_mass_mask=catalogue_subhalo[f'ApertureMeasurements/Mass/030kpc_4']>=10**mcut/10**10
    snap2_com_mask_1=np.logical_and.reduce([catalogue_subhalo[f'CentreOfPotential_{x}']>=ixmin for x,ixmin in zip('xyz',[xmin,ymin,zmin])])
    snap2_com_mask_2=np.logical_and.reduce([catalogue_subhalo[f'CentreOfPotential_{x}']<=ixmax for x,ixmax in zip('xyz',[xmax,ymax,zmax])])
    snap2_com_mask=np.logical_and.reduce([snap2_com_mask_1,snap2_com_mask_2,snap2_mask,snap2_mass_mask])
    numgal_subvolume=np.sum(snap2_com_mask);numgal_total=np.sum(np.logical_and(snap2_mask,snap2_mass_mask))
    logging.info(f'Using {numgal_subvolume} of {numgal_total} valid galaxies from box [runtime = {time.time()-t0:.2f}s]')

    #initialise output
    initfields=['nodeIndex','GroupNumber','SubGroupNumber']
    gasflow_df=catalogue_subhalo.loc[snap2_com_mask,initfields]

    gasflow_df.loc[:,'Inflow-ISM_HMS']=np.nan
    gasflow_df.loc[:,'Inflow-ISM_30kpc']=np.nan
    gasflow_df.loc[:,'Outflow-ISM_HMS']=np.nan
    gasflow_df.loc[:,'Outflow-ISM_30kpc']=np.nan

    r200_facs=[0.125,0.25,0.5,0.75,1]
    for fac in r200_facs:
        gasflow_df.loc[:,f'Inflow-{fac:.3f}R200']=np.nan
        gasflow_df.loc[:,f'Outflow-{fac:.3f}R200']=np.nan

    success=[]
    #Main halo loop
    for iigalaxy,(igalaxy_snap2,galaxy_snap2) in enumerate(catalogue_subhalo.loc[snap2_com_mask,:].iterrows()):
        
        nodeidx=galaxy_snap2['nodeIndex']
        subgroupnumber=galaxy_snap2['SubGroupNumber']
        progidx=find_progidx(catalogue_subhalo,nodeidx=nodeidx,snapidx_delta=snapidx_delta)
        print(progidx)

        if subgroupnumber==0:
            icen=True
        else:
            icen=False

        #ensuring there has been a progenitor found
        if np.sum(progidx==catalogue_subhalo['nodeIndex']):
            pass           
        else:
            logging.info(f'Skipping galaxy {iigalaxy+1} of {numgal_subvolume} - could not find progenitor')
            success.append(0)
            continue

        galaxy_snap1=catalogue_subhalo.loc[progidx==catalogue_subhalo['nodeIndex'],:]
        com_snap2=[galaxy_snap2[f"CentreOfPotential_{x}"] for x in 'xyz']
        com_snap1=[galaxy_snap1[f"CentreOfPotential_{x}"].values[0] for x in 'xyz']

        vcom_snap2=[galaxy_snap2[f"Velocity_{x}"] for x in 'xyz']
        vcom_snap1=[galaxy_snap1[f"Velocity_{x}"].values[0] for x in 'xyz']

        #select particles in halo-size sphere
        hostradius=(np.float(galaxy_snap2['Group_R_Crit200'])+np.float(galaxy_snap1['Group_R_Crit200']))/2
        hmsradius=(np.float(galaxy_snap2['HalfMassRad_4'])+np.float(galaxy_snap1['HalfMassRad_4']))/2
        if icen:
            candidate_radius=hostradius
        else:
            candidate_radius=8*hmsradius

        part_idx_candidates_snap2=kdtree_snap2_periodic.query_ball_point(com_snap2,candidate_radius)
        part_idx_candidates_snap1=kdtree_snap1_periodic.query_ball_point(com_snap1,candidate_radius)
        part_IDs_candidates_all=np.unique(np.concatenate([particledata_snap2.loc[part_idx_candidates_snap2,"ParticleIDs"].values,particledata_snap1.loc[part_idx_candidates_snap1,"ParticleIDs"].values])).astype(np.int64)

        part_idx_candidates_snap1=particledata_snap1['ParticleIDs'].searchsorted(part_IDs_candidates_all)
        part_idx_candidates_snap2=particledata_snap2['ParticleIDs'].searchsorted(part_IDs_candidates_all)
        part_data_candidates_snap1=particledata_snap1.loc[part_idx_candidates_snap1,:]
        part_data_candidates_snap2=particledata_snap2.loc[part_idx_candidates_snap2,:]
        
        #needed if using subfind particle data
        if True:
            matches=part_data_candidates_snap2.loc[:,"ParticleIDs"].values==part_data_candidates_snap1.loc[:,"ParticleIDs"].values
            matchrate=np.sum(matches)/len(matches)
            if matchrate<0.9:
                logging.info(f'Skipping galaxy {iigalaxy+1} of {numgal_subvolume} - poorly matched ({matchrate*100:.1f}%)')
                logging.info(f'')
                success.append(0)
                continue
            part_data_candidates_snap2=part_data_candidates_snap2.loc[matches,:]
            part_data_candidates_snap1=part_data_candidates_snap1.loc[matches,:]

        #adding rcom and vrad
        part_data_candidates_snap2.loc[:,"r_com"]=np.sqrt(np.sum(np.square(np.column_stack([part_data_candidates_snap2[f'Coordinates_{x}']-com_snap2[ix] for ix,x in enumerate('xyz')])),axis=1))#Mpc
        # part_data_candidates_snap2.loc[:,[f"runit_{x}rel" for x in 'xyz']]=np.column_stack([(part_data_candidates_snap2[f'Coordinates_{x}']-com_snap2[ix])/part_data_candidates_snap2[f'r_com'] for ix,x in enumerate('xyz')])
        # part_data_candidates_snap2.loc[:,[f"Velocity_{x}rel" for x in 'xyz']]=np.column_stack([part_data_candidates_snap2[f'Velocity_{x}']-vcom_snap2[ix] for ix,x in enumerate('xyz')])
        # part_data_candidates_snap2["vrad_inst"]=np.sum(np.multiply(np.column_stack([part_data_candidates_snap2[f"Velocity_{x}rel"] for x in 'xyz']),np.column_stack([part_data_candidates_snap2[f"runit_{x}rel"] for x in 'xyz'])),axis=1)
        part_data_candidates_snap1.loc[:,"r_com"]=np.sqrt(np.sum(np.square(np.column_stack([part_data_candidates_snap1[f'Coordinates_{x}']-com_snap1[ix] for ix,x in enumerate('xyz')])),axis=1))#Mpc
        # part_data_candidates_snap1.loc[:,[f"runit_{x}rel" for x in 'xyz']]=np.column_stack([(part_data_candidates_snap1[f'Coordinates_{x}']-com_snap1[ix])/part_data_candidates_snap1[f'r_com'] for ix,x in enumerate('xyz')])
        # part_data_candidates_snap1.loc[:,[f"Velocity_{x}rel" for x in 'xyz']]=np.column_stack([part_data_candidates_snap1[f'Velocity_{x}']-vcom_snap1[ix] for ix,x in enumerate('xyz')])
        # part_data_candidates_snap1["vrad_inst"]=np.sum(np.multiply(np.column_stack([part_data_candidates_snap1[f"Velocity_{x}rel"] for x in 'xyz']),np.column_stack([part_data_candidates_snap1[f"runit_{x}rel"] for x in 'xyz'])),axis=1)
        
        # part_data_candidates_snap2["vrad_ave"]=(part_data_candidates_snap2["r_com"].values-part_data_candidates_snap1["r_com"].values)/delta_lt*978.5#to km/s
        # part_data_candidates_snap1["vrad_ave"]=(part_data_candidates_snap2["r_com"].values-part_data_candidates_snap1["r_com"].values)/delta_lt*978.5#to km/s

        #removing temporary data
        # for dset in np.concatenate([[f"runit_{x}rel" for x in 'xyz'],[f"Velocity_{x}rel" for x in 'xyz'],[f'Coordinates_{x}' for x in 'xyz'],[f'Velocity_{x}' for x in 'xyz']]):
        #     del part_data_candidates_snap1[dset]
        #     del part_data_candidates_snap2[dset]

        gas_snap1=part_data_candidates_snap1["ParticleTypes"].values==0

        #sfr criterion
        # part_data_candidates_snap1["starforming-ism-hms"]=np.logical_and.reduce([part_data_candidates_snap1["Density"].values*nh_conversion>=0.1*(part_data_candidates_snap1["Metallicity"].values)**(-0.64),
        #                                                                      part_data_candidates_snap1["Temperature"].values<=tfloor(part_data_candidates_snap1["Density"].values*nh_conversion)*10**0.5,
        #                                                                      part_data_candidates_snap1["r_com"].values<=4*hmsradius]).astype(int)

        # part_data_candidates_snap2["starforming-ism-hms"]=np.logical_and.reduce([part_data_candidates_snap2["Density"].values*nh_conversion>=0.1*(part_data_candidates_snap2["Metallicity"].values)**(-0.64),
        #                                                                      part_data_candidates_snap2["Temperature"].values<=tfloor(part_data_candidates_snap2["Density"].values*nh_conversion)*10**0.5,
        #                                                                      part_data_candidates_snap2["r_com"].values<=4*hmsradius]).astype(int)

        # part_data_candidates_snap1["starforming-ism-30kpc"]=np.logical_and.reduce([part_data_candidates_snap1["Density"].values*nh_conversion>=0.1*(part_data_candidates_snap1["Metallicity"].values)**(-0.64),
        #                                                                      part_data_candidates_snap1["Temperature"].values<=tfloor(part_data_candidates_snap1["Density"].values*nh_conversion)*10**0.5,
        #                                                                      part_data_candidates_snap1["r_com"].values<=0.03]).astype(int)

        # part_data_candidates_snap2["starforming-ism-30kpc"]=np.logical_and.reduce([part_data_candidates_snap2["Density"].values*nh_conversion>=0.1*(part_data_candidates_snap2["Metallicity"].values)**(-0.64),
        #                                                                      part_data_candidates_snap2["Temperature"].values<=tfloor(part_data_candidates_snap2["Density"].values*nh_conversion)*10**0.5,
        #                                                                      part_data_candidates_snap2["r_com"].values<=0.03]).astype(int)
 


        #atomic phase
        part_data_candidates_snap1["atomic-ism-hms"]=np.logical_and.reduce([part_data_candidates_snap1["Density"].values*nh_conversion>=0.01,
                                                                             part_data_candidates_snap1["Temperature"].values<=tfloor(part_data_candidates_snap1["Density"].values*nh_conversion)*10**0.5,
                                                                             part_data_candidates_snap1["r_com"].values<=6*hmsradius]).astype(int)

        part_data_candidates_snap2["atomic-ism-hms"]=np.logical_and.reduce([part_data_candidates_snap2["Density"].values*nh_conversion>=0.01,
                                                                             part_data_candidates_snap2["Temperature"].values<=tfloor(part_data_candidates_snap2["Density"].values*nh_conversion)*10**0.5,
                                                                             part_data_candidates_snap2["r_com"].values<=6*hmsradius]).astype(int)
        
        part_data_candidates_snap1["atomic-ism-30kpc"]=np.logical_and.reduce([part_data_candidates_snap1["Density"].values*nh_conversion>=0.01,
                                                                             part_data_candidates_snap1["Temperature"].values<=tfloor(part_data_candidates_snap1["Density"].values*nh_conversion)*10**0.5,
                                                                             part_data_candidates_snap1["r_com"].values<=0.03]).astype(int)

        part_data_candidates_snap2["atomic-ism-30kpc"]=np.logical_and.reduce([part_data_candidates_snap2["Density"].values*nh_conversion>=0.01,
                                                                             part_data_candidates_snap2["Temperature"].values<=tfloor(part_data_candidates_snap2["Density"].values*nh_conversion)*10**0.5,
                                                                             part_data_candidates_snap2["r_com"].values<=0.03]).astype(int)


    
        #ism def - star forming at all within 8 hms or atomic within 4 hms
        ism_hms_snap1=np.logical_or.reduce([part_data_candidates_snap1["atomic-ism-hms"].values])
        ism_hms_snap2=np.logical_or.reduce([part_data_candidates_snap2["atomic-ism-hms"].values])
        
        ism_30kpc_snap1=np.logical_or.reduce([part_data_candidates_snap1["atomic-ism-30kpc"].values])
        ism_30kpc_snap2=np.logical_or.reduce([part_data_candidates_snap2["atomic-ism-30kpc"].values])

        #new ism particles
        ism_partidx_in_hms=np.logical_and.reduce([np.logical_not(ism_hms_snap1),ism_hms_snap2,gas_snap1])
        ism_partidx_in_30kpc=np.logical_and.reduce([np.logical_not(ism_30kpc_snap1),ism_30kpc_snap2,gas_snap1])

        #removed ism particles
        ism_partidx_out_hms=np.logical_and.reduce([ism_hms_snap1,np.logical_not(ism_hms_snap2),gas_snap1])
        ism_partidx_out_30kpc=np.logical_and.reduce([ism_30kpc_snap1,np.logical_not(ism_30kpc_snap2),gas_snap1])

        gasflow_df.loc[igalaxy_snap2,'Inflow-ISM_HMS']=np.sum(part_data_candidates_snap2.loc[ism_partidx_in_hms,'Mass'])
        gasflow_df.loc[igalaxy_snap2,'Outflow-ISM_HMS']=np.sum(part_data_candidates_snap2.loc[ism_partidx_out_hms,'Mass'])
        gasflow_df.loc[igalaxy_snap2,'Inflow-ISM_30kpc']=np.sum(part_data_candidates_snap2.loc[ism_partidx_in_30kpc,'Mass'])
        gasflow_df.loc[igalaxy_snap2,'Outflow-ISM_30kpc']=np.sum(part_data_candidates_snap2.loc[ism_partidx_out_30kpc,'Mass'])

        #halo def (if central)
        if icen:
            for fac in r200_facs:
                halo_snap1=np.logical_and.reduce([part_data_candidates_snap1["r_com"].values<fac*hostradius])
                halo_snap2=np.logical_and.reduce([part_data_candidates_snap2["r_com"].values<fac*hostradius])
                
                #new halo particles
                halo_partidx_in=np.logical_and(np.logical_not(halo_snap1),halo_snap2)
                #removed halo particles
                halo_partidx_out=np.logical_and(halo_snap1,np.logical_not(halo_snap2)) 
                
                #sum masses
                gasflow_df.loc[igalaxy_snap2,f'Inflow-{fac:.3f}R200']=np.sum(part_data_candidates_snap2.loc[halo_partidx_in,'Mass'])
                gasflow_df.loc[igalaxy_snap2,f'Outflow-{fac:.3f}R200']=np.sum(part_data_candidates_snap2.loc[halo_partidx_out,'Mass'])

        if icen:
            logging.info(f'Done with galaxy {iigalaxy+1} of {numgal_subvolume} for this subvolume - CENTRAL [runtime = {time.time()-t0:.2f}s]')
        else:
            logging.info(f'Done with galaxy {iigalaxy+1} of {numgal_subvolume} for this subvolume - SATELLITE [runtime = {time.time()-t0:.2f}s]')

        logging.info(f'')
        success.append(1)

    logging.info(f'{np.sum(success):.0f} of {len(success):.0f} galaxies were successfully processed ({np.nanmean(success)*100:.1f}%) [runtime = {time.time()-t0:.2f}s]')


    gasflow_df.to_hdf(output_fname,key='Flux')
    print(gasflow_df)

def combine_catalogues(mcut,snapidxs,nvol,snapidx_deltas=[1]):
    outname=f'catalogues/catalogue_gasflow_nvol_{str(nvol).zfill(2)}_mcut_{str(mcut).zfill(2)}.hdf5'
    catalogue_subhalo=pd.read_hdf('catalogues/catalogue_subhalo.hdf5',key='Subhalo')
    catalogue_subhalo=catalogue_subhalo.loc[np.logical_and(np.logical_or.reduce([catalogue_subhalo['snapshotidx']==snapidx for snapidx in snapidxs]),catalogue_subhalo['ApertureMeasurements/Mass/030kpc_4']>=10**mcut/10**10),:]
    catalogue_subhalo.reset_index()

    accretion_fields=['Inflow-ISM_HMS',
                      'Inflow-ISM_30kpc', 'Outflow-ISM_HMS', 'Outflow-ISM_30kpc',
                      'Inflow-0.125R200', 'Outflow-0.125R200', 'Inflow-0.250R200',
                      'Outflow-0.250R200', 'Inflow-0.500R200', 'Outflow-0.500R200',
                      'Inflow-0.750R200', 'Outflow-0.750R200', 'Inflow-1.000R200',
                      'Outflow-1.000R200']
    
    accfile_data_vols=[]

    ifile=0
    isnap=0
    for snapidx in snapidxs:
        ivol=0
        for ivol in range(nvol**3):
            idelta=0
            for delta in snapidx_deltas:
                accretion_fields_idelta=[accretion_field+f'-delta_{str(delta).zfill(2)}' for accretion_field in accretion_fields]
                fields_idelta=np.concatenate([["nodeIndex"],accretion_fields_idelta])
                fields_orig=np.concatenate([["nodeIndex"],accretion_fields])
                if idelta==0:
                    print(f'Loading volume {ivol+1}/{nvol**3} for snap {snapidx} (delta {delta})')
                try:
                    accfile_data_file=pd.read_hdf(f'catalogues/gasflow/gasflow_snapidx_{snapidx}_delta_{str(delta).zfill(3)}_n_{str(nvol).zfill(2)}_volume_{str(ivol).zfill(3)}.hdf5',key='Flux')
                    # print(accfile_data_file.shape[0],' subhaloes in subvolume')
                except:
                    print(f'Could not load volume {ivol}')
                    continue
                
                accfile_data_new=pd.DataFrame(accfile_data_file.loc[:,fields_orig].values,columns=fields_idelta)

                if idelta==0:
                    accfile_data_isnap_ivol=accfile_data_new
                else:
                    accfile_data_isnap_ivol.loc[:,accretion_fields_idelta]=accfile_data_new.loc[:,accretion_fields_idelta].values

                idelta+=1
            accfile_data_vols.append(accfile_data_isnap_ivol)
            ivol+=1
        isnap+=1

    accfile_data=pd.concat(accfile_data_vols,ignore_index=True)
    print(accfile_data)

    ngal=accfile_data.shape[0]
    iigal=0
    for igal, gal in accfile_data.iterrows():
        if iigal%1000==0:
            print(f'{iigal/ngal*100:.1f}% done with matching ...')
        nodeidx=gal['nodeIndex']
        match=nodeidx==catalogue_subhalo['nodeIndex']
        accfields=list(accfile_data)
        catalogue_subhalo.loc[match,accfields]=gal.values
        iigal+=1

    if os.path.exists(outname):
        os.remove(outname)
    
    catalogue_subhalo.to_hdf(outname,key='Subhalo')





#lower level

def ivol_gen(ix,iy,iz,nvol):
    ivol=ix*nvol**2+iy*nvol+iz
    ivol_str=str(ivol).zfill(3)
    return ivol_str

def ivol_idx(ivol,nvol):
    if type(ivol)==str:
        ivol=int(ivol)
    ix=int(np.floor(ivol/nvol**2))
    iz=int(ivol%nvol)
    iy=int((ivol-ix*nvol**2-iz)/nvol)
    return (ix,iy,iz)

def tfloor_eagle(nh,norm=17235.4775202):
    T=np.zeros(np.shape(nh))+8000
    dense=np.where(nh>=10**-1)
    T[dense]=norm*nh[dense]**(1/3)

    return T

def r200(m200,rhocrit):
    r200_cubed=3*m200/(800*np.pi*rhocrit)
    return r200_cubed**(1/3)
    
def find_progidx(catalogue_subhalo,nodeidx,snapidx_delta):
    nodeidx_depth=nodeidx
    nodeidx_depths=[nodeidx]
    for idepth in range(snapidx_delta):
        matchingnode=nodeidx_depth==catalogue_subhalo['nodeIndex'].values
        if np.sum(matchingnode)==1:
            nodeidx_depth=catalogue_subhalo.loc[matchingnode,'mainProgenitorIndex'].values[0]
        else:
            nodeidx_depth=None
            break
        
        nodeidx_depths.append(nodeidx_depth)
    
    if len(nodeidx_depths)==snapidx_delta+1:
        return nodeidx_depths[-1]
    else:
        return None

def BaryMP(x,y,eps=0.01,grad=1):
	"""
	Find the radius for a galaxy from the BaryMP method
	x = r/r_200
	y = cumulative baryonic mass profile
	eps = epsilon, if data 
	"""
	dydx = np.diff(y)/np.diff(x)
	
	maxarg = np.argwhere(dydx==np.max(dydx))[0][0] # Find where the gradient peaks
	xind = np.argwhere(dydx[maxarg:]<=grad)[0][0] + maxarg # The index where the gradient reaches 1
	
	x2fit_new, y2fit_new = x[xind:], y[xind:] # Should read as, e.g., "x to fit".
	x2fit, y2fit = np.array([]), np.array([]) # Gets the while-loop going
	
	while len(y2fit)!=len(y2fit_new):
		x2fit, y2fit = np.array(x2fit_new), np.array(y2fit_new)
		p = np.polyfit(x2fit, y2fit, 1)
		yfit = p[0]*x2fit + p[1]
		chi = abs(yfit-y2fit) # Separation in the y-direction for the fit from the data
		chif = (chi<eps) # Filter for what chi-values are acceptable
		x2fit_new, y2fit_new = x2fit[chif], y2fit[chif]
	
	r_bmp = x2fit[0] # Radius from the baryonic-mass-profile technique, returned as a fraction of the virial radius!
	Nfit = len(x2fit) # Number of points on the profile fitted to in the end

	return r_bmp, Nfit





# orbweaver
def ReadOrbitData(filenamelist,iFileno=False,apsispoints=True,crossingpoints=True,endpoints=True,desiredfields=[]):

	"""
	Function to read in the data from the .orbweaver.orbitdata.hdf files
	Parameters
	----------
	filenamelist : str
		The file containing the basenames for the orbweaver catalogue (can use the file generated from the creation of the (preprocessed) orbit catalogue)
	iFileno : bool, optional
		If True, the file number where the each of the entries came from is to be outputted.
	apsispoints : bool, optional
		If False, a boolean selection is done on the data to be loaded so the apsispoints are not loaded in. It is done so it also reduces the memory used. But this means the reading takes significatly longer.
	crossingpoints : bool, optional
		If False, a boolean selection is done on the data to be loaded so the crossingpoints are not loaded in. It is done so it also reduces the memory used. But this means the reading takes significatly longer.
	endpoints : bool, optional
		If False, a boolean selection is done on the data to be loaded so the endpoints are not loaded in. It is done so it also reduces the memory used. But this means the reading takes significatly longer.
	desiredfields : list, optional
		A list containing the desired fields to put returned, please see the FieldsDescriptions.md for the list of fields availible. If not supplied then all fields are returned
	Returns
	-------
	orbitdata : dict
		Dictionary of the fields to be outputted, where each field is a ndarray .
	"""
	start =  time.time()

	#See if any of the desired datasets are false
	createselection=False
	if((apsispoints==False) | (crossingpoints==False) | (endpoints==False)):
		createselection  = True

	#First see if the file exits
	if(os.path.isfile(filenamelist)==False):
		raise IOError("The filelist",filenamelist,"does not exist")

	filelist = open(filenamelist,"r")

	#Try and read the first line as int
	try:
		numfiles = int(filelist.readline())
	except ValueError:
		raise IOError("The first line of the filelist (which says the number of files), cannot be interpreted as a integer")

	if(iFileno): fileno = np.zeros(numfiles,dtype=np.int32)
	numentries = np.zeros(numfiles,dtype=np.uint64)
	maxorbitIDs = np.zeros(numfiles,dtype=np.uint64)
	prevmaxorbitID = np.uint64(0)
	filenames = [""]*numfiles
	orbitdatatypes = {}
	orbitdatakeys = None
	if(createselection): selArray = [[] for i in range(numfiles)]

	#Loop through the filelist and read the header of each file
	for i in range(numfiles):

		#Extract the filename
		filename = filelist.readline().strip("\n")
		filename+=".orbweaver.orbitdata.hdf"

		if(os.path.isfile(filename)==False):
			raise IOError("The file",filename,"does not exits")

		#Open up the file
		hdffile = h5py.File(filename,"r")

		#Read the header information
		if(iFileno): fileno[i] = np.int32(hdffile.attrs["Fileno"][...])
		numentries[i] =  np.uint64(hdffile.attrs["Number_of_entries"][...])
		maxorbitIDs[i] = prevmaxorbitID
		prevmaxorbitID += np.uint64(hdffile.attrs["Max_orbitID"][...])

		#Use the entrytype dataset to find points to be extracted
		if(createselection):

			#Load in the entrytype dataset to create the selection
			ientrytype = np.asarray(hdffile["entrytype"],dtype=np.float64)

			#Create an array the size of the number of entries to load in the data
			sel = np.zeros(numentries[i],dtype=bool)

			#If want apsis points
			if(apsispoints):
				sel = (np.round(np.abs(ientrytype),1) == 99.0)

			#If crossing points are also desired
			if(crossingpoints):
				sel = ((np.round(np.abs(ientrytype),1) != 99.0) & (np.round(np.abs(ientrytype),1) != 0.0)) | sel

			#The final endpoint for the orbiting halo
			if(endpoints):
				sel = (np.round(np.abs(ientrytype),1) == 0.0) | sel

			selArray[i] = sel

			#Update the number of entries based on the selection
			numentries[i] = np.sum(sel,dtype = np.uint64)

		#If the first file then file then find the dataset names and their datatypes
		if(i==0):
			if(len(desiredfields)>0):
				orbitdatakeys = desiredfields
			else:
				orbitdatakeys = list(hdffile.keys())

			for key in orbitdatakeys:
				orbitdatatypes[key] = hdffile[key].dtype

		hdffile.close()

		#Add this filename to the filename list
		filenames[i] = filename

	#Now can initilize the array to contain the data
	totnumentries = np.sum(numentries)
	orbitdata = {key:np.zeros(totnumentries,dtype = orbitdatatypes[key]) for key in orbitdatakeys}

	if(iFileno):
		#Add a field to contain the file number
		orbitdata["Fileno"] = np.zeros(totnumentries,dtype=np.int32)

	ioffset = np.uint64(0)

	#Now read in the data
	for i in range(numfiles):

		filename=filenames[i]

		print("Reading orbitdata from",filename)

		#Open up the file
		hdffile = h5py.File(filename,"r")

		#Get the start and end index
		startindex = ioffset
		endindex = np.uint64(ioffset+numentries[i])

		#Read the datasets
		for key in orbitdatakeys:
			if(createselection):
				orbitdata[key][startindex:endindex] = np.asarray(hdffile[key][selArray[i]],dtype=orbitdatatypes[key])
			else:
				orbitdata[key][startindex:endindex] = np.asarray(hdffile[key],dtype=orbitdatatypes[key])

		if("OrbitID" in orbitdatakeys):
			#Lets offset the orbitID to make it unique across all the data
			orbitdata["OrbitID"][startindex:endindex]+= np.uint64(maxorbitIDs[i] + i)

		if(iFileno):
			#Set the fileno that this came from
			orbitdata["Fileno"][startindex:endindex]=fileno[i]

		hdffile.close()

		ioffset+=numentries[i]

	print("Done reading in",time.time()-start)

	return orbitdata
