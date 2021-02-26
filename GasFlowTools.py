import os
import time
import logging
import h5py
import numpy as np
import pandas as pd

def extract_tree(path,mcut,snapidxmin=0):

    outname='catalogues/catalogue_tree.hdf5'
    fields=['snapshotNumber',
            'nodeIndex',
            'fofIndex',
            'hostIndex',
            'enclosingIndex',
            'isFoFCentre',
            'descendantIndex',
            'mainProgenitorIndex',
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


def extract_subhalo(path,mcut,snapidxmin=0):
    outname='catalogues/catalogue_subhalo.hdf5'
    fields=['/Subhalo/GroupNumber',
            '/Subhalo/SubGroupNumber',
            '/Subhalo/Mass',
            '/Subhalo/MassType',
            '/Subhalo/ApertureMeasurements/Mass/030kpc',
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
            logging.info(f'Processing snap {snapidx} ({isnap+1}/{len(groupdir)} total) [runtime {time.time()-t0:.2f} sec]')
            groupdirfnames=os.listdir(groupdir)
            groupdirfnames=sorted([groupdir+'/'+groupdirfname for groupdirfname in groupdirfnames if groupdirfname.startswith('eagle_subfind')])
            groupdirfnames_n=len(groupdirfnames)

            for ifile_snap, groupdirfname in enumerate(groupdirfnames):
                groupdirifile=h5py.File(groupdirfname,'r')

                ifile_submasses=groupdirifile['/Subhalo/Mass'].value
                ifile_mask=ifile_submasses>mcut
                ifile_nfof=np.sum(ifile_mask)
    
                logging.info(f'Snap {snapidx} ({isnap+1}/{len(groupdir)} total), file {ifile_snap+1}/{groupdirfnames_n}: extracting data for {ifile_nfof:.0f} subhaloes [runtime {time.time()-t0:.2f} sec]')
                
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
        
    if os.path.exists(f'{outname}'):
        os.remove(f'{outname}')

    data.to_hdf(f'{outname}',key='Subhalo')




def match_subs(mcut,snapidxmin=0):

    outname='catalogues/catalogue_subhalo.hdf5'
    catalogue_subhalo=pd.read_hdf('catalogues/catalogue_subhalo.hdf5',key='Subhalo',mode='r')
    catalogue_tree=pd.read_hdf('catalogues/catalogue_tree.hdf5',key='Tree',mode='r')

    fields_tree=['snapshotNumber',
                 'nodeIndex',
                 'fofIndex',
                 'hostIndex',
                 'enclosingIndex',
                 'isFoFCentre',
                 'descendantIndex',
                 'mainProgenitorIndex',
                 'position',
                 'positinInCatalogue']

    snaps_subhalo=catalogue_subhalo['snapshotidx'].unique()
    snaps_tomatch=snaps_subhalo[np.where(snaps_subhalo>=snapidxmin)]
    print(snaps_tomatch)

    
    



    
    
#