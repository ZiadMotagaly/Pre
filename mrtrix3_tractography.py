#!/bin/env python

import nipype.interfaces.mrtrix3 as mrt
import nipype.pipeline.engine as pe
import nipype.interfaces.fsl as fsl
import os
import nipype.interfaces.utility as util
from nipype import DataGrabber, DataSink, IdentityInterface, MapNode, JoinNode
from typing import List
import numpy 
from nilearn import plotting
import nibabel as nib
from nipype.interfaces.utility.wrappers import Function
import nipype.interfaces.ants as ants
from nipype.interfaces.ants.base import ANTSCommand
from nipype.interfaces import fsl
import configparser
import sys
import nipype.interfaces.fsl.utils as fslu
from nipype.interfaces.fsl.utils import Merge
from nipype import config, logging


config.update_config({'logging': {'log_directory': os.path.join(os.getcwd(), 'logs'),
                                  'workflow_level': 'DEBUG',
                                  'interface_level': 'DEBUG',
                                  'log_to_file': True
                                  },
                      'execution': {'stop_on_first_crash': True},
                    })



# Define the paths to the input and output files
 
out_dir = '/data/parietal/store/work/zmohamed/mathfun/output'
PATH = '/data/parietal/store/work/zmohamed/mathfun/'
seed_num = 50000




config = configparser.ConfigParser()
config.read(sys.argv[1])


subject_list = config['DEFAULT']['id_list'].split(" ")
visits = list(config['DEFAULT']['visits'])
subjects_dir = config['DEFAULT']['subjects_dir']
sessions = list(config['DEFAULT']['sessions'])

infosource = pe.Node(IdentityInterface(fields=['subject_id', 'visit','session']),
                      name='subjects')
infosource.iterables = [('subject_id', subject_list), ('visit', visits), ('session', sessions)]

subject_id_visit = pe.Node(
        interface=Function(
            input_names=['subject_id', 'visit'], output_names=['composite_id'],
            function=lambda subject_id, visit: '{}_{}'.format(subject_id, visit)
        ),
        name='subject_id_visit'
)  

data_source = pe.Node(DataGrabber(infields=[],
                                   outfields=['dwi', 'bval', 'bvec', 'mask', 'roi','template', 'T1', 'T1_brain', 'parc']),
                       name='input_node')

data_source.inputs.sort_filelist = True
data_source.inputs.base_directory = config['DEFAULT']['base_directory']
data_source.inputs.template = ''


data_source.inputs.field_template = {
        'T1': '%s/visit%s/session%s/anat/T1w.nii',
        'dwi': '%s/visit%s/session%s/dwi/dwi_raw.nii.gz',
        'bval': '%s/visit%s/session%s/dwi/dti_raw.bvals',
        'bvec': '%s/visit%s/session%s/dwi/dti_raw.bvecs',
        
    }
data_source.inputs.template_args = {
        template: [['subject_id', 'visit','session']]
        for template in data_source.inputs.field_template.keys()
    }


roi_source = pe.Node(DataGrabber(infields=[]),
                           name='rois')
roi_source.inputs.sort_filelist = True
roi_source.inputs.base_directory = config['ROIS']['directory']
roi_source.inputs.template = 'combined_BN*bin*.nii.gz'


template_source = pe.Node(DataGrabber(infields=[], outfields=['T1', 'T1_brain', 'T1_mask', 'T2', 'T2_brain', 'T2_mask']),
                           name='mni_template')
template_source.inputs.sort_filelist = True
template_source.inputs.base_directory = config['TEMPLATE']['directory']
template_source.inputs.template = ''
template_source.inputs.field_template = {
        'T1': config['TEMPLATE']['T1'],
        'T1_brain': config['TEMPLATE']['T1_brain'],
        'T1_mask': config['TEMPLATE']['T1_mask'],
        'T2': config['TEMPLATE']['T2'],
        'T2_brain': config['TEMPLATE']['T2_brain'],
    }
template_source.inputs.template_args = {
        template: []
        for template in template_source.inputs.field_template.keys()
    }




def mergeROI (in_files):
    import nibabel as nib
    from nilearn import plotting 
    import os, os.path
    from nilearn import image
    
  
    roi_images = {
        f'img{i}': image.binarize_img(roi_path,  0) for i, roi_path in enumerate(in_files)
    }
    
    overlap = image.math_img(
         " + ".join(f"{i + 1} * img{i} * img{i+1}" for i in range(len(roi_images)-1)),
        **roi_images
    )


    colored_img = image.math_img(
        " + ".join(f"{i + 1} * img{i}" for i in range(len(roi_images))),
        **roi_images
    )

    output_image = image.math_img(f'img1 - img2', img1=colored_img, img2=overlap)



    nib.save(output_image, os.path.join(os.getcwd(),'parcel_image.nii.gz')) 
    return os.path.join(os.getcwd(),'parcel_image.nii.gz'), in_files



MergeROI = Function(
    input_names=['in_files'], output_names=['merged_file', 'roi_file_list'],
    function=mergeROI
  )

merge =  pe.Node(interface = MergeROI,
          name = 'merge_ROI')





def outMod (labels, conn_file):
    import pandas as pd
    import nibabel as nib
    import os, os.path


    
    roi_names = []

    for i in labels: 
        text = i.split('/')
        file_name = text[-1]
        roi_names.append(file_name)
    
    df = pd.read_csv(conn_file,  header=None)



    df = df.loc[~(df==0).all(axis=1)]
    df = df.loc[:, (df != 0).any(axis=0)]

    df = df.rename(columns=dict(zip(df.columns, roi_names)))

    df.index = roi_names[0:len(df.index)]
    
    df = df.div(50000) #seed number from tckgen

    print(df)

   
    # Reshape the DataFrame with combined row and column names
    reshaped_df = pd.melt(df.reset_index(), id_vars='index')   
    reshaped_df["ROI"] = reshaped_df["variable"]  +"_"+ reshaped_df['index'].astype(str) 
    desired_columns = ['ROI', 'value']
    reshaped_df = reshaped_df.reindex(columns=desired_columns)
    #print(reshaped_df)

    df_transposed = reshaped_df.set_index('ROI').transpose()

   # df_pivot = reshaped_df.pivot(columns='ROI', values='value')
   # df_pivot.index = range(len(df_pivot))

    df_transposed.to_csv(os.path.join(os.getcwd(),'labeled_conn.csv'))
                                        
    return os.path.join(os.getcwd(),'labeled_conn.csv'), df_transposed 

    

OutMod = Function(
    input_names=['labels', 'conn_file'], output_names=['labeled_conn', 'data_frame'],
    function=outMod
  )

out_mod =  pe.Node(interface = OutMod,
          name = 'out_mod')




def multi_Mod (composite_id, conn_df):

    import pandas as pd
    import os 
    
    print(composite_id)
    #print(conn_df)

    df2 = pd.concat(conn_df, ignore_index=True)
    df2['subject_id'] = composite_id
    df_renamed = df2.set_index('subject_id')

    '''

    # Get the len of df2 then iterate over the composite_id list, and set the string as index name for the len of each connectome (conn_df)
    x =len(df2)
    y=len (conn_df)
    #n = x//y
    n = len(composite_id)

    for i, name in enumerate(composite_id):
        print(name)
        start_idx = i * n  
        end_idx = (i + 1) * n 



    # Assign the new name to the corresponding rows using df.rename
        df2 = df2.rename(index=lambda x: name if start_idx <= int(x) < end_idx else x)
        df2 = df2.rename(index=lambda x: name if int(x) % n == 0 else x)



    #print(df2)

    '''



    df_renamed.to_csv(os.path.join(os.getcwd(),'multi_index_conn.csv'))


    return  os.path.join(os.getcwd(),'multi_index_conn.csv') 



Multi_Mod = Function(
    input_names=['composite_id','conn_df'], output_names=['multi_index_conn'],
    function=multi_Mod
  )


multi_mod =  JoinNode(interface = Multi_Mod, joinsource="subjects",
          joinfield = ['composite_id','conn_df'], name = 'multi_mod')







mrconvert_nifti_to_mif = pe.Node(
    interface=mrt.MRConvert( 
        out_file='dwi.mif'
        
    ),
    name='mrconvert'
)


dwidenoise = pe.Node(
    interface=mrt.DWIDenoise( 
        out_file='dwi_denoise.mif'
        
    ),
    name='mrdenoise'
)




dwiextract = pe.Node(
    interface=mrt.DWIExtract(
        bzero=True,
        out_file='b0.mif'
    ),
    name='dwiextract'
)

reduce_dimension = pe.Node(
    interface=mrt.MRMath(
        operation='mean',
        axis = 3 ,
        out_file='b0_mean.mif'
    ),
    name='reduce_dimension'
)



mrconvert_mif_to_nifti_b0 = pe.Node(
    interface=mrt.MRConvert( 
        out_file='b0.nii.gz'
        
    ),
    name='mrconvert_mif_to_nifti_b0'
)



affine_initializer = pe.Node(
    interface=ants.AffineInitializer(
        num_threads = 20

    ),                          
    name='affine_initializer')

  
    
registration_affine = pe.Node(
    interface=ants.Registration(
 

    ), name='registration_affine'
)
registration_affine.inputs.metric = ['MI'] * 2
registration_affine.inputs.metric_weight = [1] * 2
registration_affine.inputs.radius_or_number_of_bins = [32] * 2
registration_affine.inputs.sampling_strategy = ['Random', 'Random']
registration_affine.inputs.sampling_percentage = [0.05, 0.05]
registration_affine.inputs.convergence_threshold = [1.e-6] * 2
registration_affine.inputs.convergence_window_size = [10] * 2
registration_affine.inputs.transforms = ['Rigid', 'Affine']
registration_affine.inputs.output_transform_prefix = "output_"
registration_affine.inputs.transform_parameters = [(0.1,), (0.1,)]
registration_affine.inputs.number_of_iterations = [[1000, 500, 250, 0], [1000, 500, 250, 0]]
registration_affine.inputs.smoothing_sigmas = [[3, 2, 1, 0]] * 2
registration_affine.inputs.sigma_units = ['vox'] * 2
registration_affine.inputs.shrink_factors = [[8, 4, 2, 1]] * 2
registration_affine.inputs.use_estimate_learning_rate_once = [True, True]
registration_affine.inputs.use_histogram_matching = [True, True] # This is the default
registration_affine.inputs.output_warped_image = 'output_warped_image.nii.gz'




registration_nl = pe.Node(interface=ants.Registration(), name='registration_nl')
registration_nl.inputs.num_threads = 16
registration_nl.inputs.metric = ['MI']
registration_nl.inputs.metric_weight = [1]
registration_nl.inputs.radius_or_number_of_bins = [32]
registration_nl.inputs.sampling_strategy = [None]
registration_nl.inputs.sampling_percentage = [None]
registration_nl.inputs.convergence_threshold = [1.e-6]
registration_nl.inputs.convergence_window_size = [10]
registration_nl.inputs.transforms = ['SyN']
registration_nl.inputs.output_transform_prefix = "output_"
registration_nl.inputs.transform_parameters = [(0.1, 3.0, 0.0)]
registration_nl.inputs.number_of_iterations = [[1000, 700, 400, 100]]
registration_nl.inputs.smoothing_sigmas = [[3, 2, 1, 0]] 
registration_nl.inputs.sigma_units = ['vox']
registration_nl.inputs.shrink_factors = [[8, 4, 2, 1]]
registration_nl.inputs.use_estimate_learning_rate_once = [True]
registration_nl.inputs.use_histogram_matching = [True] # This is the default
registration_nl.inputs.output_warped_image = 'output_warped_image.nii.gz'



#apply_registration = pe.MapNode(interface=ants.ApplyTransforms(),
#                              name='apply_registration', iterfield=['input_image'])

apply_registration = pe.Node(interface=ants.ApplyTransforms(),
                              name='apply_registration')
apply_registration.inputs.dimension = 3
apply_registration.inputs.input_image_type = 0
apply_registration.inputs.interpolation = 'NearestNeighbor'



dwi2response = pe.Node(
    interface=mrt.ResponseSD(
        algorithm='tournier',
     ),
    name='dwi2response'
)



# Estimate the fiber orientation distribution (FOD) using constrained spherical deconvolution (CSD)
dwi2fod = pe.Node(
    interface=mrt.ConstrainedSphericalDeconvolution(
        algorithm='csd',
    
    ),
    name='dwi2fod'
)


tckgen = pe.Node(
    interface=mrt.Tractography(
        algorithm='iFOD2',
        select= 50000,
        out_file='prob_tractography.tck',
        n_trials = 50000,
    ),
    name='tckgen'
)



data_sink = pe.Node(DataSink(), name="datasink")
data_sink.inputs.base_directory = out_dir




tdimap = pe.Node(
    interface=mrt.ComputeTDI(
        out_file='tdi.nii.gz',
     ),
    name='tdimap'
)


# Convert images from floating point to integer values

def convertfloat2int(float_image):
    import nibabel as nib
    import os, os.path
    import numpy as np
 
    # Load the floating-point image
    my_float_img = nib.load(float_image)

    data = my_float_img.get_fdata()
    data = data.astype('int16')

    # Convert to integer type
    output_int_img = nib.Nifti1Image(data, my_float_img.affine)
    
    filename = f"image_{output_int_img}.nii.gz"
    output_path = os.path.join(os.getcwd(), filename)

    nib.save(output_int_img, os.path.join(os.getcwd(),'output_image.nii.gz')) 

    #my_int_img = nib.load('/data/parietal/store/work/zmohamed/mathfun/output_image.nii.gz')
 
    return os.path.join(os.getcwd(),'output_image.nii.gz')
    

Convert_float2int = Function(
    input_names=['float_image'], output_names=['output_int_img'],
    function=convertfloat2int
  )

convert_float2int =  pe.Node(interface = Convert_float2int,
          name = 'convert_float2int')

 


tckconn = pe.Node(
    interface=mrt.BuildConnectome(
        out_file='connectome.csv'
        
     ),
    name='tckconn'
)



# Quality check of B0 and TDI 

def plotTDI (TDI_file, background):
    import nibabel as nib
    from nilearn import plotting 
    import os, os.path
    output_file = os.path.join(
        os.getcwd(),
        f'{os.path.basename(TDI_file)}.png'
    )
    stat_img = nib.load(TDI_file)
    
    plotting.plot_stat_map(stat_img,bg_img = background , display_mode='mosaic', black_bg = False, cut_coords= 5, title='Statistical map', output_file= output_file)
    return output_file   

PlotTDI = Function(
    input_names=['TDI_file','background'], output_names=['output_image'],
    function=plotTDI
  )

plot_tdi =  pe.Node(interface = PlotTDI,
          name = 'plot_tdi')


# Quality check of ROI and TDI 


def plotROI (ROI_file, background):
    import nibabel as nib
    from nilearn import plotting 
    import os, os.path
    output_file = os.path.join(
        os.getcwd(),
        f'{os.path.basename(ROI_file)}.png'
    )
    roi_img = nib.load(ROI_file)
    
    plotting.plot_roi(roi_img,bg_img = background , display_mode='mosaic',black_bg = False, cut_coords= 5, title='ROI map', output_file= output_file)
    return output_file   

PlotROI = Function(
    input_names=['ROI_file','background'], output_names=['output_image'],
    function=plotROI
  )

plot_roi =  pe.Node(interface = PlotROI,
          name = 'plot_roi')




# Create a Nipype workflow

tractography_wf = pe.Workflow(name='tractography_wf',  base_dir=PATH)

tractography_wf.connect(infosource, 'subject_id', data_source, 'subject_id')
tractography_wf.connect(infosource, 'visit', data_source, 'visit')
tractography_wf.connect(infosource, 'session', data_source, 'session')

tractography_wf.connect(infosource, 'subject_id', subject_id_visit, 'subject_id')
tractography_wf.connect(infosource, 'visit', subject_id_visit, 'visit')

tractography_wf.connect(infosource, 'subject_id', multi_mod, 'composite_id')
#tractography_wf.connect(subject_id_visit, 'composite_id', multi_mod, 'composite_id')


tractography_wf.connect(data_source, 'dwi', mrconvert_nifti_to_mif, 'in_file')

tractography_wf.connect(data_source, 'bval', dwiextract, 'in_bval')
tractography_wf.connect(data_source, 'bvec', dwiextract, 'in_bvec')
tractography_wf.connect(mrconvert_nifti_to_mif, 'out_file', dwidenoise, 'in_file')
tractography_wf.connect(dwidenoise, 'out_file', dwiextract, 'in_file')

tractography_wf.connect(dwiextract, 'out_file', reduce_dimension, 'in_file')
tractography_wf.connect(reduce_dimension, 'out_file', mrconvert_mif_to_nifti_b0, 'in_file')

tractography_wf.connect(template_source, 'T1_brain', affine_initializer, 'moving_image')
tractography_wf.connect(mrconvert_mif_to_nifti_b0, 'out_file', affine_initializer, 'fixed_image')

tractography_wf.connect(template_source, 'T1_brain', registration_affine, 'moving_image')
tractography_wf.connect(mrconvert_mif_to_nifti_b0, 'out_file', registration_affine, 'fixed_image')
tractography_wf.connect(affine_initializer, 'out_file', registration_affine , 'initial_moving_transform')

tractography_wf.connect(template_source, 'T1_brain', registration_nl, 'moving_image')
tractography_wf.connect(mrconvert_mif_to_nifti_b0, 'out_file', registration_nl, 'fixed_image')

tractography_wf.connect(registration_affine, 'forward_transforms', registration_nl, 'initial_moving_transform')
tractography_wf.connect(registration_affine, 'forward_invert_flags', registration_nl, 'invert_initial_moving_transform')

tractography_wf.connect(registration_nl, 'forward_transforms', apply_registration, 'transforms')
tractography_wf.connect(registration_nl, 'forward_invert_flags', apply_registration, 'invert_transform_flags')

tractography_wf.connect(roi_source, 'outfiles', merge, 'in_files')

# tractography_wf.connect(merge, 'roi_file_list', data_sink, 'temp')
tractography_wf.connect(merge, 'merged_file', apply_registration, 'input_image')   # create the parcellations 


tractography_wf.connect(mrconvert_mif_to_nifti_b0, 'out_file', apply_registration, 'reference_image')


tractography_wf.connect(dwidenoise, 'out_file', dwi2response, 'in_file')
tractography_wf.connect(data_source, 'bval', dwi2response, 'in_bval')
tractography_wf.connect(data_source, 'bvec', dwi2response, 'in_bvec')

tractography_wf.connect(dwi2fod, 'wm_odf', tckgen, 'in_file')
tractography_wf.connect(data_source, 'bval', tckgen, 'in_bval')
tractography_wf.connect(data_source, 'bvec', tckgen, 'in_bvec')
tractography_wf.connect(dwidenoise, 'out_file', tckgen, 'seed_image')
  
tractography_wf.connect(dwidenoise, 'out_file', dwi2fod, 'in_file')
tractography_wf.connect(dwi2response, 'wm_file', dwi2fod, 'wm_txt')
tractography_wf.connect(data_source, 'bval', dwi2fod, 'in_bval')
tractography_wf.connect(data_source, 'bvec', dwi2fod, 'in_bvec')

tractography_wf.connect(apply_registration, 'output_image', convert_float2int, 'float_image')  

tractography_wf.connect(tckgen, 'out_file', tdimap, 'in_file')
tractography_wf.connect(mrconvert_mif_to_nifti_b0, 'out_file', tdimap, 'reference') #should be b0 image (or use transformation to get higher res)

tractography_wf.connect(tckgen, 'out_file', tckconn, 'in_file')
tractography_wf.connect(convert_float2int, 'output_int_img', tckconn, 'in_parc')  

tractography_wf.connect(tckgen, 'out_file', data_sink, 'prob_tractography')
tractography_wf.connect(tdimap, 'out_file', data_sink, 'tdi_out')
tractography_wf.connect(tckconn, 'out_file', data_sink, 'conn_out')

tractography_wf.connect(merge, 'roi_file_list', out_mod, 'labels')   
tractography_wf.connect(tckconn, 'out_file', out_mod, 'conn_file')

tractography_wf.connect(out_mod, 'data_frame', multi_mod, 'conn_df')
tractography_wf.connect(out_mod, 'labeled_conn', data_sink, 'labeled_conn_out')

tractography_wf.connect(multi_mod, 'multi_index_conn', data_sink, 'multi_index_conn_out')


tractography_wf.connect(tdimap, 'out_file', plot_tdi, 'TDI_file')
tractography_wf.connect(mrconvert_mif_to_nifti_b0, 'out_file', plot_tdi, 'background')
tractography_wf.connect(plot_tdi, 'output_image', data_sink, 'tdi_image')


tractography_wf.connect(apply_registration, 'output_image', plot_roi, 'ROI_file')
tractography_wf.connect(mrconvert_mif_to_nifti_b0, 'out_file', plot_roi, 'background')
tractography_wf.connect(plot_roi, 'output_image', data_sink, 'roi_image')

#tractography_wf.connect(apply_registration, 'output_image', data_sink, 'temp')
#tractography_wf.connect(mrconvert_mif_to_nifti_b0, 'out_file', data_sink, 'temp1')



# Run the workflow
tractography_wf.run(plugin='MultiProc', plugin_args={'n_procs':90, 'memory_gb': 8,'dont_resubmit_completed_jobs':True})
#tractography_wf.run(plugin='SLURMGraph',plugin_args={'dont_resubmit_completed_jobs': True})







