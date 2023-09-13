# #!/bin/env python
from nipype.interfaces.utility.wrappers import Function
from niflow.nipype1.workflows.dmri.fsl.epi import create_eddy_correct_pipeline
from nipype import IdentityInterface, Node, Workflow
from nipype import config as config_nipype, logging
from datetime import datetime
from nipype.interfaces.utility import Function
from nipype.interfaces.freesurfer import ReconAll, MRIsConvert, MRIConvert
import os
import sys
import subprocess
import configparser

from nipype import DataGrabber, DataSink, IdentityInterface, Node, Workflow, MapNode, JoinNode, Merge
from niflow.nipype1.workflows.dmri.fsl import bedpostx_parallel
from nipype.interfaces.fsl.utils import CopyGeom
from nipype.interfaces import utility
import nipype.interfaces.fsl as fsl
import nipype.interfaces.ants as ants
from nipype.interfaces.ants.base import ANTSCommand
ANTSCommand.set_default_num_threads(16)
#from diffusion_pipelines import diffusion_preprocessing

def convert_affine_itk_2_ras(input_affine):
    import subprocess
    import os
    import os.path
    output_file = os.path.join(
        os.getcwd(),
        f'{os.path.basename(input_affine)}.ras'
    )
    subprocess.check_output(
        f'c3d_affine_tool '
        f'-itk {input_affine} '
        f'-o {output_file} -info-full ',
        shell=True
    ).decode('utf8')
    return output_file


def convert_affine_itk_2_fsl(input_affine, ref_file, src_file):
    import subprocess
    import os
    import os.path
    output_file = os.path.join(
        os.getcwd(),
        f'{os.path.basename(input_affine)}.fsl'
    )
    subprocess.check_output(
        f'c3d_affine_tool '
        f'-itk {input_affine} '
        f'-ref {ref_file} -src {src_file} '
        f'-o {output_file} -info-full -ras2fsl',
        shell=True
    ).decode('utf8')
    return output_file


ConvertITKAffine2FSL = Function(
    input_names=['input_affine', 'ref_file', 'src_file'],
    output_names=['affine_fsl'],
    function=convert_affine_itk_2_fsl
)


def rotate_gradients_(input_affine, gradient_file):
    import os
    import os.path
    import numpy as np
    from scipy.linalg import polar

    affine = np.loadtxt(input_affine)
    u, p = polar(affine[:3, :3], side='right')
    gradients = np.loadtxt(gradient_file)
    new_gradients = np.linalg.solve(u, gradients.T).T
    name, ext = os.path.splitext(os.path.basename(gradient_file))
    output_name = os.path.join(
        os.getcwd(),
        f'{name}_rot{ext}'
    )
    np.savetxt(output_name, new_gradients)

    return output_name


def cross_product(lst):
    ret = [
        [i, j] for i_, i in enumerate(lst) for j_, j in enumerate(lst)
        if i_ < j_
    ]
    return ret


CrossProduct = Function(input_names=['lst'], output_names=['out'], function=cross_product)


def create_diffusion_prep_pipeline(name='dMRI_preprocessing', bet_frac=0.34):
    ConvertAffine2RAS = Function(
        input_names=['input_affine'], output_names=['affine_ras'],
        function=convert_affine_itk_2_ras
    )

    RotateGradientsAffine = Function(
        input_names=['input_affine', 'gradient_file'],
        output_names=['rotated_gradients'],
        function=rotate_gradients_
    )
    input_subject = Node(
        IdentityInterface(
            fields=['dwi', 'bval', 'bvec'],
        ),
        name='input_subject'
    )

    input_template = Node(
        IdentityInterface(
            fields=['T1', 'T2'],
        ),
        name='input_template'
    )

    output = Node(
        IdentityInterface(
            fields=[
                'dwi_rigid_registered', 'bval', 'bvec_rotated', 'mask', 'rigid_dwi_2_template',
                'dwi_subject_space', 'mask_subject_space', 'bvec_subject_space', 'transform_subject_2_template',
            ]
        ),
        name='output'
    )

    fslroi = Node(interface=fsl.ExtractROI(), name='fslroi')
    fslroi.inputs.t_min = 0
    fslroi.inputs.t_size = 1

    bet = Node(interface=fsl.BET(), name='bet')
    bet.inputs.mask = True
    bet.inputs.frac = bet_frac

    eddycorrect = create_eddy_correct_pipeline('eddycorrect')
    eddycorrect.inputs.inputnode.ref_num = 0

    rigid_registration = Node(
        interface=ants.RegistrationSynQuick(),
        name='affine_reg'
    )
    rigid_registration.inputs.num_threads = 8
    rigid_registration.inputs.transform_type = 'a'

    conv_affine = Node(
        interface=ConvertAffine2RAS,
        name='convert_affine_itk_2_ras'
    )

    rotate_gradients = Node(
        interface=RotateGradientsAffine,
        name='rotate_gradients'
    )

    transforms_to_list = Node(
        interface=utility.Merge(1),
        name='transforms_to_list'
    )

    apply_registration = Node(
        interface=ants.ApplyTransforms(),
        name='apply_registration'
    )
    apply_registration.inputs.dimension = 3
    apply_registration.inputs.input_image_type = 3
    apply_registration.inputs.interpolation = 'NearestNeighbor'

    apply_registration_mask = Node(
        interface=ants.ApplyTransforms(),
        name='apply_registration_mask'
    )
    apply_registration_mask.inputs.dimension = 3
    apply_registration_mask.inputs.input_image_type = 3
    apply_registration_mask.inputs.interpolation = 'NearestNeighbor'

    workflow = Workflow(
        name=name,
    )
    workflow.connect([
        (input_subject, fslroi, [('dwi', 'in_file')]),
        (fslroi, bet, [('roi_file', 'in_file')]),
        (input_subject, eddycorrect, [('dwi', 'inputnode.in_file')]),
        (fslroi, rigid_registration, [('roi_file', 'moving_image')]),
        (input_template, rigid_registration, [('T2', 'fixed_image')]),
        (rigid_registration, transforms_to_list, [('out_matrix', 'in1')]),
        (rigid_registration, conv_affine, [('out_matrix', 'input_affine')]),
        (input_subject, rotate_gradients, [('bvec', 'gradient_file')]),
        (conv_affine, rotate_gradients, [('affine_ras', 'input_affine')]),
        #(transforms_to_list, apply_registration, [('out', 'transforms')]),
        #(eddycorrect, apply_registration, [
        #    ('outputnode.eddy_corrected', 'input_image')]),
        #(input_template, apply_registration, [('T2', 'reference_image')]),
        (transforms_to_list, apply_registration_mask, [('out', 'transforms')]),
        (bet, apply_registration_mask, [('mask_file', 'input_image')]),
        (input_template, apply_registration_mask, [('T2', 'reference_image')]),


        (eddycorrect, output, [
            ('outputnode.eddy_corrected', 'dwi_subject_space')]),
        (input_subject, output, [('bvec', 'bvec_subject_space')]),
        (bet, output, [('mask_file', 'mask_subject_space')]),
        (transforms_to_list, output, [
         ('out', 'transform_subject_2_template')]),

        (conv_affine, output, [('affine_ras', 'rigid_dwi_2_template')]),
        #(apply_registration, output, [
        # ('output_image', 'dwi_rigid_registered')]),
        (rotate_gradients, output, [('rotated_gradients', 'bvec_rotated')]),
        (input_subject, output, [('bval', 'bval')]),
        (apply_registration_mask, output, [('output_image', 'mask')]),
    ])
    #workflow.write_graph(graph2use='colored', dotfilename='/oak/stanford/groups/menon/projects/cdla/2019_dwi_mathfun/scripts/2019_dwi_pipeline_mathfun/dmri_preprocessing_graph_orig.dot')
    return workflow


def freesurfer_get_ras_conversion_matrix(subjects_dir, subject_id):
    from os.path import join
    from os import getcwd
    import subprocess

    f = join(subjects_dir, subject_id, 'mri', 'brain.finalsurfs.mgz')
    res = subprocess.check_output('mri_info %s' % f, shell=True)
    res = res.decode('utf8')
    lines = res.splitlines()
    translations = dict()
    for c, coord in (('c_r', 'x'), ('c_a', 'y'), ('c_s', 'z')):
        tr = [l for l in lines if c in l][0].split('=')[4]
        translations[coord] = float(tr)

    output = (
        f'1 0 0 {translations["x"]}\n'
        f'0 1 0 {translations["y"]}\n'
        f'0 0 1 {translations["z"]}\n'
        f'0 0 0 1\n'
    )

    output_file = join(getcwd(), 'ras_c.mat')
    with open(output_file, 'w') as f:
        f.write(output)

    return output_file


def freesurfer_gii_2_native(freesurfer_gii_surface, ras_conversion_matrix, warps):
    from os.path import join, basename
    from os import getcwd
    import subprocess

    if isinstance(warps, str):
        warps = [warps]

    if 'lh' in freesurfer_gii_surface:
        structure_name = 'CORTEX_LEFT'
    elif 'rh' in freesurfer_gii_surface:
        structure_name = 'CORTEX_RIGHT'

    if 'inflated' in freesurfer_gii_surface:
        surface_type = 'INFLATED'
    elif 'sphere' in freesurfer_gii_surface:
        surface_type = 'SPHERICAL'
    else:
        surface_type = 'ANATOMICAL'

    if 'pial' in freesurfer_gii_surface:
        secondary_type = 'PIAL'
    if 'white' in freesurfer_gii_surface:
        secondary_type = 'GRAY_WHITE'

    output_file = join(getcwd(), basename(freesurfer_gii_surface))
    output_file = output_file.replace('.gii', '.surf.gii')
    subprocess.check_call(
        f'cp {freesurfer_gii_surface} {output_file}', shell=True)

    subprocess.check_call(
        f'wb_command -set-structure {output_file} {structure_name} '
        f'-surface-type {surface_type} -surface-secondary-type {secondary_type}',
        shell=True
    )

    subprocess.check_call(
        f'wb_command -surface-apply-affine {freesurfer_gii_surface} {ras_conversion_matrix} {output_file}',
        shell=True
    )

    # for warp in warps:
    #    subprocess.check_call(
    #        f'wb_command -surface-apply-warpfield {output_file} {warp} {output_file}',
    #        shell=True
    #    )

    return output_file


def surface_signed_distance_image(surface, image):
    from os.path import join, basename
    from os import getcwd
    import subprocess

    output_file = str(join(getcwd(), basename(surface)))
    output_file = output_file.replace('.surf.gii', 'signed_dist.nii.gz')

    subprocess.check_call(
        f'wb_command -create-signed-distance-volume {surface} {image} {output_file}',
        shell=True
    )

    return output_file


def shrink_surface_fun(surface, image, distance):
    from os.path import join, basename
    from os import getcwd
    import subprocess

    output_file = str(join(getcwd(), basename(surface)))
    output_file = output_file.replace('.surf.gii', '_shrunk.surf.gii')

    subprocess.check_call(
        f'shrink_surface -surface {surface} -reference {image} '
        f'-mm {distance} -out {output_file}',
        shell=True
    )

    if 'lh' in output_file:
        structure_name = 'CORTEX_LEFT'
    elif 'rh' in output_file:
        structure_name = 'CORTEX_RIGHT'

    if 'inflated' in output_file:
        surface_type = 'INFLATED'
    elif 'sphere' in output_file:
        surface_type = 'SPHERICAL'
    else:
        surface_type = 'ANATOMICAL'

    if 'pial' in output_file:
        secondary_type = 'PIAL'
    if 'white' in output_file:
        secondary_type = 'GRAY_WHITE'

    subprocess.check_call(
        f'wb_command -set-structure {output_file} {structure_name} '
        f'-surface-type {surface_type} -surface-secondary-type {secondary_type}',
        shell=True
    )

    return output_file


def bvec_flip(bvecs_in, flip):
    from os.path import join, basename
    from os import getcwd

    import numpy as np

    bvecs = np.loadtxt(bvecs_in).T * flip

    output_file = str(join(getcwd(), basename(bvecs_in)))
    np.savetxt(output_file, bvecs)

    return output_file


if __name__ == '__main__':
    from multiprocessing import set_start_method, Process, Manager
    set_start_method('forkserver')
    import filelock

    print('start')

    slurm_logs = (
        f' -e {os.path.join(os.getcwd(), "slurm_out")}/slurm_%40j.out ' +
        f'-o {os.path.join(os.getcwd(), "slurm_out")}/slurm_%40j.out '
    )

    dmri_preprocess_workflow = create_diffusion_prep_pipeline(
        'dmri_preprocess')
    config = configparser.ConfigParser()
    config.read(sys.argv[1])
    log_directory = os.path.abspath(
        config['DEFAULT'].get('log_directory', os.path.join(os.getcwd(), 'logs'))
    )
    config_nipype.update_config({
        'logging': {
            'log_directory': log_directory,
            'workflow_level': 'DEBUG',
            'interface_level': 'DEBUG',
            'filemanip_level': 'INFO',
            'log_to_file': True,
        },
        'execution': {
            'stop_on_first_crash': False,
            'keep_inputs': True,
            # 'job_finished_timeout': 30,
        },
        'monitoring': {
            'enabled': True,
        },
    })
    # config_nipype.enable_debug_mode()
    logging.update_logging(config_nipype)



    #PATH = '/oak/stanford/groups/menon/projects/cdla/2019_dwi_mathfun/results/'
    # PATH = '/tmp/cdla/dwi_mathfun_gpu_group/'#+sys.argv[1].split('.')[0]
    subjects = config['DEFAULT']['id_list']
    if len(subjects) == 4:
        subject_list = [subjects]
    else:
        subject_list = config['DEFAULT']['id_list'].split(' ')
        subject_list = [str(x) for x in subject_list]
    #PATH = '/tmp/cdla/dwi_mathfun_gpu_group_050420/'
   # PATH = '/scratch/users/cdla/dwi_mathfun_gpu_group_%s'%(subject_list[0])#%s/'%(datetime.now().strftime("%Y%m%d_%H%M%S"))
    PATH = '/home/parietal/dwasserm/research/data/LargeBrainNets/mathfun'

    visits = [config['DEFAULT']['visits']]
    subjects_dir = config['DEFAULT']['subjects_dir']
    sessions = [config['DEFAULT']['sessions']]
    use_cpu = config['DEFAULT']['use_cpu']
    print(subject_list)
    print(visits)
    print(sessions)
    infosource = Node(IdentityInterface(fields=['subject_id', 'visit', 'session']),
                      name='subjects')
    infosource.iterables = [('subject_id', subject_list),
                            ('visit', visits), ('session', sessions)]
    #infosource.iterables = [('visit', visits), ('session', sessions)]
    # infosource.inputs.subject_id=list(subject_list)
    # infosource.inputs.visit=['1']
    # infosource.inputs.session=['1']

    def compose_id(subject_id, visit):
        if type(subject_id) == list:
            subject_id = subject_id[0]
        if type(visit) == list:
            visit = visit[0]
        composite_id = subject_id + '_' + visit

        return composite_id

    subject_id_visit = Node(
        interface=Function(
            input_names=['subject_id', 'visit'], output_names=['composite_id'],
            function=compose_id
            #lambda subject_id, visit: '{}_{}'.format(str(subject_id), str(visit))
        ),
        name='subject_id_visit'
    )

    data_source = Node(DataGrabber(infields=['subject_id', 'visit', 'session'],
                                   outfields=['dwi', 'bval', 'bvec', 'T1']),
                       name='data_grabber')
    data_source.inputs.sort_filelist = True
    data_source.inputs.base_directory = config['DEFAULT']['base_directory']
    data_source.inputs.template = ''
    data_source.inputs.field_template = {
        'T1': '%s/visit%s/session%s/anat/T1w.nii',
        'dwi': '%s/visit%s/session%s/dwi/dwi_raw.nii.gz',
        'bval': '%s/visit%s/session%s/dwi/dti_raw.bvals',
        'bvec': '%s/visit%s/session%s/dwi/dti_raw.bvecs'
    }
    data_source.inputs.template_args = {
        template: [['subject_id', 'visit', 'session']]
        for template in data_source.inputs.field_template.keys()
    }

    data_sink = Node(DataSink(), name="datasink")
    data_sink.inputs.base_directory = os.path.abspath('results')


    flip_bvectors_node = Node(
        interface=Function(
            input_names=['bvecs_in', 'flip'], output_names=['bvecs_out'],
            function=bvec_flip
        ),
        name='flip_bvecs',
    )
    flip_bvectors_node.inputs.flip = (-1, 1, 1)

    template_source = Node(DataGrabber(infields=[], outfields=['T1', 'T1_brain', 'T1_mask', 'T2', 'T2_brain', 'T2_mask']),
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

    roi_source = Node(interface=DataGrabber(infields=[], outfields=['outfiles']),
                      name='rois')
    roi_source.inputs.sort_filelist = True
    roi_source.inputs.base_directory = config['ROIS']['directory']
    roi_source.inputs.template = '*1mm_bin.nii.gz'

    recon_all = Node(interface=ReconAll(), name='recon_all')
    recon_all.inputs.directive = 'all'
    recon_all.inputs.subjects_dir = subjects_dir
    recon_all.inputs.openmp = 16
    recon_all.inputs.mprage = True
    recon_all.inputs.parallel = True
    recon_all.interface.num_threads = 16
    recon_all.n_procs = 16
    recon_all.plugin_args = {
        'sbatch_args': '--time=48:00:00 -c 16 --mem=16G --oversubscribe --exclude=node[22-32] ',
        'overwrite': True
    }

    ras_conversion_matrix = Node(
        interface=Function(
            input_names=['subjects_dir', 'subject_id'],
            output_names=['output_mat'],
            function=freesurfer_get_ras_conversion_matrix
        ),
        name='ras_conversion_matrix'
    )

    mris_convert = MapNode(interface=MRIsConvert(),
                           name='mris_convert', iterfield=['in_file'])
    mris_convert.inputs.out_datatype = 'gii'
    mris_convert.inputs.subjects_dir = subjects_dir

    mri_convert = Node(interface=MRIConvert(), name='mri_convert')
    mri_convert.inputs.out_type = 'nii'
    mri_convert.inputs.subjects_dir = subjects_dir
    fslcpgeom_mask = Node(
        interface=CopyGeom(), name='fsl_cpgeom_mask')

    fslcpgeom_roi = MapNode(interface=CopyGeom(),
                            name='fslcpgeom_roi', iterfield=['dest_file'])

    freesurfer_surf_2_native = MapNode(
        interface=Function(
            input_names=['freesurfer_gii_surface', 'ras_conversion_matrix', 'warps'], output_names=['out_surf'],
            function=freesurfer_gii_2_native
        ),
        name='freesurfer_surf_2_native',
        iterfield=['freesurfer_gii_surface']
    )

    #bedpostx = Node(interface=fsl.BEDPOSTX5(), name='bedpostx', iterfield=['dwi'])
    bedpostx = bedpostx_parallel(
        params=dict(
            fudge=1,
            burn_in=1000,
            n_jumps=1250,
            sample_every=25,
            n_fibres=3,
        )
    )
    bedpostx.get_node('xfibres').plugin_args = {
        'sbatch_args': '--time=72:00:00 -c 4 -n 1 --mem=16G --oversubscribe --exclude=node[22-32]',
        'max_jobs': 4,
        'overwrite': True
    }

    dtifit = Node(interface=fsl.DTIFit(), name='dti')
    dtifit.inputs.save_tensor = True

    #bedpostx.inputs.n_fibres = 3
    #bedpostx.inputs.fudge = 1
    #bedpostx.inputs.burn_in = 1000
    # bedpostx.inputs.n_jumps=1250
    # bedpostx.inputs.sample_every=25
    #bedpostx.inputs.use_gpu = False
    # bedpostx.interface.num_threads=16
    # bedpostx.n_procs=16

    #bedpostx.plugin_args={'sbatch_args':'--time=4:00:00 -c 16 --mem=128G --account=menon --partition=nih_s10 --gres=gpu:1','overwrite':True}
    #bedpostx.plugin_args={'sbatch_args':'--time=8:00:00 -c 4 --mem=16G --partition=gpu --gpus=1','overwrite':True}
    #bedpostx.plugin_args={'sbatch_args':'--time=72:00:00 -c 4 -n 1 --mem=16G --oversubscribe --comment="7014"','overwrite':True}
    join_seeds = Node(
        interface=Merge(5),
        name='join_seeds',
    )

    pbx2 = Node(
        interface=fsl.ProbTrackX2(),
        name='probtrackx2',
        #iterfield=['seed']
    )
    pbx2.inputs.n_samples = 5000
    pbx2.inputs.n_steps = 2000
    pbx2.inputs.step_length = 0.5
    pbx2.inputs.omatrix1 = True
    pbx2.inputs.distthresh1 = 5
    pbx2.inputs.network = True
    pbx2.inputs.correct_path_distribution = True
    pbx2.inputs.os2t = False
    pbx2.inputs.verbose = 2
    pbx2.inputs.args = " --ompl --fibthresh=0.01 "
    pbx2.inputs.out_dir = '.'
    pbx2.plugin_args = {
        'sbatch_args': '--time=48:00:00 -c 4 --mem=16G --exclude=node[25-32] ' + slurm_logs, 'overwrite': True}
    pbx2.interface.num_threads = 16
    pbx2.n_procs = 16

    pbx2_cp = MapNode(
        interface=fsl.ProbTrackX2(),
        name='probtrackx2_bypairs',
        iterfield=['seed']
    )
    pbx2_cp.inputs.n_samples = 5000
    pbx2_cp.inputs.n_steps = 2000
    pbx2_cp.inputs.step_length = 0.5
    pbx2_cp.inputs.omatrix1 = True
    pbx2_cp.inputs.distthresh1 = 5
    pbx2_cp.inputs.network = True
    pbx2_cp.inputs.correct_path_distribution = True
    pbx2_cp.inputs.os2t = False
    pbx2_cp.inputs.verbose = 2
    pbx2_cp.inputs.args = " --ompl --fibthresh=0.01 "
    pbx2_cp.inputs.out_dir = '.'
    pbx2_cp.plugin_args = {
            'sbatch_args': '--time=48:00:00 -c 4 --mem=16G --exclude=node[25-32] ' + slurm_logs, 'overwrite': True}
    pbx2_cp.interface.num_threads = 16
    pbx2_cp.n_procs = 16

    cross_product = Node(
        interface=CrossProduct, name='seed_cross_product',
        #joinfield='lst', joinsource='apply_registration_seeds_2_dwi'
    )

    fslroi = Node(interface=fsl.ExtractROI(), name='fslroi')
    fslroi.inputs.t_min = 0
    fslroi.inputs.t_size = 1

    affine_initializer = Node(
        interface=ants.AffineInitializer(), name='affine_initializer')
    affine_initializer.inputs.num_threads = 16

    affine_initializer.interface.num_threads = 16
    affine_initializer.n_procs = 16

    registration_affine = Node(interface=ants.Registration(), name='reg_aff')
    registration_affine.inputs.num_threads = 16
    registration_affine.n_procs = 4
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
    registration_affine.inputs.number_of_iterations = [
        [1000, 500, 250, 0], [1000, 500, 250, 0]]
    registration_affine.inputs.smoothing_sigmas = [[3, 2, 1, 0]] * 2
    registration_affine.inputs.sigma_units = ['vox'] * 2
    registration_affine.inputs.shrink_factors = [[8, 4, 2, 1]] * 2
    registration_affine.inputs.use_estimate_learning_rate_once = [True, True]
    registration_affine.inputs.use_histogram_matching = [
        True, True]  # This is the default
    registration_affine.inputs.output_warped_image = 'output_warped_image.nii.gz'

    registration_nl = Node(interface=ants.Registration(), name='reg_nl')
    registration_nl.inputs.num_threads = 16
    registration_nl.interface.num_threads = 16
    registration_nl.n_procs = 4
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
    registration_nl.inputs.use_histogram_matching = [
        True]  # This is the default
    registration_nl.inputs.output_warped_image = 'output_warped_image.nii.gz'

    affine_initializer_2_dwi = Node(
        interface=ants.AffineInitializer(), name='affine_initializer_2_dwi')
    affine_initializer_2_dwi.inputs.num_threads = 16

    affine_initializer_2_dwi.interface.num_threads = 16
    affine_initializer_2_dwi.n_procs = 16

    registration_affine_2_dwi = Node(interface=ants.Registration(), name='reg_aff_2_dwi')
    registration_affine_2_dwi.inputs.num_threads = 16
    registration_affine_2_dwi.n_procs = 4
    registration_affine_2_dwi.inputs.metric = ['MI'] * 2
    registration_affine_2_dwi.inputs.metric_weight = [1] * 2
    registration_affine_2_dwi.inputs.radius_or_number_of_bins = [32] * 2
    registration_affine_2_dwi.inputs.sampling_strategy = ['Random', 'Random']
    registration_affine_2_dwi.inputs.sampling_percentage = [0.05, 0.05]
    registration_affine_2_dwi.inputs.convergence_threshold = [1.e-6] * 2
    registration_affine_2_dwi.inputs.convergence_window_size = [10] * 2
    registration_affine_2_dwi.inputs.transforms = ['Rigid', 'Affine']
    registration_affine_2_dwi.inputs.output_transform_prefix = "output_"
    registration_affine_2_dwi.inputs.transform_parameters = [(0.1,), (0.1,)]
    registration_affine_2_dwi.inputs.number_of_iterations = [
        [1000, 500, 250, 0], [1000, 500, 250, 0]]
    registration_affine_2_dwi.inputs.smoothing_sigmas = [[3, 2, 1, 0]] * 2
    registration_affine_2_dwi.inputs.sigma_units = ['vox'] * 2
    registration_affine_2_dwi.inputs.shrink_factors = [[8, 4, 2, 1]] * 2
    registration_affine_2_dwi.inputs.use_estimate_learning_rate_once = [True, True]
    registration_affine_2_dwi.inputs.use_histogram_matching = [
        True, True]  # This is the default
    registration_affine_2_dwi.inputs.output_warped_image = 'output_warped_image.nii.gz'

    registration_nl_2_dwi = Node(interface=ants.Registration(), name='reg_nl_2_dwi')
    registration_nl_2_dwi.inputs.num_threads = 16
    registration_nl_2_dwi.interface.num_threads = 16
    registration_nl_2_dwi.n_procs = 4
    registration_nl_2_dwi.inputs.metric = ['MI']
    registration_nl_2_dwi.inputs.metric_weight = [1]
    registration_nl_2_dwi.inputs.radius_or_number_of_bins = [32]
    registration_nl_2_dwi.inputs.sampling_strategy = [None]
    registration_nl_2_dwi.inputs.sampling_percentage = [None]
    registration_nl_2_dwi.inputs.convergence_threshold = [1.e-6]
    registration_nl_2_dwi.inputs.convergence_window_size = [10]
    registration_nl_2_dwi.inputs.transforms = ['SyN']
    registration_nl_2_dwi.inputs.output_transform_prefix = "output_"
    registration_nl_2_dwi.inputs.transform_parameters = [(0.1, 3.0, 0.0)]
    registration_nl_2_dwi.inputs.number_of_iterations = [[1000, 700, 400, 100]]
    registration_nl_2_dwi.inputs.smoothing_sigmas = [[3, 2, 1, 0]]
    registration_nl_2_dwi.inputs.sigma_units = ['vox']
    registration_nl_2_dwi.inputs.shrink_factors = [[8, 4, 2, 1]]
    registration_nl_2_dwi.inputs.use_estimate_learning_rate_once = [True]
    registration_nl_2_dwi.inputs.use_histogram_matching = [
        True]  # This is the default
    registration_nl_2_dwi.inputs.output_warped_image = 'output_warped_image.nii.gz'
    registration_nl_2_dwi.inputs.output_inverse_warped_image = 'output_inverse_warped_image.nii.gz'

    affine_2_dwi_itk2fsl = Node(interface=ConvertITKAffine2FSL, name='affine_2_dwi_itk2fsl')


    apply_registration_seeds_2_dwi = MapNode(interface=ants.ApplyTransforms(),
                                 name='apply_registration_seeds_2_dwi', iterfield=['input_image'])
    apply_registration_seeds_2_dwi.inputs.dimension = 3
    apply_registration_seeds_2_dwi.inputs.input_image_type = 3
    apply_registration_seeds_2_dwi.inputs.interpolation = 'NearestNeighbor'

    select_nl_transform = Node(
        interface=utility.Select(), name='select_nl_transform')
    select_nl_transform.inputs.index = [1]

    select_seed_0 = MapNode(interface=utility.Select(), name='select_seed_0', iterfield=['inlist'])
    select_seed_0.inputs.index=[0]

    select_seed_1 = MapNode(interface=utility.Select(), name='select_seed_1', iterfield=['inlist'])
    select_seed_1.inputs.index=[1]

    registration = Node(interface=ants.Registration(), name='reg')
    registration.inputs.num_threads = 16
    registration.interface.num_threads = 16
    registration.n_procs = 16
    registration.inputs.metric = ['MI', 'MI', 'MI']
    registration.inputs.metric_weight = [1] * 3
    registration.inputs.radius_or_number_of_bins = [32] * 3
    registration.inputs.sampling_strategy = ['Random', 'Random', None]
    registration.inputs.sampling_percentage = [0.05, 0.05, None]
    registration.inputs.convergence_threshold = [1.e-6] * 3
    registration.inputs.convergence_window_size = [10] * 3
    registration.inputs.transforms = ['Rigid', 'Affine', 'SyN']
    registration.inputs.output_transform_prefix = "output_"
    registration.inputs.transform_parameters = [
        (0.1,), (0.1,), (0.1, 3.0, 0.0)]
    registration.inputs.number_of_iterations = [
        [1000, 500, 250, 0], [1000, 500, 250, 0], [1000, 700, 400, 100]]
    registration.inputs.smoothing_sigmas = [[3, 2, 1, 0]] * 2 + [[3, 2, 1, 0]]
    registration.inputs.sigma_units = ['vox'] * 3
    registration.inputs.shrink_factors = [[8, 4, 2, 1]] * 2 + [[8, 4, 2, 1]]
    registration.inputs.use_estimate_learning_rate_once = [True, True, True]
    registration.inputs.use_histogram_matching = [
        True, True, True]  # This is the default
    registration.inputs.output_warped_image = 'output_warped_image.nii.gz'

    apply_registration = MapNode(interface=ants.ApplyTransforms(),
                                 name='apply_registration', iterfield=['input_image'])
    apply_registration.inputs.dimension = 3
    apply_registration.inputs.input_image_type = 3
    apply_registration.inputs.interpolation = 'NearestNeighbor'

    apply_registration_template_2_dwi = Node(
        interface=ants.ApplyTransforms(),
        name='apply_registration_template_2_dwi'
    )
    apply_registration_template_2_dwi.inputs.dimension = 3
    apply_registration_template_2_dwi.inputs.input_image_type = 3
    apply_registration_template_2_dwi.inputs.invert_transform_flags = [False]

    apply_registration_fdt = MapNode(
        interface=ants.ApplyTransforms(),
        name='apply_registration_fdt',
        iterfield=['input_image']
    )
    apply_registration_fdt.inputs.dimension = 3
    apply_registration_fdt.inputs.input_image_type = 3
    apply_registration_fdt.inputs.interpolation = 'Linear'


    template_2_dwi_transforms = Node(
        interface=utility.Merge(1),
        name='template_2_dwi_transforms'
    )
    shrink_surface_node = MapNode(
        interface=Function(
            input_names=['surface', 'image', 'distance'],
            output_names=['out_file'],
            function=shrink_surface_fun
        ),
        name='surface_shrink_surface',
        iterfield=['surface']
    )
    shrink_surface_node.inputs.distance = 3

    workflow = Workflow('diffusion_workflow_new_mgt', base_dir=PATH)
    workflow.connect([
        (infosource, data_source, [('subject_id', 'subject_id'),
                                   ('visit', 'visit'),
                                   ('session', 'session')]),
        (data_source, flip_bvectors_node, [('bvec', 'bvecs_in')]),

        (infosource, subject_id_visit, [
            ('subject_id', 'subject_id'),
            ('visit', 'visit')
        ]),
        (data_source, recon_all, [('T1', 'T1_files')]),
        (subject_id_visit, recon_all, [('composite_id', 'subject_id')]),
        (recon_all, ras_conversion_matrix, [
            ('subjects_dir', 'subjects_dir'),
            ('subject_id', 'subject_id')
        ]),
        (recon_all, mris_convert, [('white', 'in_file')]),
        (recon_all, mri_convert, [('brain', 'in_file')]),
        (mris_convert, freesurfer_surf_2_native, [
         ('converted', 'freesurfer_gii_surface')]),

        (mri_convert, affine_initializer, [('out_file', 'moving_image')]),
        (template_source, affine_initializer, [('T1_brain', 'fixed_image')]),

        (mri_convert, registration_affine, [('out_file', 'moving_image')]),
        (template_source, registration_affine, [
            ('T1_brain', 'fixed_image'),
        ]),

        (affine_initializer, registration_affine, [
         ('out_file', 'initial_moving_transform')]),

        (mri_convert, registration_nl, [('out_file', 'moving_image')]),
        (template_source, registration_nl, [
            ('T1_brain', 'fixed_image'),
        ]),
        (registration_affine, registration_nl, [
            ('forward_transforms', 'initial_moving_transform'),
            ('forward_invert_flags', 'invert_initial_moving_transform')
        ]),

        (dmri_preprocess_workflow, affine_initializer_2_dwi, [('fslroi.roi_file', 'moving_image')]),
        (template_source, affine_initializer_2_dwi, [('T2_brain', 'fixed_image')]),

        (dmri_preprocess_workflow, registration_affine_2_dwi, [('fslroi.roi_file', 'moving_image')]),
        (template_source, registration_affine_2_dwi, [
            ('T2_brain', 'fixed_image'),
        ]),

        (affine_initializer_2_dwi, registration_affine_2_dwi,
            [('out_file', 'initial_moving_transform')]
        ),

        (dmri_preprocess_workflow, registration_nl_2_dwi, [('fslroi.roi_file', 'moving_image')]),
        (template_source, registration_nl_2_dwi, [
            ('T2_brain', 'fixed_image'),
        ]),
        (registration_affine_2_dwi, registration_nl_2_dwi, [
            ('forward_transforms', 'initial_moving_transform'),
            ('forward_invert_flags', 'invert_initial_moving_transform')
        ]),

        # (registration_affine_2_dwi, affine_2_dwi_itk2fsl,
        #    [('forward_transforms', 'input_affine')]
        #),
        #(template_source, affine_2_dwi_itk2fsl,
        #    [('T2_brain', 'ref_file')]
        #),
        #(dmri_preprocess_workflow, affine_2_dwi_itk2fsl,
        #    [('fslroi.roi_file', 'src_file')]
        #),

        (
            registration_nl_2_dwi, apply_registration_seeds_2_dwi,
            [
                ('reverse_transforms', 'transforms'),
                ('reverse_invert_flags', 'invert_transform_flags')
            ]
        ),
        (
            roi_source, apply_registration_seeds_2_dwi,
            [
                ('outfiles', 'input_image')
            ]
        ),
        (
            dmri_preprocess_workflow, apply_registration_seeds_2_dwi,
            [
                ('fslroi.roi_file', 'reference_image')
            ]
        ),


        (ras_conversion_matrix, freesurfer_surf_2_native,
         [('output_mat', 'ras_conversion_matrix')]),
        (registration_nl, select_nl_transform,
         [('forward_transforms', 'inlist')]),
        (select_nl_transform, freesurfer_surf_2_native, [('out', 'warps')]),

        (registration_nl, apply_registration, [
            ('forward_transforms', 'transforms'),
            ('forward_invert_flags', 'invert_transform_flags'),
        ]),
        (roi_source, apply_registration, [('outfiles', 'input_image')]),
        (mri_convert, apply_registration, [('out_file', 'reference_image')]),

        (data_source, dmri_preprocess_workflow,
         [
             ('dwi', 'input_subject.dwi'),
             ('bval', 'input_subject.bval'),
         ]
         ),
        (flip_bvectors_node, dmri_preprocess_workflow,
         [('bvecs_out', 'input_subject.bvec')]),

        (mri_convert, dmri_preprocess_workflow, [
            ('out_file', 'input_template.T1'),
            ('out_file', 'input_template.T2')
        ]),
        (
            dmri_preprocess_workflow,
            bedpostx,
            # [('output.bval', 'bvals'),
            # ('output.bvec_rotated', 'bvecs'),
            # ('output.dwi_rigid_registered', 'dwi'),
            # ('output.mask', 'mask')],
            [('output.bval', 'inputnode.bvals'),
             ('output.bvec_subject_space', 'inputnode.bvecs'),
             ('output.dwi_subject_space', 'inputnode.dwi'),
             ('output.mask_subject_space', 'inputnode.mask')],

        ),
        (
            dmri_preprocess_workflow,
            dtifit,
            # [('output.bval', 'bvals'),
            # ('output.bvec_rotated', 'bvecs'),
            # ('output.dwi_rigid_registered', 'dwi'),
            # ('output.mask', 'mask')],
            [('output.bval', 'bvals'),
             ('output.bvec_subject_space', 'bvecs'),
             ('output.dwi_subject_space', 'dwi'),
             ('output.mask_subject_space', 'mask')],

        ),
        (
            freesurfer_surf_2_native,
            shrink_surface_node,
            [('out_surf', 'surface')],
        ),
        (
            mri_convert,
            shrink_surface_node,
            [('out_file', 'image')],
        ),
        (
            bedpostx, pbx2,
            [
                ('outputnode.merged_thsamples', 'thsamples'),
                ('outputnode.merged_fsamples', 'fsamples'),
                ('outputnode.merged_phsamples', 'phsamples'),
                ('inputnode.mask', 'mask'),
            ]
        ),
        (
            bedpostx, pbx2_cp,
            [
                ('outputnode.merged_thsamples', 'thsamples'),
                ('outputnode.merged_fsamples', 'fsamples'),
                ('outputnode.merged_phsamples', 'phsamples'),
                ('inputnode.mask', 'mask'),
            ]
        ),
        (
            apply_registration_seeds_2_dwi, cross_product,
            [
                ('output_image', 'lst'),
            ]
        ),
        (
            apply_registration_seeds_2_dwi, pbx2,
            [
                ('output_image', 'seed'),
                # ('output_image', 'target_masks')
            ]
        ),
        (
            cross_product, pbx2_cp,
            [
                ('out', 'seed'),
                # ('out', 'target_masks'),
            ]
        ),
        (
            registration_nl_2_dwi, apply_registration_fdt,
            [
                ('forward_transforms', 'transforms'),
                ('forward_invert_flags', 'invert_transform_flags')
            ]
        ),
        (
            pbx2_cp, apply_registration_fdt,
            [
                ('fdt_paths', 'input_image')
            ]
        ),
        (
            template_source, apply_registration_fdt,
            [
                ('T2_brain', 'reference_image')
            ]
        ),

        #(
        #    affine_2_dwi_itk2fsl, pbx2,
        #    [
        #        ('affine_fsl', 'thsamples'),
        #    ]
        #),

        (
            dmri_preprocess_workflow, fslcpgeom_mask,
            [
                ('output.mask', 'dest_file'),
            ]
        ),
        (
            mri_convert, fslcpgeom_mask,
            [
                ('out_file', 'in_file')
            ]
        ),
        #         (
        #             fslcpgeom_mask, pbx2,
        #             [
        #              ('out_file', 'mask')
        #             ]
        #         ),
        # (
        #    shrink_surface_node, join_seeds,
        #    [
        #        ('out_file', 'in1')
        #    ]
        # ),

        #(
        #    apply_registration, fslcpgeom_roi,
        #    [
        #        ('output_image', 'dest_file')
        #    ]
        #),
        (
            template_source, apply_registration_template_2_dwi,
            [('T1_brain', 'input_image')],
        ),
        (
            dmri_preprocess_workflow, apply_registration_template_2_dwi,
            [('fslroi.roi_file', 'reference_image')],
        ),
        (
            dmri_preprocess_workflow, template_2_dwi_transforms,
            [('affine_reg.out_matrix', 'in1')]
        ),
        (
            registration_nl, apply_registration_template_2_dwi,
            [
                ('reverse_transforms', 'transforms'),
                ('reverse_invert_flags', 'invert_transform_flags')
            ],
        ),
        #(
        #    mri_convert, fslcpgeom_roi,
        #    [
        #        ('out_file', 'in_file')
        #    ]
        #),
        # (
        #    fslcpgeom_roi, join_seeds,
        #    [
        #        ('out_file', 'in2')
        #    ]
        #),
        #         (
        #             join_seeds, pbx2,
        #             [
        #              ('out', 'seed'),
        #             ]
        #         ),
        (
            infosource, data_sink,
            [
                ('subject_id', 'container'),
            ]
        ),
        (
            bedpostx,
            data_sink,
            [
                ('outputnode.merged_thsamples', 'bedpostx.@merged_thsamples.@subject'),
            ]
        ),
        (
            dtifit,
            data_sink,
            [
                ('FA', 'dti.@fa.@subject'),
                ('MD', 'dti.@md.@subject'),
                ('MO', 'dti.@mo.@subject'),
                ('L1', 'dti.@ad.@subject'),
                ('L3', 'dti.@pd.@subject'),
                ('S0', 'dti.@s0.@subject'),
                ('tensor', 'dti.@tensor.@subject'),
            ]
        ),
        (
            dmri_preprocess_workflow, data_sink,
            [
                ('fslroi.roi_file', 'b0'),
                #('output.dwi_rigid_registered', 'dwi_acpc')
            ]
        ),
        (
            apply_registration_template_2_dwi, data_sink,
            [
                ('output_image', 'template_2_dwi')
            ]
        ),
        (
            apply_registration_seeds_2_dwi, data_sink,
            [
                ('output_image', 'seeds')
            ]
        ),
        (
            mri_convert, data_sink,
            [('out_file', 'T1_freesurfer')]
        ),
        (
            registration_nl_2_dwi, data_sink,
            [('warped_image', 'DWI_2_MNI')]
        ),
        (
            registration_nl_2_dwi, data_sink,
            [('inverse_warped_image', 'MNI_2_DWI')]
        ),
        (
            pbx2, data_sink,
            [
                ('fdt_paths', 'pbx2.@fdt_paths'),
                #('targets', 'pbx2.@targets'),
                ('matrix1_dot', 'pbx2.@matrix1'),
                ('network_matrix', 'pbx2.@matrix1_network'),
                ('lookup_tractspace', 'pbx2.@lookup_tractspace'),
                ('log', 'pbx2.@log'),
                ('way_total', 'pbx2.@way_total'),
            ]
        ),
        (
            pbx2_cp, data_sink,
            [
                ('fdt_paths', 'pbx2_cp.@fdt_paths'),
                #('targets', 'pbx2.@targets'),
                ('matrix1_dot', 'pbx2_cp.@matrix1'),
                ('network_matrix', 'pbx2_cp.@matrix1_network'),
                ('lookup_tractspace', 'pbx2_cp.@lookup_tractspace'),
                ('log', 'pbx2_cp.@log'),
                ('way_total', 'pbx2_cp.@way_total'),
            ]
        ),
        (
            apply_registration_seeds_2_dwi, data_sink,
            [
                ('output_image', 'seed.@seed'),
            ]
        ),
        (
            apply_registration_fdt, data_sink,
            [
                ('output_image', 'pair_fdt_MNI'),
            ]
        ),
        #(
        #    cross_product, select_seed_0,
        #    [('out', 'inlist')]
        #),
        #(
        #    cross_product, select_seed_1,
        #    [('out', 'inlist')]
        #),
        #(
        #    select_seed_0, data_sink,
        #    [('out', 'pbx2.seed0')],
        #),
        #(
        #    select_seed_1, data_sink,
        #    [('out', 'pbx2.seed1')],
        #),
    ])

    workflow.write_graph(format='pdf', simple_form=False)
    if False and (config['DEFAULT'].get('server', '').lower() == 'margaret'):
        workflow.run(plugin='SLURMGraph',
                     plugin_args={
                         'dont_resubmit_completed_jobs':
                         True,
                         'sbatch_args':
                         '--oversubscribe ' +
                         '-N 1 -n 1 ' +
                         '--time 5-0 ' +
                         slurm_logs +
                         '--exclude=node[25-32] '
                     })
    elif config['DEFAULT'].get('server', '').lower() == 'linear':
        workflow.run(plugin='Linear',
        #    plugin_args={
        #        'dont_resubmit_completed_jobs': True,
        #        'sbatch_args':
        #            '--mem=16G -t 6:00:00 --oversubscribe -n 2 '
        #            '--exclude=node[22-32] -c 2 ' +
        #            slurm_logs,
        #        'max_jobs': 20
        #    },
        )
    elif config['DEFAULT'].get('server', '').lower() == 'multiproc':
        workflow.run(plugin='MultiProc',
            plugin_args={
                'dont_resubmit_completed_jobs': True,
                'n_procs': 40
            },
        )

    else:
        #workflow.run(plugin='SLURM',plugin_args={'dont_resubmit_completed_jobs': True,'max_jobs':128,'sbatch_args':'-p menon'})
        #workflow.run(plugin='Linear', plugin_args={'n_procs': 20, 'memory_gb' :32})
        #workflow.write_graph(graph2use='colored', dotfilename='/oak/stanford/groups/menon/projects/cdla/2019_dwi_mathfun/scripts/2019_dwi_pipeline_mathfun/graph_orig.dot')
        #workflow.run(plugin='MultiProc', plugin_args={'n_procs':16, 'memory_gb' :64})
        #workflow.run(plugin='SLURMGraph',plugin_args={'dont_resubmit_completed_jobs': True,'sbatch_args':' -p menon -c 4 --mem=16G -t 4:00:00'})
        workflow.run(plugin='SLURMGraph', plugin_args={
            'dont_resubmit_completed_jobs': True,
            'sbatch_args':
                '--mem=16G -t 6:00:00 --oversubscribe -n 2 '
                '--exclude=node[25-32] -c 2 ' +
                slurm_logs,
            'max_jobs': 20
        },)