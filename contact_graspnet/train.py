from genericpath import exists
import os
import sys
import argparse
from datetime import datetime
import numpy as np
import time
from tqdm import tqdm

CONTACT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR))
sys.path.append(os.path.join(ROOT_DIR))
sys.path.append(os.path.join(BASE_DIR, 'pointnet2',  'models'))
sys.path.append(os.path.join(BASE_DIR, 'pointnet2',  'utils'))

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
TF2 = True
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    
import config_utils
from data import PointCloudReader, load_scene_contacts, center_pc_convert_cam
from summaries import build_summary_ops, build_file_writers
from tf_train_ops import load_labels_and_losses, build_train_op
from contact_grasp_estimator import GraspEstimator
from graspnet1b_data import GraspNet1BReader


def train(
    global_config,
    log_dir,
    resume=False,
    max_train_batches=None,
    scene_limit=None,
    debug_numerics=False,
):
    """
    Trains Contact-GraspNet

    Arguments:
        global_config {dict} -- config dict
        log_dir {str} -- Checkpoint directory
    """

    dataset_type = global_config['DATA'].get('dataset_type', 'acronym')
    if dataset_type == 'graspnet':
        if global_config['DATA'].get('train_and_test', False):
            raise ValueError('GraspNet training requires a held-out validation split; DATA.train_and_test must be false')
        num_test_samples = global_config['DATA']['num_test_scenes']
        pcreader = GraspNet1BReader(
            global_config['DATA']['data_path'],
            camera=global_config['DATA'].get('camera', 'kinect'),
            batch_size=global_config['OPTIMIZER']['batch_size'],
            num_point=global_config['DATA']['num_point'],
            num_validation_scenes=num_test_samples,
            pc_augm_config=global_config['DATA']['pc_augm'],
            depth_augm_config=global_config['DATA']['depth_augm'],
            scene_limit=scene_limit,
        )
        contact_infos = pcreader.contact_infos
        num_train_samples = pcreader._num_train_samples
        num_test_samples = pcreader._num_test_samples
    elif dataset_type == 'acronym':
        contact_infos = load_scene_contacts(
            global_config['DATA']['data_path'],
            scene_contacts_path=global_config['DATA']['scene_contacts_path'],
            excluded_scenes_file=global_config['DATA'].get('excluded_scenes_file', None),
            max_scenes=scene_limit,
        )
        num_test_samples = global_config['DATA']['num_test_scenes']
        num_train_samples = len(contact_infos) - num_test_samples
    else:
        raise ValueError(f"Unsupported training dataset type: {dataset_type}")

    if num_train_samples < 1:
        raise ValueError(
            f"Training requires at least one training scene; got {len(contact_infos)} total and "
            f"{num_test_samples} validation scenes."
        )
        
    print('using %s meshes' % (num_train_samples + num_test_samples))
    if dataset_type == 'acronym' and 'train_and_test' in global_config['DATA'] and global_config['DATA']['train_and_test']:
        num_train_samples = num_train_samples + num_test_samples
        num_test_samples = 0
        print('using train and test data')

    if dataset_type == 'acronym':
        pcreader = PointCloudReader(
            root_folder=global_config['DATA']['data_path'],
            batch_size=global_config['OPTIMIZER']['batch_size'],
            estimate_normals=global_config['DATA']['input_normals'],
            raw_num_points=global_config['DATA']['num_point'],
            use_uniform_quaternions = global_config['DATA']['use_uniform_quaternions'],
            scene_obj_scales = [c['obj_scales'] for c in contact_infos],
            scene_obj_paths = [c['obj_paths'] for c in contact_infos],
            scene_obj_transforms = [c['obj_transforms'] for c in contact_infos],
            num_train_samples = num_train_samples,
            num_test_samples = num_test_samples,
            use_farthest_point = global_config['DATA']['use_farthest_point'],
            intrinsics=global_config['DATA']['intrinsics'],
            elevation=global_config['DATA']['view_sphere']['elevation'],
            distance_range=global_config['DATA']['view_sphere']['distance_range'],
            pc_augm_config=global_config['DATA']['pc_augm'],
            depth_augm_config=global_config['DATA']['depth_augm']
        )
        pcreader.contact_infos = contact_infos

    with tf.Graph().as_default():
        
        # Build the model
        grasp_estimator = GraspEstimator(global_config)
        ops = grasp_estimator.build_network()
        
        # contact_tensors = load_contact_grasps(contact_infos, global_config['DATA'])
        
        loss_ops = load_labels_and_losses(grasp_estimator, contact_infos, global_config)

        ops.update(loss_ops)
        ops['train_op'] = build_train_op(ops['loss'], ops['step'], global_config)
        if debug_numerics:
            debug_tensors = {
                'input point cloud': grasp_estimator.placeholders['pointclouds_pl'],
                'total loss': ops['loss'],
                'direction loss': ops['dir_loss'],
                'confidence loss': ops['bin_ce_loss'],
                'offset loss': ops['offset_loss'],
                'approach loss': ops['approach_loss'],
                'adds loss': ops['adds_loss'],
                'adds gt-to-pred loss': ops['adds_gt2pred_loss'],
            }
            ops['debug_numerics'] = {
                name: tf.debugging.check_numerics(
                    tensor, 'Non-finite {}'.format(name)
                )
                for name, tensor in debug_tensors.items()
            }

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver(save_relative_paths=True, keep_checkpoint_every_n_hours=4)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)

        # Log summaries
        summary_ops = build_summary_ops(ops, sess, global_config)

        # Init/Load weights
        weights_mode = 'train_scratch' if dataset_type == 'graspnet' and not resume else 'train'
        grasp_estimator.load_weights(sess, saver, log_dir, mode=weights_mode)

        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        file_writers = build_file_writers(sess, log_dir)

    ## define: epoch = arbitrary number of views of every training scene
    cur_epoch = sess.run(ops['step']) // num_train_samples
    for epoch in range(cur_epoch, global_config['OPTIMIZER']['max_epoch']):
        log_string('**** EPOCH %03d ****' % (epoch))
        
        sess.run(ops['iterator'].initializer)
        epoch_time = time.time()
        step = train_one_epoch(
            sess, ops, summary_ops, file_writers, pcreader,
            max_batches=max_train_batches, debug_numerics=debug_numerics,
        )
        log_string('trained epoch {} in: {}'.format(epoch, time.time()-epoch_time))

        # Save the variables to disk.
        save_path = saver.save(sess, os.path.join(log_dir, "model.ckpt"), global_step=step, write_meta_graph=False)
        log_string("Model saved in file: %s" % save_path)

        if num_test_samples > 0:
            eval_time = time.time()
            eval_validation_scenes(sess, ops, summary_ops, file_writers, pcreader)
            log_string('evaluation time: {}'.format(time.time()-eval_time))

def train_one_epoch(
    sess,
    ops,
    summary_ops,
    file_writers,
    pcreader,
    max_batches=None,
    debug_numerics=False,
):
    """ ops: dict mapping from string to tf ops """
    
    log_string(str(datetime.now()))
    loss_log = np.zeros((10,7))
    get_time = time.time()
    
    num_batches = pcreader._num_train_samples
    if max_batches is not None:
        num_batches = min(num_batches, max_batches)
    for batch_idx in range(num_batches):

        batch_data, cam_poses, scene_idx = pcreader.get_scene_batch(scene_idx=batch_idx)
        
        if not getattr(pcreader, 'outputs_centered_opencv', False):
            cam_poses, batch_data = center_pc_convert_cam(cam_poses, batch_data)
        
        feed_dict = {ops['pointclouds_pl']: batch_data, ops['cam_poses_pl']: cam_poses,
                     ops['scene_idx_pl']: scene_idx, ops['is_training_pl']: True}

        sample_id = 'index={}'.format(scene_idx)
        if debug_numerics:
            if hasattr(pcreader, 'contact_infos'):
                contact_info = pcreader.contact_infos[scene_idx]
                if 'scene_id' in contact_info and 'ann_id' in contact_info:
                    sample_id = 'scene_id={}, ann_id={}'.format(
                        int(contact_info['scene_id']), int(contact_info['ann_id'])
                    )
            try:
                sess.run(list(ops['debug_numerics'].values()), feed_dict=feed_dict)
            except tf.errors.InvalidArgumentError as exc:
                error_text = str(exc)
                failed_checks = [
                    name for name in ops['debug_numerics']
                    if 'Non-finite {}'.format(name) in error_text
                ]
                check_name = ', '.join(failed_checks) or 'input/loss check'
                print(
                    'Numerics check failed for {} on {} (batch index {}): {}'.format(
                        check_name, sample_id, scene_idx, exc
                    ),
                    flush=True,
                )
                raise

        try:
            step, summary, _, loss_val, dir_loss, bin_ce_loss, \
            offset_loss, approach_loss, adds_loss, adds_gt2pred_loss, scene_idx = sess.run(
                [
                    ops['step'], summary_ops['merged'], ops['train_op'], ops['loss'],
                    ops['dir_loss'], ops['bin_ce_loss'], ops['offset_loss'],
                    ops['approach_loss'], ops['adds_loss'],
                    ops['adds_gt2pred_loss'], ops['scene_idx'],
                ],
                feed_dict=feed_dict,
            )
        except tf.errors.InvalidArgumentError:
            if debug_numerics:
                print(
                    'Training step failed on {} (batch index {}); '
                    'input/loss checks passed, inspect the gradient failure below.'
                    .format(sample_id, scene_idx),
                    flush=True,
                )
            raise
        assert scene_idx[0] == scene_idx     
        
        loss_log[batch_idx%10,:] = loss_val, dir_loss, bin_ce_loss, offset_loss, approach_loss, adds_loss, adds_gt2pred_loss
        
        if (batch_idx+1)%10 == 0:
            file_writers['train_writer'].add_summary(summary, step)
            f = tuple(np.mean(loss_log, axis=0)) + ((time.time() - get_time) / 10., )
            log_string('total loss: %f \t dir loss: %f \t ce loss: %f \t off loss: %f \t app loss: %f adds loss: %f \t adds_gt2pred loss: %f \t batch time: %f' % f)
            get_time = time.time()
            
    return sess.run(ops['step'])

def eval_validation_scenes(sess, ops, summary_ops, file_writers, pcreader, max_eval_objects=500):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    log_string(str(datetime.now()))
    loss_log = np.zeros((min(pcreader._num_test_samples, max_eval_objects),7))

    # resets accumulation of pr and auc data
    sess.run(summary_ops['pr_reset_op'])
    
    for batch_idx in np.arange(min(pcreader._num_test_samples, max_eval_objects)):

        batch_data, cam_poses, scene_idx = pcreader.get_scene_batch(scene_idx=pcreader._num_train_samples + batch_idx)

        if not getattr(pcreader, 'outputs_centered_opencv', False):
            cam_poses, batch_data = center_pc_convert_cam(cam_poses, batch_data)

        feed_dict = {ops['pointclouds_pl']: batch_data, ops['cam_poses_pl']: cam_poses,
                     ops['scene_idx_pl']: scene_idx, ops['is_training_pl']: False}

        scene_idx, step, loss_val, dir_loss, bin_ce_loss, offset_loss, approach_loss, adds_loss, adds_gt2pred_loss, pr_summary,_,_,_ = sess.run([ops['scene_idx'], ops['step'], ops['loss'], ops['dir_loss'], ops['bin_ce_loss'],
                                                                                                        ops['offset_loss'], ops['approach_loss'], ops['adds_loss'], ops['adds_gt2pred_loss'],
                                                                                                        summary_ops['merged_eval'], summary_ops['pr_update_op'], 
                                                                                                        summary_ops['auc_update_op']] + [summary_ops['acc_update_ops']], feed_dict=feed_dict)
        assert scene_idx[0] == (pcreader._num_train_samples + batch_idx)
        
        loss_log[batch_idx,:] = loss_val, dir_loss, bin_ce_loss, offset_loss, approach_loss, adds_loss, adds_gt2pred_loss

    file_writers['test_writer'].add_summary(pr_summary, step)
    f = tuple(np.mean(loss_log, axis=0))
    log_string('mean val loss: %f \t mean val dir loss: %f \t mean val ce loss: %f \t mean off loss: %f \t mean app loss: %f \t mean adds loss: %f \t mean adds_gt2pred loss: %f' % f)

    return step

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', default=None, help='Checkpoint dir')
    parser.add_argument('--data_path', type=str, default=None, help='Grasp data root dir')
    parser.add_argument('--max_epoch', type=int, default=None, help='Epochs to run')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch Size during training')
    parser.add_argument('--arg_configs', nargs="*", type=str, default=[], help='overwrite config parameters')
    parser.add_argument('--dataset_type', choices=('acronym', 'graspnet'), default='acronym',
                        help='Training data format')
    parser.add_argument('--validation_scenes', type=int, default=None,
                        help='Number of training scenes reserved for validation')
    parser.add_argument('--config', type=str, default=None,
                        help='Config file to use for a new run (checkpoint config takes precedence on resume)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume an existing GraspNet run instead of initializing from scratch')
    parser.add_argument('--smoke_test', action='store_true',
                        help='Run one epoch on two GraspNet training scenes and validate on one held-out scene')
    parser.add_argument('--debug_numerics', action='store_true',
                        help='Check input batches and individual losses for NaNs/Infs before each train step')
    parser.add_argument('--max_scenes', type=int, default=None,
                        help='Maximum number of scenes to load (default: None for all)')
    FLAGS = parser.parse_args()

    if FLAGS.dataset_type == 'graspnet':
        gn_root = os.environ.get('GN_ROOT')
        FLAGS.data_path = FLAGS.data_path or (
            os.path.join(gn_root, 'cgn_preprocessed') if gn_root else None
        )
        if FLAGS.data_path is None:
            parser.error('Set GN_ROOT or pass --data_path to the preprocessed GraspNet directory')
        FLAGS.ckpt_dir = FLAGS.ckpt_dir or 'checkpoints/graspnet_1b_from_scratch'
        FLAGS.config = FLAGS.config or os.path.join(CONTACT_DIR, 'config_graspnet1b.yaml')
        if FLAGS.smoke_test:
            FLAGS.max_epoch = 1
            FLAGS.validation_scenes = 1
        validation_scenes = FLAGS.validation_scenes if FLAGS.validation_scenes is not None else 10
    else:
        FLAGS.ckpt_dir = FLAGS.ckpt_dir or 'checkpoints/contact_graspnet'
        validation_scenes = FLAGS.validation_scenes

    ckpt_dir = FLAGS.ckpt_dir
    if not os.path.exists(ckpt_dir): 
        if not os.path.exists(os.path.dirname(ckpt_dir)):
            ckpt_dir = os.path.join(BASE_DIR, ckpt_dir)
        os.makedirs(ckpt_dir, exist_ok=True)

    if FLAGS.smoke_test and FLAGS.dataset_type == 'graspnet':
        scene_limit = 2
        max_train_batches = None
    elif FLAGS.max_scenes is not None:
        scene_limit = FLAGS.max_scenes
        max_train_batches = None
    else:
        scene_limit = None
        max_train_batches = None
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.system('cp {} {}'.format(os.path.join(CONTACT_DIR, 'contact_graspnet.py'), ckpt_dir)) # bkp of model def
    os.system('cp {} {}'.format(os.path.join(CONTACT_DIR, 'train.py'), ckpt_dir)) # bkp of train procedure

    LOG_FOUT = open(os.path.join(ckpt_dir, 'log_train.txt'), 'a' if FLAGS.resume else 'w')
    LOG_FOUT.write(str(FLAGS)+'\n')
    def log_string(out_str):
        LOG_FOUT.write(out_str+'\n')
        LOG_FOUT.flush()
        print(out_str)

    arg_configs = list(FLAGS.arg_configs)
    arg_configs.append(f'DATA.dataset_type:{FLAGS.dataset_type}')
    if validation_scenes is not None:
        arg_configs.append(f'DATA.num_test_scenes:{validation_scenes}')
    global_config = config_utils.load_config(
        ckpt_dir, batch_size=FLAGS.batch_size, max_epoch=FLAGS.max_epoch,
        data_path=FLAGS.data_path, arg_configs=arg_configs, save=True,
        config_path=FLAGS.config
    )
    
    log_string(str(global_config))
    log_string('pid: %s'%(str(os.getpid())))

    train(
        global_config,
        ckpt_dir,
        resume=FLAGS.resume,
        max_train_batches=max_train_batches,
        scene_limit=scene_limit,
        debug_numerics=FLAGS.debug_numerics,
    )

    LOG_FOUT.close()
