import pykitti
import torch
import argparse
from model_selection import select_model
import numpy as np
import h5py


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def _project_velo_to_camera_frame(velo, transformCam2Velo, intrinsicMatrix, frameShape):
        # Get LIDAR point, discard the last component that has to do with light intensity (I think)
        velopadPoints = np.pad(velo[:, :3], (0, 1), 'constant', constant_values=(0, 1))
        # Transform from LIDAR coordiate frame to the camera coordinate frame
        p3d = np.matmul(transformCam2Velo[:3], velopadPoints.T).T
        # Project from 3D onto camera sensor
        p2d = np.matmul(intrinsicMatrix, p3d.T).T
        # Remove all points behind the camera
        maskFront = (p2d[:, 2]>0).squeeze()
        p2d = p2d[maskFront]
        p3d = p3d[maskFront]
        # Scaling
        p2d = p2d[:, :2] / p2d[:, 2:]
        # Remove all points not falling into the camera sensor
        maskFrame = (p2d[:, 0]>=0) & (p2d[:,0]<=frameShape[1]) & (p2d[:, 1]>=0) & (p2d[:, 1]<=frameShape[0])
        p2d = p2d[maskFrame].astype(int)
        p3d = p3d[maskFrame]

        p3d2d = np.concatenate((p3d, p2d), axis=1)

        return p3d2d


def _find_new_feature_positions(p2dSource, disparity, targetFrameShape):
    # Calculate the x,y disparity that needs to be applied to the source in order to get 
    # the matched feature locations in the target
    dispX = disparity.squeeze()[0][p2dSource[:, 1], p2dSource[:, 0]].cpu().numpy()
    dispY = disparity.squeeze()[1][p2dSource[:, 1], p2dSource[:, 0]].cpu().numpy() 
    p2dTarget_x = (p2dSource[:, 0]+dispX).astype(np.float32)
    p2dTarget_y = (p2dSource[:, 1]+dispY).astype(np.float32)
    # Remove matched features that went out of the target frame
    mask = (p2dTarget_x>=0) & (p2dTarget_x<=targetFrameShape[1]) & (p2dTarget_y>=0) & (p2dTarget_y<=targetFrameShape[0])
    p2dTarget = np.concatenate((p2dTarget_x[mask, np.newaxis].astype(int), p2dTarget_y[mask, np.newaxis].astype(int)), axis=1)

    return p2dTarget, mask


def _subsample_or_duplicate(tensor, numP3d2dToStore, dim=0):
    len = tensor.shape[dim]
    # If more subsample
    if len>=numP3d2dToStore:
        mask = np.random.choice(len, numP3d2dToStore, replace=False)
        tensor = np.take(tensor, indices=mask, axis=dim)
    # Otherwise duplicatte
    else:
        mask = np.random.randint(0, len, (numP3d2dToStore-len))
        tensor = np.concatenate((tensor, np.take(tensor, indices=mask, axis=dim)), axis=0)
    return tensor, mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test models on a pair of images')
    parser.add_argument('--model', default='PDCNet', type=str, help='Model to use')
    parser.add_argument('--pre_trained_model', default='megadepth', type=str, help='Name of the pre-trained-model.')

    parser.add_argument('--optim_iter', type=int, default=3,
                        help='Number of optim iter for Global GOCor, if applicable')
    parser.add_argument('--local_optim_iter', dest='local_optim_iter', default=7,
                        help='Number of optim iter for Local GOCor, if applicable')

    # TODO Check out
    parser.add_argument('--flipping_condition', dest='flipping_condition',  default=False, type=boolean_string,
                        help='Apply flipping condition for semantic data and GLU-Net-based networks ? ')
    parser.add_argument('--pre_trained_models_dir', type=str, default='pre_trained_models/',
                        help='Directory containing the pre-trained-models.')

    subprasers = parser.add_subparsers(dest='network_type')
    PDCNet = subprasers.add_parser('PDCNet', help='inference parameters for PDCNet')
    PDCNet.add_argument(
        '--confidence_map_R', default=1.0, type=float,
        help='R used for confidence map computation',
    )
    PDCNet.add_argument(
        '--multi_stage_type', default='direct', type=str, choices=['direct', 'homography_from_last_level_uncertainty',
                                                                   'homography_from_quarter_resolution_uncertainty',
                                                                   'multiscale_homo_from_quarter_resolution_uncertainty'],
        help='multi stage type',
    )
    PDCNet.add_argument(
        '--ransac_thresh', default=1.0, type=float,
        help='ransac threshold used for multi-stages alignment',
    )
    PDCNet.add_argument(
        '--mask_type', default='proba_interval_1_above_5', type=str,
        help='mask computation for multi-stage alignment',
    )
    PDCNet.add_argument(
        '--homography_visibility_mask', default=True, type=boolean_string,
        help='apply homography visibility mask for multi-stage computation ?',
    )
    PDCNet.add_argument('--scaling_factors', type=float, nargs='+', default=[0.5, 0.6, 0.88, 1, 1.33, 1.66, 2],
                        help='scaling factors')
    args = parser.parse_args()

    torch.cuda.empty_cache()
    torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance
    torch.backends.cudnn.enabled = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # either gpu or cpu
    local_optim_iter = int(args.local_optim_iter)
   
    with torch.no_grad():
        network, estimate_uncertainty = select_model(
                                                     args.model, args.pre_trained_model, args, args.optim_iter, local_optim_iter,
                                                     path_to_pre_trained_models='/srv/beegfs02/scratch/deep_slam/data/vukasin/semester_project/DenseMatching/pre_trained_models/'
                                                    )

        basedir = '/srv/beegfs02/scratch/deep_slam/data/data_sets/kitti/dataset/'
        sequences = ['00']
        hf = h5py.File(f'/srv/beegfs02/scratch/deep_slam/data/vukasin/semester_project/data/correspondences/extracted_features_sequences_00.6_new_format_TMP.h5', 'w')
        
        numP3d2dToStore = 4096
        keyframeRate = 6
        for sequence in sequences:
            print('Started loading dataset.')

            dataset = pykitti.odometry(basedir, sequence)
            ds = zip(dataset.cam2, dataset.velo)
            imageFrame0, veloFrame0 = next(ds)
            transformCam2Velo = dataset.calib.T_cam2_velo
            intrinsicMatrix = dataset.calib.K_cam2
            keyframeNum = 0
                    
            # Matched frames with the previous keyframe in a forward/backward direction
            p3d2dMatchedForwardSave = np.zeros((1, numP3d2dToStore, 5))  
            print("Started extracting data.")

            # This variable holds the last keyframe.
            # Every next frame gets matched to it.
            sourceKeyFrame = np.array(imageFrame0).astype('float32') 
            sourceKeyFrame = torch.from_numpy(sourceKeyFrame).permute(2,0,1).unsqueeze(0).clone() 
 
            
            # All 3d2d points of the current keyframe
            p3d2dSourceKeyframe = _project_velo_to_camera_frame(
                velo=veloFrame0, 
                transformCam2Velo=transformCam2Velo, 
                intrinsicMatrix=intrinsicMatrix, 
                frameShape=sourceKeyFrame.shape[2:4]
            )
            # Choose a random subset of p2d3d to save and match with next keyframe
            p3d2dSourceKeyframe, maskRnd = _subsample_or_duplicate(p3d2dSourceKeyframe, numP3d2dToStore)

            for i, (image, velo) in enumerate(ds):
                # Load next frame
                targetFrame = np.array(image, copy=True).astype('float32') 
                targetFrame = torch.from_numpy(targetFrame).permute(2,0,1).unsqueeze(0).clone()  

                # TODO This function requires first source and then targer !!!!!
                # Calculate flow from the source to the target image
                estimatedFlow, uncertaintyComponents = network.estimate_flow_and_confidence_map(
                    targetFrame,
                    sourceKeyFrame,
                    mode='channel_first'
                )

                # Match target frames 2d points to the source frames 3d2d points
                p2dCurrentlyMatched, maskMatching = _find_new_feature_positions(
                    p2dSource=p3d2dSourceKeyframe[:, 3:5], 
                    disparity=estimatedFlow, 
                    targetFrameShape=targetFrame.shape[2:4]
                )
                p3dCurrentlyMatched = p3d2dSourceKeyframe[:, :3][maskMatching, :]
                p3d2dCurrentlyMatched = np.concatenate((p3dCurrentlyMatched, p2dCurrentlyMatched), axis=1)
            
                # subsample/duplicate and save for current frame
                p3d2dCurrentlyMatched, _ = _subsample_or_duplicate(p3d2dCurrentlyMatched, numP3d2dToStore, dim=0)
                p3d2dMatchedForwardSave = np.concatenate((p3d2dMatchedForwardSave, np.expand_dims(p3d2dCurrentlyMatched, axis=0)))

                if not (i+1)%keyframeRate:
                    # This new keyframe now becomes the source keyframe
                    sourceKeyFrame = targetFrame
                    keyframeNum += 1

                    print(f'Processing keyframe {keyframeNum}')

                    # All 3d2d points of the current keyframe
                    p3d2dSourceKeyframe = _project_velo_to_camera_frame(
                        velo=velo, 
                        transformCam2Velo=transformCam2Velo, 
                        intrinsicMatrix=intrinsicMatrix, 
                        frameShape=sourceKeyFrame.shape[2:4]
                    )

                    # CHoose p3d2d to save and match with next keyframe
                    p3d2dSourceKeyframe, maskRnd = _subsample_or_duplicate(p3d2dSourceKeyframe, numP3d2dToStore)

            print("Finished extracting data.")
            print("Started writing data.")

            group = hf.create_group(sequence)
            group.create_dataset('p3d2dMatchedForward', data=p3d2dMatchedForwardSave)
            group.create_dataset('calibration_matrix', data=dataset.calib.K_cam2)
            if(int(sequence)<11):
                group.create_dataset('poses', data=dataset.poses)
            print("Finished writing data.")  
        hf.close()
          

