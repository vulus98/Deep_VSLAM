import os

from numpy.lib.npyio import save
import pykitti
import torch
import argparse
import imageio
from matplotlib import pyplot as plt
from utils_flow.pixel_wise_mapping import remap_using_flow_fields
import cv2
from model_selection import select_model
from utils_flow.util_optical_flow import flow_to_image
from utils_flow.visualization_utils import overlay_semantic_mask, replace_area
import numpy as np
import h5py


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


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
                                                     path_to_pre_trained_models=args.pre_trained_models_dir
                                                    )

        basedir = '/srv/beegfs02/scratch/deep_slam/data/data_sets/kitti/dataset/'

        sequence = '00'

        print('Started loading dataset.')

        dataset = pykitti.odometry(basedir, sequence)
        ds = zip(dataset.cam2, dataset.velo)
        imageFrame0, veloFrame0 = next(ds)
        transformCam2Velo = dataset.calib.T_cam2_velo
        intrinsicMatrix = dataset.calib.K_cam2

        numFeaturesMatching = 1024
        numFeaturesTotal = 5000
        keyframeRate = 5
        bundleSize = 10

        saveImages = False

        print("Started extracting data.")

        currentKeyframe = np.array(imageFrame0).astype('float32')
        sourceImage = torch.from_numpy(currentKeyframe).permute(2,0,1).unsqueeze(0).clone()
        
        velopadPoints = np.pad(veloFrame0[:, :3], (0, 1), 'constant', constant_values=(0, 1))
        p3dKeyframe = np.matmul(transformCam2Velo[:3], velopadPoints.T).T
        proj2d = np.matmul(intrinsicMatrix, p3dKeyframe.T).T
        maskFront = (proj2d[:, 2]>0).squeeze()
        proj2d = proj2d[maskFront]
        proj2d = proj2d[:, :2] / proj2d[:, 2:]
        maskFrame = (proj2d[:, 0]>=0) & (proj2d[:,0]<=currentKeyframe.shape[1]) & (proj2d[:, 1]>=0) & (proj2d[:, 1]<=currentKeyframe.shape[0])
        proj2d = proj2d[maskFrame].astype(int)
        sample = np.random.choice(proj2d.shape[0], numFeaturesTotal, replace=False)
        p2dKeyframe = proj2d[sample]
        p3dKeyframe = p3dKeyframe[maskFront][maskFrame][sample]
        p3d2dKeyframe = np.concatenate((p3dKeyframe, p2dKeyframe), axis=1)
        
        p3d2dForward = np.zeros((1, numFeaturesMatching, 5))
        p3d2dBackward = np.zeros((1, numFeaturesMatching, 5))

        p3dBundleCloud = np.zeros((1, bundleSize, numFeaturesMatching, 3))
        p2dBundleCloud = np.zeros((1, bundleSize, numFeaturesMatching, 4))
        p2dBundle = np.zeros((numFeaturesMatching, 4))

        if saveImages:
            currentKeyframeImage = np.copy(currentKeyframe)
            for k in range(p2dKeyframe.shape[0]):
                center = (p2dKeyframe[k, 0], p2dKeyframe[k, 1])
                radius = 5
                color = (255, 0, 0)
                thickness = 2
                currentKeyframeImage = cv2.circle(currentKeyframeImage, center, radius, color, thickness)

        framesBetweenKeyframesCurrentFrame = [currentKeyframe]
        framesBetweenKeyframesTargetImage = [sourceImage]

        bundleNum = 0
        keyframeNum = 0

        sample_bundle = np.random.choice(numFeaturesTotal, numFeaturesMatching, replace=False)

        p2dBundle[:, :2] = p2dKeyframe[sample_bundle]
        p2dBundle[:, 2] = np.zeros((numFeaturesMatching))
        p2dBundle[:, 3] = np.arange(numFeaturesMatching)
        p3dBundleCloud[bundleNum, keyframeNum%bundleSize, :, :] = p3dKeyframe[sample_bundle]

        for i, (image, velo) in enumerate(ds):
            currentFrame = np.array(image, copy=True).astype('float32')
            targetImage = torch.from_numpy(currentFrame).permute(2,0,1).unsqueeze(0).clone()  
            # estimated_flow = network.estimate_flow(target_image, source_image, device, mode='channel_first')
            estimatedFlow, uncertaintyComponents = network.estimate_flow_and_confidence_map(targetImage,
                                                                                    sourceImage,
                                                                                    mode='channel_first')
            warpedSourceImage = remap_using_flow_fields(currentFrame, estimatedFlow.squeeze()[0].cpu().numpy(),
                                                        estimatedFlow.squeeze()[1].cpu().numpy())
            dispX = estimatedFlow.squeeze()[0][p2dKeyframe[:, 1], p2dKeyframe[:, 0]].cpu().numpy()
            dispY = estimatedFlow.squeeze()[1][p2dKeyframe[:, 1], p2dKeyframe[:, 0]].cpu().numpy() 
            p2d = np.zeros((numFeaturesTotal, 2), dtype=int)
            p3d = np.zeros((numFeaturesTotal, 3))
            mapX = (p2dKeyframe[:, 0]+dispX).astype(np.float32)
            mapY = (p2dKeyframe[:, 1]+dispY).astype(np.float32)
            maskFrame = (mapX>=0) & (mapX<=currentFrame.shape[1]) & (mapY>=0) & (mapY<=currentFrame.shape[0])
            mapX = mapX[maskFrame].astype(int)
            mapY = mapY[maskFrame].astype(int)
            len = maskFrame.astype(int).sum()
            if len>=numFeaturesMatching:
                sample = np.random.choice(len, numFeaturesMatching, replace=False)
                p2d[0:numFeaturesMatching, 0] = mapX[sample]
                p2d[0:numFeaturesMatching, 1] = mapY[sample]
                p3d[0:numFeaturesMatching, :] = p3dKeyframe[maskFrame, :][sample, :]
            else:
                p2d[0:len, 0] = mapX
                p2d[0:len, 1] = mapY
                p3d[0:len, :] = p3dKeyframe[maskFrame, :]
                sample = np.random.randint(0, len, (numFeaturesMatching-len))
                p2d[len:numFeaturesMatching, :] = p2d[sample, :] 
                p3d[len:numFeaturesMatching, :] = p3d[sample, :]
            p3d2d = np.concatenate((p3d[:numFeaturesMatching, :], p2d[:numFeaturesMatching, :]), axis=1)
            p3d2dForward = np.concatenate((p3d2dForward, np.expand_dims(p3d2d, axis=0)))

            if saveImages:
                currentFrameImage = np.copy(currentFrame)
                for k in range(numFeaturesMatching):
                    center = (p2d[k, 0], p2d[k, 1])
                    radius = 5
                    color = (255, 0, 0)
                    thickness = 2
                    currentFrameImage = cv2.circle(currentFrameImage, center, radius, color, thickness)
                for k in range(p2dKeyframe.shape[0]):
                    center = (p2dKeyframe[k, 0], p2dKeyframe[k, 1])
                    radius = 5
                    color = (255, 0, 0)
                    thickness = 2
                    warpedSourceImage = cv2.circle(warpedSourceImage, center, radius, color, thickness)
                fig, (axis1, axis2, axis3) = plt.subplots(1, 3, figsize=(30, 30))
                axis1.imshow(currentKeyframeImage.astype('int'))
                axis1.set_title('Source image')
                axis3.imshow(currentFrameImage.astype('int'))
                axis3.set_title('Target image')
                axis2.imshow(warpedSourceImage.astype('int'))
                axis2.set_title('Warped source image according to estimated flow by GLU-Net')
                fig.savefig(os.path.join('evaluation/', 'Warped_source_image'+str(i+1)+'.png'),
                            bbox_inches='tight')
                plt.close(fig)

            if (i+1)%keyframeRate:
                framesBetweenKeyframesCurrentFrame.append(currentFrame)
                framesBetweenKeyframesTargetImage.append(targetImage)
            else:
                currentKeyframe = currentFrame
                sourceImage = targetImage

                velopadPoints = np.pad(velo[:, :3], (0, 1), 'constant', constant_values=(0, 1))
                p3dKeyframe = np.matmul(transformCam2Velo[:3], velopadPoints.T).T
                proj2d = np.matmul(intrinsicMatrix, p3dKeyframe.T).T
                maskFront = (proj2d[:, 2]>0).squeeze()
                proj2d = proj2d[maskFront]
                proj2d = proj2d[:, :2] / proj2d[:, 2:]
                maskFrame = (proj2d[:, 0]>=0) & (proj2d[:,0]<=currentKeyframe.shape[1]) & (proj2d[:, 1]>=0) & (proj2d[:, 1]<=currentKeyframe.shape[0])
                proj2d = proj2d[maskFrame].astype(int)
                sample = np.random.choice(proj2d.shape[0], numFeaturesTotal, replace=False)
                p2dKeyframe = proj2d[sample]
                p3dKeyframe = p3dKeyframe[maskFront][maskFrame][sample]
                p3d2dKeyframe = np.concatenate((p3dKeyframe, p2dKeyframe), axis=1)

                p2dBundleMatched = np.zeros((numFeaturesMatching, 4), dtype=int)
                dispX_bundle = estimatedFlow.squeeze()[0][p2dBundle[:, 1], p2dBundle[:, 0]].cpu().numpy()
                dispY_bundle = estimatedFlow.squeeze()[1][p2dBundle[:, 1], p2dBundle[:, 0]].cpu().numpy() 
                mapX = (p2dBundle[:, 0]+dispX_bundle).astype(np.float32)
                mapY = (p2dBundle[:, 1]+dispY_bundle).astype(np.float32)
                maskFrame = (mapX>=0) & (mapX<=currentFrame.shape[1]) & (mapY>=0) & (mapY<=currentFrame.shape[0])
                mapX = mapX[maskFrame].astype(int)
                mapY = mapY[maskFrame].astype(int)
                len = maskFrame.astype(int).sum()
                if len>=numFeaturesMatching:
                    sample = np.random.choice(len, numFeaturesMatching, replace=False)
                    p2dBundleMatched[0:numFeaturesMatching, 0] = mapX[sample]
                    p2dBundleMatched[0:numFeaturesMatching, 1] = mapY[sample]
                    p2dBundleMatched[0:numFeaturesMatching, 2] = p2dBundle[maskFrame, :][sample, 2]
                    p2dBundleMatched[0:numFeaturesMatching, 3] = p2dBundle[maskFrame, :][sample, 3]                   
                else:
                    p2dBundleMatched[0:len, 0] = mapX
                    p2dBundleMatched[0:len, 1] = mapY
                    p2dBundleMatched[0:len, 2] = p2dBundle[maskFrame, 2]
                    p2dBundleMatched[0:len, 3] = p2dBundle[maskFrame, 3]                    
                    sample = np.random.randint(0, len, (numFeaturesMatching-len))
                    p2dBundleMatched[len:numFeaturesMatching, :] = p2dBundleMatched[sample, :] 

                p2dBundleCloud[bundleNum, keyframeNum, :, :] = p2dBundleMatched
                
                if saveImages:
                    currentKeyframeImage_bundle = np.copy(currentKeyframe)
                    colors = [(255,0,0), (0,255,0), (0,0,255)]
                    for k in range(p2dBundleMatched.shape[0]):
                        center = (p2dBundleMatched[k, 0], p2dBundleMatched[k, 1])
                        radius = 5
                        color = colors[int(p2dBundleMatched[k,2])]
                        thickness = 2
                        currentKeyframeImage_bundle = cv2.circle(currentKeyframeImage_bundle, center, radius, color, thickness)
                    fig = plt.figure(figsize=(30, 30))
                    plt.imshow(currentKeyframeImage_bundle.astype('int'))
                    plt.title('Image')
                    plt.savefig(os.path.join('evaluation/', 'bundle_'+str(bundleNum)+'_frame_'+str(keyframeNum)+'.png'),
                                bbox_inches='tight')
                    plt.close(fig)

                keyframeNum = (keyframeNum+1)%bundleSize
                if keyframeNum==0:
                    sample_bundle = np.random.choice(numFeaturesTotal, numFeaturesMatching, replace=False)
                    bundleNum = bundleNum + 1
                    p3dBundleCloud = np.concatenate((p3dBundleCloud, np.zeros((1, bundleSize, numFeaturesMatching, 3))))
                    p3dBundleCloud[bundleNum, keyframeNum, :, :] = p3dKeyframe[sample_bundle]
                    p2dBundleCloud = np.concatenate((p2dBundleCloud, np.zeros((1, bundleSize, numFeaturesMatching, 4))))
                    p2dBundle[:, :2] = p2dKeyframe[sample_bundle]
                    p2dBundle[:, 2] = np.zeros((numFeaturesMatching))
                    p2dBundle[:, 3] = np.arange(numFeaturesMatching)
                else:
                    sample_bundle = np.random.choice(numFeaturesTotal, numFeaturesMatching, replace=False)
                    p3dBundleCloud[bundleNum, keyframeNum, :, :] = p3dKeyframe[sample_bundle]
                    sample_cloud = np.random.choice(numFeaturesMatching, numFeaturesMatching-numFeaturesMatching//2, replace=False)
                    sample = np.random.choice(numFeaturesMatching, numFeaturesMatching//2, replace=False)
                    p2dBundle[:numFeaturesMatching//2, :] = p2dBundleMatched[sample]
                    p2dBundle[numFeaturesMatching//2:, :2] = p2dKeyframe[sample_bundle][sample_cloud, :]
                    p2dBundle[numFeaturesMatching//2:, 2] = keyframeNum * np.ones((numFeaturesMatching-numFeaturesMatching//2))
                    p2dBundle[numFeaturesMatching//2:, 3] = sample_cloud

                if saveImages:
                    currentKeyframeImage = np.copy(currentKeyframe)
                    for k in range(p2dKeyframe.shape[0]):
                        center = (p2dKeyframe[k, 0], p2dKeyframe[k, 1])
                        radius = 5
                        color = (255, 0, 0)
                        thickness = 2
                        currentKeyframeImage = cv2.circle(currentKeyframeImage, center, radius, color, thickness)

                for j, (currentFrame, targetImage) in enumerate(zip(framesBetweenKeyframesCurrentFrame, framesBetweenKeyframesTargetImage)):
                    estimatedFlow, uncertaintyComponents = network.estimate_flow_and_confidence_map(targetImage,
                                                                                            sourceImage,
                                                                                            mode='channel_first')
                    warpedSourceImage = remap_using_flow_fields(currentFrame, estimatedFlow.squeeze()[0].cpu().numpy(),
                                                                estimatedFlow.squeeze()[1].cpu().numpy())
                    dispX = estimatedFlow.squeeze()[0][p2dKeyframe[:, 1], p2dKeyframe[:, 0]].cpu().numpy()
                    dispY = estimatedFlow.squeeze()[1][p2dKeyframe[:, 1], p2dKeyframe[:, 0]].cpu().numpy() 
                    p2d = np.zeros((numFeaturesTotal, 2), dtype=int)
                    p3d = np.zeros((numFeaturesTotal, 3))
                    mapX = (p2dKeyframe[:, 0]+dispX).astype(np.float32)
                    mapY = (p2dKeyframe[:, 1]+dispY).astype(np.float32)
                    maskFrame = (mapX>=0) & (mapX<=currentFrame.shape[1]) & (mapY>=0) & (mapY<=currentFrame.shape[0])
                    mapX = mapX[maskFrame].astype(int)
                    mapY = mapY[maskFrame].astype(int)
                    len = maskFrame.astype(int).sum()
                    if len>=numFeaturesMatching:
                        sample = np.random.choice(len, numFeaturesMatching, replace=False)
                        p2d[0:numFeaturesMatching, 0] = mapX[sample]
                        p2d[0:numFeaturesMatching, 1] = mapY[sample]
                        p3d[0:numFeaturesMatching, :] = p3dKeyframe[maskFrame, :][sample, :]
                    else:
                        p2d[0:len, 0] = mapX
                        p2d[0:len, 1] = mapY
                        p3d[0:len, :] = p3dKeyframe[maskFrame, :]
                        sample = np.random.randint(0, len, (numFeaturesMatching-len))
                        p2d[len:numFeaturesMatching, :] = p2d[sample, :] 
                        p3d[len:numFeaturesMatching, :] = p3d[sample, :]
                    p3d2d = np.concatenate((p3d[:numFeaturesMatching,:], p2d[:numFeaturesMatching,:]), axis=1)
                    p3d2dBackward = np.concatenate((p3d2dBackward, np.expand_dims(p3d2d, axis=0)))

                    if saveImages:
                        currentFrameImage = np.copy(currentFrame)
                        for k in range(p2d.shape[0]):
                            center = (p2d[k, 0], p2d[k, 1])
                            radius = 5
                            color = (255, 0, 0)
                            thickness = 2
                            currentFrameImage = cv2.circle(currentFrameImage, center, radius, color, thickness)
                        for k in range(p2dKeyframe.shape[0]):
                            center = (p2dKeyframe[k, 0], p2dKeyframe[k, 1])
                            radius = 5
                            color = (255, 0, 0)
                            thickness = 2
                            warpedSourceImage = cv2.circle(warpedSourceImage, center, radius, color, thickness)
                        fig, (axis1, axis2, axis3) = plt.subplots(1, 3, figsize=(30, 30))
                        axis1.imshow(currentKeyframeImage.astype('int'))
                        axis1.set_title('Source image')
                        axis3.imshow(currentFrameImage.astype('int'))
                        axis3.set_title('Target image')
                        axis2.imshow(warpedSourceImage.astype('int'))
                        axis2.set_title('Warped source image according to estimated flow by GLU-Net')
                        fig.savefig(os.path.join('evaluation/', 'Warped_source_image'+str(i+1)+str(i-j)+'.png'),
                                    bbox_inches='tight')
                        plt.close(fig)
                framesBetweenKeyframesCurrentFrame = [currentKeyframe]
                framesBetweenKeyframesTargetImage = [sourceImage]

        print("Finished extracting data.")

        hf = h5py.File('/srv/beegfs02/scratch/deep_slam/data/data_sets/kitti/extracted_data/PDC_correspondences_intrinsics_cam2/extracted_features_full_sequence_00_bundle_size_10.h5', 'w')

        print("Started writing data.")

        group = hf.create_group(sequence)

        # group.create_dataset('image_features', data=image_features_dataset)
        group.create_dataset('p3d2d', data=p3d2dForward)
        group.create_dataset('p3d2dBackward', data=p3d2dBackward)
        group.create_dataset('p3dBundleCloud', data=p3dBundleCloud)
        group.create_dataset('p2dBundleCloud', data=p2dBundleCloud)


        hf.close()

        print("Finished writing data.")
