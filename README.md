
	Folder named semester_project contains all the necessary files, which are listed below:
•	data – folder that contains correspondences extracted using DenseMatching, folder models_all that contains trained models from supervised single-frame learning, folder models_multiframe with models trained with multiframe approach, as well as test data from both single frame supervised and multiframe approach in folders tests_all and tests_multiframe
•	data_backup is just a backup folder for data folder containing important results and models
•	deep-vslam is a folder where main codes are stored, and it has following structure:
	config folder contains .yaml configuration files. There, various hyperparams are stored and there are different .yaml files for different training modes
	src is a folder consisting of codes and that is the core of the work
	evaluate.py is a file which does the evaluation once the models are trained. It is controlled from one of the config files
	train_network.py is a main file which starts training, reads config file and runs training routines
•	deep-vslam_multiframe is a copy of the folder deep-vslam just in case of multiframe learning, where certain procedures and data processing is adapted to multiframe case. Inner structure of the folder stays the same
•	deep-vslam_multiframe_network is another copy of deep-vslam folder where instead of classic multiframe approach described in the report, poses are selected using deep network. The work on this part was cancelled due to deadline of the project
•	sbatch_multiframe and sbatch_tutorial contain .sh files for starting the training routine on cluster instead of local training
•	DenseMatching – in this folder, network which extracts correspondences from LIDAR and 2D images and store them in data is implemented, relevant file to take a look into is named extract_bundle_adjustment_data_simple/multiframe.py
•	Hierarchical-Localization is just a python library which is used to implement matching using SuperPoint and SuperGlue for extracting correspondences, this is not used in the project (ask Nikola for more details)

Training pipeline: 
outlier network - first train unsupervised, then load that model and train supervised
correction network - train  in supervised way

For succesful training, a few steps have to be executed:
1.	Set desired values of hyperparams in config files ( config files starting with train… and ending with kitti, make sure path to config file in train_network.py is correct): path to extracted data, network trained, weights of losses, type of losses, path to models (if you are loading some), path to save location for models, on which sequences to train/test…
2.	Run the .sh file corresponding to train_network.py file (make sure that you are providing proper path to .py file in .sh file)

For testing, procedure is similar but different config (test_opencv_kitti.yml) and .sh files (sbatch_test.sh) are used.

For extraction of correspondences (not necessary, since all data is already extracted, but in case it is needed) files are named sbatch_extract_correspondences_classic/superpoint/mutliframe.sh, depending on the features we extract.

General comments and more detailed comments of the code can be found in deep-vslam code folder, in deep-vslam_multiframe and deep-vslam_multiframe_network only the relevant changes with respect to basic deep-vslam are commented and explained 
