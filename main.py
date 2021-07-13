from NGSIM2HighD import NGSIM2HighD
from HighD2NGSIM import HighD2NGSIM
import HighD_Columns as HC 
import NGSIM_Columns as NC 
# ngsim_dataset_dir =  "/Volumes/GoogleDrive/My Drive/Research/Masters Project (Human-likeness & IRL)/IRL Unpredictability approach/Datasets/NGSIM/CSVs/"#"../../Dataset/FNGSIM/Traj_data/"
# # ngsim_dataset_files = ['trajectories-0400-0415.csv',
# #             'trajectories-0500-0515.csv',
# #             'trajectories-0515-0530.csv',
# #             'trajectories-0750am-0805am.csv',
# #             'trajectories-0805am-0820am.csv',
# #             'trajectories-0820am-0835am.csv']
# ngsim_dataset_files = ['trajectories-0750am-0805am.csv',
#             'trajectories-0805am-0820am.csv',
#             'trajectories-0820am-0835am.csv']
# # ngsim_dataset_files = ['i80-1600-1615.txt'
# #                         'i80-1700-1715.txt'
# #                         'i80-1715-1730.txt'
# #                         'us101-0750-0805.txt'
# #                         'us101-0805-0820.txt'
# #                         'us101-0820-0835.txt']
#
#
#
#
#
# converter = NGSIM2HighD(ngsim_dataset_dir, ngsim_dataset_files)
# #converter.infer_lane_marking()
# converter.convert_tracks_info()
# converter.convert_meta_info()
# converter.convert_static_info()

highd_dir="/Volumes/GoogleDrive/My Drive/Research/Masters Project (Human-likeness & IRL)/IRL Unpredictability approach/Datasets/highD/highD-dataset-v1.0/data/"

converter = HighD2NGSIM(range(1,61), highd_dir)

converter.convert()