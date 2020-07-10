from NGSIM2HighD import NGSIM2HighD

ngsim_dataset_dir =  "../../Dataset/NGSIM/Traj_data/"
ngsim_dataset_files = ['trajectories-0400-0415.csv', 
            'trajectories-0500-0515.csv',
            'trajectories-0515-0530.csv',
            'trajectories-0750am-0805am.csv',
            'trajectories-0805am-0820am.csv',
            'trajectories-0820am-0835am.csv']
export_dir = "./exported_ngsim/"
converter = NGSIM2HighD(ngsim_dataset_dir, export_dir, ngsim_dataset_files)
converter.save_locations()
converter.load_locations()
converter.convert_tracks_info()