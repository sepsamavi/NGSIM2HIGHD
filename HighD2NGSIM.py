import os
import pandas as pd
import numpy as np
import HighD_Columns as HC
import NGSIM_Columns as NC
import NGSIM_MetaInfo as NMeta
NC_LIST =[NC.ID,  NC.FRAME,  NC.TOTAL_FRAME,  NC.GLOBAL_TIME,  NC.X,  NC.Y,  NC.GLOBAL_X,  NC.GLOBAL_Y,  NC.LENGTH,  NC.WIDTH,  NC.CLASS,  NC.VELOCITY,  NC.ACCELERATION,  NC.LANE_ID,  NC.PRECEDING_ID,  NC.FOLLOWING_ID,  NC.LOCATION,  NC.O_ZONE,  NC.D_ZONE,  NC.INT_ID,  NC.SECTION_ID,  NC.DIRECTION,  NC.MOVEMENT,  NC.DHW, NC.THW]
class HighD2NGSIM:
    def __init__(self, highD_tracks_csv_num_list, highD_filedir):
        self.highD_filedir=highD_filedir
        self.highD_tracks_csv_num_list = highD_tracks_csv_num_list
        self.highD_recordingMetas = []
        self.highD_tracksMetas = []
        self.highD_tracks = []

    def convert(self):
        """ This method does things
        :return:
        """
        for ds_id in self.highD_tracks_csv_num_list:
            self.highD_recordingMetas.append(pd.read_csv(self.highD_filedir+"{:02d}".format(ds_id)+"_recordingMeta.csv"))
            self.highD_tracksMetas.append(pd.read_csv(self.highD_filedir+"{:02d}".format(ds_id)+"_tracksMeta.csv"))
            self.highD_tracks.append(pd.read_csv(self.highD_filedir+"{:02d}".format(ds_id)+"_tracks.csv"))
            print("Read {:02d}".format(ds_id))
            # print(self.highD_recordingMetas[ds_id-1].frameRate.iloc[0])
            ngsim_values = self.transform_values(ds_id)

        return

    def transform_values(self, ds_id):
        recordingMeta = self.highD_recordingMetas[ds_id - 1]
        tracksMeta = self.highD_tracksMetas[ds_id - 1]
        tracks = self.highD_tracks[ds_id - 1]

        ngsim_df = pd.DataFrame(columns=NC_LIST)

        # Adding a time for future re-sampling:
        # tracks['TimeIdx'] = pd.TimedeltaIndex(data=(tracks[HC.FRAME]-1) * 1/recordingMeta[HC.FRAME_RATE].iloc[0], unit='s', freq='infer')
        tracks['temp_TimeStamp'] = pd.to_datetime((tracks[HC.FRAME]-1) * 1/recordingMeta[HC.FRAME_RATE].iloc[0], unit="s")
        tracks['TimeStamp'] = (tracks[HC.FRAME] - 1) * 1 / recordingMeta[HC.FRAME_RATE].iloc[0]
        tracks = tracks.set_index('temp_TimeStamp')
        # tracks['PeriodIdx'] = pd.PeriodIndex(tracks[HC.FRAME], freq="0.04s")

        right_tracks_ids = tracksMeta.loc[tracksMeta[HC.DRIVING_DIRECTION] == 2, HC.TRACK_ID].unique() # need to rotate pi/2 CCW
        left_tracks_ids = tracksMeta.loc[tracksMeta[HC.DRIVING_DIRECTION] == 1, HC.TRACK_ID].unique() # need to rotate pi/2 CW

        right_tracks = tracks[tracks[HC.TRACK_ID].isin(right_tracks_ids)]
        left_tracks = tracks[tracks[HC.TRACK_ID].isin(left_tracks_ids)]

        right_resampled_tracks = self.resample(right_tracks)
        left_resampled_tracks = self.resample(left_tracks)




    def resample(self, tracks):
        # Resample each track in the groupby to the sampling rate of NGSIM
        # Return a new tracks dataframe
        resampled_tracks = pd.DataFrame(columns=tracks.columns)
        for track_id, track in tracks.groupby(HC.TRACK_ID):
            print("Track id: "+str(track_id))
            # Upsampling to interpolate at intervals of 0.02s, which is divisible by 0.1s
            double_index = pd.date_range(start=pd.to_datetime(track["TimeStamp"].iloc[0],unit="s"), end=pd.to_datetime(track["TimeStamp"].iloc[-1],unit="s"), freq="20L")
            upsampled_track = track.reindex(double_index)
            # interpolating to fill
            upsampled_track["TimeStamp"].interpolate(method='linear', inplace=True)
            upsampled_track[HC.FRAME].interpolate(method='linear', inplace=True)
            upsampled_track[HC.TRACK_ID].interpolate(method='linear', inplace=True)
            upsampled_track[HC.X].interpolate(method='linear', inplace=True)
            upsampled_track[HC.Y].interpolate(method='linear', inplace=True)
            upsampled_track[HC.WIDTH].interpolate(method='linear', inplace=True)
            upsampled_track[HC.HEIGHT].interpolate(method='linear', inplace=True)
            upsampled_track[HC.X_VELOCITY].interpolate(method='linear', inplace=True)
            upsampled_track[HC.Y_VELOCITY].interpolate(method='linear', inplace=True)
            upsampled_track[HC.X_ACCELERATION].interpolate(method='linear', inplace=True)
            upsampled_track[HC.Y_ACCELERATION].interpolate(method='linear', inplace=True)
            upsampled_track[HC.FRONT_SIGHT_DISTANCE].interpolate(method='linear', inplace=True)
            upsampled_track[HC.BACK_SIGHT_DISTANCE].interpolate(method='linear', inplace=True)
            upsampled_track[HC.DHW].interpolate(method='linear', inplace=True)
            upsampled_track[HC.THW].interpolate(method='linear', inplace=True)
            upsampled_track[HC.TTC].interpolate(method='linear', inplace=True)
            upsampled_track[HC.PRECEDING_X_VELOCITY].fillna(method="bfill", inplace=True)
            upsampled_track[HC.PRECEDING_ID].fillna(method="bfill", inplace=True)
            upsampled_track[HC.FOLLOWING_ID].fillna(method="bfill", inplace=True)
            upsampled_track[HC.LEFT_PRECEDING_ID].fillna(method="bfill", inplace=True)
            upsampled_track[HC.LEFT_ALONGSIDE_ID].fillna(method="bfill", inplace=True)
            upsampled_track[HC.LEFT_FOLLOWING_ID].fillna(method="bfill", inplace=True)
            upsampled_track[HC.RIGHT_PRECEDING_ID].fillna(method="bfill", inplace=True)
            upsampled_track[HC.RIGHT_ALONGSIDE_ID].fillna(method="bfill", inplace=True)
            upsampled_track[HC.RIGHT_FOLLOWING_ID].fillna(method="bfill", inplace=True)
            upsampled_track[HC.LANE_ID].fillna(method="bfill", inplace=True)
            # Downsampling to NGSIM freq
            new_index=pd.date_range(start=pd.to_datetime(track["TimeStamp"].iloc[0],unit="s"), end=pd.to_datetime(track["TimeStamp"].iloc[-1],unit="s"), freq=str(int(1/NMeta.NGSIM_FRAME_RATE*1000))+"L")
            resampled_track = upsampled_track.reindex(new_index)
            resampled_track.set_index(np.arange(0,resampled_track.shape[0]))
            resampled_tracks = resampled_tracks.append(resampled_track, ignore_index=True)
        resampled_tracks["resampled_Frame_ID"] = pd.Series(resampled_tracks["TimeStamp"]*NMeta.NGSIM_FRAME_RATE, dtype=int)

        return resampled_tracks
        # resampled_tracks.to_csv(self.highD_filedir + "together.csv")
        #Generate new Frame IDs:
