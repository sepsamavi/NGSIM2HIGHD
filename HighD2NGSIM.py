import pandas as pd
import numpy as np
import HighD_Columns as HC
import NGSIM_Columns as NC
import NGSIM_MetaInfo as NMeta
# NC_LIST =[NC.ID,  NC.FRAME,  NC.TOTAL_FRAME,  NC.GLOBAL_TIME,  NC.X,  NC.Y,  NC.GLOBAL_X,  NC.GLOBAL_Y,  NC.LENGTH,  NC.WIDTH,  NC.CLASS,  NC.VELOCITY,  NC.ACCELERATION,  NC.LANE_ID,  NC.PRECEDING_ID,  NC.FOLLOWING_ID,  NC.LOCATION,  NC.O_ZONE,  NC.D_ZONE,  NC.INT_ID,  NC.SECTION_ID,  NC.DIRECTION,  NC.MOVEMENT,  NC.DHW, NC.THW]
NC_LIST =[NC.ID,  NC.FRAME,   NC.X,  NC.Y,  NC.LENGTH,  NC.WIDTH,  NC.CLASS,  NC.VELOCITY,  NC.ACCELERATION,  NC.LANE_ID,  NC.PRECEDING_ID,  NC.FOLLOWING_ID]
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
            right_tracks_NGSIMFormat, left_tracks_NGSIMFormat = self.transform_values(ds_id)

            right_tracks_NGSIMFormat.to_csv(self.highD_filedir+"NGSIMFormat/DS{:02d}".format(ds_id)+"_right.csv")
            left_tracks_NGSIMFormat.to_csv(self.highD_filedir+"NGSIMFormat/DS{:02d}".format(ds_id)+"_left.csv")

        return

    def transform_values(self, ds_id):
        recordingMeta = self.highD_recordingMetas[ds_id - 1]
        tracksMeta = self.highD_tracksMetas[ds_id - 1]
        tracks = self.highD_tracks[ds_id - 1]

        # Adding a time for future re-sampling:
        # tracks['TimeIdx'] = pd.TimedeltaIndex(data=(tracks[HC.FRAME]-1) * 1/recordingMeta[HC.FRAME_RATE].iloc[0], unit='s', freq='infer')
        tracks["temp_TimeStamp"] = pd.to_datetime((tracks[HC.FRAME]-1) * 1/recordingMeta[HC.FRAME_RATE].iloc[0], unit="s")
        tracks["TimeStamp"] = (tracks[HC.FRAME] - 1) * 1 / recordingMeta[HC.FRAME_RATE].iloc[0]
        tracks = tracks.set_index('temp_TimeStamp')

        # tracks['PeriodIdx'] = pd.PeriodIndex(tracks[HC.FRAME], freq="0.04s")
        right_tracks_ids = tracksMeta.loc[tracksMeta[HC.DRIVING_DIRECTION] == 2, HC.TRACK_ID].unique() # need to rotate pi/2 CCW
        left_tracks_ids = tracksMeta.loc[tracksMeta[HC.DRIVING_DIRECTION] == 1, HC.TRACK_ID].unique() # need to rotate pi/2 CW

        right_tracks = tracks[tracks[HC.TRACK_ID].isin(right_tracks_ids)]
        left_tracks = tracks[tracks[HC.TRACK_ID].isin(left_tracks_ids)]

        # Resample & interpolate to the sampling rate of the NGSIM dataset
        right_resampled_tracks = self.resample(right_tracks, tracksMeta)
        left_resampled_tracks = self.resample(left_tracks, tracksMeta)

        # Rotate tracks to match NGSIM and
        right_rotated_tsfd = self.rotate_tsf_tracks(right_resampled_tracks, np.pi/2)
        left_rotated_tsfd = self.rotate_tsf_tracks(left_resampled_tracks, -np.pi/2)

        return right_rotated_tsfd, left_rotated_tsfd

    def rotate_tsf_tracks(self, tracks, angle):
        # Change locations
        # Move point from upper left corner in image coordinates to the middle front of the car
        tracks_NGSIMformat = pd.DataFrame(columns=NC_LIST)
        if angle == np.pi/2:
            tracks_X_Temp = tracks[HC.X]+tracks[HC.WIDTH]
        else:
            tracks_X_Temp = tracks[HC.X]
        tracks_Y_Temp = tracks[HC.Y]-0.5*tracks[HC.HEIGHT]

        # Rotate with respect to Image coordinates and convert to feet, use NGSIM title
        tracks_NGSIMformat[NC.X] = (tracks_X_Temp*np.cos(angle) - tracks_Y_Temp*np.sin(angle))*NMeta.FEET_PER_METRE
        tracks_NGSIMformat[NC.Y] = (tracks_X_Temp*np.sin(angle) + tracks_Y_Temp*np.cos(angle))*NMeta.FEET_PER_METRE


        ## Add rest of the relevant information in NGSIM format
        # Add vehicle ID - matching to track ID
        tracks_NGSIMformat[NC.ID] = tracks[HC.TRACK_ID]

        # Adding Frame_ID info: NB: using the resampled IDs
        tracks_NGSIMformat[NC.FRAME] = tracks["resampled_Frame_ID"]

        # Adding Global Time in milliseconds using the timestamp defined in seconds in resample function
        tracks_NGSIMformat[NC.GLOBAL_TIME] = tracks["TimeStamp"]*1000

        # Add width and length and convert to feet
        tracks_NGSIMformat[NC.WIDTH] = tracks[HC.HEIGHT]*NMeta.FEET_PER_METRE
        tracks_NGSIMformat[NC.LENGTH] = tracks[HC.WIDTH]*NMeta.FEET_PER_METRE

        # Add vehicle class
        tracks["Class_num"] = 0
        tracks.loc[(tracks[HC.CLASS].str.match("Car"), "Class_num")] = 2
        tracks.loc[(tracks[HC.CLASS].str.match("Truck"), "Class_num")] = 3
        tracks_NGSIMformat[NC.CLASS] = tracks["Class_num"]

        # Add velocity and convert to feet/s
        tracks_NGSIMformat[NC.VELOCITY] = (tracks[HC.X_VELOCITY]**2 + tracks[HC.Y_VELOCITY]**2)**0.5*NMeta.FEET_PER_METRE

        # Add acceleration and convert ot feet/s^2
        tracks_NGSIMformat[NC.ACCELERATION] = (tracks[HC.X_ACCELERATION]**2 + tracks[HC.Y_ACCELERATION]**2)**0.5*NMeta.FEET_PER_METRE

        # Add lane ID:
        # Reordering lane_id to start at the "fast lane" and end at the merge lanes, regardless of travel direction
        tracks["new_LaneId"] = 0
        min_lane_id = np.min(tracks[HC.LANE_ID].unique())
        max_lane_id = np.max(tracks[HC.LANE_ID].unique())

        if angle == np.pi/2:
            new_lane = 1
            for lane in np.arange(min_lane_id, max_lane_id+1):
                tracks.loc[(tracks[HC.LANE_ID] == lane, "new_LaneId")] = new_lane
                new_lane += 1
        else:
            new_lane = max_lane_id - min_lane_id + 1
            for lane in np.arange(min_lane_id, max_lane_id+1):
                tracks.loc[(tracks[HC.LANE_ID] == lane, "new_LaneId")] = new_lane
                new_lane -= 1

        tracks_NGSIMformat[NC.LANE_ID] = tracks["new_LaneId"]
    
        # Add preceding and following IDs:
        tracks_NGSIMformat[NC.PRECEDING_ID] = tracks[HC.PRECEDING_ID]
        tracks_NGSIMformat[NC.FOLLOWING_ID] = tracks[HC.FOLLOWING_ID]

        return tracks_NGSIMformat



    def resample(self, tracks, tracksMeta):
        # Resample each track in the groupby to the sampling rate of NGSIM
        # Return a new tracks dataframe
        resampled_tracks = pd.DataFrame(columns=tracks.columns)
        for track_id, track in tracks.groupby(HC.TRACK_ID):
            # print("Track id: "+str(track_id))
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
            # Move track class from meta information to the actual track
            resampled_track[HC.CLASS] = tracksMeta.loc[(tracksMeta[HC.TRACK_ID] == track_id,HC.CLASS)].iloc[0]
            resampled_track.set_index(np.arange(0,resampled_track.shape[0]))
            resampled_tracks = resampled_tracks.append(resampled_track, ignore_index=True)
        resampled_tracks["resampled_Frame_ID"] = pd.Series(resampled_tracks["TimeStamp"]*NMeta.NGSIM_FRAME_RATE, dtype=int)

        return resampled_tracks
        # resampled_tracks.to_csv(self.highD_filedir + "together.csv")
        #Generate new Frame IDs:
