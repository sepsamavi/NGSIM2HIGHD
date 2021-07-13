import pandas as pd
import numpy as np
import HighD_Columns as HC
import NGSIM_Columns as NC
import NGSIM_MetaInfo as NMeta
# NC_LIST =[NC.ID,  NC.FRAME,  NC.TOTAL_FRAME,  NC.GLOBAL_TIME,  NC.X,  NC.Y,  NC.GLOBAL_X,  NC.GLOBAL_Y,  NC.LENGTH,  NC.WIDTH,  NC.CLASS,  NC.VELOCITY,  NC.ACCELERATION,  NC.LANE_ID,  NC.PRECEDING_ID,  NC.FOLLOWING_ID,  NC.LOCATION,  NC.O_ZONE,  NC.D_ZONE,  NC.INT_ID,  NC.SECTION_ID,  NC.DIRECTION,  NC.MOVEMENT,  NC.DHW, NC.THW]
NC_LIST =[NC.ID,  NC.FRAME, NC.GLOBAL_TIME, NC.X,  NC.Y,  NC.LENGTH,  NC.WIDTH,  NC.CLASS,  NC.VELOCITY,  NC.ACCELERATION,  NC.LANE_ID,  NC.PRECEDING_ID,  NC.FOLLOWING_ID]
class HighD2NGSIM:
    def __init__(self, highD_tracks_csv_num_list, highD_filedir):
        self.highD_filedir=highD_filedir
        self.highD_tracks_csv_num_list = highD_tracks_csv_num_list
        self.highD_recordingMetas = [None for _ in range(max(highD_tracks_csv_num_list))]
        self.highD_tracksMetas = [None for _ in range(max(highD_tracks_csv_num_list))]
        self.highD_tracks = [None for _ in range(max(highD_tracks_csv_num_list))]

    def convert(self):
        """ This method does things
        :return:
        """
        for ds_id in self.highD_tracks_csv_num_list:
            self.highD_recordingMetas[ds_id-1] = pd.read_csv(self.highD_filedir+"{:02d}".format(ds_id)+"_recordingMeta.csv")
            self.highD_tracksMetas[ds_id-1] = pd.read_csv(self.highD_filedir+"{:02d}".format(ds_id)+"_tracksMeta.csv")
            self.highD_tracks[ds_id-1] = pd.read_csv(self.highD_filedir+"{:02d}".format(ds_id)+"_tracks.csv")
            print("Read {:02d}".format(ds_id))
            # print(self.highD_recordingMetas[ds_id-1].frameRate.iloc[0])
            right_tracks_NGSIMFormat, left_tracks_NGSIMFormat = self.transform_values(ds_id)

            right_tracks_NGSIMFormat.to_csv(self.highD_filedir+"NGSIMFormat/DS{:02d}".format(ds_id)+"_right.csv", index=False)
            left_tracks_NGSIMFormat.to_csv(self.highD_filedir+"NGSIMFormat/DS{:02d}".format(ds_id)+"_left.csv", index=False)

        return

    def transform_values(self, ds_id):
        recordingMeta = self.highD_recordingMetas[ds_id - 1]
        tracksMeta = self.highD_tracksMetas[ds_id - 1]
        tracks = self.highD_tracks[ds_id - 1]

        # Adding a time for future re-sampling:
        # tracks['TimeIdx'] = pd.TimedeltaIndex(data=(tracks[HC.FRAME]-1) * 1/recordingMeta[HC.FRAME_RATE].iloc[0], unit='s', freq='infer')
        tracks["TimeStamp"] = ((tracks[HC.FRAME] - 1)  / recordingMeta[HC.FRAME_RATE].iloc[0])#.astype(np.float64)

        right_tracks_ids = tracksMeta.loc[tracksMeta[HC.DRIVING_DIRECTION] == 2, HC.TRACK_ID].unique() # need to rotate pi/2 CCW
        left_tracks_ids = tracksMeta.loc[tracksMeta[HC.DRIVING_DIRECTION] == 1, HC.TRACK_ID].unique() # need to rotate pi/2 CW

        right_tracks = tracks[tracks[HC.TRACK_ID].isin(right_tracks_ids)]
        left_tracks = tracks[tracks[HC.TRACK_ID].isin(left_tracks_ids)]

        # Resample & interpolate to the sampling rate of the NGSIM dataset
        right_resampled_tracks = self.resample(right_tracks, tracksMeta)
        left_resampled_tracks = self.resample(left_tracks, tracksMeta)

        # Rotate tracks to match NGSIM and
        right_rotated_tsfd = self.rotate_tsf_tracks(right_resampled_tracks, np.pi/2, True)
        left_rotated_tsfd = self.rotate_tsf_tracks(left_resampled_tracks, -np.pi/2, False)

        return right_rotated_tsfd, left_rotated_tsfd

    def rotate_tsf_tracks(self, tracks, angle, right_tracks):
        # Change locations
        # Move point from upper left corner in image coordinates to the middle front of the car
        tracks_NGSIMformat = pd.DataFrame(columns=NC_LIST)
        if right_tracks:
            tracks_X_Temp = tracks[HC.X]+tracks[HC.WIDTH]
        else:
            tracks_X_Temp = tracks[HC.X]
        tracks_Y_Temp = tracks[HC.Y]+0.5*tracks[HC.HEIGHT]

        # Rotate with respect to Image coordinates and convert to feet, use NGSIM title
        if right_tracks:
            # Since highD has a left-handed frame, switch x,y. Tracks going right (ie increasing x) in highD format will now go increasing y in NGSIM format
            tracks_NGSIMformat[NC.X] = tracks_Y_Temp*NMeta.FEET_PER_METRE
            tracks_NGSIMformat[NC.Y] = tracks_X_Temp*NMeta.FEET_PER_METRE
        else:
            # Since highD has a left-handed frame, switch x,y then rotate 180 degrees to go in direction of increasing y in NGSIM format
            tracks_NGSIMformat[NC.X] = (tracks_Y_Temp * np.cos(np.pi) - tracks_X_Temp * np.sin(np.pi))*NMeta.FEET_PER_METRE
            tracks_NGSIMformat[NC.Y] = (tracks_Y_Temp * np.sin(np.pi) + tracks_X_Temp * np.cos(np.pi))*NMeta.FEET_PER_METRE


        ## Add rest of the relevant information in NGSIM format
        # Add vehicle ID - matching to track ID
        tracks_NGSIMformat[NC.ID] = tracks[HC.TRACK_ID].round(0).astype(int)

        # NB we round floating point values that should be ints to counteract floating point errors
        # Adding Global Time in milliseconds using the timestamp defined in seconds in resample function
        tracks_NGSIMformat[NC.GLOBAL_TIME] = (tracks["TimeStamp"]*1000).round(0).astype(int)

        # Adding Frame_ID info: NB: using the resampled IDs
        tracks_NGSIMformat[NC.FRAME] = (tracks_NGSIMformat[NC.GLOBAL_TIME]/100).round(0)
        tracks_NGSIMformat[NC.FRAME] = tracks_NGSIMformat[NC.FRAME].round(0).astype(int)

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

        if right_tracks:
            new_lane = 1
            for lane in np.arange(min_lane_id, max_lane_id+1):
                tracks.loc[(tracks[HC.LANE_ID] == lane, "new_LaneId")] = new_lane
                new_lane += 1
        else:
            new_lane = max_lane_id - min_lane_id + 1
            for lane in np.arange(min_lane_id, max_lane_id+1):
                tracks.loc[(tracks[HC.LANE_ID] == lane, "new_LaneId")] = new_lane
                new_lane -= 1

        tracks_NGSIMformat[NC.LANE_ID] = tracks["new_LaneId"].round(0).astype(int)

        # Add preceding and following IDs:
        tracks_NGSIMformat[NC.PRECEDING_ID] = tracks[HC.PRECEDING_ID].round(0).astype(int)
        tracks_NGSIMformat[NC.FOLLOWING_ID] = tracks[HC.FOLLOWING_ID].round(0).astype(int)

        return tracks_NGSIMformat



    def resample(self, tracks, tracksMeta):
        # Resample each track in the groupby to the sampling rate of NGSIM
        # Return a new tracks dataframe
        resampled_tracks = pd.DataFrame(columns=tracks.columns, dtype=tracks.dtypes[0])
        for track_id, track in tracks.groupby(HC.TRACK_ID):
            # print("Track id: "+str(track_id))
            # track.set_index('temp_TimeStamp')
            # Upsampling to interpolate at intervals of 0.02s, which is divisible by 0.1s
            double_index = np.arange(track.index[0], track.index[-1]-0.5, step=0.5) #NB need -0.5 to land at final index
            upsampled_track = track.reindex(double_index)
            # interpolating to fill
            upsampled_track["TimeStamp"].interpolate(method='linear', inplace=True)
            upsampled_track["TimeStamp"] = upsampled_track["TimeStamp"].round(2)
            upsampled_track[HC.FRAME].interpolate(method='linear', inplace=True)
            upsampled_track[HC.FRAME] = upsampled_track[HC.FRAME].round(1)
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

            upsampled_track[HC.TRACK_ID].fillna(method='ffill', inplace=True)
            upsampled_track[HC.TRACK_ID] = upsampled_track[HC.TRACK_ID].round(0).astype(int)
            upsampled_track[HC.PRECEDING_X_VELOCITY].fillna(method="ffill", inplace=True)
            upsampled_track[HC.PRECEDING_ID].fillna(method="ffill", inplace=True)
            upsampled_track[HC.FOLLOWING_ID].fillna(method="ffill", inplace=True)
            upsampled_track[HC.LEFT_PRECEDING_ID].fillna(method="ffill", inplace=True)
            upsampled_track[HC.LEFT_ALONGSIDE_ID].fillna(method="ffill", inplace=True)
            upsampled_track[HC.LEFT_FOLLOWING_ID].fillna(method="ffill", inplace=True)
            upsampled_track[HC.RIGHT_PRECEDING_ID].fillna(method="ffill", inplace=True)
            upsampled_track[HC.RIGHT_ALONGSIDE_ID].fillna(method="ffill", inplace=True)
            upsampled_track[HC.RIGHT_FOLLOWING_ID].fillna(method="ffill", inplace=True)
            upsampled_track[HC.LANE_ID].fillna(method="ffill", inplace=True)
            upsampled_track = upsampled_track.set_index(np.arange(0, upsampled_track.shape[0]))
            # Removing leading frames and tailing frames, such that later downsampled frame numbers line up with NGSIM frame rate
            NGSIM_ms = int(1/NMeta.NGSIM_FRAME_RATE*1000) # periods of time where NGSIM will line up in milliseconds
            starting_ts = int(round(upsampled_track["TimeStamp"].iloc[0]*1000))# milliseconds
            ending_ts = int(round(upsampled_track["TimeStamp"].iloc[-1]*1000)) # milliseconds
            if (starting_ts % NGSIM_ms) == 0:
                leading_rows_to_pop = int(0)
            else:
                leading_rows_to_pop = int(round((NGSIM_ms - (starting_ts % NGSIM_ms)) / 20) )# find modulo of first ts and NGSIM_ms, subtract to get leading ms, divide by highD period to find number of rows to pop
            trailing_rows_to_pop = int(round(-1 * (ending_ts % NGSIM_ms) / 20)) # same idea for training rows
            if leading_rows_to_pop > 0:
                upsampled_track = upsampled_track.iloc[leading_rows_to_pop: , :] # pop leading rows
            if abs(trailing_rows_to_pop) > 0:
                upsampled_track = upsampled_track.iloc[:trailing_rows_to_pop, :] # pop trailing rows

            # Downsampling to NGSIM freq
            new_index=np.arange(upsampled_track.index[0], upsampled_track.index[-1]+1, 5)
            # re_index to down sample
            resampled_track = upsampled_track.reindex(new_index)
            resampled_track = resampled_track.set_index(np.arange(0, resampled_track.shape[0]))

            resampled_track[HC.CLASS] = tracksMeta.loc[(tracksMeta[HC.TRACK_ID] == track_id,HC.CLASS)].iloc[0]
            resampled_tracks = resampled_tracks.append(resampled_track, ignore_index=True)
        return resampled_tracks
