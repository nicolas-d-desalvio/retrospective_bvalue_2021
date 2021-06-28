# -*- coding: utf-8 -*-
#Library importation
import cartopy as ct
import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy import spatial
from scipy import interpolate
from tqdm import tqdm
from obspy.geodetics import gps2dist_azimuth
import argparse
from matplotlib import rcParams
'''This code refers to three time periods: pre-, bt-, and aft-. pre stands for pre-event, bt
stands for between events (however this subset still represents the time after a single
event if there is no second earthquake), aft- means after, and is the time period after a 
second event, and only exsists for event pairs)'''
use_command_line = False
# Command Line Arguments:
# Example option use for command line
#-start_time 1825 -na_time 0.05 -precut_mag 1 -dist_thresh 3000
#-start_time -1 -na_time -1 -precut_mag -1 -dist_thresh -1 -grid 500
if use_command_line == True:
    parser = argparse.ArgumentParser()
    parser.add_argument("-start_time", type=int, help="Start Time")
    parser.add_argument("-na_time", type=float, help="No Alert Time")
    parser.add_argument("-precut_mag", type=float, help="Precut Magnitude")
    parser.add_argument("-dist_thresh", type=int, help="Distance Threshold")
    parser.add_argument("-grid", type=int, help="Number of Distance Grid Points")
    args = parser.parse_args()
    start_t = np.copy(args.start_time)
    na_time = np.copy(args.na_time)
    precut_magnitude = np.copy(args.precut_mag)
    dist_threshold = np.copy(args.dist_thresh)
    grid = np.copy(args.grid) # Grid is used in the distance threshold calculation - see the distance from fault plane function

# To run within editor and not a command line, parameters can be set using lines 37-41
else:
    start_t = 10*365
    na_time = 0.1
    precut_magnitude = 1.5
    dist_threshold = 3000
    grid = 200

print('st_' + str(start_t) + '_na_' + str(na_time) + '_pm_' + str(precut_magnitude) + '_dt_' + str(dist_threshold))
if start_t < 0:
    start_t = 'Auto'
if na_time < 0:
    na_time = 'Auto'
if precut_magnitude < 0:
    precut_magnitude = 'Auto'
if dist_threshold < 0:
    dist_threshold = 'Auto'

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']

# Data loading and Organization
# Code to load the data files
def load_data1(filename,times_filename): # Takes names of data files
    total_data = np.loadtxt(filename,skiprows=1,delimiter = ',')
    times_original = np.loadtxt(times_filename, dtype = str)
    times_dt64 = []

    for i in times_original:
        times_dt64.append(np.datetime64(i[0]+'T'+i[1]))
    times = np.array(times_dt64)

    return times, total_data  #times and total data are the original data

def pairing(data,times,start_t): # Takes data and the start time parameter
    '''Extracts all M5+ events and creates a subcatalog with them, these events become the  
    earthquakes the code will analyze. Determines event relationships: foreshock, mainshock, aftershock,
    and removes aftershocks from the M5 subcatalog, it also converts the start time parameter to a date for 
    each earthquake and checks to see if the start time must be overwritten because of an aftershock
    sequence from a previous earthquake
    Returns a data file for non-aftershock M5 events, the associated time data, and a matrix with event relationships and the 
    dates the start time parameter corresponds to'''
    '''extra_info array:
        Col 0: first or second, indicates if an event is the first or second event in a pair, standalone mainshocks also are assigned first
        Col 1: yes or no, is this event apart of a pair?
        Col 2: Start Time as a Date
        Col 3: End Time as a Date, however this end time is overwritten outside this function
        Col 4: yes or no, is this event an aftershock?
        '''
    dt_64 = np.copy(times)
    mask6 = data[:,3] >= 5.0 # Consider only EQs above 5
    Events_6_1 = data[mask6,:] # Extract data for all events above M5
    Times_6_1 = dt_64[mask6]
    mask_time = Times_6_1 >= np.datetime64('1975-01-01') # Given catalog constraints, we only analyze M5 events past 1981, but we will use data to 1975 to determine relationships
    Events_6 = Events_6_1[mask_time,:]
    Times_6 = Times_6_1[mask_time]
    Extra_6_info = np.empty((len(Times_6),5),dtype=object) # Initialize an empty array to hold the start time date, and event relationships
    
    # Set up Gardner and Knopoff (1974) Spatiotemporal Aftershock Window Interpolation
    magnitude_interp = np.arange(2.5,8.5,0.5)
    distance_function = interpolate.interp1d(magnitude_interp,np.array([19.5,22.5,26,30,35,40,47,54,61,70,81,94]))
    time_function = interpolate.interp1d(magnitude_interp,np.array([6,11.5,22,42,83,155,290,510,790,915,960,985]))
    
    t_post_event = 180 #days, Setting a standardized end time date, this was written before we varied the end time so this gets overwritten later
    
    if start_t == 'Auto':
        start_time = 30000
    else:
        start_time = int(np.copy(start_t)) 
        
    for i in range(len(Times_6)): # For Every M5+ EQ
        if Extra_6_info[i,4] == 'yes' : # If EQ has already determined to be an aftershock or a second event; identify its aftershocks
            time_cutoff3 = Times_6[i] + np.timedelta64(int(time_function(Events_6[i,3])),'D')
            distance_bound3 = distance_function(Events_6[i,3])
            for z in range(len(Times_6)): # Determine the aftershocks to this event
                if z <= i: # If the event preceeded the EQ in question, it is not an aftershock
                    pass
                else:
                    if Times_6[z] <= time_cutoff3:
                        distance_between = gps2dist_azimuth(Events_6[i,0],Events_6[i,1],Events_6[z,0],Events_6[z,1])[0] / 1000
                        if distance_between <= distance_bound3 and Events_6[z,3] < Events_6[i,3]:
                            Extra_6_info[z,4] = 'yes'
                        else:
                             pass
                    else:
                        pass
        elif Extra_6_info[i,1] == 'yes': # Case for if we already know the current event is a mainshock for a previously found foreshock
            Extra_6_info[i,3] = str(Times_6[i] + np.timedelta64(t_post_event,'D'))
            potential_second = []
            past_event_times = []
            past_events = []
            time_cutoff2 = Times_6[i] + np.timedelta64(int(time_function(Events_6[i,3])),'D') # GK Temporal Window
            distance_bound = distance_function(Events_6[i,3]) # GK Spatial Window
            for x in range(len(Times_6)): # Determine the aftershocks to this event
                if x <= i: # If the event occurred previously, it isn't an aftershock
                    pass
                else:
                    if Times_6[x] <= time_cutoff2: # Check against the temporal window limit
                        distance_between = gps2dist_azimuth(Events_6[i,0],Events_6[i,1],Events_6[x,0],Events_6[x,1])[0] / 1000
                        if distance_between <= distance_bound and Events_6[x,3] < Events_6[i,3]: # Must have a smaller magnitude to be an aftershock, and must be within the distance window
                            Extra_6_info[x,4] = 'yes' # If the above are true, then we have identified an aftershock
                        elif distance_between > distance_bound: # If outside of the distance bound, move on
                             pass
                        else:
                            if len(potential_second) == 0: # We have identified a third, larger earthquake in the sequence
                                Extra_6_info[x,4] = 'no'
                                Extra_6_info[x,0] = 'third'
                                Extra_6_info[x,1] = 'yes'
                                Extra_6_info[i,1] = 'yes'
                                potential_second.append(x)
                            else:
                                Extra_6_info[x,4] = 'yes' # The only EQs that end up in this else statement are volcanic EQs associated with the 1980 eruption of Mt St Helens. While all events prior to 1981 will be removed later, we can just set these as aftershocks for now
                                
            for y in range(len(Times_6)): #Look for past events in the same area to set the start time of b-reference data, we can't have previous sequences contaminate the reference b-value
                if y >= i: # Since we are looking for past events, we can ignore events that occurred after the target event
                    pass
                else:
                    distance_between = gps2dist_azimuth(Events_6[i,0],Events_6[i,1],Events_6[y,0],Events_6[y,1])[0] / 1000
                    if distance_between <= distance_bound: # If a previous event is within the GK distance bound, it is a candidate event
                        past_event_times.append(Times_6[y])
                        past_events.append(y)
            if len(past_event_times) != 1:
                while Extra_6_info[past_events[-1],4] == 'yes': # Delete any aftershocks from the candidate EQ list
                    past_events = np.delete(past_events,-1)
                    past_event_times = np.delete(past_event_times,-1)
                if len(past_events) > 1: # If multiple candidate events remain, we must take the most recent, index -1 corresponds to the first event in the event pair
                    tmax = past_event_times[-2]
                    tmax_index = past_events[-2]
                    if (tmax + np.timedelta64(int(time_function(Events_6[past_events[-2],3])),'D')) > Times_6[i] - np.timedelta64(start_time,'D'):
                        # The above compares the end of the candidate events GK time window, and compares it to the start time, if the GK window ends more recently than the start time, the start time must be overwritten with the end of the previous EQ's GK window time
                        aftershock_time_past_event = tmax + np.timedelta64(int(time_function(Events_6[past_events[-2],3])),'D')
                        Extra_6_info[i,2] = str(aftershock_time_past_event)
                    else: # If not the original start time can be used
                        Extra_6_info[i,2] = str(Times_6[i] - np.timedelta64(start_time,'D'))
                else: # Since index -1 corresponds to the first event in the event pair, if the past_events list has 1 entry, there are no candidate events, use the original start time
                    Extra_6_info[i,2] = str(Times_6[i] - np.timedelta64(start_time,'D'))
            else: # Since index -1 corresponds to the first event of the pair, if the past_events list has 1 entry, there are no candidate events, use the original start time
                Extra_6_info[i,2] = str(Times_6[i] - np.timedelta64(start_time,'D'))

        else: # Case for when this event has no previously determined information
            Extra_6_info[i,0] = 'first' # We know it is not the second in a pair, or we would have identified it already
            Extra_6_info[i,4] = 'no' # We also know it is not an aftershock, or we would have identified it already
            Extra_6_info[i,3] = str(Times_6[i] + np.timedelta64(t_post_event,'D')) # Set standard end time
            distance_bound = distance_function(Events_6[i,3]) # GK Distance Window
            time_cutoff = Times_6[i] + np.timedelta64(int(time_function(Events_6[i,3])),'D') # GK Temporal Window
            past_events = []
            past_event_times = []
            potential_second = []
            for x in range(len(Times_6)): # Determine the aftershocks to this event
                if x <= i: # If an event occurred before the event in question, it can't be an aftershock
                    pass
                else:
                    if Times_6[x] <= time_cutoff: # Check against the temporal window limit
                        distance_between = gps2dist_azimuth(Events_6[i,0],Events_6[i,1],Events_6[x,0],Events_6[x,1])[0]/1000
                        if distance_between <= distance_bound and Events_6[x,3] < Events_6[i,3]: # Must have a smaller magnitude to be an aftershock, and must be within the distance window
                            Extra_6_info[x,4] = 'yes' # If the above are true, then we have identified an aftershock
                        elif distance_between > distance_bound: # If outside of the distance bound, move on
                             pass
                        else:
                            if len(potential_second) == 0: # We have identified a second, larger earthquake in the sequence
                                Extra_6_info[x,4] = 'no'
                                Extra_6_info[x,0] = 'second'
                                Extra_6_info[x,1] = 'yes'
                                Extra_6_info[i,1] = 'yes'
                                potential_second.append(x)
                            else: # Everything that falls into this category post-1980 are large aftershocks of the second event in the pair
                                Extra_6_info[x,4] = 'yes'
            if Extra_6_info[i,1] == None: # After completing the previous loop, if a pair was not found, then it is not a part of a pair
                Extra_6_info[i,1] = 'no'
            
            for y in range(len(Times_6)): #Look for past events in the same area to set the start time of b-reference data
                if y >= i: # Since we are looking for past events, we can ignore events that occurred after the target event
                    pass
                else:
                    time_cutoff = Times_6[y] + np.timedelta64(int(time_function(Events_6[y,3])),'D')
                    distance_between = gps2dist_azimuth(Events_6[i,0],Events_6[i,1],Events_6[y,0],Events_6[y,1])[0]/1000
                    if distance_between <= distance_bound: # If a previous event is within the GK distance bound, it is a candidate event
                        past_event_times.append(Times_6[y])
                        past_events.append(y)
            if len(past_event_times) != 0: # If there is a previous event to consider, pull the most recent one
                tmax = np.max(past_event_times)
                tmax_index = np.where(past_event_times == tmax)[0][0]
                if (tmax + np.timedelta64(int(time_function(Events_6[past_events[tmax_index],3])),'D')) > Times_6[i] - np.timedelta64(start_time,'D'):
                # The above compares the end of the candidate events GK time window, and compares it to the start time, if the GK window ends more recently than the start time, the start time must be overwritten with the end of the previous EQ's GK window time
                    aftershock_time_past_event = tmax + np.timedelta64(int(time_function(Events_6[past_events[tmax_index],3])),'D')
                    Extra_6_info[i,2] = str(aftershock_time_past_event)
                else: # If the candidate GK window ends before the original start time, we use the original start time
                    Extra_6_info[i,2] = str(Times_6[i] - np.timedelta64(start_time,'D'))
            else: # If there are no candidate previous events, we use the original start time
                Extra_6_info[i,2] = str(Times_6[i] - np.timedelta64(start_time,'D'))
                
    # Remove all earthquakes that were deemed aftershocks     
    aftershock_mask = Extra_6_info[:,4] == 'no'
    Extra_6_info = Extra_6_info[aftershock_mask]
    Times_6 = Times_6[aftershock_mask]
    Events_6 = Events_6[aftershock_mask,:]
    # We want to analyze earthquakes past 1981, so we remove any that came before it
    mask_time = Times_6 >= np.datetime64('1981-01-01')
    Events_6 = Events_6[mask_time,:]
    Times_6 = Times_6[mask_time]
    Extra_6_info = Extra_6_info[mask_time]
    
    # For convenience, we swap rows in the data such that all foreshock-mainshock pairs are next to eachother
    def row_swap(i,j,Extra_6_info,Times_6,Events_6):
        Extra_6_info[[i, j]] = Extra_6_info[[j, i]]
        Times_6[[i, j]] = Times_6[[j, i]]
        Events_6[[i, j]] = Events_6[[j, i]]
        return Extra_6_info,Times_6,Events_6
    # Execute the row swaps, the specific rows were determined manually
    Extra_6_info,Times_6,Events_6 = row_swap(45,48,Extra_6_info,Times_6,Events_6)
    Extra_6_info,Times_6,Events_6 = row_swap(48,49,Extra_6_info,Times_6,Events_6)
    Extra_6_info,Times_6,Events_6 = row_swap(115,116,Extra_6_info,Times_6,Events_6)
    Extra_6_info,Times_6,Events_6 = row_swap(112,115,Extra_6_info,Times_6,Events_6)
    
    # One should save these files one time, especially to save the relationships between events
    save_M5_catalogs = False
    if save_M5_catalogs == True:
       np.savetxt("usgs_full_catalog_M5.csv",Events_6, delimiter=",",header = 'Latitude, Longitude, Depth (km), Magnitude, NP1 Strike, NP1 Dip, NP1 rake, NP2 Strike, NP2 Dip, NP2 Rake')
       file0 = open("usgs_full_catalog_M5_info_sample.txt",'w')
       for i in Extra_6_info: 
           print(str(i))
           file0.write(str(i).replace('\n','').replace('[','').replace(']','').replace('\'','') + '\n')
       file0.close()
       file = open('usgs_full_catalog_M5_times.txt','w')
       for i in Times_6:
           file.write(str(i) + '\n')
       file.close()
    '''Events_6 is the data file for non-aftershock M5 events, Times_6 contains the time data, Extra_6_info
    contains event relationships and the dates the start time parameter corresponds to'''
    return Events_6, Times_6, Extra_6_info

def data_splitting(main_events,main_event_times,grand_data,extra_info,grand_times,i,start_t,parameters):
    '''Creates a subcatalog of data for a specific earthquake, with general distance criteria and 
    the start and end time. Inputs: All full, and M5+ data catalogs, the index i for this earthquake's
    position in the M5 catalog, and the start time parameter. Returns the data and times for the event-specific earthquake catalog'''
    event_lon = main_events[i,1]
    event_lat = main_events[i,0]
    
    # Set up SubCatalog for the Event, cutting by a loose distance to make the distance to fault plane calculation faster
    subcat_data0 = grand_data[(grand_data[:,1] > event_lon - 2),:]
    subcat_times0 = grand_times[(grand_data[:,1] > event_lon - 2)]
    subcat_data1 = subcat_data0[(subcat_data0[:,1] < event_lon + 2),:]
    subcat_times1 = subcat_times0[(subcat_data0[:,1] < event_lon + 2)]
    
    subcat_data2 = subcat_data1[(subcat_data1[:,0] < event_lat + 2),:]
    subcat_times2 = subcat_times1[(subcat_data1[:,0] < event_lat + 2)]
    subcat_data3 = subcat_data2[(subcat_data2[:,0] > event_lat - 2),:]
    subcat_times3 = subcat_times2[(subcat_data2[:,0] > event_lat - 2)]
    
    # Overwrite the start time with the Automatic Start Time if desired
    # Then, cut the event subcatalog by the start time, and end time.
    if start_t == 'Auto':
        start_time = automatic_start_time(subcat_data3,subcat_times3,i,extra_info,main_event_times,parameters)
        subcat_data4 = subcat_data3[subcat_times3 > np.datetime64(start_time)]
        subcat_times4 = subcat_times3[subcat_times3 > np.datetime64(start_time)]
        subcat_data = subcat_data4[subcat_times4 < (np.datetime64(extra_info[i,3]))]
        subcat_times = subcat_times4[subcat_times4 < (np.datetime64(extra_info[i,3]))]
    else:
        subcat_data4 = subcat_data3[subcat_times3 > np.datetime64(extra_info[i,2])]
        subcat_times4 = subcat_times3[subcat_times3 > np.datetime64(extra_info[i,2])]
        subcat_data = subcat_data4[subcat_times4 < (np.datetime64(extra_info[i,3]))]
        subcat_times = subcat_times4[subcat_times4 < (np.datetime64(extra_info[i,3]))]
    '''Returns the data and times for the event-specific earthquake catalog'''
    return subcat_data,subcat_times

def automatic_start_time(data,dt_64,i,extra_info,main_event_times,parameters):
    '''Function to determine an automatic start time based on an increase in regional 
    catalog completeness. We take overlapping periods of 10 years, calculate MC,
    and take the largest decrease in MC as the start time. We stop the calculation
    4 years before the earthquake to ensure that there are 4 years of background
    data. We select 4 years as the minimum because GW19 used 4 years of data for one of their
    earthquakes. Inputs: data and dt_64 are the data subcatalog specific to the event,
    i is the index of the earthquake in the M5+ catalog, extra_info contains the start time
   date-important if there is an aftershock sequence that provides an upper limit on the
   start time, end time, and event relationships, main_event_times is the time data for
   M5+ events
   Returns the start time to be used in the analysis as a date'''
    year = np.datetime64('1970')
    nxt_year = np.datetime64('1980')
    nxt_yr_list = []
    mc_list = []
    while int(str(nxt_year)[0:4]) < int(str(main_event_times[i])[0:4])-4: # The -4 ensures there are at least 4 years of data used in the reference calculation
        mask_1 = dt_64 >= year # Select the 10 year subcatalog
        times_1 = dt_64[mask_1]
        data_1 = data[mask_1,:]
        mask_2 = times_1 < nxt_year
        data_subset = data_1[mask_2]
        nxt_yr_list.append(str(nxt_year)) # We will associate the MC with the last year in the 10-year period, so we save this year
        mc_list.append(MC_function(data_subset,np.round(np.arange(1,6,0.1),1),parameters)) # Calculate MC for this subset
        
        year += np.timedelta64(2,'Y') # Increment the 10 year window
        nxt_year += np.timedelta64(2,'Y')
        
    differences = [] # List to hold the differences in MC, we are looking for a MC drop
    for j in range(len(mc_list)):
        if j == 0: 
            continue
        else:
            differences.append(mc_list[j]-mc_list[j-1]) # Calculate the MC differences
    if len(differences) == 0: #If there are not any differences, default to 10 years. This is often the case for events in 1981-1990
        # Default to 10 years because data catalog starts in 1970, so we have 10 years for a 1981 event
        start_time = str(np.datetime64(main_event_times[i]) - np.timedelta64(3650,'D'))
    else:
        index = np.where(differences == np.min(differences))[0][0] # The largest, negative difference is the largest drop, which we take as our start time
        start_time = nxt_yr_list[index+1]
    # Now make sure we aren't going to overwrite a start time that was previously set to avoid an aftershock sequence
    if np.datetime64(extra_info[i,2]) > np.datetime64(start_time): # If the original start time is more recent than Auto, then we use the original
        start_time = extra_info[i,2]
    
    return start_time

def organize_data(second_quake_exsists,main_events,main_event_times,i,Event_of_interest,parameters): #Input, whether or not there is a second quake, M5 catalog, the event index within the M5 catalog, and whether or not the event is a first or second earthquake in a pair
    '''Extract important information from the M5 catalog and place into the parameters dictionary for easier acsess. Returns
    the updated parameters dictionary'''
    if Event_of_interest == 'first':
            parameters['Magnitude_event1'] = main_events[i,3]
            parameters['Time_event1'] = main_event_times[i]
            parameters['Time_LE1'] = main_event_times[i] - np.timedelta64(1,'s') # May want to add a more accurate way to do that
            parameters['Lat_event1'] = main_events[i,0]
            parameters['Lon_event1'] = main_events[i,1]
            parameters['hypo_depth1'] = main_events[i,2] * -1000
            parameters['FP_strike1_event1'] = main_events[i,4]
            parameters['FP_dip1_event1'] = main_events[i,5]
            parameters['FP_strike2_event1'] = main_events[i,7]
            parameters['FP_dip2_event1'] = main_events[i,8]
            parameters['FP_rake1_event1'] = main_events[i,6]
            parameters['FP_rake2_event1'] = main_events[i,9]
            if second_quake_exsists == 'yes':
                parameters['Magnitude_event2'] = main_events[i+1,3]
                parameters['Time_event2'] = main_event_times[i+1]
                parameters['Time_LE2'] = main_event_times[i+1] - np.timedelta64(1,'s') # May want to add a more accurate way to do that
                parameters['Lat_event2'] = main_events[i+1,0]
                parameters['Lon_event2'] = main_events[i+1,1]
                parameters['hypo_depth2'] = main_events[i+1,2] * -1000
                parameters['FP_strike1_event2'] = main_events[i+1,4]
                parameters['FP_dip1_event2'] = main_events[i+1,5]
                parameters['FP_strike2_event2'] = main_events[i+1,7]
                parameters['FP_dip2_event2'] = main_events[i+1,8]
                parameters['FP_rake1_event2'] = main_events[i+1,6]
                parameters['FP_rake2_event2'] = main_events[i+1,9]
    else:
        parameters['Magnitude_event1'] = main_events[i-1,3]
        parameters['Time_event1'] = main_event_times[i-1]
        parameters['Time_LE1'] = main_event_times[i-1] - np.timedelta64(1,'s') # May want to add a more accurate way to do that
        parameters['Lat_event1'] = main_events[i-1,0]
        parameters['Lon_event1'] = main_events[i-1,1]
        parameters['hypo_depth1'] = main_events[i-1,2] * -1000
        parameters['FP_strike1_event1'] = main_events[i-1,4]
        parameters['FP_dip1_event1'] = main_events[i-1,5]
        parameters['FP_strike2_event1'] = main_events[i-1,7]
        parameters['FP_dip2_event1'] = main_events[i-1,8]
        parameters['FP_rake1_event1'] = main_events[i-1,6]
        parameters['FP_rake2_event1'] = main_events[i-1,9]
        if second_quake_exsists == 'yes':
            parameters['Magnitude_event2'] = main_events[i,3]
            parameters['Time_event2'] = main_event_times[i]
            parameters['Time_LE2'] = main_event_times[i] - np.timedelta64(1,'s') # May want to add a more accurate way to do that
            parameters['Lat_event2'] = main_events[i,0]
            parameters['Lon_event2'] = main_events[i,1]
            parameters['hypo_depth2'] = main_events[i,2] * -1000
            parameters['FP_strike1_event2'] = main_events[i,4]
            parameters['FP_dip1_event2'] = main_events[i,5]
            parameters['FP_strike2_event2'] = main_events[i,7]
            parameters['FP_dip2_event2'] = main_events[i,8]
            parameters['FP_rake1_event2'] = main_events[i,6]
            parameters['FP_rake2_event2'] = main_events[i,9]

    parameters['hypocenter1'] = [parameters['Lon_event1'],parameters['Lat_event1'],parameters['hypo_depth1']]
    if second_quake_exsists == 'yes':
        parameters['hypocenter2'] = [parameters['Lon_event2'],parameters['Lat_event2'],parameters['hypo_depth2']]
   
    return parameters # Dictionary containing pertinent event information

def Wells_Coppersmith(magnitude,rake):
    """Calculate the rupture length at depth and rupture width using empirical scalings from
    Wells and Coppersmith (year). The rake should be specified in degrees according to the convention in
    Aki and Richards (2002).Returns the rupture length at depth and rupture width"""
    #Wells and Coppersmith Constants -- Do Not Change
    RLD_a_ss = -2.57
    RLD_b_ss = 0.62
    RLD_a_r = -2.42
    RLD_b_r = 0.58
    RLD_a_n = -1.88
    RLD_b_n = 0.5

    RW_a_ss = -0.76
    RW_b_ss = 0.27
    RW_a_r = -1.61
    RW_b_r = 0.41
    RW_a_n = -1.14
    RW_b_n = 0.35
    
    if rake >= 45 and rake <= 135:
        #Reverse Fault
        rld = (10.**(RLD_a_r + magnitude * RLD_b_r)) * 1000.
        rw = (10.**RW_a_r + magnitude *RW_b_r) * 1000.
    elif rake <= -45. and rake >= -135.:
        #Normal Fault
        rld = (10.**(RLD_a_n + magnitude * RLD_b_n)) * 1000.
        rw = (10.**(RW_a_n + magnitude *RW_b_n)) * 1000.
    else:
        # Strike-Slip
        rld = (10.**(RLD_a_ss + magnitude * RLD_b_ss)) * 1000.
        rw = (10.**(RW_a_ss + magnitude *RW_b_ss)) * 1000.
    return rld,rw # Rupture length at depth and rupture width

def calc_strike_dip_vectors(parameters,Event_of_interest): 
    '''This function calculates strike and dip unit vectors using the focal mechanism for each
    nodal plane (denoted as 1 or 2). Returns unit strike and dip vectors'''
    if Event_of_interest == 'first':
        strike_prime1 = 90 - parameters['FP_strike1_event1']
        strike_prime2 = 90 - parameters['FP_strike2_event1']
    else:
        strike_prime1 = 90 - parameters['FP_strike1_event2']
        strike_prime2 = 90 - parameters['FP_strike2_event2']
    strike_vector1 = np.array([np.cos(np.radians(strike_prime1)),np.sin(np.radians(strike_prime1)),0])
    strike_vector2 = np.array([np.cos(np.radians(strike_prime2)),np.sin(np.radians(strike_prime2)),0])

    if Event_of_interest == 'first':
        dip_vector2 = np.array([np.cos(np.radians(strike_prime2-90)), np.sin(np.radians(strike_prime2 - 90)), ((-np.tan(np.radians(parameters['FP_dip2_event1']))))])
        dip_vector1 = np.array([np.cos(np.radians(strike_prime1-90)), np.sin(np.radians(strike_prime1 - 90)), ((-np.tan(np.radians(parameters['FP_dip1_event1']))))])
    else:
        dip_vector2 = np.array([np.cos(np.radians(strike_prime2-90)), np.sin(np.radians(strike_prime2 - 90)), ((-np.tan(np.radians(parameters['FP_dip2_event2']))))])
        dip_vector1 = np.array([np.cos(np.radians(strike_prime1-90)), np.sin(np.radians(strike_prime1 - 90)), ((-np.tan(np.radians(parameters['FP_dip1_event2']))))])
    normalization1 = np.linalg.norm(dip_vector1)
    dip_vector1 = dip_vector1 / normalization1
    
    normalization2 = np.linalg.norm(dip_vector2)
    dip_vector2 = dip_vector2 / normalization2
    
    plane_vectors = [[strike_vector1,dip_vector1],[strike_vector2,dip_vector2]]
    return plane_vectors # Unit strike and dip vectors

def FP_dimensions(parameters,Event_of_interest):
    """This function calculates the fault plane dimensions in units of (x)
    input:
        parameters - a dictionary containing the event magnitudes and rakes
        Event of interst:
            'first' - use the first event
            any other value defaults to using the second event
    output arguments:
        a vector containing [rupture length at depth, rupture width] for each of the two nodal planes."""
   
    if Event_of_interest == 'first':
        rld1,rw1 = Wells_Coppersmith(parameters['Magnitude_event1'],parameters['FP_rake1_event1'])
        rld2,rw2 = Wells_Coppersmith(parameters['Magnitude_event1'],parameters['FP_rake2_event1'])    
    else:
        rld1,rw1 = Wells_Coppersmith(parameters['Magnitude_event2'],parameters['FP_rake1_event2'])
        rld2,rw2 = Wells_Coppersmith(parameters['Magnitude_event2'],parameters['FP_rake2_event2'])
    fault_dimensions = [[rld1,rw1],[rld2,rw2]]
       
    return fault_dimensions

def distance_from_plane_cut(plane_vectors,parameters,Event_of_interest,total_data,times,fault_dimensions,dist_threshold,grid): #Takes the strike and dip vectors, and data (along with event of interest and the parameters dictionary)
    '''Function to determine the true fault plane, and cut the subcatalog by distance from that fault plane
    Returns the cut data, times, and the distance array for all of the data, in case more EQs are needed later'''    
    #First, define the new projection and transform the location data into the projection
    #Must distinguish between first and second event, since they require different parameter entries
    RLD_1 = fault_dimensions[0][0]
    RLD_2 = fault_dimensions[1][0]
    RW_1 = fault_dimensions[0][1]
    RW_2 = fault_dimensions[1][1]
    strike_vector1 = plane_vectors[0][0]
    dip_vector1 = plane_vectors[0][1]
    strike_vector2 = plane_vectors[1][0]
    dip_vector2 = plane_vectors[1][1]
    if Event_of_interest == 'first':
        projection = ct.crs.TransverseMercator(central_longitude = parameters['hypocenter1'][0],central_latitude = parameters['hypocenter1'][1]) #Transform Long/Lat to x,y coordinates
        global_projection = ct.crs.PlateCarree()
        x_longitude = []
        y_latitude = []
        for i,j in zip(total_data[:,1],total_data[:,0]):
            a,b = projection.transform_point(i,j,global_projection)
            x_longitude.append(a)
            y_latitude.append(b)
        #Set a new hypocenter for the new projection
        xh,yh = projection.transform_point(parameters['hypocenter1'][0],parameters['hypocenter1'][1],global_projection)
        hypocenter_new = [xh,yh,parameters['hypocenter1'][2]]

    else:
        projection = ct.crs.TransverseMercator(central_longitude = parameters['hypocenter2'][0],central_latitude = parameters['hypocenter2'][1]) #Transform Long/Lat to x,y coordinates
        global_projection = ct.crs.PlateCarree()
        x_longitude = []
        y_latitude = []
        for i,j in zip(total_data[:,1],total_data[:,0]):
            a,b = projection.transform_point(i,j,global_projection)
            x_longitude.append(a)
            y_latitude.append(b)
        #Set a new hypocenter for the new projection
        xh,yh = projection.transform_point(parameters['hypocenter2'][0],parameters['hypocenter2'][1],global_projection)
        hypocenter_new = [xh,yh,parameters['hypocenter2'][2]]
    
    nx =np.copy(grid)
    ny = np.copy(grid) + 1
    x = np.linspace(-RLD_1/2.,RLD_1/2.0,nx)# note that rld and rw are in meters
    y = np.linspace(-RW_1/2.,RW_1/2.0,ny)
   
    #Extend the length and width of the fault plane in the 4 different directions. Each of these sets of for loops creates a set of points for 1/4 of the plane, upper left, upper right, etc
    points = np.zeros((3,nx,ny))
    for i in range(nx):
        for j in range(ny):
            points[:,i,j] = hypocenter_new + x[i]*strike_vector1 + y[j]*dip_vector1
    new_points=points.reshape((3,nx*ny))
    tree = spatial.cKDTree(np.transpose(new_points))
    # Put the hypocenters of all the earthquakes into a array
    pts = [] 
    for i in range(len(x_longitude)):
        pts.append([x_longitude[i],y_latitude[i],-1000*total_data[:,2][i]])
    pts = np.array(pts)
    distance_from_plane_1 = tree.query(pts)[0]
    
    x = np.linspace(-RLD_2/2.,RLD_2/2.0,nx)#note that rld and rw are in meters
    y = np.linspace(-RW_2/2.,RW_2/2.0,ny)
   
    #Extend the length and width of the fault plane in the 4 different directions. Each of these sets of for loops creates a set of points for 1/4 of the plane, upper left, upper right, etc
    points = np.zeros((3,nx,ny))
    for i in range(nx):
        for j in range(ny):
            points[:,i,j] = hypocenter_new + x[i]*strike_vector2 + y[j]*dip_vector2
    new_points=points.reshape((3,nx*ny))
    tree = spatial.cKDTree(np.transpose(new_points))
    # Put the hypocenters of all the earthquakes into a array
    pts = [] 
    for i in range(len(x_longitude)):
        pts.append([x_longitude[i],y_latitude[i],-1000*total_data[:,2][i]])
    pts = np.array(pts)
    distance_from_plane_2 = tree.query(pts)[0]
    
    if dist_threshold == 'Auto':
        def dist_fun(M):
            k = 0.53
            d0 = 3
            M0 = 6
            Mref = -(np.log10(d0)/k)+M0
            D = 10 ** (k*(M-Mref))
            return D
    
        if Event_of_interest == 'first':
            threshhold = 1000 * dist_fun(parameters['Magnitude_event1'])
        else:
            threshhold = 1000 * dist_fun(parameters['Magnitude_event2'])
    else:
        thresh_6 = 3000
        thresh_5 = np.copy(dist_threshold)
        if Event_of_interest == 'first':

            if parameters['Magnitude_event1'] >= 6:
                threshhold = thresh_6
            else:
                threshhold = thresh_5
        else:
            if parameters['Magnitude_event2'] >= 6:
                threshhold = thresh_6
            else:
                threshhold = thresh_5
    # Now to determine which is the true fault plane by counting earthquakes. The FP with more is taken as the true fault plane, we begin counting at an hour after the earthquake and require at least 50 events along a single plane
    hour_choice = 1
    while True:
        if hour_choice == 360*2: # Go out a month in time, if still can't determine FP, we stop
            print('Nodal Plane 1: Amount of Earthquakes',len(total_data_first_hour[distance_mask_1_fh,3]))
            print('Nodal Plane 2: Amount of Earthquakes',len(total_data_first_hour[distance_mask_2_fh,3]))
            break
        # Create masks to weed out earthquakes that are too far
        distance_mask_1 = distance_from_plane_1 <= threshhold
        distance_mask_2 = distance_from_plane_2 <= threshhold

        #Mask the data for the first hour after the main earthquake, only use this for fault plane determination
        if Event_of_interest == 'first':
            first_hour = times > parameters['Time_event1']
            total_data_first_hour = total_data[first_hour,:]
            times_first_hour = times[first_hour]
            distance_mask_1_fh = distance_mask_1[first_hour]
            distance_mask_2_fh = distance_mask_2[first_hour]
            first_hour = times_first_hour < (parameters['Time_event1'] + np.timedelta64(hour_choice,'h'))
            total_data_first_hour = total_data_first_hour[first_hour,:]
            times_first_hour = times_first_hour[first_hour]
            distance_mask_1_fh = distance_mask_1_fh[first_hour]
            distance_mask_2_fh = distance_mask_2_fh[first_hour]
        else:
            first_hour = times > parameters['Time_event2']
            total_data_first_hour = total_data[first_hour,:]
            times_first_hour = times[first_hour]
            distance_mask_1_fh = distance_mask_1[first_hour]
            distance_mask_2_fh = distance_mask_2[first_hour]
            first_hour = times_first_hour < (parameters['Time_event2'] + np.timedelta64(hour_choice,'h'))
            total_data_first_hour = total_data_first_hour[first_hour,:]
            times_first_hour = times_first_hour[first_hour]
            distance_mask_1_fh = distance_mask_1_fh[first_hour]
            distance_mask_2_fh = distance_mask_2_fh[first_hour]
       
        if (len(total_data_first_hour[distance_mask_1_fh,3])>50 or len(total_data_first_hour[distance_mask_2_fh,3])>50) and len(total_data_first_hour[distance_mask_1_fh,3]) != len(total_data_first_hour[distance_mask_2_fh,3]):
            print('Nodal Plane 1: Amount of Earthquakes',len(total_data_first_hour[distance_mask_1_fh,3]))
            print('Nodal Plane 2: Amount of Earthquakes',len(total_data_first_hour[distance_mask_2_fh,3]))
            break # Leave the while loop, we have enough information to find the FP
        else: # If niether have more than 50 EQs, we do not have enough information to determine the fault plane, increment the hour by 1
            hour_choice += 1
            continue
    #Following selects the fault plane based off which plane has more events. Then it cuts the entire dataset to be within the distance threshold of the selected fault plane
    if len(total_data_first_hour[distance_mask_1_fh,3]) > (len(total_data_first_hour[distance_mask_2_fh,3])):
        total_data = total_data[distance_mask_1]
        times = times[distance_mask_1]
        total_distances = distance_from_plane_1
        if Event_of_interest == 'first':
            print('The Selected Strike is',parameters['FP_strike1_event1'])
            
        else:
            print('The Selected Strike is',parameters['FP_strike1_event2'])
    elif len(total_data_first_hour[distance_mask_1_fh,3]) < (len(total_data_first_hour[distance_mask_2_fh,3])):
        total_data = total_data[distance_mask_2]
        times = times[distance_mask_2]
        total_distances = distance_from_plane_2
        if Event_of_interest == 'first':
            print('The Selected Strike is',parameters['FP_strike2_event1'])
        else:
            print('The Selected Strike is',parameters['FP_strike2_event2'])
    else:
        print('Insufficient Data: Not Enough Events to Determine Fault Plane')
        raise IndexError
    if (len(total_data_first_hour[distance_mask_1_fh,3])>50 or len(total_data_first_hour[distance_mask_2_fh,3])>50) and len(total_data_first_hour[distance_mask_1_fh,3]) != len(total_data_first_hour[distance_mask_2_fh,3]):
            pass
    else:
        print('Insufficient Data: Not Enough Events to Determine Fault Plane')
        raise IndexError
    return total_data, times, total_distances #Returns the cut data, times, and the distance array for all of the data, in case more points are needed later

def mc_ts(total_data,times,parameters,Event_of_interest,second_quake_exsists,na_time,precut_magnitude): # Requires the data, no alert time, precut magnitude
    '''A function to determine a magnitude of completeness time series during the event. This function accomodates the automatic 
    no alert time and precut magnitude. Returns data cut at the automatic Mc and/or NA time, and an updated parameters dictionary'''
    # Need to only look post-event for this: So cut the data accordingly
    if Event_of_interest == 'first':
        mask = times > parameters['Time_event1']
        post_event_data = total_data[mask,:]
        post_event_times = times[mask]
        if second_quake_exsists == 'yes':
            mask2 = post_event_times < parameters['Time_event2']
            post_event_data = post_event_data[mask2,:]
            post_event_times = post_event_times[mask2]
    else:
        mask = times > parameters['Time_event2']
        post_event_data = total_data[mask,:]
        post_event_times = times[mask]
    #Need a background Mc to compare to, so cut the catalog to get this data
    pre_mask = times <= parameters['Time_LE1']
    pre_data= total_data[pre_mask,:]
    pre_times = times[pre_mask]
    pre_mask2 = pre_times >= parameters['Tmin']
    pre_data = pre_data[pre_mask2,:]
    pre_times = pre_times[pre_mask2]
    
    # Define the bins
    Mrange = np.round(np.arange(np.min(total_data[:,3]),np.max(total_data[:,3])+0.1,0.1),2)
    mc_ts = []
    mc_ts_times = []
    #Create a time series with 300-events each, moving forward 1 at a time
    if len(post_event_times) > 300:
        window = 300
    else: # If less than 300 events, use the minimum 50 as the window
        window = 50
    for i in range((len(post_event_times)-window)):
        subset = post_event_data[i:i+window,:]
        # Calculate mc with maximum curvature method for each time step
        bin_count,edges = np.histogram(subset,np.append(Mrange,np.array([Mrange[-1]+0.1]))- 0.1/2)
   
        max_bin_count = np.max(bin_count)
        max_bin_index = np.where(bin_count == max_bin_count)[0]
        mc = Mrange[max_bin_index] + parameters['mc_correction']
        mc_ts.append(mc[0])
        mc_ts_times.append(post_event_times[i+window]) #Assign a time to the mc
    mc_ts = np.array(mc_ts)
    mc_ts_times = np.array(mc_ts_times)
   # Calculate the background mc for comparison
    Mrange = np.round(np.arange(np.min(total_data[:,3]),np.max(total_data[:,3])+parameters['bin_width'],parameters['bin_width']),1)
    background_mc = MC_function(pre_data,Mrange,parameters)
    if len(mc_ts) == 0:
        print('No Mc_ts available')
        raise ValueError()
    try: # Take the auto NA time as the time where the MC returns to the pre-event level
        index = np.where(mc_ts <= background_mc)[0][0] 
    except:
        index = np.where(mc_ts <= background_mc)[0]
    if na_time == 'Auto':
        if Event_of_interest == 'first':
            x = np.timedelta64(mc_ts_times[index] - parameters['Time_event1'],'h') # Take the auto NA time as the time where the MC returns to the pre-event level

            hours = x.astype('timedelta64[h]')

            parameters['days_exclude1'] = float(hours / np.timedelta64(24,'h'))
            parameters['days_exclude2'] = float(hours / np.timedelta64(24,'h'))
        else:
            x = np.timedelta64(mc_ts_times[index] - parameters['Time_event2'],'h')

            hours = x.astype('timedelta64[h]')

            parameters['days_exclude2'] = float(hours / np.timedelta64(24,'h'))
            parameters['days_exclude1'] = float(hours / np.timedelta64(24,'h'))

        if parameters['days_exclude1'] < 0.05:
            parameters['days_exclude1'] = 0.05
            parameters['days_exclude2'] = 0.05

    
# Pre-cut at background Mc, comment if you don't want to do this
    if precut_magnitude == 'Auto':
        precut = total_data[:,3] >= background_mc
        total_data = total_data[precut,:]
        times = times[precut]

    if len(times) == 0:
        print('0 Data above the pre-event completeness after the automatic no-alert time')
        raise ValueError
    return total_data, times, parameters # Returns data cut at the automatic Mc and/or NA time, and an updated parameters dictionary

def decday_to_smaller_units(decday): 
    '''A function to convert from .5, or 1.3 days, to days, hours, minutes, seconds. Returns the number of days, hours, minutes, and 
    seconds that make up the number inputted in decimel day notation'''
    d = np.floor(decday)
    h = np.floor(decday*24)
    m = np.floor((decday*24 - h) * 60)
    s = np.floor((((decday*24 - h) * 60) - m) * 60)
    return int(d), int(h), int(m), int(s) #Returns the number of days, hours, minutes, seconds from the inputted day in decimel form

def seperate_data(times,total_data,parameters,second_quake_exsists): #Takes the distance-cut times and data
    '''This function creates the pre, bt, and aft subsections. Returns the data and times subsections as well as copies of the original times
    for the pre and bt subsections, and the magnitude bins'''
    #Defines the bins
    Mrange = np.round(np.arange(np.min(total_data[:,3]),np.max(total_data[:,3])+parameters['bin_width'],parameters['bin_width']),1)
    # Create the pre subsection
    pre_mask = times <= parameters['Time_LE1'] # Use LE1 for both first and second event, that way we stop the reference calculation for the second earthquake when the first in the pair occurs
    pre_data= total_data[pre_mask,:]
    pre_times = times[pre_mask]
    pre_mask2 = pre_times >= parameters['Tmin']
    pre_data = pre_data[pre_mask2,:]
    pre_times = pre_times[pre_mask2]
    pre_data_orig = pre_data
    pre_times_orig = pre_times
    
    # Create the bt subsection
    if int(parameters['days_exclude1']) % parameters['days_exclude1'] == 0 and parameters['days_exclude1'] >= 1:
        parameters['days_exclude1'] = int(parameters['days_exclude1'])
    if float(parameters['days_exclude1']).is_integer():
        bt_mask = times >= (parameters['Time_event1'] + np.timedelta64(parameters['days_exclude1'],'D'))
    else:   
        d1,h1,m1,s1 = decday_to_smaller_units(parameters['days_exclude1'])
        bt_mask = times >= (parameters['Time_event1'] + np.timedelta64(d1,'D') + np.timedelta64(h1,'h') + np.timedelta64(m1,'m') + np.timedelta64(s1,'s'))
    bt_data = total_data[bt_mask,:]
    bt_times = times[bt_mask]
    if second_quake_exsists == 'yes':
        bt_mask_2 = bt_times <= parameters['Time_LE2']
        bt_data = bt_data[bt_mask_2]
        bt_times = bt_times[bt_mask_2]

    bt_data_orig = bt_data
    bt_times_orig = bt_times
    # Create the aft subsection
    if second_quake_exsists == 'yes':
        if int(parameters['days_exclude2']) % parameters['days_exclude2'] == 0 and parameters['days_exclude2'] >= 1:
            parameters['days_exclude2'] = int(parameters['days_exclude2'])
        if float(parameters['days_exclude2']).is_integer():
            aft_mask = times >= (parameters['Time_event2'] + np.timedelta64(parameters['days_exclude2'],'D'))
        else:   
            d2,h2,m2,s2 = decday_to_smaller_units(parameters['days_exclude1'])
            aft_mask = times >= (parameters['Time_event2'] + np.timedelta64(d2,'D') + np.timedelta64(h2,'h') + np.timedelta64(m2,'m') + np.timedelta64(s2,'s'))


        aft_data = total_data[aft_mask,:]
        aft_data
        aft_times = times[aft_mask]
        
    else:
        aft_data = np.nan
        aft_times = np.nan
        
    return pre_data, pre_times, pre_data_orig, bt_data, bt_times, bt_data_orig, aft_data, aft_times, Mrange, bt_times_orig, pre_times_orig
    # Returns the data and times subsections (pre_data,pre_times,bt_times,bt_data,aft_times,aft_data), as well as copies of the original times for pre- and bt- (pre_times_orig, etc). also returns the bins Mrange
    
def MC_function(data,Mrange,parameters): #Takes a subset of Data, Mrange (bins)
    '''Calculates Magnitude of Completeness based off the Maximum Curvature method, using a 
    correction factor. Returns the magnitude of completeness'''
    corr = parameters['mc_correction'] # Correction Factor
    bin_count,edges = np.histogram(data,np.append(Mrange,np.array([Mrange[-1]+parameters['bin_width']]))- parameters['bin_width']/2)
    max_bin_count = np.max(bin_count)
    max_bin_index = np.where(bin_count == max_bin_count)[0]
    mc = Mrange[max_bin_index] + corr
    return mc[0] # Returns the magnitude of completeness

def seperated_data_MC_determination(pre_data,pre_times,bt_data,bt_times,aft_data,aft_times,second_quake_exsists,parameters,Mrange):
    '''Takes the data for each subset (pre_data,pre_times,bt_data,etc), and bins. The function
    Calculates the overall Mc for each subset and cuts each subset by that Mc
    Returns the newly cut data and overall Mcs for each subset'''
    
    #Pre-event Subset
    mc_pre = MC_function(pre_data[:,3],Mrange,parameters)
    mc_pre = round(mc_pre,1)
    print('Overall Magnitude of Completeness for Pre-Times', mc_pre)
    mc_pre_mask = pre_data[:,3] >= (mc_pre - parameters['mc_correction'])
    pre_data = pre_data[mc_pre_mask,:]
    pre_times = pre_times[mc_pre_mask]
    
    # Between-event or post-event 1 subset
    mc_bt = MC_function(bt_data[:,3],Mrange,parameters)
    mc_bt = round(mc_bt,1)
    print('Overall Magnitude of Completeness for Between-Times', mc_bt)
    mc_bt_mask = bt_data[:,3] >= (mc_bt)
    bt_data = bt_data[mc_bt_mask,:]
    bt_times = bt_times[mc_bt_mask]
    
    #aft Subset
    if second_quake_exsists == 'yes':
        mc_aft = MC_function(aft_data[:,3],Mrange,parameters)
        mc_aft = round(mc_aft,1)
        print('Overall Magnitude of Completeness for After-Times', mc_aft)
        mc_aft_mask = aft_data[:,3] >= (mc_aft - parameters['mc_correction'])
        aft_data = aft_data[mc_aft_mask,:]
        aft_times = aft_times[mc_aft_mask]    
    else:
        mc_aft = np.nan
    #Returns the newly cut data, and overall Mcs for each subset
    return pre_data, pre_times, mc_pre, bt_data, bt_times, mc_bt, aft_data, aft_times, mc_aft
    
    
def time64todecyear(time_original,single = 'single'):  #Input a date(s) in numpy datetime 64, for multiple dates, set single to 'multiple'
    '''Takes a numpy datetime64 and converts to a datetime so we can use the %j command to get day of year,
    and then converts the time to a float, or decimel year notation. Returns a date(s) in decimel year'''
    #https://stackoverflow.com/questions/13703720/converting-between-datetime-timestamp-and-datetime64
    unix_epoch = np.datetime64(0, 's')
    one_second = np.timedelta64(1, 's')
    new_times = []
    if single == 'single': #From https://stackoverflow.com/questions/6451655/how-to-convert-python-datetime-dates-to-decimal-float-years
        seconds_since_epoch = (time_original - unix_epoch) / one_second
        d = datetime.datetime.utcfromtimestamp(seconds_since_epoch)
        b = (((float(d.strftime("%j")))/365.25)+((float(d.strftime("%H")))/24/365.25)+((float(d.strftime("%M")))/60/24/365.25)+((float(d.strftime("%S")))/3600/60/24/365.25)) + float(d.strftime("%Y"))
        new_times.append(b)
    else:
        for i in time_original:
            seconds_since_epoch = (i - unix_epoch) / one_second
            d = datetime.datetime.utcfromtimestamp(seconds_since_epoch)
            b = (((float(d.strftime("%j")))/365.25)+((float(d.strftime("%H")))/24/365.25)+((float(d.strftime("%M")))/60/24/365.25)+((float(d.strftime("%S")))/3600/60/24/365.25)) + float(d.strftime("%Y"))
            new_times.append(b)
    return np.array(new_times) # Returns a date(s) in decimel year

def linearity_test(data,mc,parameters): #Input: Data, Magnitude of Completeness from b-time function
    '''This function takes a subset of data, and asses its linearity. Adapted from GW19 (NLIndex),
    original source is Tormann et al (2014).
    returns the Mc originally provided, the b-value for that Mc and this data set, the linearity test result, and the standard deviation of that b'''
    binnumb=5 #Required Mcs to cycle through

    flag = np.nan #Initialize the Flag
    if len(data[:,0]) < parameters['Nmin']: # If the data is not longer than 50 Events then Flag 1 and nans for the values
        flag = 1
        bestmc = np.nan
        bestb = np.nan
        bestsig = np.nan
    #Define bins
    Mrange = np.round(np.arange(mc,np.max(data[:,3])+parameters['bin_width'],parameters['bin_width']),1)
    
    #Histogram of the events
    Numb, edges = np.histogram(data[:,3],np.append(Mrange,np.array([Mrange[-1]+parameters['bin_width']]))- parameters['bin_width']/2)
    Numbh = np.flip(Numb)
    Ncumh = np.cumsum(Numbh) #Calculates Cumulative Sum
    Ncum = np.flip(Ncumh) #This is N-Larger
    nmin_mask = Ncum >= parameters['Nmin'] #Look only at magnitude bins that have over Nmin/50 events
    Mcrange = Mrange[nmin_mask] 
    if len(Mcrange) < binnumb: #Ensure there are at least binnumb (5) magnitudes post-nminmask 
        # If not, assign Flag 2, and calculate the b-value and standard deviation for this data set
        # Not enough to compute a Non-Linearity Index, but not necessarily a bad dataset
        flag = 2
        bestmc = mc
        bestb = ((1/(np.mean(data[:,3])-(bestmc-(parameters['bin_width']/2))))*np.log10(np.e))
        sig1 = (sum((data[:,3]-np.mean(data[:,3]))**2))/(len(data[:,3])*(len(data[:,3])-1)) 
        sig1 = np.sqrt(sig1)
        bestsig = 2.30*sig1*bestb**2
    else: # Here there are enough events to compute a Non-Linearity Index
        # Initialize an array to update with data
        b = np.ones((len(Mcrange),3))*np.nan
        for i in np.arange(0,len(Mcrange)):  # for each index in the range of Mcrange
            mc_mask = data[:,3] >= round(Mcrange[i],1) # Cut the data at this Mc
            # Then calculate b and standard deviation at this Mc-level
            b1 = (1/(np.mean(data[mc_mask,3])-(Mcrange[i]-(parameters['bin_width']/2))))*np.log10(np.e)
            sig1 = (sum((data[mc_mask,3]-np.mean(data[mc_mask,3]))**2))/(sum(mc_mask)*(sum(mc_mask)-1)) 
            sig1 = np.sqrt(sig1)
            sig1 = 2.30*sig1*b1**2
            # Append these results to the matrix
            b[i,:] = [Mcrange[i],b1,sig1]
            # Repeat for each Mc in Mcrange
            
        # Now to assess Linearity    
        k = 0 # Initial Index for the for-loop below
        for_loop_upper_limit = np.max(Mcrange) - (0.1 * binnumb - parameters['mc_correction']) #Not entirely sure about why the subtracted term, but in Matlab code
        marker = np.ones((len(np.arange(mc,for_loop_upper_limit+0.1,0.1)),5)) # Create an Array to hold the data
        for mcut in np.round(np.arange(mc,for_loop_upper_limit+0.1,0.1),1): # mcut in a range of Mcs
            NLIndex = np.std(b[k:,1]) / np.max(b[k:,2]) #calculate NL Index
            NLIndexw = (1 / (len(Mcrange) - k)) * NLIndex #calculates NLIndexw
            marker[k,:] = [mcut,b[k,1],b[k,2],NLIndex,NLIndexw] #Updates the data to the marker array
            k += 1 #Updates the index
        bestmc = marker[0,0] #Sets mc back to the initial mc provided
        bestb = marker[0,1] # Sets the b-value back to the initial b-value calculated. The purpose of this function is whether or not they are useable values
        bestsig = marker[0,2]
        # If the NLIndex of the data set is under 1; the data set is linear and can be used. Flag = 3
        if marker[0,3] <= 1:
            flag = 3
    return bestmc, bestb, flag, bestsig # returns the Mc originally provided, the b-value for that Mc and this data set, the linearity test flag (result), and the standard deviation of that b

def b_time(data,times,N,parameters): #Data, Npre = 250 or Npost = 400 ie the number of events for each time series calculation,
    '''Adapted from GW19, (abwithtime). This function calculates a b-value time series, and
    asseses linearity at each point.
    Returns the time series of Mcs, b-values, standard deviations, a-values, and linearity test results'''
    step = 1
    newt2 = np.copy(data) # so newt2 = data, using newt2 to be consistent with GW19
    # Define bins
    Mrange = np.round(np.arange(np.min(newt2[:,3]),np.max(newt2[:,3])+parameters['bin_width'],parameters['bin_width']),1)
    
    timev = np.copy(times)
    time_length = len(timev) #No Estimates Pre in Matlab
    
    # Initialize Arrays to modify later
    magco_median = np.ones((time_length,1)) * np.nan
    bv_median = np.ones((time_length,1)) * np.nan
    av_ann = np.ones((time_length,1)) * np.nan
    result_flag_median = np.ones((time_length,1)) * np.nan
    sig1_median = np.ones((time_length,1)) * np.nan
    av_ann_median = np.ones((time_length,1)) * np.nan
    
    # j is number, j goes from 0 to length of data/time arrays
    for j in np.arange(0,time_length,step):
        ind1 = np.where(timev <= timev[j]) #Index in the times, where the current time in the loop is first greater than or equal to
        ind1 = ind1[-1][-1]
        if ind1 < parameters['Nmin']: #Require a set of 50 events, so check if the ind1 is less than 50
            b_orig = []
        elif (ind1 >= parameters['Nmin']) and (ind1 < N): #If between 50 and 250 (pre) or 400 (bt,aft), then take the data set to be these ind1 events
            b_orig = newt2[0:ind1,:]
            time_orig = timev[0:ind1]
        else: #Once greater than 250 (pre) or 400 (aft,bt) then move through them 1 at a time in 250 or 400 event windows
            b_orig = newt2[ind1-N+1:ind1+1,:]
            time_orig = timev[ind1-N+1:ind1+1]
        if len(b_orig) != 0: #If the first if above went off, this part of the code won't run
            if len(b_orig[:,1]) >= parameters['Nmin']: #Ensure the dataset is longer than 50
                bv = np.nan #Initialize values
                av_ann = np.nan
                result_flag = np.nan
                b = b_orig
                F = 1 # Annual vs Daily (F=365)
                T = F * (time64todecyear(timev[j],'single') - time64todecyear(np.min(time_orig),'single')) #Must convert to Decimel Year, time span of the 250 or 400 events

                #Re-assess Completeness
                magco = np.round(MC_function(b[:,3],Mrange,parameters),1)
                
                b_mask = b[:,3] >= magco
                b = b[b_mask,:]
                N3 = sum(b_mask)
                # Ensure that after cutting for the new Mc there are still over 50 events
                if N3 >= parameters['Nmin']:
                    bestmc,bv,result_flag,sig1 = linearity_test(b,magco,parameters) #If so, then calculate the linearity test on this subset of events
                    # bestmc = magnitude of completeness of the subset, bv = b-value of the subset, result_flag is the linearity test result, sig1 is the standard deviation of the b-value
                    #print('Flag + b', result_flag,bv)
                    if (result_flag == 2) or (result_flag == 3): #Accept Flags 2 and 3 from Linearity Test
                        magco = np.round(bestmc,1)
                        b_mask = b[:,3] >= magco #Cut by Mc
                        N_b = sum(b_mask)
                        av_ann = np.log10(N_b/T) + bv*magco #Calculate the annualized a-value
                        
                    else:
                        bv = np.nan # If the subset did not have 50 events, assign nans
                        magco = bestmc
                        av_ann = np.nan
                        sig1 = np.nan
                    # Update the time series
                    magco_median[j] = magco # Mc
                    bv_median[j] = bv # b-value
                    sig1_median[j] = sig1 #Standard deviation of the b-value
                    av_ann_median[j] = av_ann # Annualized a-value
                    result_flag_median[j] = result_flag #Linearity Test Flag
    return magco_median, bv_median, sig1_median, av_ann_median, result_flag_median #Returns the time series of Mcs, b-values, standard deviations, a-values, and result flags

def Breg_NLI(data,times,parameters,Event_of_interest): # This function is called if there aren't enough events to compute a background-b-value
    '''Adapted from GW19 function Breg_NLI, computes the regional b-value
        Returns the new regional background b-value, the new original pre data and times, the new overall pre-Mc, and the uncertainty in b'''
    # First define some necessary variables
    b_orig = data 
    N = len(b_orig[:,0]) # Length of dataset
    Tmax = time64todecyear(parameters['Time_event1']) # last pre-event time, converted to decimel
    if Event_of_interest == 'second':
        Mtarg = parameters['Magnitude_event2']
    else:
        Mtarg = parameters['Magnitude_event1']
    Tmin = time64todecyear(np.min(times)) # Time of first event in the dataset
    
    T = Tmax - Tmin # The range of times
   
    Nmin = parameters['Nmin']
    
    corr = parameters['mc_correction']
    sig = np.nan
    bv = np.nan
    av_ann = np.nan
    pr = np.nan
    result_flag = np.nan
    #Define bins
    Mrange = np.round(np.arange(np.min(b_orig[:,3]),np.max(b_orig[:,3])+parameters['bin_width'],parameters['bin_width']),1)
    
    # Compute Histogram
    cntnumb_orig,edges = np.histogram(b_orig[:,3],np.append(Mrange,np.array([Mrange[-1]+parameters['bin_width']]))- parameters['bin_width']/2)
    cntnumbA_orig = cntnumb_orig / T #Normalize the histogram
    
    b = b_orig # data
    # Mc Calculation
    magco = np.round(MC_function(b[:,3],Mrange,parameters),1)
    
    # Cut the data at the Mc
    mask = b[:,3] >= magco
    b = b[mask,:]
    N_b = np.sum(mask)
    
    if N_b >= Nmin: # Ensure more events than Nmin
        bestmc,bv,result_flag,sig = linearity_test(b,magco,parameters) # Check for Linearity
        if (result_flag == 3 or result_flag == 2): # If Linear keep the b-value and cut 
            magco = bestmc 
            mask2 = b[:,3] >= magco # Cut at the magco from the Linearity Test (In this case its repetitive since it will always be the same)
            N_b = sum(mask2) # sum the data that survives the magco cut
            
            av_ann = np.log10(N_b/T) + bv*magco # Annualized a-value
            av = np.log10(N_b) + bv*magco # non-annualized a -value
        else:
            bv = np.nan
            magco = bestmc
            av_ann = np.nan
            sig = np.nan
    
    mask3 = b_orig[:,3] >= magco # Cut the original input data by this Mc to create a new original pre-catalog, since the old one was insufficient
    cat_Reg = b_orig[mask3,:]
    times_1 = times[mask3]
    return bv,cat_Reg,magco,times_1,sig # Return the new regional background b-value, the new original pre data and times, the new overall pre-Mc, and the uncertainty in b

def find_closest_quakes(times_orig,total_data_orig,total_distances,parameters,Event_of_interest): # Need to take in the unmodified data that was created right after the distance cut, and the total distances for each
    '''When there are not sufficient earthquakes near the fault plane for a reference b-value, we increase the distance until we have
    250 events to calculate a regional b-value.
    Returns the regional background b-value, the new original pre-event dataset, the new overall Mc for the pre-events, and the uncertainty in b'''
    # Cut this total data to be before the first event, and after the Tmin time.
    pre_mask = times_orig <= parameters['Time_LE1']
    total_data_orig = total_data_orig[pre_mask,:]
    times_orig = times_orig[pre_mask]
    total_distances = total_distances[pre_mask]
    pre_mask2 = times_orig >= parameters['Tmin']
    total_data_orig = total_data_orig[pre_mask2,:]
    times_orig = times_orig[pre_mask2]
    total_distances = total_distances[pre_mask2]

    # Sort the data by distance, so the first event is the closest to the FP, last is farthest
    times_orig_sort = times_orig[total_distances.argsort()]
    total_data_orig_sort = total_data_orig[total_distances.argsort(),:]
    if len(times_orig_sort) >= parameters['Nreg']: 
        print('Extra Distance from Plane', total_distances[total_distances.argsort()][parameters['Nreg']]/1000,'km')
    else:
        print('Extra Distance from Plane', total_distances[total_distances.argsort()][-1]/1000,'km')
    if len(times_orig_sort) >= parameters['Nreg']: #Check to see if there are the required number of events in this catalog. (Nreg/250). If not it uses what it has
        # If so, then it takes the closest Nreg/250 events
        closest_quakes = total_data_orig_sort[0:parameters['Nreg'],:]
        closest_times = times_orig_sort[0:parameters['Nreg']]
        # Now resort the new 250-event catalog by time
        closest_quakes_final = closest_quakes[closest_times.argsort(),:]
        closest_times_final = closest_times[closest_times.argsort()]
    else:
        print('Not 250 Events to compute regional b-value, consider widening initial search')
        raise IndexError
    # Call Breg_NLI to compute the background b-value on this dataset
    bpre_ts,pre_data_orig,mc_pre,pre_times_orig,sig = Breg_NLI(closest_quakes_final,closest_times_final,parameters,Event_of_interest)
    return bpre_ts,pre_data_orig,mc_pre,pre_times_orig,sig # Return the bpre_ts (regional background b-value), and the new original pre-event dataset, the new overall Mc for the pre-events, and the uncertainty in b

def b_reference_uncertainity(bpre_ts,sig1pre_ts): # Requires the time series of b-values, and the time series of their uncertainties
    '''Calculates and returns the uncertainty of the mean from the pre-event time series'''
    b_array = []
    for i in range(len(bpre_ts)): #Need to weed out nan values from the arrays
        if np.isnan(bpre_ts[i]) == False and np.isnan(sig1pre_ts[i]) == False:
            b_array.append(bpre_ts[i])
    b_array = np.array(b_array)

    sd = np.std(b_array,ddof=1)
    uncertainty = sd / np.sqrt(len(b_array))
    return uncertainty

def calculate_aft_stats(baft_ts,Mcaft_ts,aft_times,avannaft_ts): # takes the aft b times series, as well as the mc, a-value time series. aft_times are the aft-times dataset series
    '''Function to carry out the final lines of GW19 Script Run_TLS_Gulia_Wiemer. The function extracts necessary values from the post
    second event (aft) results. Returns maximum aft b-value, the Mc at that time, the time of that b-value, and the a-value at this time'''
    baft_max = np.nanmax(baft_ts) # Takes the maximum b-value in the time series, and then gets all the corresponding information to that b-value
    index = np.where(baft_ts == baft_max)[0]
    magcopostmax = Mcaft_ts[index][0][0]
    taft_max = aft_times[index][0]
    a_aftmax = avannaft_ts[index][0][0]
    return baft_max, magcopostmax, taft_max, a_aftmax # Returns maximum aft b-value, the Mc at that time, the time of that b-value, and the a-value at this time

def pre_plot_pre_prep(pre_data_orig,parameters,mc_pre,pre_times_orig,total_data): #Takes the original pre data, times, and mc. As well as the parameters dictionary
    '''Function to extract pre-event results for plotting, and this does repeat a b-value calculation but the number is always identical to the
    previously determined pre-event b from the time series.
    #Returns pre-event b and standard deviation, the FMD plot label, Nlarger for the data, the time range of the data, 
    the Nlarger for the predicted data,and the Magnitudes above Mc. Nlarger is the N in the GR formula'''
    mask = pre_data_orig[:,3]>=mc_pre # Cuts the original_pre data at the Mc
    pre_data_orig = pre_data_orig[mask]
    pre_times_orig = pre_times_orig[mask]
    #Defines Bins
    Mrange = np.round(np.arange(np.min(total_data[:,3]),np.max(total_data[:,3])+parameters['bin_width'],parameters['bin_width']),1)
    # Compute Histogram to get N-larger (number of EQs > M_i)
    cntnumb, edges = np.histogram(pre_data_orig[:,3],np.append(Mrange,np.array([Mrange[-1]+parameters['bin_width']]))- parameters['bin_width']/2)
    Numbh = np.flip(cntnumb)
    Ncumh = np.cumsum(Numbh)
    Ncumpre = np.flip(Ncumh) # N-larger

    pre_times_dec = time64todecyear(pre_times_orig[-1],'single')
    Tmin_dec = time64todecyear(pre_times_orig[0],'single')
    Tcatpre = pre_times_dec - Tmin_dec # Range of times in this catalog to annualize it

    if len(pre_times_orig) >= parameters['Nmin']: # ensure more than 50 events
        # Then calculate the pre b-value, annualized a-value, and the range of magnitudes above the Mc
        bv = (1 / (np.mean(pre_data_orig[:,3]) - (mc_pre - (parameters['bin_width'] / 2))))*np.log10(np.e)
        av_ann = np.log10(len(pre_data_orig[:,3])/Tcatpre) + bv*mc_pre
        Mrange_pre = Mrange[Mrange>= mc_pre]
        N1 = 10**(av_ann - bv*Mrange_pre) # Nlarger predicted by the a and b values
    else:
        print('Insufficient Data: Not More than 50 (Nmin) Events:',len(pre_times_orig),'/ 50')
        raise IndexError
    # calculate the standard deviation for the pre-event b-value
    sig1_pre = np.sum((pre_data_orig[:,3] - np.mean(pre_data_orig[:,3]))**2) / (len(pre_data_orig[:,3])* (len(pre_data_orig[:,3])-1))
    sig1_pre = np.sqrt(sig1_pre)
    sig1_pre = 2.3 * sig1_pre * bv**2

    # Set up FMD Label for pre-events
    label_pre = 'Pre: b = {:.02f}'.format(bv) + ' $ \pm $ ' + '{:.02f}'.format(sig1_pre)  
    print('pre'+ str(pre_times_orig[0])[0:10] + ' to ' + str(pre_times_orig[-1])[0:10])
    return bv,sig1_pre,label_pre,Ncumpre,Tcatpre,N1,Mrange_pre #Returns pre-event b and standard deviation, the FMD plot label, Nlarger for the data, the time range, the Nlarger for the predicted data,and the Magnitudes above Mc

def pre_plot_bt_prep(bt_data_orig,parameters,mc_bt,bt_times_orig,Mrange,bt_times): # This does same as above function, but for bt.
    '''Function to extract bt-event, or post-event 1 results for plotting, and calculates an overall b-value for this data used to create the
    TLS score
    Returns the bt b-value, its uncertainty, FMD plot label, Nlarger for the bt data, the time range for this data, Nlarger predicted by the GR law, and the magnitude range'''
    mc_bt2 = mc_bt + parameters['mc_correction']
    mask = bt_data_orig[:,3]>=mc_bt2 
    bt_data_orig = bt_data_orig[mask,:]
    bt_times_orig = bt_times_orig[mask]
    # Compute Histogram to get N-larger (number of EQs > M_i)
    cntnumb, edges = np.histogram(bt_data_orig[:,3],np.append(Mrange,np.array([Mrange[-1]+parameters['bin_width']]))- parameters['bin_width']/2)
    Numbh = np.flip(cntnumb)
    Ncumh = np.cumsum(Numbh)
    Ncumbt = np.flip(Ncumh) # N-Larger

    bt_times_dec = time64todecyear(np.max(bt_times_orig),'single')
    Tmin_dec = time64todecyear(np.min(bt_times_orig),'single')
    Tcatbt = bt_times_dec - Tmin_dec # Range of times in this catalog to annualize it

    if len(bt_times_orig) >= parameters['Nmin']: # ensure more than 50 events
        # Then calculate the bt b-value, annualized a-value, and the range of magnitudes above the Mc
        bv_bt = (1 / (np.mean(bt_data_orig[:,3]) - (mc_bt2 - (parameters['bin_width'] / 2))))*np.log10(np.e)
        av_ann_bt = np.log10(len(bt_data_orig[:,3])/Tcatbt) + bv_bt*mc_bt2
        Mrange_bt = Mrange[Mrange >= mc_bt2]
        N2 = 10**(av_ann_bt - bv_bt*Mrange_bt) # Nlarger predicted by the a and b values
    else:
        print('Insufficient Data: Fewer than Nmin Events in the Between Catalog. There are:',len(bt_times_orig))
        raise IndexError
    # calculate the standard deviation for the bt-event b-value
    sig1_bt = np.sum((bt_data_orig[:,3] - np.mean(bt_data_orig[:,3]))**2) / (len(bt_data_orig[:,3])* (len(bt_data_orig[:,3])-1))
    sig1_bt = np.sqrt(sig1_bt)
    sig1_bt = 2.3 * sig1_bt * bv_bt**2
    # Set up FMD Label for pre-events
    label_bt = 'Post: b = {:.02f}'.format(bv_bt) + ' $ \pm $ ' + '{:.02f}'.format(sig1_bt) 
    print('Bt'+str(bt_times[0])[0:10] + ' to ' + str(bt_times[-1])[0:10])
    return bv_bt,sig1_bt,label_bt,Ncumbt,Tcatbt,N2,Mrange_bt # Returns the bt b-value, its uncertainty, FMD plot label, Nlarger for the bt data, the time range for this data, Nlarger predicted by the GR law, and the magnitude range'''

def pre_plot_aft_prep(aft_data,parameters,magcopostmax,aft_times,taft_max,baft_max,a_aftmax,Mrange): #Takes the after-data, and all of the variables associated with the maximum b-value post-event 2, determined in the aft_stats function
    '''Function to extract aft-event, or post-event 2 results for plotting
    Returns the FMD plot Label, Nlarger from a and b, the standard deviation of the aft-bvalue, the time range, Nlarger for the data, and the magnitude range above Mc'''
    mask = aft_times <= taft_max #Look only at times before the maximum-b time
    catmax = aft_data[mask]
    catmax_time = aft_times[mask]
    if len(catmax[:,0]) > parameters['Npost']: 
        catmax = catmax[-1:-parameters['Npost']-1:-1,:]
        catmax_time = catmax_time[-1:-parameters['Npost']-1:-1]

    tdiff = time64todecyear(taft_max,'single') - time64todecyear(np.min(catmax_time))
    # Calculate a histogram for N-larger
    cntnumb, edges = np.histogram(catmax[:,3],np.append(Mrange,np.array([Mrange[-1]+parameters['bin_width']]))- parameters['bin_width']/2)
    Numbh = np.flip(cntnumb)
    Ncumh = np.cumsum(Numbh)
    Ncumaft = np.flip(Ncumh) # N-Larger

    Mrange_aft = Mrange[Mrange>= magcopostmax] # Range of Magnitudes above Mc
    N3 = 10**(a_aftmax - baft_max*Mrange_aft) # Nlarger calculated from the a and b values

    # Calculate Standard Deviatiom
    sig1_aft = np.sum((catmax[:,3] - np.mean(catmax[:,3]))**2) / (len(catmax[:,3])* (len(catmax[:,3])-1))
    sig1_aft = np.sqrt(sig1_aft)
    sig1_aft = 2.3 * sig1_aft * baft_max**2
    # Set up FMD after Label
    label_aft = 'Post-Second Event: b = {:.02f}'.format(baft_max) + ' $\pm $ ' + '{:.02f}'.format(sig1_aft)
    print('Aft'+ ' ' + str(aft_times[0])[0:10] + ' to ' + str(taft_max)[0:10])
    return label_aft, N3, sig1_aft,tdiff, Ncumaft, Mrange_aft #Return the Label, N3 (Nlarger from a and b), the standard deviation, the time range, Nlarger for the data, and the magnitude range above Mc

def TLS(b,breference,b_sig,bref_sig):
    '''Function to calculate the percent change and TLS score (different representations of the same thing), and their uncertainties.
    Returns the percent change, TLS score, and their uncertainties'''
    percent_change = 100 * ((b - breference) / breference)
    a = np.sqrt((b_sig**2) + (bref_sig**2)) #UncertaintyError for b-bref
    percent_change_uncertainty = np.sqrt(((a/(b-breference))**2) + ((bref_sig/(breference))**2)) * 100 * np.abs((b - breference) / breference) #Uncertainty for dividing by bref and mult by 100
    
    TLS = 100 * (b / breference)
    TLS_uncertainty = np.sqrt(((b_sig/(b))**2) + ((bref_sig/(breference))**2)) * np.abs(TLS)
    return percent_change, TLS, percent_change_uncertainty, TLS_uncertainty

def plotting(parameters,Mrange,Ncumpre,Tcatpre,Mrange_pre,N1,label_pre,Mrange_bt,N2,Ncumbt,Tcatbt,tdiff,Mrange_aft,N3,label_aft,times,breference,bt_times,pre_times,aft_times,bpre_ts,bbt_ts,baft_ts,sig1bt_ts,sig1pre_ts,sig1aft_ts,percent_change,percent_change2,percent_change_uncertainty,percent_change2_uncertainty,TLS1,TLS1_uncertainty,TLS2,TLS2_uncertainty,bv_bt,bv,label_bt,second_quake_exsists,Event_of_interest,i,path,Ncumaft):
    '''Function that creates FMD, b-value time series graphs, and uncertainty graphs. It is recommended that this only be used if you are
    running a snall number of earthquakes, an alert threshold does need to be specified for certain plots.'''
    alert_threshold = 5
    fig, ax = plt.subplots(figsize = (2.5,3.5))
    ax.plot(Mrange_pre,N1,color = 'blue',label = label_pre,lw=0.5)
    ax.scatter(Mrange,Ncumpre/Tcatpre,color = 'blue',marker = '.',s=3)
    
    if bv_bt != -123456789:
        ax.plot(Mrange_bt,N2,color = 'gray',label = label_bt,lw=0.5)
        ax.scatter(Mrange,Ncumbt/Tcatbt,color = 'gray',marker = '.',s=3)
    if second_quake_exsists == 'yes':
        ax.scatter(Mrange,Ncumaft/tdiff,color = 'red',marker = '.',s=3)
        ax.plot(Mrange_aft,N3,color = 'red',label = label_aft)
    ax.set_yscale('log')
    ax.legend(fontsize=8,loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True,framealpha=1)
    ax.set_ylim(1e-5,1e6)
    ax.set_yticks((1e-4,1e-2,1e0,1e2,1e4))
    ax.set_xlim(1,6)
    ax.set_xticks((1,2,3,4,5,6))
    plt.xlabel('M')
    plt.title('Chalfant Valley M5.9 Foreshock FMD')
    plt.ylabel('Annualized Cumulative Number')
    plt.tight_layout()
    plt.savefig(path+'FMD/'+'{:03d}'.format(i)+'_FMD.png')
    plt.savefig(path+'FMD/'+'{:03d}'.format(i)+'_FMD.svg')


    #Time Series Plot
    plt.figure(figsize = (8,5))
    
    plt.hlines(breference,parameters['Time_event1']-np.timedelta64(110,'D'),np.max(times)+np.timedelta64(100,'D'),color = 'blue',label='Reference')
    plt.vlines(parameters['Time_event1'],0.5,2)
    plt.annotate('$ M_w = $ ' + str(parameters['Magnitude_event1']), ((parameters['Time_event1'] - np.timedelta64(25,'D')), 1.5),rotation = 90)
    plt.annotate('b = ' + '{:.2f}'.format(breference), ((parameters['Time_event1'] - np.timedelta64(110,'D')), breference + 0.025),color = 'blue')
    plt.annotate('reference', ((parameters['Time_event1'] - np.timedelta64(110,'D')), breference - 0.075),color = 'blue')
    plt.scatter(bt_times,bbt_ts,color = 'gray',marker = '.',s=5,label='Post First-Event')
    if len(bbt_ts) > 0:
        plt.fill_between(bt_times,(bbt_ts-sig1bt_ts)[:,0],(bbt_ts+sig1bt_ts)[:,0],color='gray',alpha=0.2,label= '1-$ \sigma $ Uncertainty')
#   plot a pre-event b time series?
    pre_event_TS_plot = False
    if pre_event_TS_plot == True:
        try:
            plt.hlines(breference,pre_times[0],np.max(times)+np.timedelta64(100,'D'),color = 'blue')
            plt.scatter(pre_times,bpre_ts,color = 'orange',marker = '.',s=5)
            plt.fill_between(pre_times,(bpre_ts-sig1pre_ts)[:,0],(bpre_ts+sig1pre_ts)[:,0],color='orange',alpha=0.2)
        except:
            pass
    
    if second_quake_exsists == 'yes':
        plt.vlines(parameters['Time_event2'],0.5,2)
        plt.scatter(aft_times,baft_ts,color = 'red',marker = '.',s=5,label='Post-Second Event')
        if len(baft_ts) > 0:
            plt.fill_between(aft_times,(baft_ts-sig1aft_ts)[:,0],(baft_ts+sig1aft_ts)[:,0],color='red',alpha=0.2,label='1-$ \sigma $ Uncertainty')
        
        plt.annotate('$ M_w = $ ' + str(parameters['Magnitude_event2']), ((parameters['Time_event2'] - np.timedelta64(25,'D')), 1.5),rotation = 90)
    plt.title('b-value Time Series')
    plt.xlabel('Time')
    plt.ylabel('b')
    plt.legend()
    plt.savefig(path+'TS/'+'{:03d}'.format(i)+'_TS.png')
    #plt.savefig(path+'TS/'+'{:03d}'.format(i)+'_TS.svg')
    plt.show()

    #Zoomed in Time Series Plot
    if second_quake_exsists == 'yes':
        plt.figure(figsize = (8,5))
        plt.hlines(breference,parameters['Time_event1']-np.timedelta64(50,'D'),np.max(times)+np.timedelta64(100,'D'),color = 'blue',label='Reference')
        plt.vlines(parameters['Time_event1'],0.5,2)
        plt.annotate('$ M_w = $ ' + str(parameters['Magnitude_event1']), ((parameters['Time_event1'] + np.timedelta64(2,'h')), 1.5),rotation = 90)
        plt.scatter(bt_times,bbt_ts,color = 'gray',marker = '.',s=5,label='Post-First Event')
        if len(bbt_ts) > 0:
            plt.fill_between(bt_times,(bbt_ts-sig1bt_ts)[:,0],(bbt_ts+sig1bt_ts)[:,0],color='gray',alpha=0.2,label='1-$ \sigma $ Uncertainty')
        if second_quake_exsists == 'yes':
            plt.vlines(parameters['Time_event2'],0.5,2)
            plt.annotate('$ M_w = $ ' + str(parameters['Magnitude_event2']), ((parameters['Time_event2'] + np.timedelta64(2,'h')), 1.5),rotation = 90)
        plt.xlim(parameters['Time_event1']-np.timedelta64(1,'D'),parameters['Time_event2']+np.timedelta64(1,'D'))
        plt.title('Between b-value Time Series')
        plt.xlabel('Time')
        plt.xticks([parameters['Time_event1'],parameters['Time_event2']])
        plt.ylabel('b')
        plt.legend(loc=9)
        plt.savefig(path+'TS/'+'{:03d}'.format(i)+'_TS_zoom.png')
        #plt.savefig(path+'TS/'+'{:03d}'.format(i)+'_TS_zoom.svg')
        plt.show()
    
    parameters = percent_threshhold(parameters,Event_of_interest,alert_threshold)
    # Traffic Light Plot for Event 1
    if Event_of_interest == 'first':
        if percent_change >= parameters['Yellow-Green']:
            plt.figure()
            plt.xlim(0,1)
            plt.ylim(0,1)
            plt.fill_between((0,1),0,1,color='green')
            plt.annotate('Clear', (0.30,0.45),size = 50)
            plt.annotate('First Event', (0.30,0.75),size = 20)
            plt.annotate('{:.2f}'.format(percent_change)+'% Change', (0.275,0.25),size = 20)
            plt.axis('off')
            plt.savefig(path+'Alert/'+'{:03d}'.format(i)+'_Alert.png')
            #plt.savefig(path+'Alert/'+'{:03d}'.format(i)+'_Alert.svg')
            plt.show()
            print('Green - All Clear!, percent change is '+ '{:.2f}'.format(percent_change) + ' +/- ' +'{:.2f}'.format(percent_change_uncertainty) +'%')
        elif percent_change <= parameters['Yellow-Red']:
            plt.figure()
            plt.xlim(0,1)
            plt.ylim(0,1)
            plt.fill_between((0,1),0,1,color='red')
            plt.annotate('ALERT', (0.25,0.45),size = 50)
            plt.annotate('First Event', (0.30,0.75),size = 20)
            plt.annotate('{:.2f}'.format(percent_change)+'% Change', (0.225,0.25),size = 20)
            plt.axis('off')
            plt.savefig(path+'Alert/'+'{:03d}'.format(i)+'_Alert.png')
            #plt.savefig(path+'Alert/'+'{:03d}'.format(i)+'_Alert.svg')
            plt.show()
            print('Red - Larger Earthquake Likely, percent change is '+ '{:.2f}'.format(percent_change) + ' +/- ' +'{:.2f}'.format(percent_change_uncertainty) +'%')
        else:
            plt.figure()
            plt.xlim(0,1)
            plt.ylim(0,1)
            plt.fill_between((0,1),0,1,color='yellow')
            plt.annotate('Warning', (0.20,0.45),size = 50)
            plt.annotate('First Event', (0.30,0.75),size = 20)
            plt.annotate('{:.2f}'.format(percent_change)+'%'+' Change', (0.30,0.25),size = 20)
            plt.axis('off')
            plt.savefig(path+'Alert/'+'{:03d}'.format(i)+'_Alert.png')
            #plt.savefig(path+'Alert/'+'{:03d}'.format(i)+'_Alert.svg')
            plt.show()
            print('Yellow, percent change is ' + '{:.2f}'.format(percent_change) + ' +/- ' +'{:.2f}'.format(percent_change_uncertainty) +'%')

        print('TLS number is '+ '{:.2f}'.format(TLS1) +  ' +/- ' +'{:.2f}'.format(TLS1_uncertainty) + '%')
        
        plt.figure(figsize=(3,5))
        plt.fill_between((0,1),-100,parameters['Yellow-Red'],color='red',alpha=0.6)
        plt.fill_between((0,1),parameters['Yellow-Red'],parameters['Yellow-Green'],color='yellow',alpha=0.6)
        plt.fill_between((0,1),parameters['Yellow-Green'],100,color='green',alpha=0.6)
        plt.errorbar(0.5,percent_change,yerr=percent_change_uncertainty,color='black',fmt='.')
        plt.xlabel('Event')
        plt.ylabel('Percent Change in b-value')
        plt.title('Alert Uncertainty')
        plt.xticks([])
        plt.tight_layout()
        plt.savefig(path+'alert_unc/'+'{:03d}'.format(i)+'_alert_unc.png')
        #plt.savefig(path+'alert_unc/'+'{:03d}'.format(i)+'_alert_unc.svg')
        plt.show()
        
    else:
        if percent_change2 >= parameters['Yellow-Green']:
            plt.figure()
            plt.xlim(0,1)
            plt.ylim(0,1)
            plt.fill_between((0,1),0,1,color='green')
            plt.annotate('Clear', (0.30,0.45),size = 50)
            plt.annotate('Second Event', (0.30,0.75),size = 20)
            plt.annotate('{:.2f}'.format(percent_change2)+'% Change', (0.275,0.25),size = 20)
            plt.axis('off')
            plt.savefig(path+'Alert/'+'{:03d}'.format(i)+'_Alert.png')
           # plt.savefig(path+'Alert/'+'{:03d}'.format(i)+'_Alert.svg')
            plt.show()
            print('Green - All Clear!, percent change is '+ '{:.2f}'.format(percent_change2) + ' +/- ' +'{:.2f}'.format(percent_change2_uncertainty) +'%')
        elif percent_change2 <= parameters['Yellow-Red']:
            plt.figure()
            plt.xlim(0,1)
            plt.ylim(0,1)
            plt.fill_between((0,1),0,1,color='red')
            plt.annotate('ALERT', (0.25,0.45),size = 50)
            plt.annotate('Second Event', (0.30,0.75),size = 20)
            plt.annotate('{:.2f}'.format(percent_change2)+'% Change', (0.225,0.25),size = 20)
            plt.axis('off')
            plt.savefig(path+'Alert/'+'{:03d}'.format(i)+'_Alert.png')
            #plt.savefig(path+'Alert/'+'{:03d}'.format(i)+'_Alert.svg')
            plt.show()
            print('Red - Larger Earthquake Likely, percent change is '+ '{:.2f}'.format(percent_change2) + ' +/- ' +'{:.2f}'.format(percent_change2_uncertainty) +'%')
        else:
            plt.figure()
            plt.xlim(0,1)
            plt.ylim(0,1)
            plt.fill_between((0,1),0,1,color='yellow')
            plt.annotate('Warning', (0.20,0.45),size = 50)
            plt.annotate('Second Event', (0.30,0.75),size = 20)
            plt.annotate('{:.2f}'.format(percent_change2)+'%'+' Change', (0.30,0.25),size = 20)
            plt.axis('off')
            plt.savefig(path+'Alert/'+'{:03d}'.format(i)+'_Alert.png')
            #plt.savefig(path+'Alert/'+'{:03d}'.format(i)+'_Alert.svg')
            plt.show()
            print('Yellow, percent change is ' + '{:.2f}'.format(percent_change2) + ' +/- ' +'{:.2f}'.format(percent_change2_uncertainty) +'%')

        plt.figure(figsize=(3,5))
        plt.fill_between((0,1),-100,parameters['Yellow-Red'],color='red',alpha=0.6)
        plt.fill_between((0,1),parameters['Yellow-Red'],parameters['Yellow-Green'],color='yellow',alpha=0.6)
        plt.fill_between((0,1),parameters['Yellow-Green'],100,color='green',alpha=0.6)
        plt.errorbar(0.5,percent_change2,yerr=percent_change2_uncertainty,color='black',fmt='.')
        plt.xlabel('Event')
        plt.ylabel('Percent Change in b-value')
        plt.title('Alert Uncertainty')
        plt.xticks([])
        plt.tight_layout()
        plt.savefig(path+'alert_unc/'+'{:03d}'.format(i)+'_alert_unc.png')
        #plt.savefig(path+'alert_unc/'+'{:03d}'.format(i)+'_alert_unc.svg')
        plt.show()

    if second_quake_exsists == 'yes':
        try:
            print('TLS 2 number is '+ '{:.2f}'.format(TLS2) + ' +/- ' +'{:.2f}'.format(TLS2_uncertainty) + '%')
        except:
            pass

def organize_results(result_summary,bv_bt,baft_max,precent_change,percent_change2,Event_of_interest,i,breference,percent_change,rs_index_list,sig1_pre,sig1_bt,sig1_aft,percent_change_uncertainty,percent_change2_uncertainty):
    '''Saves important results. Returns an updated result_summary matrix storing the results'''
    result_summary[i,rs_index_list[1]] = '{:.2f}'.format(breference)
    result_summary[i,rs_index_list[2]] = '{:.2f}'.format(sig1_pre)
    if Event_of_interest == 'first':
        result_summary[i,rs_index_list[3]] = '{:.2f}'.format(bv_bt)
        result_summary[i,rs_index_list[4]] = '{:.2f}'.format(sig1_bt)
        result_summary[i,rs_index_list[5]] = '{:.2f}'.format(percent_change)
        result_summary[i,rs_index_list[6]] = '{:.2f}'.format(percent_change_uncertainty)
    else:
        result_summary[i,rs_index_list[3]] = '{:.2f}'.format(baft_max)
        result_summary[i,rs_index_list[4]] = '{:.2f}'.format(sig1_aft)
        result_summary[i,rs_index_list[5]] = '{:.2f}'.format(percent_change2)
        result_summary[i,rs_index_list[6]] = '{:.2f}'.format(percent_change2_uncertainty)
    return result_summary

def percent_threshhold(parameters,Event_of_interest,alert_threshold):
    '''Function that converts the alert threshold to percentage bounds for the different alert levels
    Returns an updated parameters dictionary'''
    new_YG = alert_threshold
    new_YR = -alert_threshold
    if Event_of_interest == 'first':
        if parameters['Magnitude_event1'] >= 6:
            parameters['Yellow-Green'] = 10
            parameters['Yellow-Red'] = -10
        else:
            parameters['Yellow-Green'] = new_YG
            parameters['Yellow-Red'] = new_YR
    else:
        if parameters['Magnitude_event2'] >= 6:
            parameters['Yellow-Green'] = 10
            parameters['Yellow-Red'] = -10
        else:
            parameters['Yellow-Green'] = new_YG
            parameters['Yellow-Red'] = new_YR
    return parameters

def main(na_time,dist_threshold,precut_magnitude,start_t,grid,path):
    '''Function to conduct the full analysis, takes the parameter choices. grid = the number of points on the simulated fault plane to determine
    distance from earthquakes to a mainshock fault plane, higher is generally better but that adds time,path is the file path where you would like
    figures to be saved.
    Returns the result_summary, a matrix storing analysis results, and a M5+ event subcatalog'''
    make_plots = True # Do you want figures to be generated for every earthquake?
    filename = 'usgs_full_catalog_cmt.csv'
    times_filename = 'usgs_full_catalog_times_cmt.txt'
    # Initiate the parameters dictionary to hold pertinent event information
    parameters = {}
    parameters['days_exclude1'] = np.copy(na_time) #Number of days to exclude after first M6+ event
    parameters['days_exclude2'] = np.copy(na_time) #Number of days to exclude after second M6+ event
    parameters['mc_correction'] = 0.2 # Maximum Curvature Method Mc correction factor
    parameters['bin_width'] = 0.1 # Magnitude bin width
    parameters['Npre'] = 250 # Number of events in each b-value calculation for the pre-event time series
    parameters['Npost'] = 400 # Number of events in each b-value calculation for the bt-event and aft-event time series
    parameters['Nreg'] = 250 # Minimum number of events to establish a reference b-value
    parameters['Nmin'] = 50 # Required number of events to make any b-value calculation

    grand_times, grand_data = load_data1(filename,times_filename) # Load the EQ catalog
    main_events, main_event_times, extra_info = pairing(grand_data,grand_times,start_t)
    
    # The catalog has one grouping of three events, but both foreshocks never have sufficient data, therefore, we re-write the relationships
    # so that the code knows how to handle this one case.
    extra_info[115,0] = 'first' # First, we seperate the initial earthquake from the sequence and have the code process it as if it's standalone
    extra_info[115,1] = 'no' #Get rid of the relationship, this will be changed back at the end for the accuracy calculation
    extra_info[116,0] = 'first' # Treat the second and third event like a normal event pair
    extra_info[117,0] = 'second'
    rs_col = 9 #Data Columns in the output file for each time step, used for indexing
    num_time_steps = 18 # Length of the number of time steps calculated total, minus 1
    result_summary = np.empty((len(main_events[:,0]),3+(rs_col*(num_time_steps+1))),dtype='object') # Initiate matrix to store results

    for i in tqdm(range(len(main_events[:,0]))):
        if i == 19:# or i == 121: # This line is if you wish to single out a specific earthquake, for all Eqs use the next line instead
        #if isinstance(i,int) == True:
            print('Analyzing a New Event')
            print(main_event_times[i])
            Event_of_interest = extra_info[i,0]

            second_quake_exsists = extra_info[i,1]
            parameters = organize_data(second_quake_exsists,main_events,main_event_times,i,Event_of_interest,parameters)
            if i == 116 or i == 117: # Three event sequence: since the sequence actually starts at event 115 but are treating 116 and 117 normally, we overwrite the time of the last event so the reference b isn't contaminated by the first foreshock
                parameters['Time_LE1'] = main_event_times[115]
            result_summary[i,0] = str(main_events[i,0])
            result_summary[i,1] = str(main_events[i,3])
            result_summary[i,2] = str(main_event_times[i])[0:10]
            # Initiate Lists to save important information on how an alert changes over time, these lists are used to create a alerts vs time graph
            days_after = [] # Days after the event at which an alert is produced (end time)
            TLS_List = [] # The percent change at a specific end time
            TLS_times = [] # Same as days_after but the actual date/time
            TLS_unc = [] # Uncertainty of the values in TLS_list
            
            stop = '' # A key that will be turned on if there are reasons not to calculate an alert for all end times.
            
            magnitude_interp = np.arange(2.5,8.5,0.5) # Magnitude Range for interpolation
            time_function = interpolate.interp1d(magnitude_interp,np.array([6,11.5,22,42,83,155,290,510,790,915,960,985])) # Gardner and Knopoff (1974) Temporal Window
            r_loop_count = -1 # Counter
            for r in [0.75]:
                if r < 1: # First, convert end time to a timedelta, and obtain the time between event pairs, which will be used as a stop criterion for the first event in these pairs
                    if r == 0.25:
                        r_timedelta = np.timedelta64(6,'h')
                    elif r == 0.5:
                        r_timedelta = np.timedelta64(12,'h')
                    else:
                        r_timedelta = np.timedelta64(18,'h')
                    if Event_of_interest == 'first' and second_quake_exsists == 'yes':
                        time_between = np.timedelta64(parameters['Time_event2']-parameters['Time_event1'],'h')
                else:
                    r_timedelta = np.timedelta64(r,'D')
                    if Event_of_interest == 'first' and second_quake_exsists == 'yes':
                        time_between = np.timedelta64(parameters['Time_event2']-parameters['Time_event1'],'D')
                r_loop_count += 1
                rs_index_list = [3+(rs_col*r_loop_count),4+(rs_col*r_loop_count),5+(rs_col*r_loop_count),6+(rs_col*r_loop_count),7+(rs_col*r_loop_count),8+(rs_col*r_loop_count),9+(rs_col*r_loop_count),10+(rs_col*r_loop_count),11+(rs_col*r_loop_count)] # Indicies for saving results to the final csv
                if Event_of_interest == 'first' and second_quake_exsists == 'yes':
                     if r_timedelta > time_between: # Once the end time surpasses the second EQ for EQ pairs, we need to turn on the stop sequence
                         extra_info[i,3] = str(parameters['Time_event2'] - np.timedelta64(1,'s')) # Set the end time date for the EQ
                         r = np.timedelta64(parameters['Time_event2']-parameters['Time_event1'],'D').astype(int) # This last time is a unique time step, right before the second earthquake
                         TLS_times.append(parameters['Time_event2'] - np.timedelta64(1,'s')) 
                         stop = 'stop'
                     else: # Continue as normal in this case
                         extra_info[i,3] = str(parameters['Time_event1'] + r_timedelta) # Convert end time to a date
                         TLS_times.append(parameters['Time_event1'] + r_timedelta)
                elif r >= time_function(main_events[i,3]): # If the current end time exceeds the GK window, we initiate the stop sequence and do a final calculation at the end of the window
                    r = int(round(float(time_function(main_events[i,3]))))
                    stop = 'stop'
                    if Event_of_interest == 'first':
                        extra_info[i,3] = str(parameters['Time_event1'] + r_timedelta)
                        TLS_times.append(parameters['Time_event1'] + r_timedelta)
                    else:
                        extra_info[i,3] = str(parameters['Time_event2'] + r_timedelta)
                        TLS_times.append(parameters['Time_event2'] + r_timedelta)
                else: # If niether of the above are true, then continue as normal without initiating the stop sequence
                    if Event_of_interest == 'first':
                        extra_info[i,3] = str(parameters['Time_event1'] + r_timedelta)
                        TLS_times.append(parameters['Time_event1'] + r_timedelta)
                    else:
                        extra_info[i,3] = str(parameters['Time_event2'] + r_timedelta)
                        TLS_times.append(parameters['Time_event2'] + r_timedelta)
                if stop == 'stop': # For final calculation put the results in the rightmost csv column because the end times are unique to each event
                    rs_index_list = [3+(rs_col*num_time_steps),4+(rs_col*num_time_steps),5+(rs_col*num_time_steps),6+(rs_col*num_time_steps),7+(rs_col*num_time_steps),8+(rs_col*num_time_steps),9+(rs_col*num_time_steps),10+(rs_col*num_time_steps),11+(rs_col*num_time_steps)]
                
                # Begin rest of calculation
                result_summary[i,rs_index_list[0]] = str(r)
                total_data, times = data_splitting(main_events,main_event_times,grand_data,extra_info,grand_times,i,start_t,parameters)
                parameters['Tmin'] = np.datetime64(extra_info[i,2]) # Start time date
                parameters['Tmax'] = np.datetime64(extra_info[i,3]) # End time date
                times_orig = np.copy(times)
                total_data_orig = np.copy(total_data)
                
                # Check for Moment Tensor Information
                if  np.isnan(main_events[i,-1]) == True:
                    print('Insufficient Data: No Moment Tensor Information')
                    result_summary[i,rs_index_list[-1]] = 'Insufficient Data'
                    continue
                plane_vectors = calc_strike_dip_vectors(parameters,Event_of_interest)
    
                fault_dimensions = FP_dimensions(parameters,Event_of_interest)
                '''When there is insufficient data to make a calculation is present, it often produces errors, and we use try/except logic to catch some of these, we address others through if/else statements'''
                try:
                    total_data, times, total_distances = distance_from_plane_cut(plane_vectors,parameters,Event_of_interest,total_data,times,fault_dimensions,dist_threshold,grid)
                except IndexError:
                    result_summary[i,rs_index_list[-1]] = 'Insufficient Data'
                    continue
    
                    #This is the code for the automatic mc-time series
                    # This provides automatic precutting, and no-alert time determination
                if na_time == 'Auto' or precut_magnitude == 'Auto':    
                    try: 
                        total_data, times, parameters = mc_ts(total_data,times,parameters,Event_of_interest,second_quake_exsists,na_time,precut_magnitude)
                    except ValueError:
                        print('Insufficient Data: Not enough events')
                        result_summary[i,rs_index_list[-1]] = 'Insufficient Data'
                        continue
                # Apply the non-automatic precut magnitude
                if precut_magnitude != 'Auto':
                    precut = total_data[:,3] >= precut_magnitude
                    total_data = total_data[precut,:]
                    times = times[precut]
              
                pre_data, pre_times, pre_data_orig, bt_data, bt_times, bt_data_orig, aft_data, aft_times, Mrange,bt_times_orig,pre_times_orig = seperate_data(times,total_data,parameters,second_quake_exsists)

                pre_data, pre_times, mc_pre, bt_data, bt_times, mc_bt, aft_data, aft_times, mc_aft = seperated_data_MC_determination(pre_data,pre_times,bt_data,bt_times,aft_data,aft_times,second_quake_exsists,parameters,Mrange)
    
                if len(bt_data) == 0:
                    if Event_of_interest == 'first': #Bt data isn't neccessary to produce an alert for a 2nd event, but it is for a first event
                        print('0 Events within the distance and completeness post-event')
                        result_summary[i,rs_index_list[-1]] = 'Insufficient Data'
                        continue
                    else:
                        bbt_ts = []
                bpre_sig = np.nan
                if len(pre_times) >= parameters['Npre']: # Require enough events to compute a background b-value the normal way
                    Mcpre_ts, bpre_ts, sig1pre_ts, avannpre_ts, flagpre_ts = b_time(pre_data,pre_times,parameters['Npre'],parameters)
                else: # If not enough events, must increase the distance range to calculate it
                    try:
                        bpre_ts,pre_data_orig,mc_pre,pre_times_orig,bpre_sig = find_closest_quakes(times_orig,total_data_orig,total_distances,parameters,Event_of_interest)
                        sig1pre_ts = np.nan
                    except IndexError:
                        print('No Pre-data')
                        result_summary[i,rs_index_list[-1]] = 'Insufficient Data'
                        continue
    
                breference = np.nanmean(bpre_ts)
                if np.isnan(breference) == True:
                    print('Nan Reference B-value: Failed Linearity Test')
                    result_summary[i,rs_index_list[-1]] = 'Insufficient Data'
                    continue
    
                if np.isnan(bpre_sig) == True:
                    bpre_sig = b_reference_uncertainity(bpre_ts,sig1pre_ts)
    
                print('Reference B-value ','{:.3f}'.format(breference), ' +/- ','{:.3f}'.format(bpre_sig))
    
                if len(bt_times) <= 1:
                    if Event_of_interest == 'first': #Bt data isn't crucial for a 2nd event but is for the first event
                            print('Insufficient Data: Not Enough Events in bt-catalog')
                            result_summary[i,rs_index_list[-1]] = 'Insufficient Data'
                            continue
                    else:
                        bbt_ts = np.nan
    
                else:
    
                    Mcbt_ts, bbt_ts, sig1bt_ts, avannbt_ts, flagbt_ts = b_time(bt_data[1:,:],bt_times,parameters['Npost'],parameters)
    
                if second_quake_exsists == 'yes':
                    if len(aft_data) == 0 and Event_of_interest == 'second': #After data isn't crucial to a first event
                        print('Insufficient Data: After Catalog Contains 0 Events')
                        result_summary[i,rs_index_list[-1]] = 'Insufficient Data'
                        continue
                    if len(aft_data) == 0 and Event_of_interest == 'first':
                        # These values need to be reset so they don't contain information from the previous earthquake
                        baft_ts = []
                        Mrange_aft = np.empty((len(Mrange)))*np.nan
                        Ncumaft=np.empty((len(Mrange)))*np.nan
                        tdiff = np.empty((len(Mrange)))*np.nan
                        N3 = np.empty((len(Mrange)))*np.nan
                        label_aft = ''
                        baft_max = np.nan
                        percent_change2 = np.nan
                        sig1aft_ts = []
                        percent_change2_uncertainty = []
                        TLS2_uncertainty = []
                        TLS2 = np.nan
                        sig1_aft = np.nan
                    else:
                    
                        Mcaft_ts, baft_ts, sig1aft_ts, avannaft_ts, flagaft_ts = b_time(aft_data,aft_times,parameters['Npost'],parameters)
                        
                        if np.all(np.isnan(baft_ts)) == True:
                            if Event_of_interest == 'second':
                                print('This Earthquake has insufficient data: Aft-TS doesn\'t pass the Linearity Test')
                                result_summary[i,rs_index_list[-1]] = 'Insufficient Data'
                                continue
                            else:
                                Mrange_aft = np.empty((len(Mrange)))*np.nan
                                Ncumaft=np.empty((len(Mrange)))*np.nan
                                tdiff = np.empty((len(Mrange)))*np.nan
                                N3 = np.empty((len(Mrange)))*np.nan
                                label_aft = ''
                                baft_max = np.nan
                                percent_change2 = np.nan
                                sig1aft_ts = []
                                percent_change2_uncertainty = []
                                TLS2_uncertainty = []
                                TLS2 = np.nan
                                sig1_aft = np.nan
                        else:
                            baft_max, magcopostmax, taft_max, a_aftmax = calculate_aft_stats(baft_ts,Mcaft_ts,aft_times,avannaft_ts)
    
                            label_aft, N3, sig1_aft,tdiff, Ncumaft, Mrange_aft = pre_plot_aft_prep(aft_data,parameters,magcopostmax,aft_times,taft_max,baft_max,a_aftmax,Mrange)
    
                            percent_change2, TLS2,percent_change2_uncertainty, TLS2_uncertainty = TLS(baft_max,breference,sig1_aft,bpre_sig)
                        
    
                try: #
                    bv,sig1_pre,label_pre,Ncumpre,Tcatpre,N1,Mrange_pre = pre_plot_pre_prep(pre_data_orig,parameters,mc_pre,pre_times_orig,total_data)
                except IndexError: # Created by raise command 
                    result_summary[i,rs_index_list[-1]] = 'Insufficient Data'
                    continue
    
                try:
                    bv_bt,sig1_bt,label_bt,Ncumbt,Tcatbt,N2,Mrange_bt = pre_plot_bt_prep(bt_data_orig,parameters,mc_bt,bt_times_orig,Mrange,bt_times)
                except IndexError: # vreated by raise command
                    if Event_of_interest == 'first':
                        result_summary[i,rs_index_list[-1]] = 'Insufficient Data'
                        continue
                    else:
                        bv_bt = -123456789
                        sig1_bt = -123456789
                        Mrange_bt = np.nan
                        N2 = np.nan
                        label_bt = np.nan
                        Ncumbt = np.nan
                        Tcatbt = np.nan
                except ValueError: # This error occurs if there is no bt-data for a second event, bt data isn't crucial for a 2nd event
                    bv_bt = -123456789
                    sig1_bt = -123456789
                    Mrange_bt = np.nan
                    N2 = np.nan
                    label_bt = np.nan
                    Ncumbt = np.nan
                    Tcatbt = np.nan
                percent_change, TLS1, percent_change_uncertainty, TLS1_uncertainty = TLS(bv_bt,breference,sig1_bt,bpre_sig)
    
                if np.isnan(np.nanmean(bbt_ts)) == True and Event_of_interest == 'first':
                    result_summary[i,rs_index_list[-1]] = 'Insufficient Data'
                    print('First Event with a non-linear time series post-event')
                    continue
    
                elif np.isnan(np.nanmean(bbt_ts)) == True and Event_of_interest == 'second':
                    bbt_ts = []
                    bt_times = []
                if second_quake_exsists == 'yes':
                    if np.isnan(np.nanmean(baft_ts)) == True and Event_of_interest == 'second':
                        result_summary[i,rs_index_list[-1]] = 'Insufficient Data'
                        print('Second Event with a non-linear time series post-event')
                        continue
                    elif np.isnan(np.nanmean(baft_ts)) == True and Event_of_interest == 'first':
                        baft_ts = []
                        aft_times = []
                else:
                    Mrange_aft = np.empty((len(Mrange)))*np.nan
                    Ncumaft=np.empty((len(Mrange)))*np.nan
                    tdiff = np.empty((len(Mrange)))*np.nan
                    N3 = np.empty((len(Mrange)))*np.nan
                    label_aft = ''
                    baft_max = np.nan
                    percent_change2 = np.nan
                    baft_ts = []
                    aft_times = []
                    sig1aft_ts = []
                    percent_change2_uncertainty = []
                    TLS2_uncertainty = []
                    TLS2 = np.nan
                    sig1_aft = np.nan   
                if make_plots == True:
                    plotting(parameters=parameters,Mrange=Mrange,Ncumpre=Ncumpre,Tcatpre=Tcatpre,Mrange_pre=Mrange_pre,N1=N1,
                             label_pre=label_pre,Mrange_bt=Mrange_bt,N2=N2,Ncumbt=Ncumbt,Tcatbt=Tcatbt,tdiff=tdiff,
                             Mrange_aft=Mrange_aft,N3=N3,label_aft=label_aft,times=times,breference=breference,
                             bt_times=bt_times,pre_times=pre_times,aft_times=aft_times,bpre_ts=bpre_ts,bbt_ts=bbt_ts,
                             baft_ts=baft_ts,sig1bt_ts=sig1bt_ts,sig1pre_ts=sig1pre_ts,sig1aft_ts=sig1aft_ts,
                             percent_change=percent_change,percent_change2=percent_change2,
                             percent_change_uncertainty=percent_change_uncertainty,percent_change2_uncertainty=percent_change2_uncertainty,
                             TLS1=TLS1,TLS1_uncertainty=TLS1_uncertainty,TLS2=TLS2,TLS2_uncertainty=TLS2_uncertainty,bv_bt=bv_bt,
                             bv=bv,label_bt=label_bt,second_quake_exsists=second_quake_exsists,Event_of_interest=Event_of_interest,
                             i=i,path=path,Ncumaft=Ncumaft)
                if second_quake_exsists == 'no':
                    baft_max = ''
                    percent_change2 = ''
                
                # Save values for alerts vs time graph
                days_after.append(r) # If we get to this point, we have sucsessfully generated an alert for the event, so we save the end time to this list for an alerts vs time graph
                if Event_of_interest == 'first':
                    TLS_List.append(percent_change)
                    TLS_unc.append(percent_change_uncertainty)
                else:
                    TLS_List.append(percent_change2)
                    TLS_unc.append(percent_change2_uncertainty)
            
                result_summary = organize_results(result_summary,bv_bt,baft_max,percent_change,percent_change2,Event_of_interest,i,breference,percent_change,rs_index_list,sig1_pre,sig1_bt,sig1_aft,percent_change_uncertainty,percent_change2_uncertainty)
                if stop == 'stop': # If the stop sequence was initiated, then we break the end time loop here.
                    break
           

# Code for Alerts vs Time Graph
            make_plots = False
            if len(TLS_List) > 0 and make_plots == True:
                plt.figure(figsize=(5,2.5))
                plt.fill_between((0,370),-100,-10,color='red',alpha=0.6)
                plt.fill_between((0,370),-10,10,color='yellow',alpha=0.6)
                plt.fill_between((0,370),10,100,color='green',alpha=0.6)
                plt.scatter(days_after,TLS_List,color='black',marker='.')
                plt.errorbar(days_after,TLS_List,yerr=TLS_unc,color='black',fmt='.')
                plt.xlabel('Days After Event')
                plt.ylabel('Percent Change in b-value')
                plt.title('Alerts vs Time over 15 days',fontsize=12)
                plt.xlim(0,25)
                plt.ylim(-75,75)
                plt.tight_layout()
                plt.savefig(path+'{:03d}'.format(i)+'short_time_variability.png')
                plt.savefig(path+'{:03d}'.format(i)+'short_time_variability.svg')

    return result_summary, main_events # Return the results and the M5 catalog
path = 'b_output/'
result_summary, main_events = main(na_time,dist_threshold,precut_magnitude,start_t,grid,path) # Call the analysis
loc = np.loadtxt('renamed_eq_locations.txt',dtype='str',delimiter='\n') # Load earthquake names for the result files
for i in range(len(loc)): # Populate the result files with the earthquake names
    result_summary[i,0] = np.copy(loc[i])

# Convert parameter choices to more readable values for the file names
if start_t == 25550:
    start_t = 1971
elif start_t == 'Auto':
    pass
else:
    start_t = start_t / 365
if dist_threshold == 'Auto':
    pass
else:
    dist_threshold = dist_threshold / 1000
# Save the Data
num_time_steps = 18   
basic_header = 'Event Location, Magnitude, Date,'
time_step_header = ' Day After Event, Reference b, Reference b uncertainty, Post-Event b, Post-Event b uncertainty, Percent Change in b, Percent Change in b uncertainty, Traffic Light, Accuracy,'
headers = basic_header + (1+num_time_steps)*time_step_header

#np.savetxt(path+'eq_locations.txt',result_summary[:,0],fmt='%s')
title = 'st_' + str(start_t) + '_na_' + str(na_time) + '_pm_' + str(precut_magnitude) + '_dt_' + str(dist_threshold)
#np.savetxt(path+title+'.csv',result_summary,delimiter=',',header=headers,fmt='%s')
