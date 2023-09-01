# Name: Sitong Mu; Github username: edsml-sm1122
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from statsmodels.graphics.tsaplots import plot_acf
import folium
import math
import requests
import os


def checkdim(target_contour, data_folder='../../data/20230608/TimeSeries_Data_20230607', parameter_types=['_MWD_', '_PWP_', '_SWH_', '_TWD_', '_WiD_', '_WiS_']):
    """
    Check if all the .csv files in the given directory have the same shape and if all tracks are the same.

    :param target_contour: Target contour to check.
    :type target_contour: str
    :param data_folder: Path to the data folder.
    :type data_folder: str
    :param parameter_types: List of parameter types to check.
    :type parameter_types: list of str
    :return: Maximum time dimension, space dimension, and track dimension found among all CSV files.
    :rtype: tuple of int
    """
    # Placeholder for the actual dimensions
    time_dim = space_dim = track_dim = 0

    # A dictionary to store the shapes of each file
    file_shapes = {}

    # Store the order of the tracks for each parameter
    track_order = {}

    for root, dirs, files in os.walk(data_folder):
        for parameter in parameter_types:
            if parameter in root and target_contour in root:
                csv_files = sorted([f for f in files if f.endswith('.csv')], key=lambda f: int(os.path.splitext(f)[0][:-3]))
                track_dim = max(track_dim, len(csv_files))
                for file in csv_files:
                    data = np.genfromtxt(os.path.join(root, file), delimiter=',')
                    file_shapes[os.path.join(root, file)] = data.shape
                    time_dim = max(time_dim, data.shape[0])
                    space_dim = max(space_dim, data.shape[1])
                # Store the order of tracks for this parameter
                track_order[parameter] = csv_files

    # Check if all shapes are the same
    shapes = set(file_shapes.values())

    # Count the number of files with each shape
    shape_counts = {shape: list(file_shapes.values()).count(shape) for shape in shapes}

    # Find the shape with the largest number of files
    max_shape = max(shape_counts, key=shape_counts.get)

    if len(shapes) > 1:
        print("Error: Not all .csv files have the same shape.")
        for file, shape in file_shapes.items():
            if shape != max_shape:
                print(f"File {file} has a different shape: {shape}")
    else:
        print("All .csv files have the same shape.")

    # Print the count of files with each shape
    for shape, count in shape_counts.items():
        print(f"Shape {shape}: {count} files")

        
    # Check if the order of the tracks is the same for all parameters
    print('-'*20)

    order_correct = True
    first_param_tracks = track_order[parameter_types[0]]
    for parameter, tracks in track_order.items():
        if tracks != first_param_tracks:
            print(f"Error: The order of tracks for parameter {parameter} is different.")
            order_correct = False
    if order_correct:
        print('the code read track file in the same order (ascending track number) which is \n')
        print(first_param_tracks)
        
    return time_dim,space_dim,track_dim






def read_save(target_contour, savepath, track_dim, time_dim, space_dim, data_folder='../../data/20230608/TimeSeries_Data_20230607', parameter_types=['_MWD_', '_PWP_', '_SWH_', '_TWD_', '_WiD_', '_WiS_']):
    """
    Create a zero-filled array with the maximum dimension, then fill in the data from the .csv files.

    :param target_contour: Target contour to read.
    :type target_contour: str
    :param savepath: Path to save the numpy array.
    :type savepath: str
    :param track_dim: Number of tracks.
    :type track_dim: int
    :param time_dim: Maximum number of time points.
    :type time_dim: int
    :param space_dim: Maximum number of space points.
    :type space_dim: int
    :param data_folder: Path to the data folder.
    :type data_folder: str
    :param parameter_types: List of parameter types to read.
    :type parameter_types: list of str
    :return: 0 indicating success.
    :rtype: int
    """    
    
    # Create a 4D numpy array.
    data_array = np.empty((track_dim, time_dim, space_dim, len(parameter_types)))

    # Second pass to fill the array
#     parameter_index = 0
    for root, dirs, files in os.walk(data_folder):
        for parameter_index,parameter in enumerate(parameter_types):
            if parameter in root and target_contour in root:
                csv_files = sorted([f for f in files if f.endswith('.csv')], key=lambda f: int(os.path.splitext(f)[0][:-3]))
                for track_index, file in enumerate(csv_files):
    #                 print(os.path.join(root, file))
                    data = np.genfromtxt(os.path.join(root, file), delimiter=',',dtype='float64')
                    (a,b) = data.shape
                    data_array[track_index, :a, :b, parameter_index] = data
#             print()
#             parameter_index += 1
    np.save(savepath,data_array)
    return 0



def track_order(target_contour, data_folder='../../data/20230608/TimeSeries_Data_20230607', parameter_types=['_MWD_', '_PWP_', '_SWH_', '_TWD_', '_WiD_', '_WiS_']):
    """
    Check the order of the tracks for a given contour.

    :param target_contour: Target contour to check order for.
    :type target_contour: str
    :param data_folder: Path to the data folder.
    :type data_folder: str
    :param parameter_types: List of parameter types to check.
    :type parameter_types: list of str
    :return: Order of the tracks for the given contour if same order.
    :rtype: list of str
    """
    
    # Store the order of the tracks for each parameter
    track_order = {}

    for root, dirs, files in os.walk(data_folder):
        for parameter in parameter_types:
            if parameter in root and target_contour in root:
                csv_files = sorted([f for f in files if f.endswith('.csv')], key=lambda f: int(os.path.splitext(f)[0][:-3]))
                # Store the order of tracks for this parameter
                track_order[parameter] = csv_files

    # Check if the order of the tracks is the same for all parameters
    print('-'*20)

    order_correct = True
    first_param_tracks = track_order[parameter_types[0]]
    for parameter, tracks in track_order.items():
        if tracks != first_param_tracks:
            print(f"Error: The order of tracks for parameter {parameter} is different.")
            order_correct = False
            
    if order_correct:
#         print(f"Error: The order of tracks for parameter {parameter} is different.")
        return first_param_tracks
    

# def get_variable_name(variable):
#     """
#     Returns the name of a variable.
    
#     :param variable: The variable for which the name is desired.
#     :type variable: Any data type
#     :return: Name of the variable if found, else None.
#     :rtype: str or None
#     """
#     for name in globals():
#         if globals()[name] == variable:
#             return name
#     return None

def checkdup(arr):
    """
    Checks the percentage of duplicate data in the provided numpy array.

    :param arr: Numpy array with data.
    :type arr: numpy.ndarray
    :return: Percentage of duplicated data.
    :rtype: float
    """
    arr_rs = arr.reshape(np.prod(arr.shape[:2]),arr.shape[2],arr.shape[3])
    uni = np.unique(arr_rs,axis=0)
    dupratio = (len(arr_rs)-len(uni))/len(arr_rs)
    print('percentage of duplication:')
    print("{:.2%}".format(dupratio))
    
    return dupratio

def check_nan(arr,cols=['MWD(° from north)','PWP(s)','SWH(m)','TWD(m)','WiD(° from north)','WiS(m/s)']):
    """
    Checks the number of NaN values in a provided numpy array.

    :param arr: Numpy array with data.
    :type arr: numpy.ndarray
    :return: 0 indicating success.
    :rtype: int
    """
    nan_count = np.sum(np.isnan(arr), axis=(0,1,2))
    
    print('Nan count for each features')
    print(cols)
    print('nan count:', nan_count)
    print('ratio of nan:', nan_count/np.prod(arr.shape[:-1]))
    return 0

def plotarr(arr,cols=['MWD(° from north)','PWP(s)','SWH(m)','TWD(m)','WiD(° from north)','WiS(m/s)']):
    """
    Plots the quantiles (25%, 50%, 75%) and the mean for each feature in the given array.

    :param arr: Numpy array with data.
    :type arr: numpy.ndarray
    :return: 0 indicating success.
    :rtype: int
    """

    # Transpose the array to shape (188, 145, 98, 6)
    transposed_data = np.transpose(arr, (0, 2, 1, 3))

    # Flatten the array along the first two dimensions
    shape = transposed_data.shape
    flattened_data = np.reshape(transposed_data, (shape[0]*shape[1], shape[2], shape[3]))

    # Calculate the quantiles and mean along the flattened dimension
    quantiles = np.percentile(flattened_data, [25, 50, 75], axis=0)
    mean = np.mean(flattened_data, axis=0)

    # Create x-coordinate values
    x = np.arange(flattened_data.shape[1])
    fig = plt.figure(figsize=(12, 8))

    # Generate six plots for the last dimension
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.plot(x, quantiles[:, :, i][0], label='0.25 quantile')
        plt.plot(x, quantiles[:, :, i][1], label='0.50 quantile')
        plt.plot(x, quantiles[:, :, i][2], label='0.75 quantile')
        plt.plot(x, mean[:, i], label='Mean')
        plt.xlabel('time step') # Tstart: 1945-10-07 12:00:00 dt: 1800s nt: 97
        plt.ylabel('Values')
        plt.title(f'Plot for {cols[i]}'.format(i))
        plt.legend()

    plt.tight_layout()
    plt.show()
    return 0


def plot_elevation(data_array):
    """
    Plot the elevation values from a provided data array.

    :param data_array: A numpy array where the first column represents IDs and the last column 
                       represents elevations. The array should be structured as [ID, ..., Elevation].
    :type data_array: numpy.ndarray
    :return: 0 indicating success.
    :rtype: int
    
    .. note:: This function assumes that the elevation is the last column in the data_array.
              The plot generated will have IDs on the x-axis and elevations on the y-axis.
    """
    
    # Split the data
    ids = data_array[:, 0]
    elevations = data_array[::-1, -1]  # Assuming elevation is the last column
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    x = np.arange(len(data_array))
    plt.plot(x[::-1], elevations, marker='o', linestyle='-', color='b') #south to north
#     plt.plot(np.gradient(elevations))
    # Add labels and title
    plt.xlabel('ID')
    plt.ylabel('Elevation (m)')
    plt.title('Elevation by ID')
    
    # Show the plot
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return 0

def haversine_distance(loc1, loc2):
    """
    Calculates the Haversine distance between two points on the Earth.

    :param loc1: First location in the format [points ID, latitude, longitude].
    :type loc1: list
    :param loc2: Second location in the format [points ID, latitude, longitude].
    :type loc2: list
    :return: Distance in kilometers between the two points.
    :rtype: float
    """
    R = 6371  # Earth's radius in kilometers

    lat1, lon1 = np.radians(loc1[1]), np.radians(loc1[2])
    lat2, lon2 = np.radians(loc2[1]), np.radians(loc2[2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    distance = R * c
    return distance

def autocorr(data, name,cols):
    """
    Compute and plot the autocorrelation for time series data.

    :param data: A numpy array with shape (tracks, locations, time, parameters). The data represents 
                 multiple tracks, with multiple locations, over time, and for various parameters.
    :type data: numpy.ndarray
    :param name: A string representing the title of the plots.
    :type name: str
    :param name: A list of strings representing the subtitles of the plots.
    :type name: list
    :return: A list of numpy arrays, where each array is the autocorrelation for a given parameter.
    :rtype: list[numpy.ndarray]

    .. note:: The function computes the autocorrelation for each parameter and location. It then plots the 
              average autocorrelation for each parameter across all tracks and locations.
    """
    # swape sapce and time dim to make the time the third
    data = data.transpose(0, 2, 1,3)
    # Create a placeholder for the results
    average_autocorr = np.zeros((data.shape[-1], data.shape[2]))

    # Initialize an empty list to store the autocorrelations for each track/location
    autocorr = []
    
    for i in range(data.shape[-1]):
        # Combine all tracks and locations for the current parameter
        parameter_data = data[..., i].reshape(-1, data.shape[2]) # 2 is the time dimention

        for x in parameter_data:
            # Check if variance is zero
            if np.isclose(x.var(), 0):
                # If variance is zero, autocorrelation is not defined, so we'll use NaNs
                autocorr.append(np.full_like(x, np.nan))
            else:
                # If variance is non-zero, calculate autocorrelation as before
                autocorrelation = correlate(x, x, mode='full')[x.size-1:]
                normalized_autocorrelation = autocorrelation / x.var() / x.size
                autocorr.append(normalized_autocorrelation)
                
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.flatten()

    for i in range(data.shape[-1]):
        # Combine all tracks and locations for the current parameter
        parameter_data = data[..., i].mean(axis=0).mean(axis=1)

        plot_acf(parameter_data, ax=axs[i])
        axs[i].set_title(cols[i])
        axs[i].grid(True)  # Add a grid to the plot
        axs[i].set_ylim([0,1])
    plt.suptitle(name, fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # leave space for the super title
    fig.text(0.5, 0.01, 'Lags (blue area 95% CI)', ha='center',size=15)
    plt.show()
    
    return autocorr



def convert_to_numpy(df):
    """
    Convert the DataFrame into a numpy array containing only ID, X, and Y columns.

    :param df: DataFrame to convert.
    :type df: pandas.DataFrame
    :return: Numpy array with columns ID, X, and Y.
    :rtype: numpy.ndarray
    """
    return df[['ID', 'X', 'Y']].to_numpy()



def add_points_to_map(arr, map_obj, color="blue"):
    """
    Add points from a numpy array to a folium map.

    :param arr: Numpy array containing the points with columns ID, latitude, and longitude.
    :type arr: numpy.ndarray
    :param map_obj: Folium map object.
    :type map_obj: folium.folium.Map
    :param color: Color for the points, default is "blue".
    :type color: str
    :return: Always returns 0.
    :rtype: int
    """
    for row in arr:
        _, x, y = row
        folium.CircleMarker(
            location=[y, x],
            radius=1,  # you can change the radius if you like
            color=color,
            fill=True,
            fill_color=color
        ).add_to(map_obj)

    return 0   
    
    
    
def get_elevation(lat, lon, api_key):
    """
    Get the elevation in meters for a given latitude and longitude using Google Maps Elevation API.

    :param lat: Latitude coordinate.
    :type lat: float
    :param lon: Longitude coordinate.
    :type lon: float
    :param api_key: API key for the Google Maps Elevation API.
    :type api_key: str
    :return: Elevation in meters or False if there's an error.
    :rtype: float or False
    """
    base_url = "https://maps.googleapis.com/maps/api/elevation/json"
    params = {
        "locations": f"{lat},{lon}",
        "key": api_key
    }

    response = requests.get(base_url, params=params).json()
    
    if response["status"] == "OK":
        return response["results"][0]["elevation"]
    else:
        print("Error:", response["status"])
        return False
    
    

def get_elevation_for_array(arr, api_key):
    """
    Get elevation data for all points in a numpy array.

    :param arr: Numpy array with columns ID, latitude, and longitude.
    :type arr: numpy.ndarray
    :param api_key: API key for the Google Maps Elevation API.
    :type api_key: str
    :return: A new numpy array containing ID, latitude, longitude, and elevation columns.
    :rtype: numpy.ndarray
    """
    
    elevations = []
    for row in arr:
        _, lat, lon = row
        elevation = get_elevation(lat, lon, api_key)
        if elevation is not None:
            elevations.append(elevation)
        else:
            elevations.append(np.nan)  # Using NaN for error/missing elevations

    # Convert elevations to numpy array and reshape to match the original array's shape
    elevations = np.array(elevations).reshape(-1, 1)
    
    return np.hstack((arr, elevations))

