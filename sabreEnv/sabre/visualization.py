import pandas as pd
import matplotlib.pyplot as plt


def diffBandwidths():
    # Load the datasets
    file_paths = [
        "sabreEnv/sabre/data/visualizationData/sabreMetricsBW64.csv",
        "sabreEnv/sabre/data/visualizationData/sabreMetricsBW128.csv",
        "sabreEnv/sabre/data/visualizationData/sabreMetricsBW256.csv",
        "sabreEnv/sabre/data/visualizationData/sabreMetricsBW512.csv",
        "sabreEnv/sabre/data/visualizationData/sabreMetricsBW1024.csv"
    ]
    # Store the data from each file along with its bandwidth
    data = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        bandwidth = file_path.split('sabreMetricsBW')[-1].split('.csv')[0]  # Extract bandwidth from filename
        data.append((bandwidth, df))

    # Prepare to plot
    plt.figure(figsize=(12, 6))

    # Loop through each dataset and plot
    for bandwidth, df in data:
        # Filter rows where status is 'downloadSegment'
        filtered_df = df[df['status'] == 'downloadedSegment']

        # Plot time_average_score against the count of occurrences
        plt.plot(filtered_df['time_average_score'].values, label=f'Bandwidth {bandwidth} kbps')

    # Setting up the plot
    plt.xlabel('Downloaded Segment Count')
    plt.ylabel('Time Average Score')
    plt.title('Time Average Score for Different Bandwidths')
    plt.legend()

    # Show the plot
    plt.show()


def diffLatencies():
    # Load the datasets
    file_paths = [
        "sabreEnv/sabre/data/visualizationData/sabreMetricsLat64.csv",
        "sabreEnv/sabre/data/visualizationData/sabreMetricsLat128.csv",
        "sabreEnv/sabre/data/visualizationData/sabreMetricsLat256.csv",
        "sabreEnv/sabre/data/visualizationData/sabreMetricsLat512.csv",
        "sabreEnv/sabre/data/visualizationData/sabreMetricsLat1024.csv"
    ]
    # Store the data from each file along with its bandwidth
    data = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        latency = file_path.split('sabreMetricsLat')[-1].split('.csv')[0]  # Extract bandwidth from filename
        data.append((latency, df))

    # Prepare to plot
    plt.figure(figsize=(12, 6))

    # Loop through each dataset and plot
    for latency, df in data:
        # Filter rows where status is 'downloadSegment'
        filtered_df = df[df['status'] == 'downloadedSegment']

        # Plot time_average_score against the count of occurrences
        plt.plot(filtered_df['time_average_score'].values, label=f'Latency {latency} ms')

    # Setting up the plot
    plt.xlabel('Downloaded Segment Count')
    plt.ylabel('Time Average Score')
    plt.title('Time Average Score for Different Latencies')
    plt.legend()

    # Show the plot
    plt.show()

diffLatencies()
# diffBandwidths()