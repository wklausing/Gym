import csv
import os
from datetime import datetime

class Util:

    def __init__(self, savingPath='sabreEnv/gymSabre/data/', filePrefix='', gridWidth=100, gridHeight=100):
        # For CSVs
        time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # self.clientFilename = 'data/client_' + time + '.csv'
        # self.cdnFilename = 'data/cdn_' + time + '.csv'
        # self.cpFilename = 'data/cp_' + time + '.csv'
        
        self.gridWidth = gridWidth
        self.gridHeight = gridHeight


        # Remove later
        time = 'test'
        self.clientFilename = savingPath + filePrefix + 'client_' + time + '.csv'
        self.cdnFilename = savingPath + filePrefix + 'cdn_' + time + '.csv'
        self.cpFilename = savingPath + filePrefix + 'cp_' + time + '.csv'
        
        file_exists = os.path.exists(self.clientFilename)
        if file_exists:
            os.remove(self.clientFilename)
            print(f"File {self.clientFilename} has been removed.")
        else:
            print(f"File {self.clientFilename} does not exist.")
        file_exists = os.path.exists(self.cdnFilename)
        if file_exists:
            os.remove(self.cdnFilename)
            print(f"File {self.cdnFilename} has been removed.")
        else:
            print(f"File {self.cdnFilename} does not exist.")
        file_exists = os.path.exists(self.cpFilename)
        if file_exists:
            os.remove(self.cpFilename)
            print(f"File {self.cpFilename} has been removed.")
        else:
            print(f"File {self.cpFilename} does not exist.")
        # Remove later ^
        
        # Shared variables
        self.episodeCounter = 0

    def cdnCsvExport(self, stepData):
        allCdnKeys = ['episode', 'time', 'id', 'clientsCount', 'clients', 'bandwidth_kbps', 'currentBandwidth', 'price', 'money', 'contigent']
        file_exists = os.path.exists(self.cdnFilename)
        with open(self.cdnFilename, 'a' if file_exists else 'w', newline='') as file:
            writer = csv.writer(file)
            if not file_exists: writer.writerow(allCdnKeys)
            writer.writerows(stepData)

    def clientCsvExport(self, stepData):
        allClientKeys = ['id','location','episode','gymTime','qoe','normalized_qoe','qoeFlag','estimate','manifest','alive',\
                         'cdn_location', 'cdn_id', 'status','size','download_time',\
                            'total_play_time_chunks','total_play_time','total_played_utility','total_log_bitrate_change',\
                                'total_reaction_time','time_average_played_utility','time_average_rebuffer_events','time_average_score',\
                                    'buffer_size','total_rebuffer_events','time_average_played_bitrate','time_average_log_bitrate_change',\
                                        'rebuffer_ratio','total_bitrate_change','time_average_bitrate_change','total_played_bitrate',\
                                            'time_average_rebuffer','total_rebuffer', 'buffer_level', 'delay', 'latency', 'bandwidth']
        file_exists = os.path.exists(self.clientFilename)
        with open(self.clientFilename, 'a' if file_exists else 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=allClientKeys)
            if not file_exists: writer.writeheader()
            writer.writerows(stepData)