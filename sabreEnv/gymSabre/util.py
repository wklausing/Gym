import sqlite3
import csv
import os
from datetime import datetime

class Util:

    def __init__(self):
        # For CSVs
        time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.clientFilename = 'data/client_' + time + '.csv'
        self.cdnFilename = 'data/cdn_' + time + '.csv'
        self.cpFilename = 'data/cp_' + time + '.csv'

        self.dbPath = 'data/sabregym.db'

        # Shared variables
        self.episodeCounter = 0

    def connect(self):
        self.conn = sqlite3.connect(self.dbPath)
        self.c = self.conn.cursor()

    def close(self):
        self.conn.close()

    def insert(self, table, columns, values):
        self.connect()
        self.c.execute('INSERT INTO %s (%s) VALUES (%s)' % (table, columns, values))
        self.conn.commit()
        self.close()

    def cdnCsvExport(self, stepData):
        allCdnKeys = ['episode', 'time', 'id', 'clientsCount', 'clients', 'bandwidth_kbps', 'currentBandwidth', 'price', 'money', 'contigent']
        file_exists = os.path.exists(self.cdnFilename)
        with open(self.cdnFilename, 'a' if file_exists else 'w', newline='') as file:
            writer = csv.writer(file)
            if not file_exists: writer.writerow(allCdnKeys)
            writer.writerows(stepData)

    def clientCsvExport(self, stepData):
        allClientKeys = ['id','location','episode','gymTime','qoe','qoeFlag','estimate','manifest','alive',\
                         'cdn_location', 'cdn_id', 'status','size','download_time',\
                            'total_play_time_chunks','total_play_time','total_played_utility','total_log_bitrate_change',\
                                'total_reaction_time','time_average_played_utility','time_average_rebuffer_events','time_average_score',\
                                    'buffer_size','total_rebuffer_events','time_average_played_bitrate','time_average_log_bitrate_change',\
                                        'rebuffer_ratio','total_bitrate_change','time_average_bitrate_change','total_played_bitrate',\
                                            'time_average_rebuffer','total_rebuffer', 'delay', 'latency', 'bandwidth']
        
        #allClientKeys = set(key for d in merged_dicts for key in d.keys())
        file_exists = os.path.exists(self.clientFilename)
        with open(self.clientFilename, 'a' if file_exists else 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=allClientKeys)
            if not file_exists: writer.writeheader()
            writer.writerows(stepData)