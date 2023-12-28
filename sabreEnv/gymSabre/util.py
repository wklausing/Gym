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

    def clientCsvExport(self, metaData, stepData):
        allClientKeys = ['id','location','episode','time','qoe','estimate','manifest','alive','edgeServer_location','status','size','edgeServer_id','total_play_time_chunks','total_play_time','total_played_utility','total_log_bitrate_change','total_reaction_time','time_average_played_utility','time_average_rebuffer_events','time_average_score','buffer_size','total_rebuffer_events','time_average_played_bitrate','time_average_log_bitrate_change','rebuffer_ratio','total_bitrate_change','time_average_bitrate_change','total_played_bitrate','time_average_rebuffer','total_rebuffer', 'delay']
        merged_dicts = [{**{key: None for key in allClientKeys}, **d, **metaData} for d in stepData]
        #allClientKeys = set(key for d in merged_dicts for key in d.keys())
        file_exists = os.path.exists(self.clientFilename)
        with open(self.clientFilename, 'a' if file_exists else 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=allClientKeys)
            if not file_exists: writer.writeheader()
            writer.writerows(merged_dicts)

    def cdnCsvExport(self, metaData, stepData):
        merged_dicts = [{**d, **metaData} for d in stepData]
        all_keys = set(key for d in merged_dicts for key in d.keys())
        with open(self.cdnFilename, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(merged_dicts)