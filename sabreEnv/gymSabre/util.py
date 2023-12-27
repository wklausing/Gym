import sqlite3
import csv
import os
from datetime import datetime

class Util:

    def __init__(self, dbPath='data/sabregym.db'):
        # For CSVs
        time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.clientFilename = 'data/client_' + time + '.csv'
        self.cdnFilename = 'data/cdn_' + time + '.csv'
        self.cpFilename = 'data/cp_' + time + '.csv'

        self.dbPath = dbPath

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

    def clientCsvExport(self, metaData, stepData, ):
        merged_dicts = [{**d, **metaData} for d in stepData]
        all_keys = set(key for d in merged_dicts for key in d.keys())
        with open(self.clientFilename, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(merged_dicts)

    def cdnCsvExport(self, metaData, stepData):
        merged_dicts = [{**d, **metaData} for d in stepData]
        all_keys = set(key for d in merged_dicts for key in d.keys())
        with open(self.cdnFilename, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(merged_dicts)