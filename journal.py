# journal.py
import csv
import os
from datetime import datetime

class TradeJournal:
    def __init__(self, filename="trade_journal.csv"):
        self.filename = filename
        self.init_file()

    def init_file(self):
        if not os.path.exists(self.filename):
            with open(self.filename, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Time", "Direction", "Size", "Consensus", "Regime", "MetaResult", "Status"])

    def log_trade(self, direction, size, consensus, regime, meta_result, status="SUCCESS"):
        with open(self.filename, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), direction, size, consensus, regime, meta_result, status])
