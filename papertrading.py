import pandas as pd
import sqlite3
from datetime import datetime

class PaperTrader:
    def __init__(self, db_path="papertrades.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database and trades table."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    Trade_ID INTEGER PRIMARY KEY AUTOINCREMENT,
                    Symbol TEXT,
                    Side TEXT,
                    Entry_Date TEXT,
                    Entry_Price REAL,
                    Quantity INTEGER,
                    Exit_Date TEXT,
                    Exit_Price REAL,
                    Status TEXT
                )
            """)

    def take_trade(self, symbol, side, entry_price, quantity):
        """Open a new trade and store it in the database."""
        entry_date = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO trades (Symbol, Side, Entry_Date, Entry_Price, Quantity, Status)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (symbol, side, entry_date, entry_price, quantity, 'OPEN'))
            trade_id = cursor.lastrowid
        return trade_id

    def close_trade(self, trade_id, exit_price):
        """Close an open trade by trade_id."""
        exit_date = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE trades
                SET Exit_Date = ?, Exit_Price = ?, Status = ?
                WHERE Trade_ID = ? AND Status = 'OPEN'
            """, (exit_date, exit_price, 'CLOSED', trade_id))

    def get_open_trades(self):
        """Return all open trades as a DataFrame."""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query("SELECT * FROM trades WHERE Status = 'OPEN'", conn)
        return df

    def get_trade_log(self):
        """Return all trades as a DataFrame."""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query("SELECT * FROM trades", conn)
        return