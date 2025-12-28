#!/usr/bin/env python3
"""
CSV Stock Data Updater - Gets ALL missing trading days
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import time
import sys
import argparse

# Your portfolio
DEFAULT_STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA']

# Data directory
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def update_stock_csv(symbol: str) -> bool:
    """Update CSV with ALL missing data from yfinance"""
    
    # Find CSV
    csv_paths = [
        DATA_DIR / f"{symbol}.csv",
        Path(f"data/stock_data/{symbol}.csv"),
        Path(f"stock_data/{symbol}.csv"),
    ]
    
    csv_path = None
    for path in csv_paths:
        if path.exists():
            csv_path = path
            break
    
    if csv_path is None:
        print(f"  ❌ No CSV found")
        return False
    
    try:
        # Load CSV
        df = pd.read_csv(csv_path)
        original_len = len(df)
        df.columns = df.columns.str.lower()
        
        # Find date column
        date_col = None
        for col in ['date', 'datetime', 'timestamp', 'price']:
            if col in df.columns:
                date_col = col
                break
        if date_col is None:
            date_col = df.columns[0]
        
        # Parse dates
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        df = df.sort_values(date_col)
        
        last_date = df[date_col].max()
        today = pd.Timestamp.now().normalize()
        days_old = (today - last_date).days
        
        print(f"  📅 CSV Last: {last_date.strftime('%Y-%m-%d')} ({days_old}d ago)")
        
        # Check if weekend
        is_weekend = today.dayofweek >= 5  # 5=Sat, 6=Sun
        last_was_friday = last_date.dayofweek == 4  # 4=Friday
        
        if is_weekend and last_was_friday and days_old <= 2:
            print(f"  ✅ Up-to-date (Weekend, last trading day was Friday)")
            return True
        
        if days_old == 0:
            print(f"  ✅ Up-to-date (Today's data)")
            return True
        
        # Fetch from yfinance
        print(f"  🔄 Fetching from yfinance...", end="", flush=True)
        
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        
        # Fetch from last CSV date to now (with margin)
        start_date = last_date - timedelta(days=2)  # 2 days before for safety
        end_date = datetime.now() + timedelta(days=1)  # Tomorrow for safety
        
        # Try history with date range first
        new_data = ticker.history(start=start_date, end=end_date)
        
        # If empty, try period
        if new_data.empty:
            new_data = ticker.history(period='1mo')
        
        if new_data.empty:
            print(" ✗")
            print(f"  ⚠️  No data returned from yfinance")
            return True
        
        print(" ✓")
        
        # Show what we got
        yf_first = new_data.index[0]
        yf_last = new_data.index[-1]
        print(f"  📊 yfinance data: {yf_first.strftime('%Y-%m-%d')} to {yf_last.strftime('%Y-%m-%d')} ({len(new_data)} days)")
        
        # Process new data
        new_data = new_data.reset_index()
        new_data.columns = new_data.columns.str.lower()
        
        # Handle multi-level columns
        if isinstance(new_data.columns, pd.MultiIndex):
            new_data.columns = new_data.columns.get_level_values(0)
        
        # Rename date column
        for col in ['date', 'index']:
            if col in new_data.columns:
                new_data = new_data.rename(columns={col: date_col})
                break
        
        # Remove timezone from date
        if date_col in new_data.columns:
            new_data[date_col] = pd.to_datetime(new_data[date_col])
            if hasattr(new_data[date_col].dtype, 'tz') and new_data[date_col].dtype.tz:
                new_data[date_col] = new_data[date_col].dt.tz_localize(None)
        
        # Filter only dates AFTER CSV last date
        new_rows = new_data[new_data[date_col] > last_date].copy()
        
        if new_rows.empty:
            print(f"  ℹ️  No NEW trading days after {last_date.strftime('%Y-%m-%d')}")
            if is_weekend:
                print(f"  ✅ This is expected (Weekend - market closed)")
            return True
        
        print(f"  ➕ Found {len(new_rows)} new trading day(s):")
        for _, row in new_rows.iterrows():
            print(f"     {row[date_col].strftime('%Y-%m-%d')} → ${row['close']:.2f}")
        
        # Keep only matching columns
        common_cols = [col for col in df.columns if col in new_rows.columns]
        new_rows = new_rows[common_cols]
        
        # Combine
        combined = pd.concat([df, new_rows], ignore_index=True)
        combined = combined.drop_duplicates(subset=[date_col], keep='last')
        combined = combined.sort_values(date_col)
        
        # Save
        combined.to_csv(csv_path, index=False)
        
        added = len(combined) - original_len
        latest = combined[date_col].max()
        latest_price = float(combined['close'].iloc[-1])
        
        print(f"  ✅ Added {added} row(s) to CSV")
        print(f"  💾 Saved: Latest now {latest.strftime('%Y-%m-%d')} @ ${latest_price:.2f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Update stock CSVs with missing data")
    parser.add_argument("-s", "--stocks", nargs="+", help="Stocks to update")
    args = parser.parse_args()
    
    symbols = [s.upper() for s in args.stocks] if args.stocks else DEFAULT_STOCKS
    
    print("\n" + "="*70)
    print("📊 STOCK CSV UPDATER")
    print("="*70)
    print(f"⏰ {datetime.now().strftime('%A, %B %d, %Y - %I:%M %p')}")
    print(f"📈 Stocks: {', '.join(symbols)}")
    
    # Show market status
    now = datetime.now()
    is_weekend = now.weekday() >= 5
    if is_weekend:
        print(f"⚠️  Today is {now.strftime('%A')} - Market Closed")
        print(f"   Next trading day: Monday")
    
    print("="*70 + "\n")
    
    success = 0
    failed = 0
    
    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] {symbol}")
        
        if update_stock_csv(symbol):
            success += 1
        else:
            failed += 1
        
        print()
        
        if i < len(symbols):
            time.sleep(0.5)
    
    print("="*70)
    print(f"✅ Updated: {success}/{len(symbols)}")
    if failed > 0:
        print(f"❌ Failed: {failed}/{len(symbols)}")
    print("="*70)
    
    if success > 0:
        print("\n💡 Next: python predict.py --portfolio\n")


if __name__ == "__main__":
    main()
