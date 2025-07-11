#!/usr/bin/env python3
"""
Cleanup Non-Trading Day JSON Files

This script removes JSON files for dates that are not trading days:
- Weekends (Saturday, Sunday)
- Major US holidays
- Any other non-trading days

Usage: python cleanup_non_trading_days.py [TICKER]
"""

import os
import sys
import glob
from datetime import datetime
from pathlib import Path

def is_trading_day(date):
    """Check if a date is a trading day (not weekend or major US holiday)"""
    # Check if it's a weekend
    if date.weekday() >= 5:  # 5=Saturday, 6=Sunday
        return False
    
    # Check major US stock market holidays
    year = date.year
    month = date.month
    day = date.day
    
    # Major holidays (simplified list)
    holidays = [
        (1, 1),   # New Year's Day
        (7, 4),   # Independence Day
        (12, 25), # Christmas Day
        (6, 19),  # Juneteenth
        (11, 28), # Thanksgiving (approximate - 4th Thursday)
        (11, 29), # Black Friday (day after Thanksgiving)
    ]
    
    # Check Martin Luther King Jr. Day (3rd Monday in January)
    if month == 1 and date.weekday() == 0:  # Monday
        # Find the 3rd Monday
        first_monday = 1
        while datetime(year, 1, first_monday).weekday() != 0:
            first_monday += 1
        third_monday = first_monday + 14
        if day == third_monday:
            return False
    
    # Check Presidents' Day (3rd Monday in February)
    if month == 2 and date.weekday() == 0:  # Monday
        first_monday = 1
        while datetime(year, 2, first_monday).weekday() != 0:
            first_monday += 1
        third_monday = first_monday + 14
        if day == third_monday:
            return False
    
    # Check Memorial Day (last Monday in May)
    if month == 5 and date.weekday() == 0:  # Monday
        # Find the last Monday
        last_day = 31
        while True:
            try:
                last_monday = datetime(year, 5, last_day)
                if last_monday.weekday() == 0:
                    break
                last_day -= 1
            except ValueError:
                last_day -= 1
        if day == last_day:
            return False
    
    # Check Labor Day (1st Monday in September)
    if month == 9 and date.weekday() == 0:  # Monday
        first_monday = 1
        while datetime(year, 9, first_monday).weekday() != 0:
            first_monday += 1
        if day == first_monday:
            return False
    
    for holiday_month, holiday_day in holidays:
        if month == holiday_month and day == holiday_day:
            return False
    
    return True

def clean_non_trading_day_files(ticker=None):
    """Remove JSON files for non-trading days"""
    print(f"🧹 Cleaning non-trading day JSON files")
    if ticker:
        print(f"📊 Ticker: {ticker}")
    
    # Common output directories
    output_dirs = [
        os.path.expanduser("~/Workspace/zzsheepTrader/analysis_results/json"),
        os.path.expanduser("~/zzsheepTrader/analysis_results/json"),
        "./analysis_results/json",
        "../zzsheepTrader/analysis_results/json",
        "C:/Users/USER/Workspace/zzsheepTrader/analysis_results/json"
    ]
    
    files_removed = 0
    files_checked = 0
    
    for output_dir in output_dirs:
        if os.path.exists(output_dir):
            print(f"📁 Checking directory: {output_dir}")
            
            # Create pattern based on ticker
            if ticker:
                pattern = os.path.join(output_dir, f"{ticker}-*.json")
            else:
                pattern = os.path.join(output_dir, "*.json")
            
            json_files = glob.glob(pattern)
            print(f"   Found {len(json_files)} JSON files")
            
            for json_file in json_files:
                files_checked += 1
                try:
                    # Extract date from filename
                    filename = os.path.basename(json_file)
                    print(f"   📄 Checking: {filename}")
                    
                    # Try to extract date from different filename patterns
                    date_extracted = False
                    file_date = None
                    
                    # Pattern: TICKER-YYYY-MM-DD_*.json
                    if '-' in filename and '_' in filename:
                        parts = filename.split('-')
                        if len(parts) >= 4:  # TICKER-YYYY-MM-DD
                            try:
                                date_str = f"{parts[1]}-{parts[2]}-{parts[3].split('_')[0]}"
                                file_date = datetime.strptime(date_str, '%Y-%m-%d')
                                date_extracted = True
                            except (ValueError, IndexError):
                                pass
                    
                    # Pattern: TICKER-YYYY-MM-DD.json
                    if not date_extracted and '-' in filename:
                        parts = filename.split('-')
                        if len(parts) >= 4:
                            try:
                                date_str = f"{parts[1]}-{parts[2]}-{parts[3].split('.')[0]}"
                                file_date = datetime.strptime(date_str, '%Y-%m-%d')
                                date_extracted = True
                            except (ValueError, IndexError):
                                pass
                    
                    if date_extracted and file_date:
                        day_name = file_date.strftime('%A')
                        if not is_trading_day(file_date):
                            os.remove(json_file)
                            files_removed += 1
                            print(f"      🗑️  REMOVED: {filename} ({day_name} - Non-trading day)")
                        else:
                            print(f"      ✅ KEPT: {filename} ({day_name} - Trading day)")
                    else:
                        print(f"      ⚠️  Could not parse date from filename: {filename}")
                        
                except Exception as e:
                    print(f"      ❌ Error processing file {json_file}: {e}")
    
    print(f"\n📊 Summary:")
    print(f"   📄 Files checked: {files_checked}")
    print(f"   🗑️  Files removed: {files_removed}")
    print(f"   ✅ Files kept: {files_checked - files_removed}")

def main():
    """Main function"""
    ticker = None
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
    
    print(f"🧹 Non-Trading Day JSON Cleanup Tool")
    print("=" * 50)
    
    clean_non_trading_day_files(ticker)
    
    print(f"\n🎉 Cleanup Complete!")

if __name__ == "__main__":
    main() 