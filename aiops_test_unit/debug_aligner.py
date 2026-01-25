import os
import glob
import pandas as pd
import numpy as np

# === Configuration ===
METRIC_DIR = "data/processed_metrics"
LOG_DIR = "data/processed_logs"
WINDOW_SIZE = 100 

def extract_service_prefix(filename):
    """
    Extracts the core service name from the complex metric filename.
    Example: 'adservice-0.destination.frontend.csv' -> 'adservice-0'
    """
    # Strategy: The log file names seem to be "service-index".
    # We can split by '.' and take the first part if it matches that pattern.
    parts = filename.split('.')
    potential_name = parts[0] # e.g., "adservice-0"
    return potential_name

def debug_one_pair():
    print("🔬 [Debug] Starting Single File Alignment Diagnosis (Fuzzy Matching)...")
    
    m_files = glob.glob(os.path.join(METRIC_DIR, "*.csv"))
    target_metric_path = None
    target_log_path = None
    
    print(f"   - Found {len(m_files)} Metric files.")
    
    # Try to find a match using the prefix logic
    for f in m_files:
        metric_filename = os.path.basename(f)
        # Logic: Extract "adservice-0" from "adservice-0.destination..."
        core_name = extract_service_prefix(metric_filename)
        
        # Construct expected log path
        log_filename = f"{core_name}.csv"
        potential_log_path = os.path.join(LOG_DIR, log_filename)
        
        if os.path.exists(potential_log_path):
            print(f"\n✅ MATCH FOUND!")
            print(f"   - Core Name:   {core_name}")
            print(f"   - Metric File: {metric_filename}")
            print(f"   - Log File:    {log_filename}")
            target_metric_path = f
            target_log_path = potential_log_path
            break
    
    if target_metric_path is None:
        print("❌ Fatal Error: Still cannot match any files even with prefix logic.")
        return

    # --- Proceed with Data Loading & Alignment Check ---
    df_m = pd.read_csv(target_metric_path)
    df_l = pd.read_csv(target_log_path)
    
    m_ts = df_m['timestamp'].values.astype(np.float64)
    l_ts = df_l['timestamp'].values.astype(np.float64)
    l_vals = df_l['event_id'].values
    
    print("\n⏱️ Timestamp Sample (First 5):")
    print(f"   - Metric: {m_ts[:5]}")
    print(f"   - Log:    {l_ts[:5]}")
    
    # Align
    print(f"\n⚙️ Executing Alignment (Window={WINDOW_SIZE})...")
    
    left_indices = np.searchsorted(l_ts, m_ts)
    right_indices = np.searchsorted(l_ts, m_ts + WINDOW_SIZE)
    
    total_logs_found = 0
    valid_windows = 0
    
    # Check first 10 windows
    print("   - Inspecting first 10 windows:")
    for i in range(min(10, len(m_ts))):
        start = left_indices[i]
        end = right_indices[i]
        count = end - start
        if count > 0:
            valid_windows += 1
            total_logs_found += count
            print(f"     Window {i}: Logs Found = {count} (Indices {start}-{end})")
        else:
            print(f"     Window {i}: Logs Found = 0")

    print(f"\n📊 Diagnosis Conclusion:")
    print(f"   - Total Windows: {len(m_ts)}")
    print(f"   - Windows with Logs: {valid_windows}")
    
    if valid_windows > 0:
        print("✅ SUCCESS! The naming mismatch was the cause.")
    else:
        print("❌ FAILURE: Files matched, but time ranges still don't overlap.")

if __name__ == "__main__":
    debug_one_pair()