import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

def process_dataset(name, csv_path, out_png):
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found.")
        return
    
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        print(f"No data in {csv_path}.")
        return
        
    avg_time = df['time_ms'].mean()
    tpr = df['is_detected'].mean()
    
    print(f"--- {name} ---")
    print(f"Average computing time: {avg_time:.2f} ms")
    print(f"Overall TPR: {tpr:.2%} ({df['is_detected'].sum()} / {len(df)})")
    
    # Pivot for heatmap
    # X axis: distance, Y axis: angle
    pivot_df = df.pivot(index='angle', columns='distance', values='is_detected')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_df, annot=True, cmap='RdYlGn', cbar=False, fmt='g')
    plt.title(f"{name} Detection Heatmap\nAvg Time: {avg_time:.2f}ms | TPR: {tpr:.2%}")
    plt.xlabel("Distance")
    plt.ylabel("Angle")
    
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"Saved heatmap to {out_png}\n")

def main():
    datasets = [
        ("Marker", "results_marker.csv", "heatmap_marker.png"),
        ("Fractal", "results_fractal.csv", "heatmap_fractal.png"),
        ("Raruco", "results_raruco.csv", "heatmap_raruco.png")
    ]
    
    for name, csv_path, out_png in datasets:
        process_dataset(name, csv_path, out_png)

if __name__ == "__main__":
    main()
