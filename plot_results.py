#!/usr/bin/env python3
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def main():
    curves_dir = "catkin_ws/src/gtrl/scripts/SAC/final_curves"
    files = glob.glob(os.path.join(curves_dir, "*.npy"))
    
    if not files:
        print("No reward curves found yet. Train for longer to generate data.")
        return

    # Find the most recently modified numpy file
    latest_file = max(files, key=os.path.getctime)
    print(f"Plotting training data from: {latest_file}")
    
    data = np.load(latest_file, allow_pickle=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(data, label='Mean Reward (last 20 eps)', color='blue')
    plt.title('GoT-GTRL Training Performance')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
