import os
import re
import matplotlib.pyplot as plt 
import datetime
import glob
import math
import sklearn 
from sklearn.cluster import DBSCAN

class LidarMeasurement:
    def __init__(self):
        self.angle = 0.0
        self.avg_distance = 0.0
        self.distances = []
        self.coordinates = []
        self.raw_coordinates = []
        self.centroids = [] 
        self.centroid = (0.0, 0.0)

    def clear(self):
        self.angle = 0.0
        self.distances.clear()
        self.coordinates.clear()
        self.centroids.clear()
        self.centroid = (0.0, 0.0)

class LidarSweep:
    def __init__(self):
        self.timestamp = 0 
        self.accelerometerAvailable = 0
        self.sonar_distance = 0.0
        self.area = 0.0
        self.streamWidth = 0.0
        self.left_bank = (0.0, 0.0)
        self.right_bank = (0.0, 0.0)
        self.measurements = []

    def clear(self):
        self.accelerometerAvailable = 0
        self.sonar_distance = 0.0
        self.area = 0.0
        self.streamWidth = 0.0
        self.left_bank = (0.0, 0.0)
        self.right_bank = (0.0, 0.0)
        self.measurements.clear()


num_angles = 0
angles = []
sweep = LidarSweep()

def calculate_coordinates():

    n1 = 1.00
    n2 = 1.33
    water_level = sweep.sonar_distance

    for meas in sweep.measurements:

        rad_angle = math.radians(meas.angle)

        beta = math.asin(n1 * math.sin(rad_angle) / n2)
        da = water_level / math.cos(rad_angle)
        db = meas.avg_distance - da
        xa = math.sin(rad_angle) * da

        xb = math.sin(rad_angle if db <= 0 else beta) * db / (n1 if db <= 0 else n2)
        yb = math.cos(rad_angle if db <= 0 else beta) * db / (n1 if db <= 0 else n2)

        meas.coordinates.append((-1 * (xa + xb), -1 * (water_level - yb)))

        x = meas.avg_distance * math.sin(rad_angle)
        y = meas.avg_distance * math.cos(rad_angle)
        
        meas.raw_coordinates.append((x, y))

# Currently not working
def remove_outliers():

    for meas in sweep.measurements:
        x_coords = []
        y_coords = []
        for x, y in meas.coordinates:
            x_coords.append(x)
            y_coords.append(y)
        
        x_coords = [[x] for x in x_coords]
        y_coords = [[y] for y in y_coords]

        db = DBSCAN(eps=10, min_samples=5).fit(x_coords, y_coords)

        labels = db.labels_

        for i, label in enumerate(labels):
            if label == -1:
                meas.coordinates.pop(i)
                meas.raw_coordinates.pop(i)

def execute():
    
    file_pattern = "/home/streamscope/data/streamscope_log_*.txt"
    file_list = glob.glob(file_pattern)
    for file_name in file_list:
        try:
            with open(file_name, 'r') as file:
                lines = file.readlines()
                print("Processing file: ", file_name)
                for i, line in enumerate(lines):
                    
                    line = line.strip()
                    
                    if line.startswith("Date:") and "Time:" in lines[i+1]:
                        date_str = line.split("Date:")[-1].strip()
                        time_str = lines[i+1].split("Time:")[-1].strip()
                        timestamp = f"{date_str} {time_str}"
                        timestamp = datetime.datetime.strptime(timestamp, "%d/%m/%Y %H:%M UTC")
                        sweep.timestamp = timestamp
                    if line.startswith("Number of Angles:"):
                        num_angles = int(line.split("Number of Angles:")[-1].strip())
                    if line.startswith("Sonar Distance (mm):"):
                        sweep.sonar_distance = int(line.split("Sonar Distance (mm):")[-1].strip())
                    if line.startswith("Recorded Angles (degrees):"):
                        angles = list(map(int, re.findall(r"-?\d+", line)))  
                    if line.startswith("Raw Measurements"):
                        for j in range(i+1, len(lines)):
                            if lines[j].startswith("Angle (degrees):"):
                                meas = LidarMeasurement()
                                meas.angle = int(lines[j].split("Angle (degrees):")[-1].strip())

                                distances_line = lines[j + 1]
                                distances = list(map(int, distances_line.split(":")[-1].strip().split(", ")))
                        
                                valid_distances = [d for d in distances if d <= 5000]

                                if valid_distances:
                                    meas.avg_distance = sum(valid_distances) / len(valid_distances)
                                
                                sweep.measurements.append(meas)

                calculate_coordinates()
                # remove_outliers()
                graph()
                sweep.clear()
                        
        except FileNotFoundError:
            print(f"Error: File {file_name} not found.")

def graph():
    
    x_coords = []
    y_coords = []
    
    for measurement in sweep.measurements:
        for x, y in measurement.coordinates:
            x_coords.append(x)
            y_coords.append(y)

    plt.figure(figsize=(8, 6))
    
    y_coords = [-y for y in y_coords]

    plt.scatter(x_coords, y_coords, label="Lidar Measurements", color='blue', s=10)

    plt.axhline(y=sweep.sonar_distance, color='red', linestyle='--', label="Sonar Distance")

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title(f"StreamScope Cross-Section (Accumulation Day and Night)")
    plt.legend()
    plt.grid(True)

    plt.show()
    
           
if __name__ == "__main__":
    
    while True:
        print("StreamScope Cross-Section.\n")
        print("1. Cross-Section View.\n")
        print("2. Exit.\n")
        choice = input("Select an option: ") 
        
        if choice == '1':
            execute()
        elif choice == '2':
            break

