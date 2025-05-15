"""
StreamScope Cross-Section Analysis Script

This script processes data from StreamScope sweeps to analyze stream cross-sections.
It reads measurement logs, computes coordinates for points, removes outliers, clusters points to find centroids,
identifies stream banks, and visualizes the cross-section using matplotlib.

Classes:
    LidarPoint: Represents a single coordinate point in 2D space.
    LidarMeasurement: Holds all measurements for a single angle sweep.
    LidarSweep: Represents a full sweep of laser measurements at a given timestamp.

Functions:
    calculate_coordinates(sweeps): Computes (x, y) coordinates for all measurements, accounting for refraction.
    remove_outliers(sweep, eps=50, min_samples=3): Removes outlier points from measurements using DBSCAN clustering.
    process_centroids(sweep): Clusters inlier points to find centroids for each angle.
    find_banks(sweep): Identifies the left and right stream banks based on centroids and water detection.
    graph(sweeps): Visualizes the processed data, including raw points, centroids, and banks.
    execute(): Main data processing and visualization pipeline.
"""

import os
import re
import matplotlib.pyplot as plt
import datetime
import glob
import math
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans

class LidarPoint:

    """
    Represents a single LIDAR point in 2D space.

    Attributes:
        x (float): X coordinate.
        y (float): Y coordinate.
        distance (float): Distance from origin.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.distance = math.sqrt(x**2 + y**2)

    def clear(self):
        """
        Resets the point's coordinates and distance to zero.
        """
        self.x = 0.0
        self.y = 0.0
        self.distance = 0.0

class LidarMeasurement:

    """
    Holds all measurements for a single angle sweep.

    Attributes:
        sonar_distance (int): Sonar distance for this measurement.
        angle (float): Angle of the measurement.
        distances (list): List of raw distance readings.
        raw_points (list): List of LidarPoint objects.
        inliers (list): List of inlier points after clustering.
        outliers (list): List of outlier points after clustering.
        certain_centroid (tuple): Centroid considered certain (likely bank or bottom).
        uncertain_centroid (tuple): Centroid considered uncertain.
        in_water (bool): Whether this measurement is in water.
    """

    def __init__(self):
        self.sonar_distance = 0
        self.angle = 0.0
        self.distances = []
        self.raw_points = []
        self.inliers = []
        self.outliers = []
        self.certain_centroid = (0.0, 0.0)
        self.uncertain_centroid = (0.0, 0.0)
        self.in_water = False

    def clear(self):
        """
        Resets all measurement data to initial state.
        """
        self.sonar_distance = 0
        self.angle = 0.0
        self.distances.clear()
        self.raw_points.clear()
        self.inliers.clear()
        self.outliers.clear()
        self.certain_centroid = (0.0, 0.0)
        self.uncertain_centroid = (0.0, 0.0)
        self.in_water = False

class LidarSweep:

    """
    Represents a full sweep of laser measurements at a given timestamp.

    Attributes:
        timestamp (datetime): Timestamp of the sweep.
        accelerometer_available (int): Accelerometer status.
        sonar_distances (list): List of sonar distances for the sweep.
        area (float): Calculated area (if applicable).
        stream_width (float): Calculated stream width (if applicable).
        left_bank (tuple): Coordinates of the left bank.
        right_bank (tuple): Coordinates of the right bank.
        measurements (list): List of LidarMeasurement objects.
    """

    def __init__(self):
        self.timestamp = 0 
        self.accelerometer_available = 0
        self.sonar_distances = []
        self.area = 0.0
        self.stream_width = 0.0
        self.left_bank = (0.0, 0.0)
        self.right_bank = (0.0, 0.0)
        self.measurements = []

    def clear(self):
        """
        Resets all sweep data to initial state.
        """
        self.accelerometer_available = 0
        self.sonar_distances = []
        self.area = 0.0
        self.stream_width = 0.0
        self.left_bank = (0.0, 0.0)
        self.right_bank = (0.0, 0.0)
        self.measurements.clear()

def calculate_coordinates(sweeps):

    """
    Computes (x, y) coordinates for all laser measurements in the provided sweeps,
    accounting for refraction at the water surface.

    Args:
        sweeps (list): List of LidarSweep objects.
    """

    n1 = 1.00
    n2 = 1.33

    for sweep in sweeps:
        for meas in sweep.measurements:
            water_level = meas.sonar_distance
            rad_angle = math.radians(meas.angle)
            da = water_level / math.cos(rad_angle)
            db = getattr(meas, "avg_distance", 0) - da

            if db <= 0:
                xa = math.sin(rad_angle) * getattr(meas, "avg_distance", 0)
                ya = math.cos(rad_angle) * getattr(meas, "avg_distance", 0)
                point = LidarPoint(-1 * xa, -1 * ya)
                meas.raw_points.append(point)
            else:
                beta = math.asin(n1 * math.sin(rad_angle) / n2)
                xa = math.sin(rad_angle) * da
                xb = math.sin(beta) * db / n2
                yb = math.cos(beta) * db / n2
                point = LidarPoint(-1 * (xa + xb), -1 * (water_level + yb))
                meas.raw_points.append(point)
                meas.in_water = True

def remove_outliers(sweep, eps=50, min_samples=3):

    """
    Removes outlier points from each measurement in the sweep using DBSCAN clustering.

    Args:
        sweep (LidarSweep): The sweep to process.
        eps (float): DBSCAN epsilon parameter.
        min_samples (int): Minimum samples for DBSCAN.
    """

    for measurement in sweep.measurements:
        if len(measurement.raw_points) < 2:
            continue

        coords = np.array([(point.x, point.y) for point in measurement.raw_points])
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        labels = dbscan.labels_

        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)

        if len(unique_labels) == 0:
            continue

        largest_cluster_label = max(unique_labels, key=list(labels).count)
        measurement.inliers = [coords[i] for i in range(len(labels)) if labels[i] == largest_cluster_label]
        measurement.outliers = [coords[i] for i in range(len(labels)) if labels[i] == -1]

def process_centroids(sweep):

    """
    Clusters inlier points for each measurement to find certain and uncertain centroids.

    Args:
        sweep (LidarSweep): The sweep to process.
    """

    angle_to_points = {meas.angle: list(meas.raw_points) for meas in sweep.measurements}

    missing_centroids = []

    for meas in sweep.measurements:
        if hasattr(meas, "inliers"):
            if meas.in_water:
                meas.inliers = [pt for pt in meas.inliers if meas.in_water]
            elif any(m.in_water and m.angle == meas.angle for m in sweep.measurements):
                meas.inliers = []

        if hasattr(meas, "inliers") and len(meas.inliers) >= 3:
            coords = np.array(meas.inliers)
            k = min(3, len(coords))
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=0).fit(coords)
            centroids = kmeans.cluster_centers_

            if len(centroids) > 2:
                centroids = sorted(centroids, key=lambda c: c[1])
                centroids = [centroids[0], centroids[-1]]
            if len(centroids) == 1:
                meas.certain_centroid = tuple(centroids[0])
                farthest_pt = max(meas.raw_points, key=lambda pt: abs(pt.y), default=None)
                meas.uncertain_centroid = (farthest_pt.x, farthest_pt.y) if farthest_pt else (0.0, 0.0)
            elif len(centroids) == 2:
                c0, c1 = centroids
                if abs(c0[1]) > abs(c1[1]):
                    farther, closer = c0, c1
                else:
                    farther, closer = c1, c0
                meas.uncertain_centroid = tuple(farther)
                meas.certain_centroid = tuple(closer)

                dist = np.linalg.norm(np.array(meas.uncertain_centroid) - np.array(meas.certain_centroid))
                if dist < 50:
                    furthest_outlier = max(getattr(meas, "outliers", []), key=lambda pt: abs(pt[1]), default=None)
                    if furthest_outlier is not None:
                        meas.uncertain_centroid = (furthest_outlier[0], furthest_outlier[1])
                    else:
                        furthest_raw = max(meas.raw_points, key=lambda pt: abs(pt.y), default=None)
                        meas.uncertain_centroid = (furthest_raw.x, furthest_raw.y) if furthest_raw else (0.0, 0.0)
            else:
                farthest_pt = max(meas.raw_points, key=lambda pt: max(pt.y), default=None)
                meas.uncertain_centroid = (farthest_pt.x, farthest_pt.y) if farthest_pt else (0.0, 0.0)
        else:
            farthest_pt = max(meas.raw_points, key=lambda pt: abs(pt.y), default=None)
            meas.uncertain_centroid = (farthest_pt.x, farthest_pt.y) if farthest_pt else (0.0, 0.0)

        if meas.certain_centroid == (0.0, 0.0) or meas.uncertain_centroid == (0.0, 0.0):
            missing_centroids.append(meas.angle)

    if missing_centroids:
        for meas in sweep.measurements:
            if meas.angle in missing_centroids and meas.raw_points:
                furthest_raw = max(meas.raw_points, key=lambda pt: abs(pt.y), default=None)
                if furthest_raw:
                    if meas.uncertain_centroid == (0.0, 0.0):
                        meas.uncertain_centroid = (furthest_raw.x, furthest_raw.y)

    for meas in sweep.measurements:
        if meas.uncertain_centroid != (0.0, 0.0):
            sonar_dist = sweep.sonar_distances[0] if sweep.sonar_distances else 0
            if abs(meas.uncertain_centroid[1]) < sonar_dist:
                meas.uncertain_centroid = (0.0, 0.0)

def find_banks(sweep):

    """
    Identifies the left and right stream banks based on centroids and water detection.

    Args:
        sweep (LidarSweep): The sweep to process.
    """

    sorted_measurements = sorted(sweep.measurements, key=lambda m: m.angle)
    n = len(sorted_measurements)

    def is_underwater(meas):
        return meas.in_water

    left_bank = (0.0, 0.0)
    left_angle = None
    for idx, meas in enumerate(sorted_measurements):
        if is_underwater(meas):
            prev_idx = idx - 1
            if prev_idx >= 0:
                prev_meas = sorted_measurements[prev_idx]
                if hasattr(prev_meas, "certain_centroid") and prev_meas.certain_centroid != (0.0, 0.0):
                    left_bank = prev_meas.certain_centroid
                    left_angle = prev_meas.angle
                elif hasattr(prev_meas, "uncertain_centroid") and prev_meas.uncertain_centroid != (0.0, 0.0):
                    left_bank = prev_meas.uncertain_centroid
                    left_angle = prev_meas.angle
                else:
                    left_bank = meas.certain_centroid if hasattr(meas, "certain_centroid") and meas.certain_centroid != (0.0, 0.0) else meas.uncertain_centroid
                    left_angle = meas.angle
            else:
                left_bank = meas.certain_centroid if hasattr(meas, "certain_centroid") and meas.certain_centroid != (0.0, 0.0) else meas.uncertain_centroid
                left_angle = meas.angle
            break

    right_bank = (0.0, 0.0)
    right_angle = None
    n = len(sorted_measurements)
    found_underwater = False
    underwater_angle = None
    underwater_centroid = (0.0, 0.0)

    for idx in reversed(range(n)):
        meas = sorted_measurements[idx]
        if not found_underwater:
            if is_underwater(meas):
                found_underwater = True
                underwater_angle = meas.angle
                underwater_centroid = meas.certain_centroid if hasattr(meas, "certain_centroid") and meas.certain_centroid != (0.0, 0.0) else meas.uncertain_centroid
    
    for idx in reversed(range(n)):
        meas = sorted_measurements[idx]
        if meas.angle > underwater_angle:
            candidate_centroid = None
            if hasattr(meas, "certain_centroid") and meas.certain_centroid != (0.0, 0.0):
                candidate_centroid = meas.certain_centroid
            elif hasattr(meas, "uncertain_centroid") and meas.uncertain_centroid != (0.0, 0.0):
                candidate_centroid = meas.uncertain_centroid
            closest_candidate = None
            closest_angle = None
            min_dist = float('inf')
            for idx2 in reversed(range(n)):
                meas2 = sorted_measurements[idx2]
                if meas2.angle > underwater_angle:
                    cand_centroid = None
                    if hasattr(meas2, "certain_centroid") and meas2.certain_centroid != (0.0, 0.0):
                        cand_centroid = meas2.certain_centroid
                    elif hasattr(meas2, "uncertain_centroid") and meas2.uncertain_centroid != (0.0, 0.0):
                        cand_centroid = meas2.uncertain_centroid
                    if cand_centroid and abs(cand_centroid[0]) > abs(underwater_centroid[0]):
                        dist = abs(cand_centroid[0] - underwater_centroid[0])
                        if dist < min_dist:
                            min_dist = dist
                            closest_candidate = cand_centroid
                            closest_angle = meas2.angle
            if closest_candidate:
                right_bank = closest_candidate
                right_angle = closest_angle

    sweep.left_bank = left_bank
    sweep.right_bank = right_bank

def graph(sweeps):

    """
    Visualizes the processed data, including raw points, centroids, and banks.

    Args:
        sweeps (list): List of LidarSweep objects to visualize.
    """

    raw_x = []
    raw_y = []
    inlier_x = []
    inlier_y = []
    outlier_x = []
    outlier_y = []
    centroid_x = []
    centroid_y = []
    uncertain_centroid_x = []
    uncertain_centroid_y = []

    for sweep in sweeps:
        for measurement in sweep.measurements:
            for point in getattr(measurement, "raw_points", []):
                raw_x.append(-point.x)
                raw_y.append(-point.y)
            for inlier in getattr(measurement, "inliers", []):
                inlier_x.append(-inlier[0])
                inlier_y.append(-inlier[1])
            for outlier in getattr(measurement, "outliers", []):
                outlier_x.append(-outlier[0])
                outlier_y.append(-outlier[1])
            if hasattr(measurement, "certain_centroid") and measurement.certain_centroid != (0.0, 0.0):
                centroid_x.append(-measurement.certain_centroid[0])
                centroid_y.append(-measurement.certain_centroid[1])
            if hasattr(measurement, "uncertain_centroid") and measurement.uncertain_centroid != (0.0, 0.0):
                uncertain_centroid_x.append(-measurement.uncertain_centroid[0])
                uncertain_centroid_y.append(-measurement.uncertain_centroid[1])

    plt.figure(figsize=(15, 8))

    if hasattr(sweep, "left_bank") and sweep.left_bank != (0.0, 0.0):
        plt.axvline(x=sweep.left_bank[0]*-1, color='purple', linestyle='-.', linewidth=2, label="Left Bank")
    if hasattr(sweep, "right_bank") and sweep.right_bank != (0.0, 0.0):
        plt.axvline(x=sweep.right_bank[0]*-1, color='brown', linestyle='-.', linewidth=2, label="Right Bank")

    sonar_distances = [d for sweep in sweeps if hasattr(sweep, "sonar_distances") for d in sweep.sonar_distances]
    if sonar_distances:
        min_sonar = min(sonar_distances)
        max_sonar = max(sonar_distances)
        plt.axhspan(max_sonar, min_sonar, color='red', alpha=0.15, label="Sonar Distance Range")

    if raw_x and raw_y:
        pass
        plt.scatter(raw_x, raw_y, s=8, color='black', alpha=0.2, label="Raw Points")

    if inlier_x and inlier_y:
        pass
        # plt.scatter(inlier_x, inlier_y, s=8, color='blue', alpha=0.5, label="Inliers", zorder=5)

    if outlier_x and outlier_y:
        pass
        # plt.scatter(outlier_x, outlier_y, s=8, color='red', alpha=0.5, label="Outliers", zorder=5)

    if centroid_x and centroid_y:
        pass
        plt.scatter(centroid_x, centroid_y, s=80, color='green', marker='X', label="Certain Centroids", zorder=6, edgecolors='black', linewidths=1.2)
    if uncertain_centroid_x and uncertain_centroid_y:
        plt.scatter(uncertain_centroid_x, uncertain_centroid_y, s=80, color='orange', marker='P', label="Uncertain Centroids", zorder=6, edgecolors='black', linewidths=1.2)

    if centroid_x and centroid_y:
        angle_centroid_pairs = []
        for sweep in sweeps:
            for measurement in sweep.measurements:
                if hasattr(measurement, "certain_centroid") and measurement.certain_centroid != (0.0, 0.0):
                    angle_centroid_pairs.append((measurement.angle, measurement.certain_centroid))
        angle_centroid_pairs.sort(key=lambda x: x[0])
        if angle_centroid_pairs:
            cx, cy = zip(*[c for _, c in angle_centroid_pairs])
            cy = [-y for y in cy]
            cx = [-x for x in cx]
            plt.plot(cx, cy, color='lime', linewidth=2, label="Certain Centroids Contour")

    if uncertain_centroid_x and uncertain_centroid_y:
        angle_centroid_pairs = []
        for sweep in sweeps:
            for measurement in sweep.measurements:
                if hasattr(measurement, "uncertain_centroid") and measurement.uncertain_centroid != (0.0, 0.0):
                    angle_centroid_pairs.append((measurement.angle, measurement.uncertain_centroid))
        angle_centroid_pairs.sort(key=lambda x: x[0])
        if angle_centroid_pairs:
            cx, cy = zip(*[c for _, c in angle_centroid_pairs])
            cy = [-y for y in cy]
            cx = [-x for x in cx]

            plt.plot(cx, cy, color='orange', linewidth=2, label="Uncertain Centroids Contour")
        
    plt.title("StreamScope Cross-Section (2025-05-14)")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.xlim(-2800, 2800)
    plt.ylim(5600, 0)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def execute():

    """
    Main data processing and visualization pipeline.
    Reads log files, processes sweeps, computes coordinates, removes outliers,
    finds centroids and banks, and visualizes the results.
    """

    file_pattern = "/home/braden/StreamScopeDeploy/v2_data/streamscope_log_*.txt"
    file_list = glob.glob(file_pattern)
    all_sweeps = [] 

    for file_name in file_list:
        try:
            with open(file_name, 'r') as file:
                lines = file.readlines()
                sweep = LidarSweep()
                
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
                                angle_val = int(lines[j].split("Angle (degrees):")[-1].strip())
                                if angle_val < -90 or angle_val > 90:
                                    continue
                                meas = LidarMeasurement()
                                meas.sonar_distance = sweep.sonar_distance
                                meas.angle = angle_val
                                distances_line = lines[j + 1]
                                distances = list(map(int, distances_line.split(":")[-1].strip().split(", ")))
                                valid_distances = [d for d in distances if d > 500 and d < 5000]
                                if valid_distances:
                                    meas.avg_distance = sum(valid_distances) / len(valid_distances)
                                sweep.measurements.append(meas)

                all_sweeps.append(sweep) 

        except FileNotFoundError:
            print(f"Error: File {file_name} not found.")

    calculate_coordinates(all_sweeps)

    combined_by_angle = {}
    for sweep in all_sweeps:
        for meas in sweep.measurements:
            angle = meas.angle
            if angle not in combined_by_angle:
                combined_by_angle[angle] = []
            combined_by_angle[angle].append(meas)

    unique_sonar_distances = set()
    for sweep in all_sweeps:
        if hasattr(sweep, "sonar_distance"):
            unique_sonar_distances.add(sweep.sonar_distance)
        if hasattr(sweep, "sonar_distances"):
            unique_sonar_distances.update(sweep.sonar_distances)
    unique_sonar_distances = {d for d in unique_sonar_distances if d}

    combined_sweep = LidarSweep()
    combined_sweep.sonar_distances = list(unique_sonar_distances)
    for angle, measurements in combined_by_angle.items():
        combined_meas = LidarMeasurement()
        combined_meas.angle = angle

        for m in measurements:
            if hasattr(m, "distances"):
                combined_meas.distances.extend(m.distances)
            if hasattr(m, "raw_points"):
                combined_meas.raw_points.extend(m.raw_points)
            if hasattr(m, "in_water"):
                if m.in_water is True:
                    combined_meas.in_water = True

        combined_sweep.measurements.append(combined_meas)

    remove_outliers(combined_sweep, eps=10, min_samples=5)
    process_centroids(combined_sweep)
    find_banks(combined_sweep)
    graph([combined_sweep])

if __name__ == "__main__":
    while True:
        print("\nStreamScope Cross-Section.\n")
        print("1. Cross-Section View.")
        print("2. Exit.")
        choice = input("Select an option: ") 
        
        if choice == '1':
            execute()
        elif choice == '2':
            break
        else:
            print("Invalid choice. Please try again.\n")
