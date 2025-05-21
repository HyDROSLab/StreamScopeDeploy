import os
import re
import matplotlib.pyplot as plt 
import datetime
import glob
import math
import numpy as np
import pandas as pd
import csv
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
        self.uncertain = ("None", "None") # ("True", "Raw") or ("False", "Inlier")

    def clear(self):
        """
        Resets the point's coordinates and distance to zero.
        """
        self.x = 0.0
        self.y = 0.0
        self.distance = 0.0

    def as_tuple(self):
        return (self.x, self.y)

    def __eq__(self, other):
        if isinstance(other, LidarPoint):
            return np.allclose([self.x, self.y], [other.x, other.y])
        elif isinstance(other, tuple) or isinstance(other, list):
            return np.allclose([self.x, self.y], other)
        return False

    def __hash__(self):
        return hash((round(self.x, 6), round(self.y, 6)))

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
        certain_centroid (LidarPoint): Centroid considered certain (likely bank or bottom).
        uncertain_centroid (LidarPoint): Centroid considered uncertain.
        in_water (bool): Whether this measurement is in water.
    """

    def __init__(self):
        self.timestamp = None
        self.sonar_distance = 0
        self.angle = 0.0
        self.avg_distance = 0.0
        self.distances = []
        self.raw_points = []
        self.inliers = []
        self.outliers = []
        self.certain_centroid = None
        self.uncertain_centroid = None
        self.in_water = False

    def clear(self):
        """
        Resets all measurement data to initial state.
        """
        self.timestamp = None
        self.sonar_distance = 0
        self.angle = 0.0
        self.distances.clear()
        self.raw_points.clear()
        self.inliers.clear()
        self.outliers.clear()
        self.certain_centroid = None
        self.uncertain_centroid = None
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
        left_bank (LidarPoint): Coordinates of the left bank.
        right_bank (LidarPoint): Coordinates of the right bank.
        measurements (list): List of LidarMeasurement objects.
    """

    def __init__(self):
        self.timestamps = []
        self.accelerometer_available = 0
        self.sonar_distances = []
        self.area = 0.0
        self.stream_width = 0.0
        self.left_bank = None
        self.right_bank = None
        self.measurements = []

    def clear(self):
        """
        Resets all sweep data to initial state.
        """
        self.timestamps.clear()
        self.accelerometer_available = 0
        self.sonar_distances = []
        self.area = 0.0
        self.stream_width = 0.0
        self.left_bank = None
        self.right_bank = None
        self.measurements.clear()


num_angles = 0
angles = []
sweep = LidarSweep()

def calculate_coordinates():

    n1 = 1.00
    n2 = 1.33

    for sweep in sweeps:
        for meas in sweep.measurements:
            water_level = meas.sonar_distance
            rad_angle = math.radians(meas.angle)
            for dist in meas.distances:
                if 500 < dist < 5000:
                    da = water_level / math.cos(rad_angle)
                    db = dist - da

                    if db <= 0:
                        xa = math.sin(rad_angle) * dist
                        ya = math.cos(rad_angle) * dist
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

        coords = np.array([point.as_tuple() for point in measurement.raw_points])
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        labels = dbscan.labels_

        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)

        if len(unique_labels) == 0:
            continue

        largest_cluster_label = max(unique_labels, key=list(labels).count)
        inlier_tuples = [coords[i] for i in range(len(labels)) if labels[i] == largest_cluster_label]
        outlier_tuples = [coords[i] for i in range(len(labels)) if labels[i] == -1]
        measurement.inliers = [LidarPoint(x, y) for x, y in inlier_tuples]
        measurement.outliers = [LidarPoint(x, y) for x, y in outlier_tuples]

def process_centroids(sweep):
    """
    Clusters inlier points for each measurement to find certain and uncertain centroids.

    Args:
        sweep (LidarSweep): The sweep to process.
    """

    angle_to_points = {meas.angle: list(meas.raw_points) for meas in sweep.measurements}

    missing_centroids = []

    for meas in sweep.measurements:
        sonar_dist = getattr(meas, "sonar_distance", 0)
        if hasattr(meas, "inliers"):
            if meas.in_water:
                meas.inliers = [pt for pt in meas.inliers if meas.in_water]
            elif any(m.in_water and m.angle == meas.angle for m in sweep.measurements):
                meas.inliers = []

        if hasattr(meas, "inliers") and len(meas.inliers) >= 10:
            coords = np.array([pt.as_tuple() for pt in meas.inliers])
            k = min(3, len(coords))
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=0).fit(coords)
            centroids = kmeans.cluster_centers_
            labels = kmeans.labels_

            # Convert centroids to LidarPoint objects
            centroid_points = [LidarPoint(c[0], c[1]) for c in centroids]

            # Find the centroid with the highest density (most points assigned)
            label_counts = np.bincount(labels)
            densest_idx = np.argmax(label_counts)
            certain_centroid = centroid_points[densest_idx]

            # For uncertain centroid, pick the farthest centroid from origin (or sonar_dist)
            farthest_idx = np.argmax([abs(pt.y) for pt in centroid_points])
            uncertain_candidate = centroid_points[farthest_idx]
            if abs(uncertain_candidate.y) > sonar_dist:
                uncertain_centroid = uncertain_candidate
            else:
                uncertain_centroid = None

            meas.certain_centroid = certain_centroid
            meas.uncertain_centroid = uncertain_centroid

            # If only one centroid, fallback to farthest inlier or raw point for uncertain
            if len(centroid_points) == 1:
                farthest_pt = max(
                    (pt for pt in meas.inliers if abs(pt.y) > sonar_dist),
                    key=lambda pt: abs(pt.y),
                    default=None
                )
                if farthest_pt:
                    meas.uncertain_centroid = farthest_pt
                else:
                    farthest_pt = max((pt for pt in meas.raw_points if abs(pt.y) > sonar_dist), key=lambda pt: abs(pt.y), default=None)
                    meas.uncertain_centroid = farthest_pt if farthest_pt else None
        else:
            farthest_pt = max(
                (pt for pt in meas.raw_points if abs(pt.y) > sonar_dist),
                key=lambda pt: abs(pt.y),
                default=None
            )
            meas.uncertain_centroid = farthest_pt if farthest_pt else None
            meas.certain_centroid = None

        # Ensure uncertain centroid does not exist at y values greater than sonar distance
        if meas.uncertain_centroid is not None and abs(meas.uncertain_centroid.y) <= sonar_dist:
            meas.uncertain_centroid = None

        if meas.certain_centroid is None or meas.uncertain_centroid is None:
            missing_centroids.append(meas.angle)

    if missing_centroids:
        for meas in sweep.measurements:
            sonar_dist = getattr(meas, "sonar_distance", 0)
            if meas.angle in missing_centroids and meas.raw_points:
                furthest_raw = max(
                    (pt for pt in meas.raw_points if abs(pt.y) > sonar_dist),
                    key=lambda pt: abs(pt.y),
                    default=None
                )
                if furthest_raw and abs(furthest_raw.y) > sonar_dist:
                    meas.uncertain_centroid = furthest_raw
                    meas.certain_centroid = None
                else:
                    meas.uncertain_centroid = None
                    meas.certain_centroid = None

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

    left_bank = None
    left_angle = None
    for idx, meas in enumerate(sorted_measurements):
        if is_underwater(meas):
            prev_idx = idx - 1
            if prev_idx >= 0:
                prev_meas = sorted_measurements[prev_idx]
                if hasattr(prev_meas, "certain_centroid") and prev_meas.certain_centroid is not None:
                    left_bank = prev_meas.certain_centroid
                    left_angle = prev_meas.angle
                elif hasattr(prev_meas, "uncertain_centroid") and prev_meas.uncertain_centroid is not None:
                    left_bank = prev_meas.uncertain_centroid
                    left_angle = prev_meas.angle
                else:
                    left_bank = meas.certain_centroid if hasattr(meas, "certain_centroid") and meas.certain_centroid is not None else meas.uncertain_centroid
                    left_angle = meas.angle
            else:
                left_bank = meas.certain_centroid if hasattr(meas, "certain_centroid") and meas.certain_centroid is not None else meas.uncertain_centroid
                left_angle = meas.angle
            break

    right_bank = None
    right_angle = None
    n = len(sorted_measurements)
    found_underwater = False
    underwater_angle = None
    underwater_centroid = None

    for idx in reversed(range(n)):
        meas = sorted_measurements[idx]
        if not found_underwater:
            if is_underwater(meas):
                found_underwater = True
                underwater_angle = meas.angle
                underwater_centroid = meas.certain_centroid if hasattr(meas, "certain_centroid") and meas.certain_centroid is not None else meas.uncertain_centroid
    
    for idx in reversed(range(n)):
        meas = sorted_measurements[idx]
        if underwater_angle is not None and meas.angle > underwater_angle:
            candidate_centroid = None
            if hasattr(meas, "certain_centroid") and meas.certain_centroid is not None:
                candidate_centroid = meas.certain_centroid
            elif hasattr(meas, "uncertain_centroid") and meas.uncertain_centroid is not None:
                candidate_centroid = meas.uncertain_centroid
            closest_candidate = None
            closest_angle = None
            min_dist = float('inf')
            for idx2 in reversed(range(n)):
                meas2 = sorted_measurements[idx2]
                if meas2.angle > underwater_angle:
                    cand_centroid = None
                    if hasattr(meas2, "certain_centroid") and meas2.certain_centroid is not None:
                        cand_centroid = meas2.certain_centroid
                    elif hasattr(meas2, "uncertain_centroid") and meas2.uncertain_centroid is not None:
                        cand_centroid = meas2.uncertain_centroid
                    if cand_centroid and underwater_centroid and abs(cand_centroid.x) > abs(underwater_centroid.x):
                        dist = abs(cand_centroid.x - underwater_centroid.x)
                        if dist < min_dist:
                            min_dist = dist
                            closest_candidate = cand_centroid
                            closest_angle = meas2.angle
            if closest_candidate:
                right_bank = closest_candidate
                right_angle = closest_angle

    sweep.left_bank = left_bank
    sweep.right_bank = right_bank

    for meas in sweep.measurements:
        if meas.uncertain_centroid is not None:
            sonar_dist = sweep.sonar_distances[0] if hasattr(sweep, "sonar_distances") and sweep.sonar_distances else 0
            uc = meas.uncertain_centroid
            if sweep.left_bank and sweep.right_bank:
                min_x = min(sweep.left_bank.x, sweep.right_bank.x)
                max_x = max(sweep.left_bank.x, sweep.right_bank.x)
                if uc.x < min_x or uc.x > max_x:
                    meas.uncertain_centroid = None
                    continue
            if abs(uc.y) < sonar_dist:
                meas.uncertain_centroid = None
        
        if meas.certain_centroid is not None:
            sonar_dist = sweep.sonar_distances[0] if hasattr(sweep, "sonar_distances") and sweep.sonar_distances else 0
            cc = meas.certain_centroid
            if sweep.left_bank and sweep.right_bank:
                min_x = min(sweep.left_bank.x, sweep.right_bank.x)
                max_x = max(sweep.left_bank.x, sweep.right_bank.x)
                if cc.x > min_x and cc.x < max_x and abs(cc.y) < sonar_dist:
                    meas.certain_centroid = None
                    continue

    # Rechoosing uncertain centroids if they are too close to certain centroids
    for meas in sweep.measurements:
        if meas.uncertain_centroid is not None and meas.certain_centroid is not None:
            uc = meas.uncertain_centroid
            cc = meas.certain_centroid
            if abs(uc.y - cc.y) < 100:
                if meas.inliers:
                    farthest_inlier = max(
                        (pt for pt in meas.inliers if abs(pt.y) > sweep.sonar_distances[0] and pt.x > min(sweep.left_bank.x, sweep.right_bank.x) and pt.x < max(sweep.left_bank.x, sweep.right_bank.x)),
                        key=lambda pt: abs(pt.y),
                        default=None
                    )
                    if abs(farthest_inlier.y - cc.y) > 100:
                        meas.uncertain_centroid = farthest_inlier
                    else:
                        farthest_raw = max(
                            (pt for pt in meas.raw_points if abs(pt.y) > sweep.sonar_distances[0] and pt.x > min(sweep.left_bank.x, sweep.right_bank.x) and pt.x < max(sweep.left_bank.x, sweep.right_bank.x)),
                            key=lambda pt: abs(pt.y),
                            default=None
                        )
                        meas.uncertain_centroid = farthest_raw if farthest_raw else None

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
                raw_x.append(point.x)
                raw_y.append(-point.y)
            for inlier in getattr(measurement, "inliers", []):
                inlier_x.append(inlier.x)
                inlier_y.append(-inlier.y)
            for outlier in getattr(measurement, "outliers", []):
                outlier_x.append(outlier.x)
                outlier_y.append(-outlier.y)
            if hasattr(measurement, "certain_centroid") and measurement.certain_centroid is not None:
                centroid_x.append(measurement.certain_centroid.x)
                centroid_y.append(-measurement.certain_centroid.y)
            if hasattr(measurement, "uncertain_centroid") and measurement.uncertain_centroid is not None:
                uncertain_centroid_x.append(measurement.uncertain_centroid.x)
                uncertain_centroid_y.append(-measurement.uncertain_centroid.y)

    left_bank = getattr(sweep, "left_bank", None)
    right_bank = getattr(sweep, "right_bank", None)

    plt.figure(figsize=(8, 8))

    if left_bank is not None:
        plt.axvline(x=left_bank.x, color='purple', linestyle='-.', linewidth=2, label="Left Bank")
    if right_bank is not None:
        plt.axvline(x=right_bank.x, color='brown', linestyle='-.', linewidth=2, label="Right Bank")

    sonar_distances = [d for sweep in sweeps if hasattr(sweep, "sonar_distances") for d in sweep.sonar_distances]
    if sonar_distances:
        min_sonar = min(sonar_distances)
        max_sonar = max(sonar_distances)
        plt.axhspan(max_sonar, min_sonar, color='red', alpha=0.15, label="Sonar Distance Range")

    if raw_x and raw_y:
        plt.scatter(raw_x, raw_y, s=8, color='black', alpha=0.2, label="Raw Points")

    if inlier_x and inlier_y:
        pass
        # plt.scatter(inlier_x, inlier_y, s=8, color='blue', alpha=0.5, label="Inliers", zorder=5)

    if outlier_x and outlier_y:
        pass
        # plt.scatter(outlier_x, outlier_y, s=8, color='red', alpha=0.5, label="Outliers", zorder=5)

    if centroid_x and centroid_y:
        filtered_centroids = []
        for sweep in sweeps:
            for measurement in sweep.measurements:
                if (
                    hasattr(measurement, "certain_centroid")
                    and measurement.certain_centroid is not None
                ):
                    filtered_centroids.append(measurement.certain_centroid)
        if filtered_centroids:
            fx = [pt.x for pt in filtered_centroids]
            fy = [-pt.y for pt in filtered_centroids]
            plt.scatter(fx, fy, s=80, color='green', marker='X', label="Certain Centroids", zorder=90, edgecolors='black', linewidths=1.2)
    if uncertain_centroid_x and uncertain_centroid_y:
        plt.scatter(uncertain_centroid_x, uncertain_centroid_y, s=80, color='orange', marker='P', label="Uncertain Centroids", zorder=6, edgecolors='black', linewidths=1.2)

    # Add angle labels for each certain centroid
    for sweep in sweeps:
        for measurement in sweep.measurements:
            if centroid_x and centroid_y:
                angle_centroid_pairs = []
                for sweep in sweeps:
                    for measurement in sweep.measurements:
                        if (
                            hasattr(measurement, "certain_centroid")
                            and measurement.certain_centroid is not None
                        ):
                            angle_centroid_pairs.append((measurement.angle, measurement.certain_centroid))
                angle_centroid_pairs.sort(key=lambda x: x[0])
                if angle_centroid_pairs:
                    cx = [c.x for _, c in angle_centroid_pairs]
                    cy = [-c.y for _, c in angle_centroid_pairs]
                    handles, labels = plt.gca().get_legend_handles_labels()
                    if "Certain Centroids Contour" not in labels:
                        plt.plot(cx, cy, color='lime', linewidth=2, label="Certain Centroids Contour")
                    else:
                        plt.plot(cx, cy, color='lime', linewidth=2)

    if uncertain_centroid_x and uncertain_centroid_y:
        angle_centroid_pairs = []
        for sweep in sweeps:
            for measurement in sweep.measurements:
                if hasattr(measurement, "uncertain_centroid") and measurement.uncertain_centroid is not None:
                    angle_centroid_pairs.append((measurement.angle, measurement.uncertain_centroid))
        angle_centroid_pairs.sort(key=lambda x: x[0])
        if angle_centroid_pairs:
            cx = [c.x for _, c in angle_centroid_pairs]
            cy = [-c.y for _, c in angle_centroid_pairs]

            plt.plot(cx, cy, color='orange', linewidth=2, label="Uncertain Centroids Contour")
            

    # Determine date and time range from sweeps
    timestamps = getattr(sweep, "timestamps", [])
    if timestamps:
        timestamps = sorted(timestamps)
        date_str = timestamps[0].strftime("%Y-%m-%d")
        start_time = timestamps[0].strftime("%H:%M")
        end_time = timestamps[-1].strftime("%H:%M")
        plt.title(f"StreamScope Cross-Section ({date_str}, {start_time}–{end_time} UTC)")
    else:
        plt.title("StreamScope Cross-Section")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.xlim(-2800, 2800)
    plt.ylim(5600, 0)
    plt.grid(True)
    plt.tight_layout()
    plt.gca().set_aspect('equal', adjustable='box')

    # Show the main legend first
    main_legend = plt.legend(loc='lower left')
    plt.gca().add_artist(main_legend)

    # Error, Valid, and Timeout Inset Plot
    # Count valid, error, and timeout points
    total_points = 0
    valid_points = 0
    error_points = 0
    timeout_points = 0
    for sweep in sweeps:
        for measurement in sweep.measurements:
            for d in getattr(measurement, "distances", []):
                total_points += 1
                if d == 9999:
                    timeout_points += 1
                elif d > 9990:
                    error_points += 1
                else:
                    valid_points += 1

    if total_points > 0:
        valid_pct = valid_points / total_points * 100
        error_pct = error_points / total_points * 100
        timeout_pct = timeout_points / total_points * 100
    else:
        valid_pct = error_pct = timeout_pct = 0

    # Inset axes for pie chart
    ax = plt.gca()
    axins = inset_axes(ax, width="18%", height="18%", loc='lower center', borderpad=2)
    sizes = [valid_pct, error_pct, timeout_pct]
    counts = [valid_points, error_points, timeout_points]
    labels = ['Valid', 'Error', 'Timeout']
    colors = ['#4CAF50', '#FF9800', '#F44336']

    def make_autopct(sizes, counts):
        def my_autopct(pct):
            total = sum(counts)
            count = int(round(pct * total / 100.0))
            idx = int(round(pct * len(counts) / 100.0))
            my_autopct.counter += 1
            i = my_autopct.counter - 1
            if i < len(counts):
                return f"{pct:.1f}%\n({counts[i]})"
            else:
                return f"{pct:.1f}%"
        my_autopct.counter = 0
        return my_autopct

    wedges, texts, autotexts = axins.pie(
        sizes, labels=labels, colors=colors,
        autopct=make_autopct(sizes, counts), startangle=90, textprops={'fontsize': 9}
    )
    axins.set_title("Point Type %", fontsize=10)

    # Pie chart for uncertain centroid types (lower right)
    # Classify each uncertain centroid as 'Centroid', 'Inlier', or 'Raw Point'
    uc_types = {'Centroid': 0, 'Inlier': 0, 'Raw Point': 0}
    total_uc = 0

    for sweep in sweeps:
        for measurement in sweep.measurements:
            uc = getattr(measurement, "uncertain_centroid", None)
            if uc is not None:
                total_uc += 1
                # Check if uncertain centroid matches a certain centroid
                if hasattr(measurement, "certain_centroid") and measurement.certain_centroid is not None and uc == measurement.certain_centroid:
                    uc_types['Centroid'] += 1
                # Check if uncertain centroid is in inliers
                elif hasattr(measurement, "inliers") and any(uc == inl for inl in measurement.inliers):
                    uc_types['Inlier'] += 1
                # Check if uncertain centroid is in raw_points
                elif hasattr(measurement, "raw_points") and any(uc == pt for pt in measurement.raw_points):
                    uc_types['Raw Point'] += 1
                else:
                    uc_types['Raw Point'] += 1  # fallback

    # Filter out types with 0 count
    uc_labels = []
    uc_sizes = []
    uc_counts = []
    uc_colors = []
    color_map = {'Centroid': '#8BC34A', 'Inlier': '#2196F3', 'Raw Point': '#9E9E9E'}
    label_map = {'Centroid': 'Centroid', 'Inlier': 'Inlier', 'Raw Point': 'Raw'}
    for k in ['Centroid', 'Inlier', 'Raw Point']:
        if uc_types[k] > 0:
            uc_labels.append(label_map[k])
            uc_counts.append(uc_types[k])
            uc_sizes.append(uc_types[k] / total_uc * 100 if total_uc > 0 else 0)
            uc_colors.append(color_map[k])

    if uc_sizes:
        axins2 = inset_axes(ax, width="18%", height="18%", loc='lower right', borderpad=2)

        def make_uc_autopct(sizes, counts):
            def my_autopct(pct):
                total = sum(counts)
                i = my_autopct.counter
                my_autopct.counter += 1
                if i < len(counts):
                    return f"{pct:.1f}%\n({counts[i]})"
                else:
                    return f"{pct:.1f}%"
            my_autopct.counter = 0
            return my_autopct

        wedges2, texts2, autotexts2 = axins2.pie(
            uc_sizes, labels=uc_labels, colors=uc_colors,
            autopct=make_uc_autopct(uc_sizes, uc_counts), startangle=90, textprops={'fontsize': 9}
        )
        axins2.set_title("Uncertain Centroid Type %", fontsize=10)

    plt.show()

def execute(folder):
    """
    Main data processing and visualization pipeline.
    Reads log files, processes sweeps, computes coordinates, removes outliers,
    finds centroids and banks, and visualizes the results.
    """

    file_pattern = f"/home/braden/StreamScopeDeploy/{folder}/streamscope_log_*.txt"
    file_list = glob.glob(file_pattern)
    print(f"Found {len(file_list)} files matching pattern: {file_pattern}")
    all_sweeps = [] 

    num_distances = 0

    for file_name in file_list:
        try:
            with open(file_name, 'r') as file:
                print(f"Processing file: {file_name}")
                lines = file.readlines()
                print("Processing file: ", file_name)
                for i, line in enumerate(lines):
                    
                    line = line.strip()
                    
                    if line.startswith("Date:") and "Time:" in lines[i+1]:
                        date_str = line.split("Date:")[-1].strip()
                        time_str = lines[i+1].split("Time:")[-1].strip()
                        timestamp = f"{date_str} {time_str}"
                        timestamp = datetime.datetime.strptime(timestamp, "%d/%m/%Y %H:%M UTC")
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
                                meas.timestamp = timestamp
                                meas.sonar_distance = sweep.sonar_distance
                                meas.angle = angle_val
                                distances_line = lines[j + 1]
                                distances = list(map(int, distances_line.split(":")[-1].strip().split(", ")))
                                meas.distances = distances
                                num_distances += len(distances)
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
    all_timestamps = []
    for sweep in all_sweeps:
        if hasattr(sweep, "sonar_distance"):
            unique_sonar_distances.add(sweep.sonar_distance)
        if hasattr(sweep, "sonar_distances"):
            unique_sonar_distances.update(sweep.sonar_distances)
        # Collect all timestamps from measurements, avoiding duplicates
        for meas in getattr(sweep, "measurements", []):
            if hasattr(meas, "timestamp") and meas.timestamp is not None:
                if meas.timestamp not in all_timestamps:
                    all_timestamps.append(meas.timestamp)
    unique_sonar_distances = {d for d in unique_sonar_distances if d}

    combined_sweep = LidarSweep()
    combined_sweep.sonar_distances = list(unique_sonar_distances)
    combined_sweep.timestamps = all_timestamps
    for angle, measurements in combined_by_angle.items():
        combined_meas = LidarMeasurement()
        combined_meas.angle = angle

        for m in measurements:
            if hasattr(m, "distances") and m.distances:
                combined_meas.distances.extend(m.distances)
            if hasattr(m, "raw_points") and m.raw_points:
                combined_meas.raw_points.extend(m.raw_points)
            if hasattr(m, "in_water"):
                if m.in_water is True:
                    combined_meas.in_water = True

        combined_sweep.measurements.append(combined_meas)

    remove_outliers(combined_sweep, eps=10, min_samples=5)
    process_centroids(combined_sweep)
    find_banks(combined_sweep)

    def print_combined_sweep_summary(sweep):
        print("=== Combined Sweep Summary ===")
        timestamps = [getattr(sweep, 'timestamps', None)]
        if timestamps:
            timestamps = sorted(timestamps)
            print(f"Timestamps: {timestamps[0]} to {timestamps[-1]}")
        else:
            print("Timestamps: None")
        print(f"Accelerometer Available: {getattr(sweep, 'accelerometer_available', None)}")
        print(f"Sonar Distances: {getattr(sweep, 'sonar_distances', None)} (length: {len(getattr(sweep, 'sonar_distances', []))})")
        print(f"Area: {getattr(sweep, 'area', None)}")
        print(f"Stream Width: {getattr(sweep, 'stream_width', None)}")
        print(f"Left Bank: {getattr(sweep, 'left_bank', None)}")
        print(f"Right Bank: {getattr(sweep, 'right_bank', None)}")
        print(f"Number of Measurements: {len(getattr(sweep, 'measurements', []))}")
        print()

        total_distances = 0
        total_valid = 0
        total_errors = 0
        total_timeouts = 0
        total_certain_centroids = 0
        total_uncertain_centroids = 0
        total_raw_points = 0

        for meas in getattr(sweep, 'measurements', []):
            if hasattr(meas, "distances") and isinstance(meas.distances, list):
                distances = meas.distances
                total_distances += len(distances)
                total_errors += sum(1 for d in distances if d > 9990 and d != 9999)
                total_timeouts += sum(1 for d in distances if d == 9999)
                total_valid += sum(1 for d in distances if d < 9990)
            if hasattr(meas, "raw_points") and isinstance(meas.raw_points, list):
                total_raw_points += len(meas.raw_points)
            if hasattr(meas, "certain_centroid") and meas.certain_centroid is not None:
                total_certain_centroids += 1
            if hasattr(meas, "uncertain_centroid") and meas.uncertain_centroid is not None:
                total_uncertain_centroids += 1

        print(f"Total distances: {total_distances}")
        print(f"  Valid: {total_valid}")
        print(f"  Errors: {total_errors}")
        print(f"  Timeouts: {total_timeouts}")
        print(f"Total certain centroids: {total_certain_centroids}")
        print(f"Total uncertain centroids: {total_uncertain_centroids}")
        print(f"Total raw points: {total_raw_points}")
        print("=== End of Summary ===\n")

    print_combined_sweep_summary(combined_sweep)
    graph([combined_sweep])

if __name__ == "__main__":
    
    while True:
        print("StreamScope Cross-Section.\n")
        print("1. Cross-Section View.\n")
        print("2. Exit.\n")
        choice = input("Select an option: ") 
        
        if choice == '1':
            date_input = input("Enter date (MMDDYYYY): ").strip()
            base_dir = "/home/braden/StreamScopeDeploy/"
            folder_pattern = os.path.join(base_dir, f"{date_input}")
            if not os.path.isdir(folder_pattern):
                print(f"Folder for date {date_input} not found at {folder_pattern}. Please try again.\n")
                continue
            else:
                execute(date_input)
        elif choice == '2':
            print("Exiting the program.")
            break
