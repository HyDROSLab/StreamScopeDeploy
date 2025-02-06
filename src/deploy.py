from pymodbus.client import ModbusSerialClient as ModbusClient
import pysftp
import time
import sys
import glob
import csv
import os
import logging
from datetime import datetime

# Add project folder to path
sys.path.insert(0,'../')

class LidarMeasurement:
    def __init__(self):
        self.angle = 0.0
        self.distances = []
        self.coordinates = []
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

UPDATE_TIME_REG = 0
CURR_HOUR_REG = 1
CURR_MIN_REG = 2
CURR_DAY_REG = 3
CURR_MONTH_REG = 4
CURR_YEAR_REG = 5
SWEEP_TYPE_REG = 6
NUM_MEASUREMENTS_REG = 7
NUM_ANGLES_REG = 8
NUM_SWEEPS_REG = 9
MEASUREMENTS_READY_REG = 10
SPECIFIC_ANGLES_START_REG = 11
STAGE_MEASUREMENT_REG = 100
AREA_MEASUREMENT_REG = 101
STREAM_WIDTH_MEASUREMENT_REG = 102
CENTROIDS_START_REG = 103
LEFT_BANK_START_REG = 200
RIGHT_BANK_START_REG = 202
RAW_RESULTS_START_REG = 300
EEPROM_INITIALIZE_REG = 2000
EEPROM_NUM_ANGLES_REG = 2001
EEPROM_NUM_MEASUREMENTS_REG = 2002
EEPROM_ANGLES_START_REG = 2003
ACCELEROMETER_AVAILABLE_REG = 2100
STATE_REG = 2101
RESET_REG = 2102

angles = [angle + 32768 for angle in [-40, -30, -20, -10, 0, 10, 20, 30, 40]]
num_angles = 9
num_measurements = 30
num_sweeps = 1

static_ip = '13.83.7.88'

log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, 'deploy.log')
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

client = ModbusClient(port='/dev/ttyUSB0', baudrate=19200, timeout=1)
client.connect()

def write_results(sweep):

    current_time = datetime.now()
    out_folder = "./data"
    file_name = f"{out_folder}/streamscope_log_{current_time.strftime('%Y%m%d_%H%M%S')}.txt"

    if not os.path.isfile(file_name):
        with open(file_name, 'a') as newfile:
            sdBuffer = ""
            sdBuffer += "----------------------------------------\n"
            sdBuffer += "       StreamScope Sweep Log File       \n"
            sdBuffer += "----------------------------------------\n\n"
            sdBuffer += f"Date: {current_time.strftime('%d/%m/%Y')}\n"
            sdBuffer += f"Time: {current_time.strftime('%H:%M')}\n\n"
            sdBuffer += "----------------------------------------\n"
            sdBuffer += "            System Details              \n"
            sdBuffer += "----------------------------------------\n\n"
            sdBuffer += f"Accelerometer Available: {'Yes' if sweep.accelerometerAvailable == 1 else 'No'}\n\n"
            sdBuffer += "----------------------------------------\n"
            sdBuffer += "            Sweep Details               \n"
            sdBuffer += "----------------------------------------\n\n"
            sdBuffer += f"Number of Angles: {num_angles}\n"
            sdBuffer += f"Number of Measurements at Each Angle: {num_measurements}\n"
            sdBuffer += "Requested Angles (degrees): "
            for i in range(num_angles):
                sdBuffer += str(angles[i] - 32768)
                if i != num_angles - 1:
                    sdBuffer += ", "
            sdBuffer += "\n\n"
            sdBuffer += "----------------------------------------\n"
            sdBuffer += "         General Sweep Results          \n"
            sdBuffer += "----------------------------------------\n\n"
            sdBuffer += f"Sonar Distance (mm): {sweep.sonar_distance}\n"
            sdBuffer += f"Left Bank Coordinates (mm): {sweep.left_bank}\n"
            sdBuffer += f"Right Bank Coordinates (mm): {sweep.right_bank}\n"
            sdBuffer += f"Stream Width (mm): {sweep.streamWidth}\n"
            sdBuffer += f"Area (mm^2): {sweep.area}\n"
            sdBuffer += "Recorded Angles (degrees): "
            for i, measurement in enumerate(sweep.measurements):
                sdBuffer += str(measurement.angle)
                if i != len(sweep.measurements) - 1:
                    sdBuffer += ", "
            sdBuffer += "\n"
            sdBuffer += "Centroid Coordinates (corresponding to each recorded angle) (mm): \n"
            for measurement in sweep.measurements:
                for centroid in measurement.centroids:
                    sdBuffer += f"(X: {centroid[0]}, Y: {centroid[1]})\n"
            sdBuffer += "\n"
            sdBuffer += "----------------------------------------\n"
            sdBuffer += "            Raw Measurements            \n"
            sdBuffer += "----------------------------------------\n\n"
            for measurement in sweep.measurements:
                sdBuffer += f"Angle (degrees): {measurement.angle}\n"
                sdBuffer += "Distances (mm): "
                for j, distance in enumerate(measurement.distances):
                    sdBuffer += str(distance)
                    if j != len(measurement.distances) - 1:
                        sdBuffer += ", "
                sdBuffer += "\n\n"
            newfile.write(sdBuffer)
    
    if not os.path.isfile(file_name):
        with open(file_name, 'rb') as f:
            files = {'file': (file_name, f, 'text/plain')}
            logging.info("Results written to file successfully.")

    # Push output file to server via SFTP
    srv = pysftp.Connection(host=static_ip, username="pinpoint", password="Pinpoint2025#")
    try:
        with srv.cd("data/fallscreek"):
            srv.put(file_name)
        srv.close()
        logging.info("Results written to file and sent to server successfully.")
    except e:
        logging.error(e)
        logging.error("Results written to file but failed to send to server.")

def write_sweep_parameters():
    client.write_register(NUM_ANGLES_REG, num_angles)
    client.write_register(NUM_MEASUREMENTS_REG, num_measurements)
    client.write_register(NUM_SWEEPS_REG, num_sweeps)
    client.write_registers(SPECIFIC_ANGLES_START_REG, angles)
    client.write_register(SWEEP_TYPE_REG, 2)

def deploy():

    logging.info("Writing sweep parameters.")
    write_sweep_parameters()

    while True:

        try:
            if client.read_holding_registers(MEASUREMENTS_READY_REG, 1).registers[0] == 0:
                logging.info("Measurements are not ready yet.")
                time.sleep(15)
            elif client.read_holding_registers(MEASUREMENTS_READY_REG, 1).registers[0] == 1:
                logging.info("Measurements ready. Reading in measurements.")
                sweep = LidarSweep()
                sweep.sonar_distance = client.read_holding_registers(STAGE_MEASUREMENT_REG, 1).registers[0] - 32768
                sweep.accelerometerAvailable = client.read_holding_registers(ACCELEROMETER_AVAILABLE_REG, 1).registers[0]
                sweep.left_bank = (client.read_holding_registers(LEFT_BANK_START_REG, 1).registers[0] - 32768, client.read_holding_registers(LEFT_BANK_START_REG + 1, 1).registers[0] - 32768)
                sweep.right_bank = (client.read_holding_registers(RIGHT_BANK_START_REG, 1).registers[0] - 32768, client.read_holding_registers(RIGHT_BANK_START_REG + 1, 1).registers[0] - 32768)
                sweep.streamWidth = client.read_holding_registers(STREAM_WIDTH_MEASUREMENT_REG, 1).registers[0] - 32768
                sweep.area = client.read_holding_registers(AREA_MEASUREMENT_REG, 1).registers[0] - 32768   
                logging.info("Reading in raw measurements.")
                raw_index = RAW_RESULTS_START_REG
                for i in range(num_angles):
                    measurement = LidarMeasurement()
                    measurement.angle = client.read_holding_registers(raw_index, 1).registers[0] - 32768
                    raw_index += 1
                    for j in range(num_measurements):
                        measurement.distances.append(client.read_holding_registers(raw_index, 1).registers[0] - 32768)
                        raw_index += 1
                    sweep.measurements.append(measurement)
                centroids_index = CENTROIDS_START_REG
                for i in range(num_angles):
                    sweep.measurements[i].centroids.append((client.read_holding_registers(centroids_index, 1).registers[0] - 32768, client.read_holding_registers(centroids_index + 1, 1).registers[0] - 32768))
                    centroids_index += 2
                client.write_register(MEASUREMENTS_READY_REG, 0)
                logging.info("Writing results to file.")
                write_results(sweep)
                break
            else:
                logging.warning("Measurement register in an undefined state.")
        except Exception as e:
            logging.error(e)
            time.sleep(10)


def main():
    deploy()

if __name__ == "__main__":
    try:
        main()
    finally:
        client.close()
