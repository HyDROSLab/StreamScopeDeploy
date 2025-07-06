from pymodbus.client import ModbusSerialClient as ModbusClient
import pysftp
import time
import sys
import glob
import csv
import os
import logging
from datetime import datetime, timezone
import socket

# Add project folder to path
sys.path.insert(0, '/home/braden/StreamScopeDeploy')

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
SWEEP_TYPE_REG = 6  # General Measurement = 1, Specific Measurement = 2, Configured Measurement = 3, 4 = Full Sweep
NUM_MEASUREMENTS_REG = 7
NUM_ANGLES_REG = 8
NUM_SWEEPS_REG = 9
MEASUREMENTS_READY_REG = 10
SPECIFIC_ANGLES_START_REG = 11
STAGE_MEASUREMENT_REG = 100
SERVO_CONTROL_TYPE_REG = 101
ACCELEROMETER_AVAILABLE_REG = 102
STATE_REG = 103
RESET_REG = 104
DEBUG_REG = 105
TEMPERATURE_REG = 106
CONTROL_TYPE_TWO_REG = 107  # 1 = Continuous, 2 = Single
CONTROL_TYPE_FIVE_REG = 108  # 1 = Single/Slow, 2 = Single/Fast, 3 = Single/Auto, 4 = Continuous/Slow, 5 = Continuous/Fast, 6 = Continuous/Auto
LASER_TIMEOUT_REG = 109
RAW_RESULTS_START_REG = 110
EEPROM_INITIALIZE_REG = 4000
EEPROM_NUM_ANGLES_REG = 4001
EEPROM_NUM_MEASUREMENTS_REG = 4002
EEPROM_ANGLES_START_REG = 4003

# Write to SWEEP_TYPE_REG with value 4. Don't write to NUM_ANGLES_REG, NUM_MEASUREMENTS_REG, NUM_SWEEPS_REG.
# StreamScope will use it's internal values for these parameters. The variables below are for reading back data.
# If full sweep needs to be changed, it must be done in the firmware.
full_sweep = False

# Winter precip experiment
angles = [angle + 32768 for angle in [-20, 0, 20]]

# Inverse cosine experiment angles
# angles = [angle + 32768 for angle in [-38, -34, -30, -26, -22, -18, -14, -10, -6, -2, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38]]
# angles = [angle + 32768 for angle in [-40, -36, -32, -28, -24, -20, -16, -12, -8, -4, 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40]]

temperature = 9999

if full_sweep:
    num_angles = 101
    num_measurements = 30
else:
    num_angles = len(angles)
    num_measurements = 30

num_sweeps = 1

# static_ip = '13.83.7.88'

log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, 'deploy.log')
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_lock(process_name):
    global lock_socket   # Without this our lock gets garbage collected
    lock_socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        lock_socket.bind('\0' + process_name)
    except socket.error:
        return False
    return True


if not get_lock("StreamScopeDeploy"):
    logging.error("Could not aquire lock; seems like prior process is still running?")
    sys.exit()

client = ModbusClient(port='/dev/ttyUSB0', baudrate=19200, timeout=1)
client.connect()

def write_results(sweep):

    current_time = datetime.now(timezone.utc)
    out_folder = "/home/braden/StreamScopeDeploy/data"
    file_name = f"{out_folder}/streamscope_log_{current_time.strftime('%Y%m%d_%H%M%S')}.txt"

    if not os.path.isfile(file_name):
        with open(file_name, 'a') as newfile:
            sdBuffer = ""
            sdBuffer += "----------------------------------------\n"
            sdBuffer += "       StreamScope Sweep Log File       \n"
            sdBuffer += "----------------------------------------\n\n"
            sdBuffer += f"Date: {current_time.strftime('%d/%m/%Y')}\n"
            sdBuffer += f"Time: {current_time.strftime('%H:%M')} UTC\n\n"
            sdBuffer += "----------------------------------------\n"
            sdBuffer += "            System Details              \n"
            sdBuffer += "----------------------------------------\n\n"
            sdBuffer += f"Internal Temperature (C): {temperature}\n\n"
            sdBuffer += "----------------------------------------\n"
            sdBuffer += "            Sweep Details               \n"
            sdBuffer += "----------------------------------------\n\n"
            sdBuffer += f"Number of Angles: {num_angles}\n"
            sdBuffer += f"Number of Measurements at Each Angle: {num_measurements}\n"
            sdBuffer += "Requested Angles (degrees): "
            if full_sweep:
                sdBuffer += "Full Sweep (default angles)\n"
            else:
                for i in range(num_angles):
                    sdBuffer += str(angles[i] - 32768)
                    if i != num_angles - 1:
                        sdBuffer += ", "
            sdBuffer += "\n\n"
            sdBuffer += "----------------------------------------\n"
            sdBuffer += "         General Sweep Results          \n"
            sdBuffer += "----------------------------------------\n\n"
            sdBuffer += f"Sonar Distance (mm): {sweep.sonar_distance}\n"
            sdBuffer += "Recorded Angles (degrees): "
            for i, measurement in enumerate(sweep.measurements):
                sdBuffer += str(measurement.angle)
                if i != len(sweep.measurements) - 1:
                    sdBuffer += ", "
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
    '''
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
    '''
def write_sweep_parameters():
    try:
        if client.read_holding_registers(DEBUG_REG, 1).registers[0] == 0:
            logging.info("Running in non-debug mode. Soft-resetting Teensy.")
            client.write_register(RESET_REG, 1)
        else:
            logging.info("Running in DEBUG mode.")
        time.sleep(3)
        if full_sweep:
            logging.info("Writing full sweep.")
            client.write_register(CONTROL_TYPE_FIVE_REG, 5);
            client.write_register(LASER_TIMEOUT_REG, 1000)
            client.write_register(SERVO_CONTROL_TYPE_REG, 2)
            client.write_register(SWEEP_TYPE_REG, 4)
        else:
            logging.info("Writing sweep parameters.")
            client.write_register(CONTROL_TYPE_FIVE_REG, 5) # 4 = Continuous/Auto
            client.write_register(LASER_TIMEOUT_REG, 1000)
            client.write_register(NUM_ANGLES_REG, num_angles)
            client.write_register(NUM_MEASUREMENTS_REG, num_measurements)
            client.write_register(NUM_SWEEPS_REG, num_sweeps)
            client.write_registers(SPECIFIC_ANGLES_START_REG, angles)
            client.write_register(SERVO_CONTROL_TYPE_REG, 2)
            client.write_register(SWEEP_TYPE_REG, 2)
    except Exception as e:
        logging.error(f"Error writing sweep parameters: {e}")

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
                temperature = client.read_holding_registers(TEMPERATURE_REG, 1).registers[0] - 32768
                sweep.sonar_distance = client.read_holding_registers(STAGE_MEASUREMENT_REG, 1).registers[0] - 32768
                sweep.accelerometerAvailable = client.read_holding_registers(ACCELEROMETER_AVAILABLE_REG, 1).registers[0]
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
