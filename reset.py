from pymodbus.client import ModbusSerialClient as ModbusClient

RESET_REG = 5102

def reset():
    client = ModbusClient(port='/dev/ttyUSB0', baudrate=9600, timeout=1)
    client.connect()
    print("Resetting...")
    client.write_register(RESET_REG, 1)
    client.close()
    print("Reset complete.")

if __name__ == "__main__":
    reset()
