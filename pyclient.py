#!/usr/bin/env python3
import sys
import argparse
import socket
import driver  # Assuming driver.py is available
import keyboard
import msgParser
import csv
import os

if __name__ == '__main__':
    pass

strParser = msgParser.MsgParser()

# Configure argument parser
parser = argparse.ArgumentParser(description='Python client to connect to the TORCS SCRC server.')
parser.add_argument('--host', action='store', dest='host_ip', default='localhost',
                    help='Host IP address (default: localhost)')
parser.add_argument('--port', action='store', type=int, dest='host_port', default=3001,
                    help='Host port number (default: 3001)')
parser.add_argument('--id', action='store', dest='id', default='SCR',
                    help='Bot ID (default: SCR)')
parser.add_argument('--maxEpisodes', action='store', dest='max_episodes', type=int, default=1,
                    help='Maximum number of learning episodes (default: 1)')
parser.add_argument('--maxSteps', action='store', dest='max_steps', type=int, default=0,
                    help='Maximum number of steps (default: 0)')
parser.add_argument('--track', action='store', dest='track', default=None,
                    help='Name of the track')
parser.add_argument('--stage', action='store', dest='stage', type=int, default=3,
                    help='Stage (0 - Warm-Up, 1 - Qualifying, 2 - Race, 3 - Unknown)')

arguments = parser.parse_args()

# Print summary
print('Connecting to server host ip:', arguments.host_ip, '@ port:', arguments.host_port)
print('Bot ID:', arguments.id)
print('Maximum episodes:', arguments.max_episodes)
print('Maximum steps:', arguments.max_steps)
print('Track:', arguments.track)
print('Stage:', arguments.stage)
print('*********************************************')

try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
except socket.error as msg:
    print('Could not make a socket.')
    sys.exit(-1)

# One second timeout
sock.settimeout(1.0)

shutdownClient = False
curEpisode = 0

verbose = False

d = driver.Driver(arguments.stage)

current_direction = None
# def handle_keypress(event):
#     global current_direction
#     if event.name == "up":
#         current_direction = "UP"
#     elif event.name == "down":
#         current_direction = "DOWN"
#     elif event.name == "left":
#         current_direction = "LEFT"
#     elif event.name == "right":
#         current_direction = "RIGHT"
#     elif event.name == "q":
#         # print("Exiting...")
#         current_direction = None
#         exit()

# keyboard.on_press(handle_keypress)

def get_direction():
    """Detects multiple key presses and returns combined direction.
    Now supports both arrow keys and WASD keys where WASD bypasses AI control."""
    directions = []

    # Arrow keys 
    if keyboard.is_pressed("up"):
        directions.append("W")
    if keyboard.is_pressed("down"):
        directions.append("S")
    if keyboard.is_pressed("left"):
        directions.append("A")
    if keyboard.is_pressed("right"):
        directions.append("D")
    
    # WASD keys 
    if keyboard.is_pressed("w"):
        directions.append("UP")
    if keyboard.is_pressed("s"):
        directions.append("DOWN")
    if keyboard.is_pressed("a"):
        directions.append("LEFT")
    if keyboard.is_pressed("d"):
        directions.append("RIGHT")



    return directions


def make_feature_vector(receiving_buffer):
    fv = []

    # 1) car‐centric sensors
    fv.append(receiving_buffer['angle'])
    fv.append(receiving_buffer['gear'])

    # 2) 36 opponents sensors
    fv.extend(receiving_buffer['opponents'])

    # 3) scalar race / car state
    fv.append(receiving_buffer['racePos'])
    fv.append(receiving_buffer['rpm'])
    fv.append(receiving_buffer['speedX'])
    fv.append(receiving_buffer['speedY'])
    fv.append(receiving_buffer['speedZ'])

    # 4) 19 track‐border sensors
    fv.extend(receiving_buffer['track'])

    # 5) track position
    fv.append(receiving_buffer['trackPos'])

    # 6) 4 wheel spin velocities
    fv.extend(receiving_buffer['wheelSpinVel'])

    return fv    # length == 2 + 36 + 5 + 19 p+ 1 + 4 == 67



def append_data_to_csv(receiving_buffer, sending_buffer_str, csv_filename="driving_data.csv"):
    """
    Append receiving and sending buffer data to a CSV file.
    
    Args:
        receiving_buffer (dict): Dictionary containing car state data
        sending_buffer_str (str): String containing car control commands
        csv_filename (str): Name of the CSV file to append to
    """
    # Parse the sending buffer string to get a dictionary
    sending_buffer = strParser.parse(sending_buffer_str) if sending_buffer_str else {}
    
    # Create a combined dictionary with all data
    combined_data = {}
    
    # Add timestamp
    from datetime import datetime
    combined_data['timestamp'] = datetime.now().isoformat()
    
    # Process receiving buffer - flatten nested lists to single values
    for key, value in receiving_buffer.items():
        if isinstance(value, list):
            if len(value) == 1:  # Single value in list
                combined_data[f"recv_{key}"] = value[0]
            else:  # Multiple values in list
                for i, v in enumerate(value):
                    combined_data[f"recv_{key}_{i}"] = v
    
    if sending_buffer:
        for key, value in sending_buffer.items():
            if isinstance(value, list):
                if len(value) == 1:  # Single value in list
                    # Convert acceleration to binary (0 or 1) in CSV only
                    if key == "accel":
                        accel_value = float(value[0])
                        combined_data[f"send_{key}"] = 1 if accel_value > 0.2 else 0
                    else:
                        combined_data[f"send_{key}"] = value[0]
                else:  # Multiple values in list
                    for i, v in enumerate(value):
                        combined_data[f"send_{key}_{i}"] = v
            elif key == "accel":  # Handle non-list accel values too
                accel_value = float(value)
                combined_data[f"send_{key}"] = 1 if accel_value > 0.2 else 0
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(csv_filename)
    
    # Write to CSV file
    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=combined_data.keys())
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        # Write the data row
        writer.writerow(combined_data)



while not shutdownClient:
    receivingBuffer = {}
    sendingBuffer = None
    while True:
        print('Sending id to server:', arguments.id)
        buf = arguments.id + d.init()
        print('Sending init string to server:', buf)
        
        try:
            sock.sendto(buf.encode(), (arguments.host_ip, arguments.host_port))
        except socket.error as msg:
            print("Failed to send data...Exiting...")
            sys.exit(-1)
            
        try:
            buf, addr = sock.recvfrom(1000)
            buf = buf.decode()
        except socket.error as msg:
            print("Didn't get response from server...")
    
        if '***identified***' in buf:
            print('Received:', buf)
            break

    currentStep = 0
    
    while True:
        # Wait for an answer from server
        buf = None
        try:
            buf, addr = sock.recvfrom(1000)
            buf = buf.decode()
        except socket.error as msg:
            print("Didn't get response from server...")
        
        if verbose:
            print('Received:', buf)
        
        if buf:
            receivingBuffer = strParser.parse(buf)
            

        if buf and '***shutdown***' in buf:
            d.onShutDown()
            shutdownClient = True
            print('Client Shutdown')
            break
        
        if buf and '***restart***' in buf:
            d.onRestart()
            print('Client Restart')
            break
        
        currentStep += 1
        if currentStep != arguments.max_steps:
            if buf:
                current_direction = None
                current_direction = get_direction()
                buf = d.manualDrive(buf, current_direction)  # Use aiDrive with key direction detection
                # buf = d.drive(buf)  # This would use pure AI without any manual input

        else:
            buf = '(meta 1)'
        
        if verbose:
            print('Sending:', buf)
        
        if buf:
            sendingBuffer = buf
            # Append data to CSV
            append_data_to_csv(receivingBuffer, sendingBuffer, csv_filename=f"dirt.csv")
            try:
                sock.sendto(buf.encode(), (arguments.host_ip, arguments.host_port))
            except socket.error as msg:
                print("Failed to send data...Exiting...")
                sys.exit(-1)
    
    curEpisode += 1
    
    if curEpisode == arguments.max_episodes:
        shutdownClient = True

sock.close()
