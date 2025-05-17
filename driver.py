import os
import joblib
import numpy as np
import torch
import torch.nn.functional as F
import msgParser
import carState
import carControl
import time  # Added for obstacle avoidance timing
from model import ActionClassifier, GearNet, SteeringNet

TRACK = 'track'

class Driver(object):
    '''
    A driver object for the Simulated Car Racing Championship (SCRC).
    Uses a trained PyTorch model to select controls based on sensor inputs.
    '''

    def __init__(self, stage):
        # Stages
        self.WARM_UP    = 0
        self.QUALIFYING = 1
    
        RACE       = 2
        self.UNKNOWN    = 3
        self.stage      = stage

        # Interfaces
        self.parser  = msgParser.MsgParser()
        self.state   = carState.CarState()
        self.control = carControl.CarControl()

        # Paths
        base_dir    = os.path.dirname(__file__)
        scaler_path = os.path.join(base_dir, f'torcs_input_scalers.pkl')
        model_action_path = os.path.join(base_dir, f'{TRACK}_model_action.pkl')
        model_gear_path = os.path.join(base_dir, f'{TRACK}_model_gear.pkl')
        model_steer_path = os.path.join(base_dir, f'{TRACK}_model_steer.pkl')

        # Load scaler
        self.scalers = joblib.load(scaler_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Feature definitions from model.py
        self.accelbrake_features = [
            'speedX', 'speedY', 'speedZ', 'angle', 
            'wheelSpinVel_0', 'wheelSpinVel_1', 'wheelSpinVel_2', 'wheelSpinVel_3',
        ] + [f'track_{i}' for i in range(19)]

        self.gear_features = [
         'speedX', 'rpm', 'gear', 'angle',
        ] + [f'track_{i}' for i in range(19)] 

        self.steer_features = [
            'angle', 'speedX', 'speedY', 'speedZ', 'wheelSpinVel_0', 'wheelSpinVel_1', 'wheelSpinVel_2', 'wheelSpinVel_3', 
        ] + [f'track_{i}' for i in range(19)] 

        # Instantiate models
        self.model_action = ActionClassifier(len(self.accelbrake_features)).to(self.device)
        self.model_gear = GearNet(len(self.gear_features)).to(self.device)
        self.model_steer = SteeringNet(len(self.steer_features)).to(self.device)

        # Load model weights
        self.model_action.load_state_dict(torch.load(model_action_path, map_location=self.device))
        self.model_gear.load_state_dict(torch.load(model_gear_path, map_location=self.device))
        self.model_steer.load_state_dict(torch.load(model_steer_path, map_location=self.device))
        
        # Set models to evaluation mode
        self.model_action.eval()
        self.model_gear.eval()
        self.model_steer.eval()
        
        # Initialize advanced stuck detection system variables
        self.position_history = []        # Track recent positions
        self.speed_history = []           # Track recent speeds
        self.angle_history = []           # Track recent steering angles
        self.history_max_len = 20         # Number of position/speed samples to keep
        self.is_stuck = False             # Current stuck state
        self.stuck_time = 0.0             # How long we've been stuck
        self.recovery_mode = False        # If we're in recovery mode
        self.recovery_start_time = 0.0    # When recovery started
        self.recovery_timeout = 5.0       # Maximum recovery time (seconds)
        self.recovery_phase = 0           # Phases of recovery (0=reverse, 1=turn, 2=forward)
        self.last_check_time = time.time()# Last time we checked stuck status
        self.movement_threshold = 3.0     # Speed below which we consider "not moving"
        self.position_change_threshold = 0.1  # Position change below which we consider "not moving"
        self.track_center_recovery = False  # Whether we're trying to recover to track center
        
        # For backward compatibility (will be deprecated after validation)
        self.stuck_detection = {
            'last_trackPos': 0,
            'last_speed': 0,
            'stuck_time': 0,
            'last_check': time.time(),
            'was_stuck': False
        }

        # Maximum speed boost
        self.max_speed = 500  # Increased from 360

    def init(self):
        """Return init string with custom rangefinder angles"""
        angles = [0]*19
        for i in range(5):
            angles[i]    = -90 + i*15
            angles[18-i] =  90 - i*15
        for i in range(5,9):
            angles[i]    = -20 + (i-5)*5
            angles[18-i] =  20 - (i-5)*5
        return self.parser.stringify({'init': angles})

    def _get_scalar(self, recv, key):
        v = recv.get(key)
        return float(v[0]) if isinstance(v, list) else float(v)

    def _get_vector(self, recv, key, length):
        v = recv.get(key)
        if isinstance(v, list):
            return [float(x) for x in v]
        return [float(recv[f'{key}_{i}']) for i in range(length)]
    
    def _make_feature_vector(self, recv):
        fv = {}
        
        scalar_keys = ['speedX', 'speedY', 'speedZ', 'angle', 'gear', 'rpm', 'distFromStart', 'trackPos']
        for key in scalar_keys:
            if key in recv:
                fv[key] = self._get_scalar(recv, key)
            else:
                # Default value if key not found
                fv[key] = 0.0
                print(f"Warning: Key '{key}' not found in data, using default 0.0")
        
        # Wheel spin velocities
        for i in range(4):
            recv_key = f'wheelSpinVel'
            if recv_key in recv and isinstance(recv[recv_key], list) and len(recv[recv_key]) > i:
                fv[f'wheelSpinVel_{i}'] = float(recv[recv_key][i])
            else:
                fv[f'wheelSpinVel_{i}'] = 0.0
                print(f"Warning: 'wheelSpinVel[{i}]' not found in data, using default 0.0")
            
        # Track edge sensors
        if 'track' in recv and isinstance(recv['track'], list):
            track_data = recv['track']
            for i in range(18):
                if i < len(track_data):
                    fv[f'track_{i}'] = float(track_data[i])
                else:
                    fv[f'track_{i}'] = 0.0
        
        # Track edge sensors
        if 'track' in recv and isinstance(recv['track'], list):
            track_data = recv['track']
            for i in range(19):
                if i < len(track_data):
                    fv[f'track_{i}'] = float(track_data[i])
                else:
                    fv[f'track_{i}'] = 0.0
                    print(f"Warning: 'track[{i}]' not found in data, using default 0.0")
        else:
            for i in range(19):
                fv[f'track_{i}'] = 0.0
                print(f"Warning: 'track' array not found in data, using default 0.0 for track_{i}")
        
        # Opponent sensors - now used in the steering network
        if 'opponents' in recv and isinstance(recv['opponents'], list):
            opponents_data = recv['opponents']
            for i in range(36):
                if i < len(opponents_data):
                    fv[f'opponents_{i}'] = float(opponents_data[i])
                else:
                    fv[f'opponents_{i}'] = 200.0  # Default 200 means no opponent in that direction
                    print(f"Warning: 'opponents[{i}]' not found in data, using default 200.0")
        else:
            for i in range(36):
                fv[f'opponents_{i}'] = 200.0
                print(f"Warning: 'opponents' array not found in data, using default 200.0 for opponents_{i}")
            
        return fv

    def _extract_model_features(self, feature_dict, feature_list):
        """Extract features for a specific model from the feature dictionary"""
        result = []
        for f in feature_list:
            # Handle missing keys by defaulting to 0.0
            if f not in feature_dict:
                print(f"Warning: Feature '{f}' not found in data, using default value 0.0")
                result.append(0.0)
            else:
                result.append(feature_dict[f])
        return result
    def manualDrive(self, msg, dir):
        # REFRESH SET FROM THE DATA RECEIVED
        self.state.setFromMsg(msg)
        speed = self.state.getSpeedX()
        accel = self.control.getAccel()
        brake = self.control.getBrake()
        rpm = self.state.getRpm()
        gear = self.state.getGear()
        angle = self.state.angle
        dist = self.state.trackPos
        steer = self.control.getSteer()

        if "UP" in dir:
            if gear < 1:
                # If in reverse or neutral, shift up when UP is pressed
                gear += 1
            else:
                accel += 0.1
                if accel > 1:
                    accel = 1.0
                
                if speed >= self.max_speed:
                    accel = 0
        else:
            if "DOWN" not in dir:
                # Only decrease acceleration if not in reverse or not pressing DOWN
                if gear != -1:
                    accel -= 0.1
                    if accel < 0:
                        accel = 0

        
        if "DOWN" in dir:
            if gear == -1:
                # Already in reverse gear, accelerate backwards continuously
                brake = 0
                accel += 0.2
                if accel > 1:
                    accel = 1
            else:
                # Not in reverse yet
                if speed > 0:
                    # Still moving forward, need to brake
                    brake += 0.1
                    if brake > 1:
                        brake = 1
                else:
                    # Speed is close to 0, put directly into reverse and start accelerating
                    gear = -1
                    accel = 0.5  # Start with higher initial acceleration for better response
                    brake = 0     
        else:
            # If not pressing DOWN but already in reverse, maintain some acceleration
            if gear == -1 and accel > 0:
                # Keep a minimum reverse acceleration unless UP is pressed
                if "UP" not in dir and accel < 0.3:
                    accel = 0.3
            else:
                brake = 0


        if "LEFT" in dir:
            steer += 0.05
            if steer > 1:
                steer = 1
        elif "RIGHT" in dir:
            steer -= 0.05
            if steer < -1:
                steer = -1
        else:
            steer = 0
        
        # GEAR SYSTEM
        if gear != -1:  # Don't apply regular gear system logic to reverse gear
            if rpm > 6000:
                gear += 1
                if gear > 6:
                    gear = 6
            if rpm <= 3000 and gear >= 2:
                gear -= 1

            # Only set to neutral if not pressing any direction keys, the car is stopped, 
            # and not already in reverse
            if rpm == 0 and speed < 0.1 and "UP" not in dir and "DOWN" not in dir and gear != -1:
                gear = 0
        elif "UP" in dir and speed < 0.1:
            # Only allow shifting out of reverse with UP key when stopped
            gear = 0  # Shift to neutral first
    

        self.control.setSteer(steer)
        self.control.setGear(gear)
        self.control.setAccel(accel)
        self.control.setBrake(brake)
        return self.control.toMsg()

    def aiDrive(self, msg, _dir):
        # 1) parse sensors
        recv = self.parser.parse(msg)
        
        # 2) build full feature vector dictionary
        feature_dict = self._make_feature_vector(recv)
        
        # Check for WASD direct control first (bypassing AI)
        wasd_control = False
        if any(key in _dir for key in ["W", "A", "S", "D"]):
            wasd_control = True
            print("WASD DIRECT CONTROL ACTIVE")
            
        # Initialize values for WASD control
        accel = 0.0
        brake = 0.0
        steer_pred = 0.0
        speed = feature_dict['speedX']
        rpm = feature_dict['rpm']
        gear_pred = feature_dict['gear']
        
        # Handle WASD direct control mode
        if wasd_control:
            # W - Accelerate
            if "W" in _dir:
                accel = 1.0
                brake = 0.0
                # If in reverse, switch to neutral first when moving forward
                if gear_pred == -1 and speed < 0.1:
                    gear_pred = 1
            
            # S - Brake/Reverse
            if "S" in _dir:
                if gear_pred == -1:
                    # Already in reverse gear, accelerate backwards
                    accel = 1.0
                    brake = 0.0
                elif speed > 0.1:
                    # Moving forward, apply brake
                    accel = 0.0
                    brake = 1.0
                else:
                    # Almost stopped, shift to reverse
                    gear_pred = -1
                    accel = 0.5  # Start with decent reverse acceleration
                    brake = 0.0
            
            # A - Steer Left
            if "A" in _dir:
                steer_pred = 1.0  # Full left steering
            
            # D - Steer Right
            if "D" in _dir:
                steer_pred = -1.0  # Full right steering
                
            # GEAR SYSTEM for WASD control
            if gear_pred != -1:  # Only handle upshifts/downshifts when not in reverse
                if rpm > 7000 and gear_pred < 6:
                    gear_pred += 1
                elif rpm < 2500 and gear_pred > 1 and "W" not in _dir:
                    gear_pred -= 1
            
            # Reset to neutral if stopped and no W/S pressed
            if speed < 0.1 and "W" not in _dir and "S" not in _dir and gear_pred != -1:
                gear_pred = 0
                
            # For debugging
            print(f"[DIRECT CONTROL] WASD Mode - Gear: {gear_pred}, Accel: {accel}, Brake: {brake}, Steer: {steer_pred}")
            
        else:
            # REGULAR AI CONTROL FLOW
            # 3) extract model-specific features
            X_ab = np.array([self._extract_model_features(feature_dict, self.accelbrake_features)], dtype=np.float32)
            X_g = np.array([self._extract_model_features(feature_dict, self.gear_features)], dtype=np.float32)
            X_s = np.array([self._extract_model_features(feature_dict, self.steer_features)], dtype=np.float32)
            
            # 4) scale features
            X_ab = self.scalers['accelbrake'].transform(X_ab)
            X_g = self.scalers['gear'].transform(X_g)
            X_s = self.scalers['steer'].transform(X_s)
            
            # 5) convert to tensors
            x_ab = torch.from_numpy(X_ab).to(self.device)
            x_g = torch.from_numpy(X_g).to(self.device)
            x_s = torch.from_numpy(X_s).to(self.device)
            
            # 6) inference
            with torch.no_grad():
                # Action prediction (0: no-op, 1: accel, 2: brake)
                action_logits = self.model_action(x_ab)
                action_pred = torch.argmax(action_logits, dim=1).item()
                action_probs = F.softmax(action_logits, dim=1)[0].cpu().numpy()
                
             
                # Regular gear model
                gear_logits = self.model_gear(x_g)
                raw_gear_pred = torch.argmax(gear_logits, dim=1).item()
                gear_pred = raw_gear_pred - 1 
                

                
                steer_pred = self.model_steer(x_s).item()
                steer_pred = 2.0 * steer_pred - 1.0  # Convert from [0,1] to [-1,1]
            
            # 7) decode outputs based on action prediction:
            if action_pred == 2:  # brake
                accel = 0.0
                brake = 1.0
            elif action_pred == 1:  # accelerate
                accel = 1.0
                brake = 0.0
            else:  # no-op
                accel = 1
                brake = 0.0
      
                
            # 8) manual overrides from key presses
            angle = feature_dict['angle']
            track_pos = feature_dict.get('trackPos', 0)
            
            if "UP" in _dir:
                accel = 1.0
                brake = 0.0
            if "DOWN" in _dir:
                accel = 0.0
                brake = 1.0
                # If DOWN is pressed and speed is low, consider switching to reverse
                if speed < 1.0 and brake > 0.5:
                    print("⬇️ DOWN + low speed: Consider reverse gear")
   
                
          
                
            if "LEFT" in _dir:
                steer_pred = min(steer_pred + 0.3, 1.0)
            if "RIGHT" in _dir:
                steer_pred = max(steer_pred - 0.3, -1.0)
                
            # 9) performance enhancements
            
            if speed > 100 and brake > 0.0 and "DOWN" not in _dir:
                brake = max(0.0, brake * 0.5)  
            
            if action_probs[1] > 0.5:  # if probability of acceleration is high (>50%)
                accel = 1.0
                brake = 0.0
            
            if rpm > 7500 and gear_pred < 6:  # Upshift earlier for better acceleration
                gear_pred += 1
            elif rpm < 2500 and speed > 50 and gear_pred > 1:  # Stay in gear longer for better power
                pass  # Don't downshift as quickly
                
            # 9.4) Dynamic steering assistance - smoother turning at high speeds
            if abs(steer_pred) > 0.5 and speed > 120:
                # Reduce steering angle slightly at high speeds for stability
                steer_pred *= 0.8
                
            # 9.5) Track and opponent awareness - correct if going off track or approaching opponents
            track_readings = [feature_dict[f'track_{i}'] for i in range(19)]
            min_track_dist = min([x for x in track_readings if x < 200])
            
            # Get opponent sensor readings (if available)
            opponent_readings = []
            for i in range(36):
                key = f'opponents_{i}'
                if key in feature_dict:
                    opponent_readings.append(feature_dict[key])
                else:
                    opponent_readings.append(200.0)  # No opponent detected
            
            # If we're close to track edge and going fast, apply appropriate correction
            if min_track_dist < 2.0 and speed > 80:
                # Find which side we're going off
                left_track = track_readings[0:9]
                right_track = track_readings[10:19]
                
                if min(left_track) < min(right_track):
                    # We're going off to the left, steer right
                    steer_pred = max(steer_pred - 0.2, -1.0)
                    accel *= 0.8  # Reduce speed for the turn
                else:
                    # We're going off to the right, steer left
                    steer_pred = min(steer_pred + 0.2, 1.0)
                    accel *= 0.8  # Reduce speed for the turn
            
            # Check for opponents and adjust steering to avoid collisions
            if opponent_readings:
                # Get closest opponent distance and its direction
                # Front opponents (index 12-24)
                front_opponents = opponent_readings[12:24]
                min_front_distance = min([x for x in front_opponents if x < 30], default=200)
                
                if min_front_distance < 30:  # If opponent is within 30 distance units
                    # Determine which side has more space
                    left_opponents = opponent_readings[0:12]  # Left side opponents
                    right_opponents = opponent_readings[24:36]  # Right side opponents
                    
                    left_space = min([x for x in left_opponents if x < 200], default=200)
                    right_space = min([x for x in right_opponents if x < 200], default=200)
                    
                    # Steer toward the side with more space if opponent is directly ahead
                    if min_front_distance < 10 and speed > 50:
                        if left_space > right_space:
                            # More space on the left, steer left
                            steer_pred = min(steer_pred + 0.3, 1.0)
                            # Reduce speed slightly when avoiding
                            accel *= 0.9
                        else:
                            # More space on the right, steer right
                            steer_pred = max(steer_pred - 0.3, -1.0)
                            # Reduce speed slightly when avoiding
                            accel *= 0.9
                    
            # Detect if car is stuck using trackPos and apply reverse
            # Store last positions and speeds to detect if stuck
            if not hasattr(self, 'stuck_detection'):
                self.stuck_detection = {
                    'last_trackPos': track_pos,
                    'last_speed': speed,
                    'stuck_time': 0,
                    'last_check': time.time(),
                    'was_stuck': False
                }
            
            current_time = time.time()
            time_diff = current_time - self.stuck_detection['last_check']
            self.stuck_detection['last_check'] = current_time
            
            # Check if car is stuck - more aggressive detection criteria
            is_stuck = ((abs(speed) < 5.0 and abs(track_pos) > 0.85 and 
                        abs(self.stuck_detection['last_trackPos'] - track_pos) < 0.15) or 
                       (abs(track_pos) > 1.3))  # Immediately consider stuck if very far off track
                
            if is_stuck:
                self.stuck_detection['stuck_time'] += time_diff
            else:
                # Reset stuck time if car is moving properly
                self.stuck_detection['stuck_time'] = 0
                self.stuck_detection['was_stuck'] = False
            
            # If stuck for more than 1.5 seconds, try to reverse towards track center
            if self.stuck_detection['stuck_time'] > 1.5 and not self.stuck_detection['was_stuck']:
                print(f"🚨 Car is stuck at trackPos={track_pos}! Attempting to reverse towards track center...")
                gear_pred, accel, brake, steer_pred = self.reverse_from_stuck(track_pos, speed)
                self.stuck_detection['was_stuck'] = True
            
            # If we were stuck but now moving, reset stuck status
            if self.stuck_detection['was_stuck'] and abs(speed) > 5.0:
                self.stuck_detection['was_stuck'] = False
                self.stuck_detection['stuck_time'] = 0
                
                # Add a recovery sequence to return to forward driving towards track center
                if gear_pred == -1 and abs(track_pos) < 1.0:  # If we're in reverse but close enough to track
                    print("Car is now unstuck! Switching to forward gear and heading to track center...")
                    gear_pred = 1  # Switch to first gear
                    accel = 0.5    # Moderate acceleration
                    brake = 0.0    # No braking
                    
                    # Steer towards center of track based on current position
                    steer_pred = max(-1.0, min(1.0, -track_pos * 0.7))
                else:
                    print("Car is now unstuck and moving!")
            
            # Update values for next check
            self.stuck_detection['last_trackPos'] = track_pos
            self.stuck_detection['last_speed'] = speed
            
            # 10) clamp values to valid ranges
            gear_pred = max(-1, min(6, gear_pred)) 

            
            # Better handling of reverse gear
            if gear_pred == -1 and speed <= 0.1:
                # In reverse gear and stopped/almost stopped, apply acceleration
                accel = 1.0  # Full acceleration in reverse
                brake = 0.0  # No braking
                
                # If we're in reverse due to being stuck, steer towards center
                if self.stuck_detection['was_stuck']:
                    # Enhanced steering correction based on track position
                    target_angle = -track_pos * 0.9  # More aggressive correction
                    steer_pred = max(-1.0, min(1.0, target_angle))
            elif gear_pred == -1 and speed > 0:
                # In reverse gear but still moving forward, apply brakes first
                brake = 1.0
                accel = 0.0
                
            brake = max(0.0, min(1.0, brake))
            steer_pred = max(-1.0, min(1.0, steer_pred))
                
            # For debugging
            print(f"Model Output - Gear: {gear_pred}, Accel: {accel}, Brake: {brake}, Steer: {steer_pred}")
            
            # Only automatically set to gear 1 if we're not intentionally in reverse
            if speed < 30 and not self.stuck_detection.get('was_stuck', False) and gear_pred != -1:
                gear_pred = 1
            elif speed < 40 and gear_pred != 2 and gear_pred != -1: 
                gear_pred = 3
        # 11) apply controls (common for both WASD and AI control)
        self.control.setGear(gear_pred)
        self.control.setAccel(accel)
        self.control.setBrake(brake)
        self.control.setSteer(steer_pred)
        
        return self.control.toMsg()

    def reverse_from_stuck(self, track_pos, speed):
        """
        Special method to handle reversal when the car is stuck
        Returns gear, accel, brake, and steering values
        """
        # Always ensure we're in reverse gear
        gear = -1
        
        # Use full acceleration when reversing from stuck position
        accel = 1.0
        brake = 0.0
        
        # Determine steering direction to return to track center
        # Negative trackPos means we're on the right side of track, need to steer left (positive)
        # Positive trackPos means we're on the left side of track, need to steer right (negative)
        # The magnitude determines how aggressive the steering should be
        
        # For trackPos > 0 (left side), we want negative steering (right)
        # For trackPos < 0 (right side), we want positive steering (left)
        base_steer = -track_pos  # Basic steering is opposite of track position
        
        # Apply more aggressive steering based on how far off-track we are
        if abs(track_pos) > 1.5:  # Very far off track
            steer = base_steer * 1.2  # 120% steering
        elif abs(track_pos) > 1.0:
            steer = base_steer * 1.0  # 100% steering
        else:
            steer = base_steer * 0.8  # 80% steering
            
        # Ensure steering is within valid range
        steer = max(-1.0, min(1.0, steer))
        
        print(f"🔄 Reversing from stuck position: trackPos={track_pos}, steering={steer}")
        
        return gear, accel, brake, steer

    def check_if_stuck(self, track_pos, speed, angle):
        """
        Advanced stuck detection using historical position and speed data.
        Returns True if the car is stuck, False otherwise.
        """
        current_time = time.time()
        time_diff = current_time - self.last_check_time
        self.last_check_time = current_time
        
        # Add current position and speed to history
        self.position_history.append(track_pos)
        self.speed_history.append(speed)
        self.angle_history.append(angle)
        
        # Keep history within max length
        if len(self.position_history) > self.history_max_len:
            self.position_history.pop(0)
            self.speed_history.pop(0)
            self.angle_history.pop(0)
            
        # Can't detect until we have enough history
        if len(self.position_history) < 5:
            return False
            
        # Get the average speed and position change over the last few samples
        recent_speeds = self.speed_history[-5:]
        avg_speed = sum(abs(s) for s in recent_speeds) / len(recent_speeds)
        
        # Calculate position change variance - low variance means not moving much
        recent_positions = self.position_history[-5:]
        pos_variance = max(recent_positions) - min(recent_positions)
        
        # Multiple stuck detection conditions
        stuck_conditions = [
            # Condition 1: Low speed and far off track
            (avg_speed < self.movement_threshold and abs(track_pos) > 0.85),
            
            # Condition 2: Not moving much in terms of track position and not at center
            (pos_variance < self.position_change_threshold and abs(track_pos) > 0.7),
            
            # Condition 3: Very far off track regardless of speed (emergency)
            (abs(track_pos) > 1.3),
            
            # Condition 4: Car is moving very slowly and not changing position
            (avg_speed < 1.0 and pos_variance < 0.05)
        ]
        
        is_stuck_now = any(stuck_conditions)
        
        # Update stuck timer
        if is_stuck_now:
            self.stuck_time += time_diff
            print(f"🔍 Possible stuck condition - timer: {self.stuck_time:.1f}s, trackPos: {track_pos:.2f}, speed: {speed:.1f}")
        else:
            # Only reset if car is moving properly
            if avg_speed > self.movement_threshold:
                self.stuck_time = 0.0
                self.is_stuck = False
                
        # Only enter stuck state after a threshold period
        stuck_threshold = 1.0  # 1 second
        if self.stuck_time > stuck_threshold and not self.is_stuck:
            print(f"🚨 STUCK DETECTED! trackPos={track_pos:.2f}, avgSpeed={avg_speed:.1f}")
            self.is_stuck = True
            return True
            
        return self.is_stuck

    def drive(self, msg):
        """
        Main driving method that uses ML models for prediction
        This method will be called automatically by the client
        """
        # Get the car state
        self.state.setFromMsg(msg)
        speed = self.state.getSpeedX()
        track_pos = self.state.trackPos
        angle = self.state.angle
        
        # Get track sensors if available
        track_sensors = None
        if hasattr(self.state, 'track'):
            track_sensors = self.state.track
        
        # Check if we're currently in recovery mode
        if self.recovery_mode:
            print(f"🔄 In recovery mode (phase {self.recovery_phase})")
            gear, accel, brake, steer = self.recover_from_stuck(track_pos, speed, angle, track_sensors)
            
            # Apply controls directly
            self.control.setGear(gear)
            self.control.setAccel(accel)
            self.control.setBrake(brake)
            self.control.setSteer(steer)
            
            return self.control.toMsg()
        
        # Use advanced stuck detection even if not already stuck
        if self.check_if_stuck(track_pos, speed, angle):
            print(f"🚨 Car is stuck - initiating recovery! trackPos={track_pos:.2f}, speed={speed:.2f}")
            gear, accel, brake, steer = self.recover_from_stuck(track_pos, speed, angle, track_sensors)
            
            # Apply controls directly
            self.control.setGear(gear)
            self.control.setAccel(accel)
            self.control.setBrake(brake)
            self.control.setSteer(steer)
            
            return self.control.toMsg()
        
        # Emergency condition even without full stuck detection (immediate response)
        if abs(track_pos) > 1.5 and abs(speed) < 2.0:
            print("🚨 EMERGENCY: Car detected far off track! Activating immediate recovery.")
            self.is_stuck = True  # Force stuck state
            gear, accel, brake, steer = self.recover_from_stuck(track_pos, speed, angle, track_sensors)
            
            # Apply controls directly
            self.control.setGear(gear)
            self.control.setAccel(accel)
            self.control.setBrake(brake)
            self.control.setSteer(steer)
            
            return self.control.toMsg()
        
        # Normal case - use AI drive with empty direction list
        return self.aiDrive(msg, [])

    def recover_from_stuck(self, track_pos, speed, angle, track_sensors=None):
        """
        Advanced recovery system with multiple phases and track awareness.
        Returns tuple of (gear, accel, brake, steer) control values.
        """
        # Initialize recovery mode if not already
        if not self.recovery_mode:
            self.recovery_mode = True
            self.recovery_start_time = time.time()
            self.recovery_phase = 0
            print(f"🔄 Starting recovery sequence at trackPos={track_pos:.2f}")
            
        # Track time in recovery mode
        recovery_time = time.time() - self.recovery_start_time
        
        # If we've been in recovery too long, try changing phases
        if recovery_time > self.recovery_timeout:
            self.recovery_phase += 1
            if self.recovery_phase > 2:
                self.recovery_phase = 0  # Reset to first phase if we've tried everything
            self.recovery_start_time = time.time()  # Reset timer for new phase
            print(f"⏱️ Recovery phase {self.recovery_phase} timeout - switching phases")
            
        # Handle recovery based on current phase
        if self.recovery_phase == 0:
            # PHASE 0: Reverse away from current position
            gear = -1
            accel = 1.0
            brake = 0.0
            
            # Calculate steering based on track position
            target_angle = -track_pos * 1.1  # More aggressive than normal steering
            steer = max(-1.0, min(1.0, target_angle))
            
            # Use track sensors for improved decision making if available
            if track_sensors and len(track_sensors) >= 19:
                # Find which side has more space (larger sensor values)
                left_space = sum(track_sensors[0:9]) / 9.0
                right_space = sum(track_sensors[10:19]) / 9.0
                
                if abs(track_pos) > 1.2:
                    # Very far off track - steer more aggressively toward center
                    if track_pos > 0:  # If on left side of track
                        steer = -0.9  # Steer right
                    else:             # If on right side of track
                        steer = 0.9   # Steer left
                else:
                    # Use sensor data to find better path
                    if left_space > right_space and track_pos < 0:
                        # More space on left and we're on right - emphasize left steer
                        steer = min(steer * 1.2, 1.0)
                    elif right_space > left_space and track_pos > 0:
                        # More space on right and we're on left - emphasize right steer
                        steer = max(steer * 1.2, -1.0)
            
            print(f"🔄 Recovery phase 0: Reversing with steer={steer:.2f}")
            
            # Criteria to move to next phase: moved enough or timer expired
            if abs(speed) > 10 and ((track_pos > 0 and track_pos < 0.8) or 
                                  (track_pos < 0 and track_pos > -0.8)):
                self.recovery_phase = 1
                self.recovery_start_time = time.time()
                print(f"✓ Moved enough in reverse, switching to phase 1")
                
        elif self.recovery_phase == 1:
            # PHASE 1: Turn vehicle toward track center, still in reverse
            gear = -1
            
            # Calculate how much we need to turn to face track center
            # Use angle (vehicle heading) and track position to determine proper alignment
            heading_correction = angle * 0.5
            position_correction = -track_pos * 0.8
            
            # Combine corrections with more weight to position_correction
            steer = max(-1.0, min(1.0, position_correction + heading_correction))
            
            # Use moderate reverse acceleration while turning
            accel = 0.7
            brake = 0.0
            
            print(f"🔄 Recovery phase 1: Turning in reverse with steer={steer:.2f}")
            
            # Criteria to move to next phase: heading roughly toward center or timer expired
            # Simple check - if sign of angle and trackPos are opposite, we're roughly facing center
            if (angle * track_pos < 0) or (abs(track_pos) < 0.4):
                self.recovery_phase = 2
                self.recovery_start_time = time.time()
                print(f"✓ Now facing toward track center, switching to phase 2")
                
        else:  # Phase 2
            # PHASE 2: Switch to forward gear and accelerate toward track center
            gear = 1
            accel = 0.8
            brake = 0.0
            
            # Continue steering toward track center
            steer = max(-1.0, min(1.0, -track_pos * 0.8))
            
            # Adjust steering based on current angle to ensure we're heading toward center
            angle_correction = -angle * 0.3  # Counter current angle
            steer = max(-1.0, min(1.0, steer + angle_correction))
            
            print(f"🔄 Recovery phase 2: Forward with steer={steer:.2f}")
            
            # Criteria to exit recovery mode completely
            if abs(track_pos) < 0.3 and abs(speed) > 10:
                print(f"✅ Recovery successful! Back on track at trackPos={track_pos:.2f}")
                self.recovery_mode = False
                self.is_stuck = False
                self.stuck_time = 0
                
        return gear, accel, brake, steer

    def onShutDown(self): pass
    def onRestart(self): pass
