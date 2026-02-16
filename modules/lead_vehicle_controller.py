"""
Lead Vehicle Controller
Spawns and controls a vehicle ahead of ego for testing obstacle detection
"""

import carla
import time
import random
import math


class LeadVehicleController:
    """Controls a dynamic lead vehicle for testing"""
    
    def __init__(self, world: carla.World, ego_vehicle: carla.Vehicle):
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.lead_vehicle = None
        self.enabled = False
        
        # Control parameters
        self.spawn_distance = 30.0  # meters ahead
        self.min_distance = 15.0    # Don't get closer
        self.max_distance = 60.0    # Don't get farther
        
        # Speed parameters (km/h)
        self.base_speed = 20.0
        self.speed_variation_range = 15.0
        self.current_target_speed = self.base_speed
        self.speed_change_interval = 5.0
        self.last_speed_change = time.time()
        
        self.behavior_mode = 'random'  # 'random', 'oscillate', 'brake_test'
        self.oscillate_phase = 0.0
    
    def spawn_lead_vehicle(self):
        """Spawn a lead vehicle ahead of ego"""
        try:
            ego_transform = self.ego_vehicle.get_transform()
            ego_location = ego_transform.location
            ego_forward = ego_transform.get_forward_vector()
            
            spawn_location = carla.Location(
                x=ego_location.x + ego_forward.x * self.spawn_distance,
                y=ego_location.y + ego_forward.y * self.spawn_distance,
                z=ego_location.z + 1.0
            )
            
            carla_map = self.world.get_map()
            spawn_waypoint = carla_map.get_waypoint(spawn_location)
            spawn_transform = spawn_waypoint.transform
            spawn_transform.location.z += 0.5
            
            bp_library = self.world.get_blueprint_library()
            vehicle_bp = bp_library.filter('vehicle.tesla.model3')[0]
            if vehicle_bp.has_attribute('color'):
                vehicle_bp.set_attribute('color', '255,0,0')  # Red
            
            self.lead_vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_transform)
            
            if self.lead_vehicle is not None:
                self.enabled = True
                print(f"✓ Lead vehicle spawned {self.spawn_distance}m ahead (RED)")
                time.sleep(0.1)
                self.world.tick()
                return True
            else:
                print("⚠️ Failed to spawn lead vehicle")
                return False
                
        except Exception as e:
            print(f"❌ Error spawning lead vehicle: {e}")
            return False
    
    def update(self):
        """Update lead vehicle - call every frame"""
        if not self.enabled or self.lead_vehicle is None:
            return
        
        try:
            if not self.lead_vehicle.is_alive:
                print("⚠️ Lead vehicle destroyed")
                self.enabled = False
                return
            
            ego_loc = self.ego_vehicle.get_transform().location
            lead_loc = self.lead_vehicle.get_transform().location
            distance = ego_loc.distance(lead_loc)
            
            current_time = time.time()
            if current_time - self.last_speed_change > self.speed_change_interval:
                self._update_target_speed()
                self.last_speed_change = current_time
            
            adjusted_speed = self._adjust_speed_for_distance(distance)
            control = self._calculate_control(adjusted_speed)
            self.lead_vehicle.apply_control(control)
            
        except Exception as e:
            print(f"⚠️ Lead vehicle update error: {e}")
    
    def _update_target_speed(self):
        """Update target speed based on behavior mode"""
        if self.behavior_mode == 'random':
            self.current_target_speed = self.base_speed + random.uniform(
                -self.speed_variation_range, self.speed_variation_range
            )
            self.current_target_speed = max(5.0, min(40.0, self.current_target_speed))
            
        elif self.behavior_mode == 'oscillate':
            self.oscillate_phase += 0.3
            variation = math.sin(self.oscillate_phase) * self.speed_variation_range
            self.current_target_speed = self.base_speed + variation
            
        elif self.behavior_mode == 'brake_test':
            if self.current_target_speed > self.base_speed:
                self.current_target_speed = self.base_speed - 10.0
            else:
                self.current_target_speed = self.base_speed + 15.0
    
    def _adjust_speed_for_distance(self, distance):
        """Adjust speed based on distance"""
        if distance < self.min_distance:
            return self.current_target_speed + 10.0
        elif distance > self.max_distance:
            return max(5.0, self.current_target_speed - 10.0)
        else:
            return self.current_target_speed
    
    def _calculate_control(self, target_speed_kmh):
        """Calculate vehicle control"""
        velocity = self.lead_vehicle.get_velocity()
        current_speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        speed_error = target_speed_kmh - current_speed
        control = carla.VehicleControl()
        
        if speed_error > 2.0:
            control.throttle = min(0.7, speed_error * 0.05)
            control.brake = 0.0
        elif speed_error < -2.0:
            control.throttle = 0.0
            control.brake = min(1.0, -speed_error * 0.08)
        else:
            control.throttle = 0.3
            control.brake = 0.0
        
        # Waypoint following
        try:
            lead_transform = self.lead_vehicle.get_transform()
            carla_map = self.world.get_map()
            current_waypoint = carla_map.get_waypoint(lead_transform.location)
            next_waypoints = current_waypoint.next(5.0)
            
            if len(next_waypoints) > 0:
                target_waypoint = next_waypoints[0]
                control.steer = self._calculate_steering(lead_transform, target_waypoint.transform.location)
        except:
            control.steer = 0.0
        
        return control
    
    def _calculate_steering(self, vehicle_transform, target_location):
        """Calculate steering angle"""
        vehicle_loc = vehicle_transform.location
        dx = target_location.x - vehicle_loc.x
        dy = target_location.y - vehicle_loc.y
        
        forward = vehicle_transform.get_forward_vector()
        angle_to_target = math.atan2(dy, dx)
        vehicle_angle = math.atan2(forward.y, forward.x)
        
        angle_diff = angle_to_target - vehicle_angle
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        steer = angle_diff * 0.5
        return max(-1.0, min(1.0, steer))
    
    def get_status(self):
        """Get status for display"""
        if not self.enabled or self.lead_vehicle is None:
            return None
        
        try:
            ego_loc = self.ego_vehicle.get_transform().location
            lead_loc = self.lead_vehicle.get_transform().location
            distance = ego_loc.distance(lead_loc)
            
            velocity = self.lead_vehicle.get_velocity()
            speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            
            return {
                'distance': distance,
                'speed': speed,
                'target_speed': self.current_target_speed,
                'mode': self.behavior_mode
            }
        except:
            return None
    
    def destroy(self):
        """Clean up lead vehicle"""
        if self.lead_vehicle is not None:
            try:
                control = carla.VehicleControl()
                control.throttle = 0.0
                control.brake = 1.0
                self.lead_vehicle.apply_control(control)
                self.world.tick()
                time.sleep(0.05)
                self.lead_vehicle.destroy()
                print("✓ Lead vehicle destroyed")
            except Exception as e:
                print(f"⚠️ Error destroying lead vehicle: {e}")
            finally:
                self.lead_vehicle = None
                self.enabled = False
