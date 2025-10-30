"""
CARLA Actor Spawner
Handles spawning of vehicles and pedestrians for testing
"""

import carla
import random
import time
import math


class CarlaSpawner:
    """Manages spawning and cleanup of actors in CARLA"""
    
    def __init__(self, world):
        """
        Initialize spawner with existing CARLA world
        
        Args:
            world: carla.World instance (NOT client)
        """
        self.world = world
        self.spawned_actors = []
    
    def spawn_traffic_obstacles(self, num_vehicles=10, num_pedestrians=0, num_static=3):
        """
        Spawn traffic obstacles for testing
        
        Args:
            num_vehicles: Number of moving vehicles
            num_pedestrians: Number of pedestrians
            num_static: Number of static obstacles in lane
        """
        print("\n🚗 Auto-spawning traffic obstacles...")
        start_time = time.time()
        
        try:
            bp = self.world.get_blueprint_library()
            spawn_points = self.world.get_map().get_spawn_points()
            
            # Spawn moving vehicles
            vehicles_spawned = self._spawn_moving_vehicles(bp, spawn_points, num_vehicles)
            
            # Spawn static obstacles
            static_spawned = self._spawn_static_obstacles(bp, spawn_points, num_static)
            
            # Spawn pedestrians
            pedestrians_spawned = self._spawn_pedestrians(bp, spawn_points, num_pedestrians)
            
            print(f"   ✓ Total obstacles: {vehicles_spawned + static_spawned + pedestrians_spawned}")
            
            # Wait for actors to settle
            if len(self.spawned_actors) > 0:
                print("   ⏳ Waiting for actors to settle...")
                for i in range(3):
                    self.world.tick()
                    time.sleep(0.1)
                print("   ✓ Ready!")
            
        except Exception as e:
            print(f"   ⚠️ Error spawning obstacles: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"   ⏱️ Spawning took {time.time() - start_time:.1f}s")
    
    def _spawn_moving_vehicles(self, bp, spawn_points, num_vehicles):
        """Spawn moving vehicles on roads"""
        vehicle_bps = bp.filter('vehicle.*')
        vehicles_spawned = 0
        
        available_points = spawn_points[1:min(num_vehicles * 2 + 1, len(spawn_points))]
        random.shuffle(available_points)
        
        for spawn_point in available_points[:num_vehicles]:
            try:
                vehicle_bp = random.choice(vehicle_bps)
                if vehicle_bp.has_attribute('color'):
                    color = random.choice(vehicle_bp.get_attribute('color').recommended_values)
                    vehicle_bp.set_attribute('color', color)
                
                vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
                if vehicle is not None:
                    self.spawned_actors.append(vehicle)
                    vehicles_spawned += 1
                    vehicle.set_autopilot(True)
            except:
                pass
        
        print(f"   ✓ Spawned {vehicles_spawned} vehicles (all on autopilot)")
        return vehicles_spawned
    
    def _spawn_static_obstacles(self, bp, spawn_points, num_static):
        """Spawn static obstacles in ego lane"""
        if num_static == 0:
            return 0
        
        print(f"   🚧 Spawning {num_static} static obstacles in driving lane...")
        
        vehicle_bps = bp.filter('vehicle.*')
        tesla_bps = [b for b in vehicle_bps if 'tesla' in b.id.lower()] or vehicle_bps
        
        ego_spawn = spawn_points[0]
        carla_map = self.world.get_map()
        ego_waypoint = carla_map.get_waypoint(ego_spawn.location)
        
        distances = [50, 80, 120]
        static_spawned = 0
        
        for i, distance in enumerate(distances[:num_static]):
            try:
                # Walk forward using waypoints
                current_waypoint = ego_waypoint
                remaining_distance = distance
                
                while remaining_distance > 0:
                    step = min(2.0, remaining_distance)
                    next_waypoints = current_waypoint.next(step)
                    if len(next_waypoints) > 0:
                        current_waypoint = next_waypoints[0]
                        remaining_distance -= step
                    else:
                        break
                
                spawn_transform = current_waypoint.transform
                spawn_transform.location.z += 0.5
                
                vehicle_bp = random.choice(tesla_bps)
                if vehicle_bp.has_attribute('color'):
                    colors = ['255,0,0', '255,255,0', '0,255,255']
                    if i < len(colors):
                        vehicle_bp.set_attribute('color', colors[i])
                
                vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_transform)
                if vehicle is not None:
                    self.spawned_actors.append(vehicle)
                    static_spawned += 1
                    print(f"      → Static obstacle {static_spawned} at ~{distance}m ahead")
                    
                    vehicle.apply_control(carla.VehicleControl(
                        throttle=0.0, brake=1.0, hand_brake=True
                    ))
            except Exception as e:
                print(f"      ⚠️ Failed at {distance}m: {e}")
        
        print(f"   ✓ Spawned {static_spawned} static obstacles")
        return static_spawned
    
    def _spawn_pedestrians(self, bp, spawn_points, num_pedestrians):
        """Spawn pedestrians"""
        if num_pedestrians == 0:
            return 0
        
        print(f"   🚶 Spawning {num_pedestrians} pedestrians...")
        walker_bps = bp.filter('walker.pedestrian.*')
        ref_loc = spawn_points[0].location
        
        pedestrians_spawned = 0
        
        for i in range(num_pedestrians * 3):
            if pedestrians_spawned >= num_pedestrians:
                break
            
            try:
                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(10, 50)
                
                spawn_loc = carla.Location(
                    x=ref_loc.x + distance * math.cos(angle),
                    y=ref_loc.y + distance * math.sin(angle),
                    z=ref_loc.z + 1.0
                )
                
                spawn_transform = carla.Transform(
                    spawn_loc, carla.Rotation(yaw=random.uniform(0, 360))
                )
                
                walker_bp = random.choice(walker_bps)
                walker = self.world.try_spawn_actor(walker_bp, spawn_transform)
                
                if walker is not None:
                    self.spawned_actors.append(walker)
                    
                    walker_controller_bp = bp.find('controller.ai.walker')
                    controller = self.world.try_spawn_actor(
                        walker_controller_bp, carla.Transform(), walker
                    )
                    
                    if controller is not None:
                        self.spawned_actors.append(controller)
                        pedestrians_spawned += 1
                        controller.start()
                        controller.set_max_speed(1.0 + random.random() * 0.5)
            except:
                pass
        
        print(f"   ✓ Spawned {pedestrians_spawned} pedestrians")
        return pedestrians_spawned
    
    def cleanup(self):
        """Destroy all spawned actors - safe version"""
        if len(self.spawned_actors) > 0:
            print(f"\n🧹 Cleaning up {len(self.spawned_actors)} spawned actors...")
            destroyed_count = 0
            already_destroyed_count = 0
            error_count = 0
            
            for actor in self.spawned_actors:
                try:
                    # Check if actor still exists in world
                    if hasattr(actor, 'is_alive') and actor.is_alive:
                        actor.destroy()
                        destroyed_count += 1
                    else:
                        already_destroyed_count += 1
                except RuntimeError as e:
                    # Actor already destroyed
                    if "destroyed actor" in str(e).lower():
                        already_destroyed_count += 1
                    else:
                        error_count += 1
                except Exception as e:
                    # Any other error - just count and continue
                    error_count += 1
            
            self.spawned_actors.clear()
            print(f"   ✓ Destroyed: {destroyed_count}, Already gone: {already_destroyed_count}, Errors: {error_count}")