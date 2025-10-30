"""
Main entry point for modular autonomous driving system
Clean architecture with separated modules
"""

import carla
import cv2
import time
import sys

from modules.driving_agent import DrivingAgent


class AutonomousDrivingSystem:
    """Main system coordinator"""
    
    def __init__(self):
        print("="*60)
        print("CARLA Autonomous Driving System - Modular Architecture")
        print("="*60)
        
        # Connect to CARLA
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world('Town04')
        
        # Spawn ego vehicle
        bp = self.world.get_blueprint_library()
        vehicle_bp = bp.filter('vehicle.tesla.model3')[0]
        spawn_points = self.world.get_map().get_spawn_points()
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_points[0])
        
        # Setup camera
        camera_bp = bp.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1280')
        camera_bp.set_attribute('image_size_y', '720')
        camera_bp.set_attribute('fov', '90')
        
        cam_transform = carla.Transform(
            carla.Location(x=2.0, z=1.4),
            carla.Rotation(pitch=-15)
        )
        self.camera = self.world.spawn_actor(camera_bp, cam_transform, attach_to=self.vehicle)
        
        self.camera_data = None
        self.camera.listen(lambda image: self._camera_callback(image))
        
        # Initialize driving agent
        self.agent = DrivingAgent(self.world, self.vehicle)
        
        print("✓ System initialized")
    
    def _camera_callback(self, image):
        """Camera callback"""
        import numpy as np
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.camera_data = array
    
    def run(self, duration=300, spawn_traffic=True):
        """Run autonomous driving"""
        print("\n" + "="*70)
        print("  CONTROLS:")
        print("  [L] = Autonomous Mode")
        print("  [M] = Manual Mode")
        print("  [V] = Toggle Lane Mask Visualization")  # NEW
        print("  [W/A/S/D] = Manual throttle/brake/steering")
        print("  [Space] = Brake")
        print("  [Q] = Quit")
        print("="*70)
        print("\n⚠️  Vehicle starts in MANUAL mode")
        print("⚠️  Press [L] to enable AUTONOMOUS driving")
        print("⚠️  Press [W] to start driving manually\n")
        
        # Wait for camera
        print("⏳ Waiting for camera...")
        wait_start = time.time()
        while self.camera_data is None:
            if time.time() - wait_start > 10:
                print("❌ Camera timeout")
                return
            time.sleep(0.1)
            self.world.tick()
        print("✓ Camera ready")
        
        # Spawn traffic
        if spawn_traffic:
            self.agent.spawn_traffic(num_vehicles=10, num_static=3)
        
        # Main loop
        start_time = time.time()
        frame_count = 0
        
        try:
            while time.time() - start_time < duration:
                if self.camera_data is None:
                    time.sleep(0.01)
                    continue
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                # Mode switching
                if key == ord('m'):
                    self.agent.set_mode('manual')
                elif key == ord('l'):
                    self.agent.set_mode('auto')
                    self.agent.handle_roi_choice_when_auto(self.camera_data)
                
                # NEW: Toggle lane mask visualization
                elif key == ord('v'):
                    self.agent.toggle_lane_mask_visualization()
                
                # ROI selection in auto mode
                if self.agent.awaiting_roi_choice:
                    self.agent.process_roi_choice_key(key)
                
                # Manual controls
                if self.agent.mode == 'manual':
                    self.agent.process_manual_keys(key)
                
                # Quit
                if key == ord('q'):
                    break
                
                # Process frame
                result = self.agent.process_frame(self.camera_data)
                
                # Apply control
                self.vehicle.apply_control(result['control'])
                
                # Visualize
                vis, _ = self.agent.visualize(self.camera_data, result)
                
                if vis is not None:
                    cv2.imshow('Autonomous Driving - Modular', vis)
                
                # Status
                if frame_count % 100 == 0:
                    print(f"[{frame_count}] {self.agent.mode.upper()}: {result['decision']}")
                
                frame_count += 1
        
        except KeyboardInterrupt:
            print("\n⏹ Interrupted")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources - safe order"""
        print("\n🧹 Cleaning up...")
        
        # 1. Close windows first
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        # 2. Stop vehicle
        try:
            if hasattr(self, 'vehicle') and self.vehicle is not None:
                control = carla.VehicleControl()
                control.throttle = 0.0
                control.brake = 1.0
                control.steer = 0.0
                self.vehicle.apply_control(control)
        except:
            pass
        
        # 3. Cleanup spawned actors (agent handles this)
        try:
            if hasattr(self, 'agent'):
                self.agent.cleanup()
        except Exception as e:
            print(f"   ⚠️ Error in agent cleanup: {e}")
        
        # 4. Destroy camera BEFORE vehicle
        try:
            if hasattr(self, 'camera') and self.camera is not None:
                self.camera.stop()  # Stop listening first
                time.sleep(0.1)  # Give it time
                self.camera.destroy()
        except Exception as e:
            print(f"   ⚠️ Camera cleanup error: {e}")
        
        # 5. Destroy vehicle last
        try:
            if hasattr(self, 'vehicle') and self.vehicle is not None:
                self.vehicle.destroy()
        except Exception as e:
            print(f"   ⚠️ Vehicle cleanup error: {e}")
        
        print("✓ Cleanup complete")


def main():
    """Main function"""
    try:
        system = AutonomousDrivingSystem()
        system.run(duration=300, spawn_traffic=True)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
