# Obstacle Spawning Guide for CARLA

## Quick Start

### Method 1: Automatic Spawning (Recommended)

The main script now automatically spawns obstacles when it starts!

**Just run:**

```powershell
python ard_man_copy.py
```

**Configure in `ard_man_copy.py` (lines 32-35):**

```python
AUTO_SPAWN_OBSTACLES = True   # Enable/disable auto-spawn
NUM_SPAWN_VEHICLES   = 15     # Number of vehicles
NUM_SPAWN_PEDESTRIANS = 5     # Number of pedestrians
```

---

### Method 2: Manual Spawning Script

Use the dedicated spawning script for more control.

**Run in separate terminal:**

```powershell
python spawn_obstacles.py
```

**Interactive Menu:**

```
1. Quick Test (10 vehicles + 5 pedestrians)
2. Heavy Traffic (30 vehicles + 15 pedestrians)
3. Static Obstacles (cars in front for testing)
4. Custom (choose your own numbers)
5. Cleanup Only (remove all spawned actors)
```

**Then run main script:**

```powershell
python ard_man_copy.py
```

---

## Spawning Options

### Quick Test

- **Vehicles**: 10 on roads with autopilot
- **Pedestrians**: 5 walking around
- **Best for**: Quick functionality test

### Heavy Traffic

- **Vehicles**: 30 distributed across map
- **Pedestrians**: 15 in various locations
- **Best for**: Stress testing detection

### Static Obstacles

- **Vehicles**: 5-10 parked in driving lane
- **Pedestrians**: 2-3 crossing road
- **Best for**: Testing stop behavior

---

## Spawned Object Behavior

### Vehicles

- **70% with autopilot**: Drive around naturally
- **30% static**: Parked obstacles
- **Random colors**: Easy to distinguish
- **Various types**: Cars, trucks, buses, motorcycles

### Pedestrians

- **Walking**: Random navigation paths
- **Speed**: 1.0-2.5 m/s (natural walking)
- **Crossing**: May cross roads
- **Various types**: Different ages, genders, clothes

---

## Configuration Examples

### Minimal (Testing)

```python
AUTO_SPAWN_OBSTACLES = True
NUM_SPAWN_VEHICLES   = 5
NUM_SPAWN_PEDESTRIANS = 2
```

### Standard (Realistic)

```python
AUTO_SPAWN_OBSTACLES = True
NUM_SPAWN_VEHICLES   = 15
NUM_SPAWN_PEDESTRIANS = 5
```

### Heavy (Challenging)

```python
AUTO_SPAWN_OBSTACLES = True
NUM_SPAWN_VEHICLES   = 30
NUM_SPAWN_PEDESTRIANS = 10
```

### Disabled

```python
AUTO_SPAWN_OBSTACLES = False
# No obstacles spawned
```

---

## Cleanup

### Automatic

All spawned actors are automatically destroyed when you:

- Press **Q** to quit
- Press **Ctrl+C**
- Script exits normally

### Manual Cleanup

If actors remain after script crash:

```powershell
python spawn_obstacles.py
# Choose option 5: Cleanup Only
```

---

## Tips

### For Testing Detection

1. Use **Static Obstacles** mode
2. Spawn 5-10 cars in front
3. Watch YOLO detect and calculate distances
4. Vehicle should stop automatically

### For Realistic Simulation

1. Use **Quick Test** or **Heavy Traffic**
2. Enable autopilot (70% of vehicles)
3. Let pedestrians walk naturally
4. Test lane keeping + obstacle avoidance

### For Performance Testing

1. Start with few obstacles (5 vehicles, 2 pedestrians)
2. Gradually increase if FPS is good
3. Monitor FPS in console output
4. Too many objects → lower FPS

---

## Troubleshooting

### No obstacles appear

- Check CARLA server is running
- Verify `AUTO_SPAWN_OBSTACLES = True`
- Check console for spawn errors
- Try manual spawning script

### Too many obstacles

- Reduce `NUM_SPAWN_VEHICLES` and `NUM_SPAWN_PEDESTRIANS`
- Some spawn locations may overlap

### Obstacles don't move

- Vehicles: 70% have autopilot by default
- Pedestrians: Controllers may fail to spawn
- Check console for "spawned X controllers"

### Script crashes on cleanup

- Already cleaned up automatically
- Run `spawn_obstacles.py` → option 5

---

## Advanced: Custom Spawning

### Spawn at Specific Location

```python
# In spawn_obstacles.py
spawner.spawn_static_obstacles_ahead(
    ego_vehicle_location=carla.Location(x=100, y=50, z=0.5),
    num_obstacles=5,
    distance_range=(10, 30)  # 10-30 meters ahead
)
```

### Spawn in Circle Around Point

```python
spawner.spawn_vehicles_in_area(
    num_vehicles=10,
    spawn_point_index=0,  # Center spawn point
    radius=50.0  # 50 meter radius
)
```

---

## Expected Behavior

### What You Should See

1. Console: "✓ Spawned X vehicles"
2. Console: "✓ Spawned Y pedestrians"
3. YOLO detection: Green/red bounding boxes
4. Distance labels: "car 12.5m", "person 8.2m"
5. Automatic stopping when obstacle <15m in lane

### Detection Examples

```
Green box (Safe):
- car 25.3m    → Outside lane or far away
- person 18.7m → Safe distance

Red box (Dangerous):
- car 8.5m     → STOP! Too close in lane
- truck 12.1m  → STOP! Approaching threshold
```

---

## Quick Reference

| Action                      | Command                            |
| --------------------------- | ---------------------------------- |
| Auto-spawn with main script | `python ard_man_copy.py`           |
| Manual spawn script         | `python spawn_obstacles.py`        |
| Quick test                  | Option 1 in spawn script           |
| Heavy traffic               | Option 2 in spawn script           |
| Cleanup all                 | Option 5 in spawn script           |
| Disable spawning            | Set `AUTO_SPAWN_OBSTACLES = False` |

---

## Performance Impact

| Obstacles | FPS Impact              | Detection Quality |
| --------- | ----------------------- | ----------------- |
| 5-10      | Minimal (~2-5 FPS)      | Excellent         |
| 10-20     | Moderate (~5-10 FPS)    | Good              |
| 20-30     | Noticeable (~10-15 FPS) | Fair              |
| 30+       | Significant (~15+ FPS)  | May lag           |

**Recommended**: Start with 10 vehicles + 5 pedestrians for best balance.
