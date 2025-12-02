# r512-model-math
This is a school project to model a motorcycle around a track

the language used is python

### How to launch:
```bash
python ShowTrack.py --mode 3d --animate --speed-options 25 35 55
```

the ShowTrack accepts multiple parameters

* --csv (default -> file.csv) Path to CSV containing s_m,x_m,y_m,theta_rad,w_m columns
* --mode (choices -> 2d 3d) (default -> 3d) Choose between 2D or 3D visualization.
* --no-normalize Keep absolute coordinates instead of shifting origin to the first point.
* --animate Animate a rider marker along the track (3D mode only).
* --speed (default -> 35.0) Playback speed for the rider animation (m/s).
* --frame-step (default -> 5) Sample every Nth point when building animation frames.
* --thickness (default -> 1.5) Track extrusion thickness for 3D mesh (meters).
* --speed-options List of speeds (m/s) exposed on the UI slider.