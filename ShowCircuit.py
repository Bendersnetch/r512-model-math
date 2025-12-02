import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
import plotly.graph_objects as go


@dataclass
class TrackPoint:
    s: float
    x: float
    y: float
    theta: float
    width: float


class Track:
    def __init__(self, points):
        self.points = points

    def _geometry(self, normalize_origin):
        s = np.array([p.s for p in self.points])
        x = np.array([p.x for p in self.points])
        y = np.array([p.y for p in self.points])
        theta = np.array([p.theta for p in self.points])
        width = np.array([p.width for p in self.points])

        if normalize_origin:
            x = x - x[0]
            y = y - y[0]

        normals = np.column_stack((-np.sin(theta), np.cos(theta)))
        half_width = (width / 2.0)[:, None]
        left_edge = np.column_stack((x, y)) + normals * half_width
        right_edge = np.column_stack((x, y)) - normals * half_width
        curvature = np.gradient(theta, s, edge_order=2)
        curvature_metric = np.abs(curvature)

        return dict(
            s=s,
            x=x,
            y=y,
            theta=theta,
            width=width,
            left_edge=left_edge,
            right_edge=right_edge,
            curvature_metric=curvature_metric,
        )

    def draw(self, normalize_origin=True, cmap='viridis'):
        if len(self.points) < 2:
            raise ValueError('Need at least two points to draw a track')

        geom = self._geometry(normalize_origin)
        s = geom['s']
        x = geom['x']
        y = geom['y']
        left_edge = geom['left_edge']
        right_edge = geom['right_edge']
        curvature_metric = geom['curvature_metric']

        quads = []
        quad_colors = []

        for i in range(len(self.points) - 1):
            quad = [
                tuple(left_edge[i]),
                tuple(left_edge[i + 1]),
                tuple(right_edge[i + 1]),
                tuple(right_edge[i]),
            ]
            quads.append(quad)
            quad_colors.append(0.5 * (curvature_metric[i] + curvature_metric[i + 1]))

        fig, ax = plt.subplots(figsize=(12, 8))
        poly = PolyCollection(quads, array=np.array(quad_colors), cmap=cmap, linewidth=0.2)
        ax.add_collection(poly)
        fig.colorbar(poly, ax=ax, label='|dθ/ds| (rad/m)')

        # Draw centerline for reference
        ax.plot(x, y, color='white', linewidth=1.5, label='Centerline')

        # Start/finish markers
        ax.plot(x[0], y[0], marker='o', color='lime', markersize=8, label='Start')
        ax.plot(x[-1], y[-1], marker='x', color='red', markersize=8, label='End')

        ax.set_xlabel('X (m)' + (' (relative)' if normalize_origin else ''))
        ax.set_ylabel('Y (m)' + (' (relative)' if normalize_origin else ''))
        ax.set_title(f'Track footprint · length≈{s[-1]:.0f} m')
        ax.set_aspect('equal', adjustable='datalim')
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def draw_interactive_3d(
        self,
        normalize_origin=True,
        thickness=1.5,
        cmap='Turbo',
        animate=False,
        speed_mps=35.0,
        frame_step=5,
        speed_options=None,
    ):
        if len(self.points) < 2:
            raise ValueError('Need at least two points to draw a track')

        geom = self._geometry(normalize_origin)
        s = geom['s']
        x = geom['x']
        y = geom['y']
        left_edge = geom['left_edge']
        right_edge = geom['right_edge']
        curvature_metric = geom['curvature_metric']

        verts = []
        intensities = []

        def vidx(idx, side, layer):
            base = idx * 4
            side_offset = 0 if side == 'left' else 1
            layer_offset = 0 if layer == 'top' else 2
            return base + side_offset + layer_offset

        for i in range(len(self.points)):
            left = left_edge[i]
            right = right_edge[i]
            for (pt, layer) in ((left, 'top'), (right, 'top'), (left, 'bottom'), (right, 'bottom')):
                z = thickness / 2 if layer == 'top' else -thickness / 2
                verts.append((pt[0], pt[1], z))
                intensities.append(curvature_metric[i])

        triangles = []

        def add_quad(a, b, c, d):
            triangles.append((a, b, c))
            triangles.append((a, c, d))

        n = len(self.points)
        for i in range(n - 1):
            next_i = i + 1
            lt0 = vidx(i, 'left', 'top')
            rt0 = vidx(i, 'right', 'top')
            lt1 = vidx(next_i, 'left', 'top')
            rt1 = vidx(next_i, 'right', 'top')
            lb0 = vidx(i, 'left', 'bottom')
            rb0 = vidx(i, 'right', 'bottom')
            lb1 = vidx(next_i, 'left', 'bottom')
            rb1 = vidx(next_i, 'right', 'bottom')

            add_quad(lt0, rt0, rt1, lt1)  # top surface
            add_quad(lb0, lb1, rb1, rb0)  # bottom surface
            add_quad(lt0, lt1, lb1, lb0)  # left wall
            add_quad(rt0, rb0, rb1, rt1)  # right wall

        # start and finish caps
        add_quad(vidx(0, 'left', 'top'), vidx(0, 'right', 'top'), vidx(0, 'right', 'bottom'), vidx(0, 'left', 'bottom'))
        add_quad(vidx(n - 1, 'left', 'top'), vidx(n - 1, 'left', 'bottom'), vidx(n - 1, 'right', 'bottom'), vidx(n - 1, 'right', 'top'))

        i_idx, j_idx, k_idx = zip(*triangles)
        x_verts, y_verts, z_verts = zip(*verts)

        mesh = go.Mesh3d(
            x=x_verts,
            y=y_verts,
            z=z_verts,
            intensity=intensities,
            colorscale=cmap,
            i=i_idx,
            j=j_idx,
            k=k_idx,
            showscale=True,
            opacity=0.95,
            lighting=dict(diffuse=0.8, specular=0.3, fresnel=0.1),
            lightposition=dict(x=0, y=0, z=1)
        )

        centerline = go.Scatter3d(
            x=x,
            y=y,
            z=np.zeros_like(x),
            mode='lines',
            line=dict(color='white', width=4),
            name='Centerline'
        )

        markers = go.Scatter3d(
            x=[x[0], x[-1]],
            y=[y[0], y[-1]],
            z=[0, 0],
            mode='markers',
            marker=dict(color=['lime', 'red'], size=6),
            name='Start/Finish'
        )

        rider_trace = go.Scatter3d(
            x=[x[0]],
            y=[y[0]],
            z=[0],
            mode='markers',
            marker=dict(color='orange', size=6),
            name='Rider'
        )

        fig = go.Figure(data=[mesh, centerline, markers, rider_trace])
        fig.update_layout(
            title=f'Interactive Track · length≈{s[-1]:.0f} m',
            scene=dict(
                xaxis_title='X (m)' + (' rel' if normalize_origin else ''),
                yaxis_title='Y (m)' + (' rel' if normalize_origin else ''),
                zaxis_title='Thickness (m)',
                aspectmode='data',
                dragmode='orbit'
            ),
            margin=dict(l=0, r=0, t=40, b=0)
        )

        if animate:
            self._attach_animation(fig, s, x, y, speed_mps, frame_step, speed_options)

        fig.show()

    def _attach_animation(self, fig, s, x, y, speed_mps, frame_step, speed_options):
        frame_step = max(1, int(frame_step))
        indices = list(range(0, len(s), frame_step))
        if indices[-1] != len(s) - 1:
            indices.append(len(s) - 1)

        total_distance = s[-1] - s[0]
        total_time = total_distance / max(speed_mps, 1e-3)
        frame_duration = max(40, int(1000 * total_time / max(len(indices) - 1, 1)))

        if not speed_options:
            speed_options = sorted({max(1.0, speed_mps)} | {20.0, 40.0, 60.0})
        else:
            speed_options = sorted({max(1.0, val) for val in speed_options})

        frames = []
        slider_steps = []
        for idx, sample in enumerate(indices):
            frame_name = f'frame-{idx}'
            frames.append(
                go.Frame(
                    name=frame_name,
                    data=[
                        go.Scatter3d(
                            x=[x[sample]],
                            y=[y[sample]],
                            z=[0],
                            mode='markers',
                            marker=dict(color='orange', size=6),
                        )
                    ],
                    traces=[3],
                )
            )
            slider_steps.append(
                dict(
                    label=f"{s[sample]:.0f} m",
                    method='animate',
                    args=[[frame_name], dict(frame=dict(duration=frame_duration, redraw=True), transition=dict(duration=0), mode='immediate')],
                )
            )

        fig.frames = frames

        distance_slider = dict(
            active=0,
            steps=slider_steps,
            x=0.05,
            y=-0.05,
            len=0.9,
            currentvalue=dict(prefix='Distance: '),
        )

        def duration_for_speed(value):
            time_span = total_distance / max(value, 1e-3)
            return max(40, int(1000 * time_span / max(len(indices) - 1, 1)))

        speed_slider_steps = []
        for val in speed_options:
            speed_slider_steps.append(
                dict(
                    label=f"{val:.0f}",
                    method='animate',
                    args=[
                        None,
                        dict(
                            frame=dict(duration=duration_for_speed(val), redraw=True),
                            transition=dict(duration=0),
                            fromcurrent=True,
                            mode='immediate',
                        ),
                    ],
                )
            )

        speed_slider = dict(
            active=speed_options.index(min(speed_options, key=lambda v: abs(v - speed_mps))),
            steps=speed_slider_steps,
            x=0.05,
            y=-0.12,
            len=0.9,
            currentvalue=dict(prefix='Speed (m/s): '),
        )

        fig.update_layout(
            updatemenus=[
                dict(
                    type='buttons',
                    showactive=False,
                    buttons=[
                        dict(
                            label='Play',
                            method='animate',
                            args=[
                                None,
                                dict(frame=dict(duration=frame_duration, redraw=True), transition=dict(duration=0), fromcurrent=True, mode='immediate'),
                            ],
                        ),
                        dict(label='Pause', method='animate', args=[[None], dict(frame=dict(duration=0, redraw=False), transition=dict(duration=0), mode='immediate')]),
                    ],
                )
            ],
            sliders=[distance_slider, speed_slider],
        )


def load_track_from_csv(path):
    required_cols = ('s_m', 'x_m', 'y_m', 'theta_rad', 'w_m')
    points = []
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        missing = [col for col in required_cols if col not in reader.fieldnames]
        if missing:
            raise ValueError(f'Missing columns in CSV: {missing}')

        for idx, row in enumerate(reader, start=2):  # account for header line
            try:
                points.append(
                    TrackPoint(
                        s=float(row['s_m']),
                        x=float(row['x_m']),
                        y=float(row['y_m']),
                        theta=float(row['theta_rad']),
                        width=float(row['w_m']),
                    )
                )
            except (ValueError, TypeError) as exc:
                print(f'Skipping line {idx}: {exc}')

    if not points:
        raise ValueError('CSV did not yield any valid track points')

    return Track(points)


def main():
    parser = argparse.ArgumentParser(description='Visualize a motorcycle track footprint.')
    parser.add_argument('--csv', type=Path, default=Path('file.csv'), help='Path to CSV containing s_m,x_m,y_m,theta_rad,w_m columns.')
    parser.add_argument('--mode', choices=['2d', '3d'], default='3d', help='Choose between 2D or 3D visualization.')
    parser.add_argument('--no-normalize', action='store_true', help='Keep absolute coordinates instead of shifting origin to the first point.')
    parser.add_argument('--animate', action='store_true', help='Animate a rider marker along the track (3D mode only).')
    parser.add_argument('--speed', type=float, default=35.0, help='Playback speed for the rider animation (m/s).')
    parser.add_argument('--frame-step', type=int, default=5, help='Sample every Nth point when building animation frames.')
    parser.add_argument('--thickness', type=float, default=1.5, help='Track extrusion thickness for 3D mesh (meters).')
    parser.add_argument('--speed-options', type=float, nargs='+', help='List of speeds (m/s) exposed on the UI slider.')
    args = parser.parse_args()

    track = load_track_from_csv(args.csv)
    normalize = not args.no_normalize

    if args.mode == '2d':
        track.draw(normalize_origin=normalize)
    else:
        track.draw_interactive_3d(
            normalize_origin=normalize,
            thickness=args.thickness,
            animate=args.animate,
            speed_mps=args.speed,
            frame_step=args.frame_step,
            speed_options=args.speed_options,
        )


if __name__ == '__main__':
    main()