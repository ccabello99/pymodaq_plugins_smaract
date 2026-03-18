# -*- coding: utf-8 -*-
"""
SmarAct MCS2 Trajectory Streaming Extension for PyMoDAQ
========================================================

Streams a pre-loaded trajectory (CSV / TXT) to a SmarAct MCS2 controller
using the hardware trajectory-streaming API.

The extension does NOT open its own connection to the hardware.  Instead it
reuses the already-open DAQ_Move module that is loaded in the dashboard,
accessing its underlying SmarActMCS2Wrapper (and therefore the device handle
and bindings) directly.

Trajectory file format
----------------------
Plain text file (CSV, TSV, space-delimited ...) where each row is one support
point (frame) and each column is one axis.  Values must be in the same
physical units that the DAQ_Move module uses -- i.e. whatever unit the MCS2
PyMoDAQ plugin is configured for (typically um for linear axes, m-deg for
rotary axes).  No unit conversion is applied; values are rounded to the
nearest integer before packing.

Frame binary format  (MCS2 Programmer's Guide section 2.15)
-----------------------------------------------------------
Each frame encodes one target position per participating channel:

  For every channel in the frame (same order in every frame):
      1 byte   channel index   (uint8, unsigned)
      8 bytes  target position (int64, little-endian)
  Total per frame = 9 x N_channels  bytes

On-the-fly Arbitrary-Axis Linear Move
--------------------------------------
The "Axis Move" panel moves the stage by a signed distance d along an
arbitrary direction defined by a direction vector (v1, v2, v3).
The vector is normalised automatically.  Given the current positions
(X1, X2, X3) the targets are:

    target = current_pos + d * (v / |v|)

A linear interpolation of N frames is streamed.  The channels for X1, X2, X3
are user-configurable; set any channel to -1 to exclude that axis.

Rotation Compensation Move
--------------------------
The "Rotation Compensation" dock handles the case where an XY positioner is
mounted on top of a rotation stage.  The calibration convention is that when
both XY axes are at zero, the target of interest sits exactly on the rotation
axis -- so a pure rotation is possible with no XY correction.

When the XY stage is displaced to (x0, y0) the target is off-axis by that
amount (in the rotating body frame).  A subsequent rotation would carry the
target in a circle.  The XY stage must compensate to keep the target fixed
in the lab frame.

The exact closed-form solution (no small-angle approximation):

    xy_cmd(theta) = R(θ0 - theta) * (x0, y0)

where R is the 2-D rotation matrix and theta is in m-deg (the native unit).

When x0 = y0 = 0 the correction vanishes identically -- pure rotation.

These are more natural than Cartesian (cx, cy) in the lab frame because they
are independent of the rotation stage's current angle.

The body-frame offset vector is:

    offset_body = R_offset * (cos(phi_offset), sin(phi_offset))

The target position in the rotating body frame (constant throughout the move):

    P_body = offset_body + (x0, y0)

The required XY stage commands to keep the target fixed in the lab frame:

    xy_cmd(theta) = R(θ0 - theta) * P_body - offset_body

where R(.) is the 2-D rotation matrix.  This is an exact closed-form solution.

Multi-module (master/slave) architecture
-----------------------------------------
Each physical axis is managed by an independent DAQ_Move module in the
PyMoDAQ dashboard (master/slave preset).  This extension references each
axis module by name to:
  - read the current position via module.get_actuator_value()  (already in
    plugin units -- no manual base-resolution arithmetic needed)
  - obtain the controller channel index via module.settings['move_settings', 'channel']
  - derive the base resolution for raw encoding via module.settings
    (or from the wrapper) -- raw = plugin_value / (10 ** base_resolution_exp)

The master module (user-selected) provides the shared hardware handle
(controller.controller_index) used for trajectory streaming.  All slave
modules reference the same underlying wrapper.

Frame units
-----------
Trajectory arrays in this extension use plugin units (um for linear axes,
m-deg for rotary axes).  Before encoding each frame the values are converted
to the MCS2 native base units (pm for linear, n-deg for rotary) using each
axis module's base_resolution exponent.

Flow
----
  1. Set module names in the Settings tree (master + per-axis) and click
     Refresh Modules.
  2. Set the stream rate (Hz).
  -- File-based workflow --
  3a. Load a trajectory CSV / TXT file, assign module names to columns in
      the Axis Mapping group, and click Start Streaming.
  -- Arbitrary-axis move workflow --
  3b. Open the "Axis Move" dock, select axis modules, define the direction
      vector, set distance and N frames, then click Generate & Stream.
  -- Rotation compensation workflow --
  3c. Open the "Rotation Compensation" dock, select axis modules and
      parameters, then click Generate & Stream.
  -- All workflows --
  4. Abort at any time with Abort Stream.
"""

import struct
import threading
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
from qtpy import QtWidgets, QtCore

from pymodaq_gui import utils as gutils
from pymodaq_utils.config import Config
from pymodaq_utils.logger import set_logger, get_module_name
from pymodaq.utils.config import Config as PyMoConfig
from pymodaq.extensions.utils import CustomExt

logger = set_logger(get_module_name(__file__))
config_pymodaq = PyMoConfig()

EXTENSION_NAME = 'MCS2 Trajectory Streaming'
CLASS_NAME = 'MCS2TrajectoryStreaming'


class MCS2TrajConfig(Config):
    config_name = 'smaract_trajectory'
    config_template_path = None

traj_config = MCS2TrajConfig()

# ---------------------------------------------------------------------------
# Frame encoding
# ---------------------------------------------------------------------------
def encode_frame(channel_positions: List[Tuple[int, int]]) -> bytes:
    """Pack one trajectory frame into the binary format expected by the MCS2.

    Each participating channel contributes a 9-byte tuple:
        1 byte  - channel index (uint8, unsigned)
        8 bytes - target position (int64, little-endian)

    Parameters
    ----------
    channel_positions : list of (channel_index: int, position: int)

    Returns
    -------
    bytes  -  9 x len(channel_positions) bytes
    """
    frame = bytearray()
    for ch, pos in channel_positions:
        frame += struct.pack('<B', int(ch))
        frame += struct.pack('<q', int(pos))
    return bytes(frame)


# ---------------------------------------------------------------------------
# Trajectory generators
# ---------------------------------------------------------------------------
def build_axis_move_trajectory(
        positions: np.ndarray,
        direction: np.ndarray,
        distance: float,
        n_frames: int,
        active_mask: np.ndarray,
) -> np.ndarray:
    """Linear move by *distance* along *direction* in N-axis space.

    The direction vector is normalised internally.  Only axes flagged True
    in *active_mask* appear as columns in the output; their order matches
    the order in *positions* / *active_mask*.

    Parameters
    ----------
    positions   : current absolute position for every physical axis, shape (N,)
    direction   : raw direction vector in the same N-axis space, shape (N,)
                  Need not be a unit vector; zero vector is rejected.
    distance    : signed scalar distance to travel (same units as positions)
    n_frames    : total frames including start and end (>= 2)
    active_mask : bool array shape (N,) -- True for axes to include in output

    Returns
    -------
    np.ndarray shape (n_frames, sum(active_mask))
    """
    if n_frames < 2:
        raise ValueError('n_frames must be >= 2.')
    norm = np.linalg.norm(direction)
    if norm == 0.0:
        raise ValueError('Direction vector must not be the zero vector.')
    unit = direction / norm
    targets = positions + distance * unit
    t = np.linspace(0.0, 1.0, n_frames)[:, np.newaxis]   # (n_frames, 1)
    full_traj = positions + t * (targets - positions)     # (n_frames, N)
    return full_traj[:, active_mask]


def build_rotation_compensation_trajectory(
        x0: float,
        y0: float,
        θ0_mdeg: float,
        theta_end_mdeg: float,
        n_frames: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a coordinated (X, Y, theta) trajectory that keeps the target
    fixed in the lab frame while the rotation stage sweeps.

    Physical model
    --------------
    The XY stage sits on top of the rotation stage.  The zero position of the
    XY stage corresponds to the target being exactly on the rotation axis.
    This is the factory/user calibration: when XY = (0, 0), rotating freely
    is a pure rotation with no sample displacement.

    When the XY stage is at (x0, y0) != (0, 0) the target is displaced from
    the rotation axis by (x0, y0) in the rotating body frame.  A rotation of
    the bottom stage by Dtheta carries the target in a circle unless the XY stage
    compensates.

    The target position in the lab frame is:

        P_lab = R(theta) * (x_cmd, y_cmd)

    where R(theta) is the 2-D rotation matrix.  To keep P_lab constant:

        (x_cmd, y_cmd) = R(θ0 - theta) * (x0, y0)

    Boundary checks:
        theta = θ0  ->  (x_cmd, y_cmd) = R(0) * (x0, y0) = (x0, y0)  [ok]
        x0 = y0 = 0    ->  (x_cmd, y_cmd) = (0, 0) for all theta  [ok] (pure rotation)

    Parameters
    ----------
    x0, y0          : current XY stage positions (um)
    θ0_mdeg     : current rotation angle (m-deg)
    theta_end_mdeg  : target rotation angle (m-deg)
    n_frames        : number of trajectory frames (>= 2)

    Returns
    -------
    x_traj     : np.ndarray shape (n_frames,) -- XY X commands (um)
    y_traj     : np.ndarray shape (n_frames,) -- XY Y commands (um)
    theta_traj : np.ndarray shape (n_frames,) -- rotation commands (m-deg)
    """
    if n_frames < 2:
        raise ValueError('n_frames must be >= 2.')

    theta_arr   = np.linspace(θ0_mdeg, theta_end_mdeg, n_frames)
    θ0_rad  = np.deg2rad(θ0_mdeg / 1000.0)
    theta_rad   = np.deg2rad(theta_arr   / 1000.0)

    delta   = θ0_rad - theta_rad   # shape (n_frames,)
    cos_d   = np.cos(delta)
    sin_d   = np.sin(delta)

    x_cmd   = cos_d * x0 - sin_d * y0
    y_cmd   = sin_d * x0 + cos_d * y0

    return x_cmd, y_cmd, theta_arr


# ===========================================================================
# Extension class
# ===========================================================================
class MCS2TrajectoryStreaming(CustomExt):
    """PyMoDAQ extension for SmarAct MCS2 trajectory streaming."""

    settings_name = 'MCS2TrajSettings'

    params = [
        # -- Master module (provides the hardware handle for streaming) --------
        {'title': 'Module Settings', 'name': 'module_settings',
         'type': 'group', 'expanded': True, 'children': [
             {'title': 'Master module:', 'name': 'master_name',
              'type': 'str', 'value': 'Target X',
              'tip': 'Name of the master DAQ_Move in the dashboard.\n'
                     'This module provides the hardware connection for streaming.'},
             {'title': 'Refresh Modules', 'name': 'refresh_modules',
              'type': 'action'},
             {'title': 'Master status:', 'name': 'master_status',
              'type': 'str', 'value': 'Not connected', 'readonly': True},
         ]},

        # -- Per-axis module names (rotation-compensation dock) ----------------
        {'title': 'RC Axis Modules', 'name': 'rc_modules',
         'type': 'group', 'expanded': True,
         'tip': 'Module names for the Rotation Compensation dock.\n'
                'Each name must match a DAQ_Move loaded in the dashboard.',
         'children': [
             {'title': 'X module:', 'name': 'x_module',
              'type': 'str', 'value': 'Target X',
              'tip': 'DAQ_Move module for the X linear axis'},
             {'title': 'Y module:', 'name': 'y_module',
              'type': 'str', 'value': 'Target Y',
              'tip': 'DAQ_Move module for the Y linear axis'},
             {'title': 'Theta module:', 'name': 'theta_module',
              'type': 'str', 'value': 'Target Rotation',
              'tip': 'DAQ_Move module for the rotation axis'},
         ]},

        # -- Per-axis module names (arbitrary-axis move dock) ------------------
        {'title': 'Axis Move Modules', 'name': 'ax_modules',
         'type': 'group', 'expanded': False,
         'tip': 'Module names for the Arbitrary Axis Move dock.',
         'children': [
             {'title': 'X1 module:', 'name': 'x1_module',
              'type': 'str', 'value': 'Target X',
              'tip': 'DAQ_Move for axis 1 (set empty to disable)'},
             {'title': 'X2 module:', 'name': 'x2_module',
              'type': 'str', 'value': 'Target Y',
              'tip': 'DAQ_Move for axis 2 (set empty to disable)'},
             {'title': 'X3 module:', 'name': 'x3_module',
              'type': 'str', 'value': '',
              'tip': 'DAQ_Move for axis 3 (set empty to disable)'},
             {'title': 'Rotation module:', 'name': 'rot_module',
              'type': 'str', 'value': 'Target Rotation',
              'tip': 'Rotation DAQ_Move for body-frame direction mode'},
         ]},

        # -- File-based axis mapping (module name per CSV column) --------------
        {'title': 'Axis Mapping (file mode)', 'name': 'axis_mapping',
         'type': 'group', 'expanded': True,
         'tip': 'Map CSV columns to DAQ_Move module names.\n'
                'Leave empty to skip that column.',
         'children': [
             {'title': 'Col 0 module:', 'name': 'col0_module',
              'type': 'str', 'value': 'Target X'},
             {'title': 'Col 1 module:', 'name': 'col1_module',
              'type': 'str', 'value': 'Target Y'},
             {'title': 'Col 2 module:', 'name': 'col2_module',
              'type': 'str', 'value': 'Target Rotation'},
         ]},

        # Stream settings
        {'title': 'Stream Settings', 'name': 'stream_settings',
         'type': 'group', 'expanded': True, 'children': [
             {'title': 'Stream Rate (Hz):', 'name': 'stream_rate',
              'type': 'int', 'value': 1000, 'min': 1, 'max': 1000,
              'tip': 'Sets STREAM_BASE_RATE on the controller'},
             {'title': 'Disable Interpolation:', 'name': 'no_interpolation',
              'type': 'bool', 'value': False,
              'tip': 'Set StreamOption.INTERPOLATION_DIS if True'},
         ]},

        # Trajectory file
        {'title': 'Trajectory File', 'name': 'traj_file', 'type': 'group',
         'expanded': True, 'children': [
             {'title': 'File Path:', 'name': 'file_path', 'type': 'str',
              'value': '', 'readonly': True},
             {'title': 'Delimiter:', 'name': 'delimiter', 'type': 'list',
              'limits': ['comma', 'tab', 'space', 'semicolon'],
              'value': 'comma'},
             {'title': 'Skip Header Rows:', 'name': 'skip_rows',
              'type': 'int', 'value': 0, 'min': 0},
             {'title': 'N Frames Loaded:', 'name': 'n_frames',
              'type': 'int', 'value': 0, 'readonly': True},
             {'title': 'Duration (s):', 'name': 'duration',
              'type': 'float', 'value': 0.0, 'readonly': True},
         ]},
    ]

    _log_signal = QtCore.Signal(str, str)

    def __init__(self, parent: gutils.DockArea, dashboard):
        super().__init__(parent, dashboard)

        # Master DAQ_Move module -- provides the hardware handle for streaming.
        # Slave modules for individual axes are stored in _axis_modules.
        self._master_module = None          # DAQ_Move  (master)
        self._axis_modules: dict = {}       # name -> DAQ_Move

        self._trajectory: Optional[np.ndarray] = None
        self._pending_channel_map: Optional[List] = None  # [(col, module), ...]
        self._stream_thread: Optional[threading.Thread] = None
        self._abort_event = threading.Event()

        self._log_signal.connect(self._on_log_signal)
        self.settings.child('module_settings', 'refresh_modules'
                            ).sigActivated.connect(self.refresh_modules)
        self.setup_ui()


    # -----------------------------------------------------------------------
    # Dock / UI setup
    # -----------------------------------------------------------------------
    def setup_docks(self):
        # Trajectory preview
        self.docks['preview'] = gutils.Dock('Trajectory Preview')
        self.dockarea.addDock(self.docks['preview'])

        preview_w = QtWidgets.QWidget()
        preview_lay = QtWidgets.QVBoxLayout()
        preview_w.setLayout(preview_lay)
        self.traj_table = QtWidgets.QTableWidget()
        self.traj_table.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers)
        self.traj_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch)
        preview_lay.addWidget(self.traj_table)
        self.docks['preview'].addWidget(preview_w)

        # Stream control
        self.docks['control'] = gutils.Dock('Stream Control')
        self.dockarea.addDock(self.docks['control'], 'right',
                              self.docks['preview'])
        self._build_control_dock()

        # Settings tree
        self.docks['settings'] = gutils.Dock('Settings')
        self.dockarea.addDock(self.docks['settings'], 'below',
                              self.docks['control'])
        self.docks['settings'].addWidget(self.settings_tree)

        # Arbitrary Axis Move dock
        self.docks['axis_move'] = gutils.Dock('Axis Move')
        self.dockarea.addDock(self.docks['axis_move'], 'below',
                              self.docks['settings'])
        self._build_axis_move_dock()

        # Rotation Compensation dock
        self.docks['rot_comp'] = gutils.Dock('Rotation Compensation')
        self.dockarea.addDock(self.docks['rot_comp'], 'below',
                              self.docks['axis_move'])
        self._build_rotation_comp_dock()

        # Status log
        self.docks['status'] = gutils.Dock('Status Log')
        self.dockarea.addDock(self.docks['status'], 'bottom',
                              self.docks['preview'])
        self.status_text = QtWidgets.QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(160)
        self.status_text.setStyleSheet(
            'background-color: #1e1e1e; color: #d4d4d4; '
            'font-family: monospace; font-size: 10pt;')
        self.docks['status'].addWidget(self.status_text)

        self.log_message('Extension initialised. Set the MCS2 module name '
                         'and click Refresh Module.')

    # -----------------------------------------------------------------------
    # Stream Control dock
    # -----------------------------------------------------------------------
    def _build_control_dock(self):
        ctrl_w = QtWidgets.QWidget()
        ctrl_lay = QtWidgets.QVBoxLayout()
        ctrl_w.setLayout(ctrl_lay)

        self.load_btn = QtWidgets.QPushButton('Load Trajectory File')
        self.load_btn.setMinimumHeight(48)
        self.load_btn.setStyleSheet(self._btn_style('#2c7be5', '#1a5cb8'))
        self.load_btn.clicked.connect(self.load_trajectory_file)
        ctrl_lay.addWidget(self.load_btn)

        ctrl_lay.addSpacing(8)

        self.stream_btn = QtWidgets.QPushButton('Start Streaming')
        self.stream_btn.setMinimumHeight(64)
        self.stream_btn.setStyleSheet(self._btn_style('#28a745', '#1e7e34'))
        self.stream_btn.setEnabled(False)
        self.stream_btn.clicked.connect(self.start_streaming)
        ctrl_lay.addWidget(self.stream_btn)

        ctrl_lay.addSpacing(4)

        self.abort_btn = QtWidgets.QPushButton('ABORT STREAM')
        self.abort_btn.setMinimumHeight(56)
        self.abort_btn.setStyleSheet(self._btn_style('#dc3545', '#a71d2a'))
        self.abort_btn.setEnabled(False)
        self.abort_btn.clicked.connect(self.abort_streaming)
        ctrl_lay.addWidget(self.abort_btn)

        ctrl_lay.addSpacing(12)

        progress_label = QtWidgets.QLabel('Streaming progress:')
        progress_label.setStyleSheet('font-weight: bold;')
        ctrl_lay.addWidget(progress_label)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setValue(0)
        ctrl_lay.addWidget(self.progress_bar)

        ctrl_lay.addSpacing(6)

        self.frame_label = QtWidgets.QLabel('Frame: 0 / 0')
        self.frame_label.setAlignment(QtCore.Qt.AlignCenter)
        self.frame_label.setStyleSheet(
            'font-size: 13pt; font-weight: bold; '
            'background: #111; color: #0f0; '
            'border: 2px solid #555; border-radius: 4px; padding: 6px;')
        ctrl_lay.addWidget(self.frame_label)

        ctrl_lay.addStretch()
        self.docks['control'].addWidget(ctrl_w)

    # -----------------------------------------------------------------------
    # Arbitrary Axis Move dock
    # -----------------------------------------------------------------------
    def _build_axis_move_dock(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout()
        lay.setSpacing(8)
        w.setLayout(lay)

        info = QtWidgets.QLabel(
            '<b>On-the-fly arbitrary-axis linear move</b><br>'
            'Define a direction vector and move the stage from its current '
            'position by distance <i>d</i> along that direction.<br><br>'
            'The direction vector can be expressed in the <b>lab frame</b> '
            '(default) or in the <b>XY stage body frame</b>.  In body-frame '
            'mode the vector is rotated by the current rotation-stage angle '
            'before use, so the motion direction is always the same relative '
            'to the sample regardless of how the rotation stage is oriented.<br><br>'
            '<tt>v_lab = R(θ0) * v_body</tt>'
        )
        info.setWordWrap(True)
        info.setStyleSheet(self._info_style('#1a2e1a', '#a0e8a0', '#2c6a2c'))
        lay.addWidget(info)

        grid = QtWidgets.QGridLayout()
        grid.setColumnStretch(1, 1)
        gr = 0  # running grid row counter

        # ---- Axis module assignments ----
        grid.addWidget(self._bold_label('Axis modules'), gr, 0, 1, 2)
        gr += 1

        grid.addWidget(QtWidgets.QLabel(
            '<i style="color:#aaa">Module names must match dashboard entries.<br>'
            'Click "Sync from Settings" to fill from the Settings tree.<br>'
            'Leave empty to exclude that axis.</i>'),
            gr, 0, 1, 2)
        gr += 1

        for label, attr, setting_key in [
                ('X1 module:', 'ax_x1_mod_edit', 'x1_module'),
                ('X2 module:', 'ax_x2_mod_edit', 'x2_module'),
                ('X3 module:', 'ax_x3_mod_edit', 'x3_module')]:
            grid.addWidget(QtWidgets.QLabel(label), gr, 0)
            edit = QtWidgets.QLineEdit()
            edit.setPlaceholderText('module name (empty = skip)...')
            edit.setText(self.settings['ax_modules', setting_key])
            edit.setToolTip(
                f'DAQ_Move module for {label[:2]} axis. '
                f'Leave empty to exclude from the stream.')
            setattr(self, attr, edit)
            grid.addWidget(edit, gr, 1)
            gr += 1

        ax_sync_btn = QtWidgets.QPushButton('Sync from Settings')
        ax_sync_btn.setToolTip('Fill module names from the Axis Move Modules group in Settings')
        ax_sync_btn.clicked.connect(self._sync_ax_modules_from_settings)
        grid.addWidget(ax_sync_btn, gr, 0, 1, 2)
        gr += 1

        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setStyleSheet('color: #444;')
        grid.addWidget(sep, gr, 0, 1, 2)
        gr += 1

        # Body-frame toggle
        grid.addWidget(self._bold_label('Reference frame'), gr, 0, 1, 2)
        gr += 1

        self.ax_body_frame_chk = QtWidgets.QCheckBox('Direction in XY body frame')
        self.ax_body_frame_chk.setToolTip(
            'When checked, the direction vector is interpreted in the rotating '
            'XY stage body frame.\n'
            'It is pre-multiplied by R(θ0) (current rotation angle) before '
            'computing the move, so the physical motion direction is always '
            'the same relative to the sample.\n\n'
            'When unchecked (default), the vector is in the lab frame.')
        self.ax_body_frame_chk.stateChanged.connect(self._update_axis_norm_label)
        grid.addWidget(self.ax_body_frame_chk, gr, 0, 1, 2)
        gr += 1

        grid.addWidget(QtWidgets.QLabel('Rotation module:'), gr, 0)
        self.ax_rot_mod_edit = QtWidgets.QLineEdit()
        self.ax_rot_mod_edit.setPlaceholderText('rotation module name…')
        self.ax_rot_mod_edit.setText(self.settings['ax_modules', 'rot_module'])
        self.ax_rot_mod_edit.setToolTip(
            'DAQ_Move module of the rotation stage.\n'
            'Used only when "Direction in XY body frame" is checked.\n'
            'Leave empty to disable body-frame mode.')
        self.ax_rot_mod_edit.textChanged.connect(self._update_axis_norm_label)
        grid.addWidget(self.ax_rot_mod_edit, gr, 1)
        gr += 1

        grid.addWidget(QtWidgets.QLabel('Current θ (m-deg):'), gr, 0)
        self.ax_theta_label = QtWidgets.QLabel('--')
        self.ax_theta_label.setStyleSheet(
            'font-family: monospace; color: #7ef; '
            'background: #111; padding: 2px 6px; border-radius: 3px;')
        grid.addWidget(self.ax_theta_label, gr, 1)
        gr += 1

        sep2 = QtWidgets.QFrame()
        sep2.setFrameShape(QtWidgets.QFrame.HLine)
        sep2.setStyleSheet('color: #444;')
        grid.addWidget(sep2, gr, 0, 1, 2)
        gr += 1

        # Direction vector
        grid.addWidget(self._bold_label('Direction vector  (v1, v2, v3)'), gr, 0, 1, 2)
        gr += 1

        for label, attr, default in [
                ('v1:', 'ax_v1_spin', 1.0),
                ('v2:', 'ax_v2_spin', 0.0),
                ('v3:', 'ax_v3_spin', 0.0)]:
            grid.addWidget(QtWidgets.QLabel(label), gr, 0)
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(-1e9, 1e9)
            spin.setDecimals(6)
            spin.setSingleStep(0.1)
            spin.setValue(default)
            spin.valueChanged.connect(self._update_axis_norm_label)
            setattr(self, attr, spin)
            grid.addWidget(spin, gr, 1)
            gr += 1

        grid.addWidget(QtWidgets.QLabel('As lab-frame unit v:'), gr, 0)
        self.ax_unit_label = QtWidgets.QLabel('--')
        self.ax_unit_label.setStyleSheet(
            'font-family: monospace; color: #a0e8a0;')
        grid.addWidget(self.ax_unit_label, gr, 1)
        gr += 1

        sep3 = QtWidgets.QFrame()
        sep3.setFrameShape(QtWidgets.QFrame.HLine)
        sep3.setStyleSheet('color: #444;')
        grid.addWidget(sep3, gr, 0, 1, 2)
        gr += 1

        # Move parameters
        grid.addWidget(self._bold_label('Move parameters'), gr, 0, 1, 2)
        gr += 1

        grid.addWidget(QtWidgets.QLabel('Distance d:'), gr, 0)
        self.ax_dist_spin = QtWidgets.QDoubleSpinBox()
        self.ax_dist_spin.setRange(-1e9, 1e9)
        self.ax_dist_spin.setDecimals(4)
        self.ax_dist_spin.setSingleStep(1.0)
        self.ax_dist_spin.setValue(100.0)
        self.ax_dist_spin.setToolTip(
            'Signed distance to travel along the direction vector '
            '(same units as the stage, typically um).\n'
            'Positive = forward along v, negative = backward.')
        grid.addWidget(self.ax_dist_spin, gr, 1)
        gr += 1

        grid.addWidget(QtWidgets.QLabel('N Frames:'), gr, 0)
        self.ax_nframes_spin = QtWidgets.QSpinBox()
        self.ax_nframes_spin.setRange(2, 1_000_000)
        self.ax_nframes_spin.setValue(500)
        self.ax_nframes_spin.setSingleStep(100)
        self.ax_nframes_spin.setToolTip('Duration = N / Stream Rate')
        grid.addWidget(self.ax_nframes_spin, gr, 1)
        gr += 1

        grid.addWidget(QtWidgets.QLabel('Duration (s):'), gr, 0)
        self.ax_duration_label = QtWidgets.QLabel('--')
        self.ax_duration_label.setStyleSheet('font-weight: bold;')
        grid.addWidget(self.ax_duration_label, gr, 1)

        lay.addLayout(grid)

        self.ax_nframes_spin.valueChanged.connect(self._update_ax_duration)
        self.settings.child('stream_settings', 'stream_rate'
                             ).sigValueChanged.connect(self._update_ax_duration)
        self._update_ax_duration()
        self._update_axis_norm_label()

        # Position read-back
        lay.addWidget(self._build_pos_readback_group(
            labels=[
                ('Current X1:', 'ax_x1_label', '#7ef'),
                ('Current X2:', 'ax_x2_label', '#7ef'),
                ('Current X3:', 'ax_x3_label', '#7ef'),
                ('Target X1:', 'ax_t1_label', '#a0e8a0'),
                ('Target X2:', 'ax_t2_label', '#a0e8a0'),
                ('Target X3:', 'ax_t3_label', '#a0e8a0'),
            ],
            title='Stage positions (updated on Preview / Generate & Stream)',
            ncols=3,
        ))

        # Buttons
        btn_lay = QtWidgets.QHBoxLayout()

        self.ax_preview_btn = QtWidgets.QPushButton('Preview')
        self.ax_preview_btn.setMinimumHeight(44)
        self.ax_preview_btn.setStyleSheet(self._btn_style('#6f42c1', '#4b2d8f'))
        self.ax_preview_btn.setToolTip(
            'Compute trajectory and show in preview table -- no streaming')
        self.ax_preview_btn.clicked.connect(self.preview_axis_move)
        btn_lay.addWidget(self.ax_preview_btn)

        self.ax_stream_btn = QtWidgets.QPushButton('Generate && Stream')
        self.ax_stream_btn.setMinimumHeight(44)
        self.ax_stream_btn.setStyleSheet(self._btn_style('#27ae60', '#1a7a42'))
        self.ax_stream_btn.setToolTip(
            'Build axis-move trajectory from current position and stream it')
        self.ax_stream_btn.clicked.connect(self.generate_and_stream_axis_move)
        btn_lay.addWidget(self.ax_stream_btn)

        lay.addLayout(btn_lay)
        lay.addStretch()
        self.docks['axis_move'].addWidget(w)

    # -----------------------------------------------------------------------
    # Rotation Compensation dock
    # -----------------------------------------------------------------------
    def _build_rotation_comp_dock(self):
        """Build the Rotation Compensation dock.

        Generates a coordinated (X, Y, theta) trajectory that keeps the target
        fixed in the lab frame while the rotation stage sweeps.

        Physical model:
            xy_cmd(theta) = R(θ0 - theta) * (x0, y0)

        When XY = (0, 0) the correction is zero -- pure rotation.
        All rotation angles in the GUI use milli-degrees (m-deg), matching
        the native unit of the MCS2 rotary axis plugin.
        """
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout()
        lay.setSpacing(8)
        w.setLayout(lay)

        # Info banner
        info = QtWidgets.QLabel(
            '<b>Rotation compensation with XY correction</b><br>'
            'Sweeps the rotation stage from its current angle to a target '
            'angle while commanding the XY stage to keep the sample fixed '
            'in the lab frame.<br><br>'
            '<b>Convention:</b> XY = (0, 0) means the target is exactly on '
            'the rotation axis -- pure rotation, no XY correction needed.  '
            'Any non-zero XY position offsets the target, and this dock '
            'generates the compensating trajectory.<br><br>'
            '<tt>xy_cmd(theta) = R(θ0 - theta) * (x0, y0)</tt><br>'
            'All rotation angles are in <b>milli-degrees (m-deg)</b>.'
        )
        info.setWordWrap(True)
        info.setStyleSheet(self._info_style('#1a1a2e', '#a0c4ff', '#2c4a8f'))
        lay.addWidget(info)

        grid = QtWidgets.QGridLayout()
        grid.setColumnStretch(1, 1)
        row = 0

        # ---- Axis module assignments ----
        grid.addWidget(self._bold_label('Axis modules'), row, 0, 1, 2)
        row += 1

        grid.addWidget(QtWidgets.QLabel(
            '<i style="color:#aaa">Module names must match dashboard entries.<br>'
            'Click "Sync from Settings" to fill from the Settings tree.</i>'),
            row, 0, 1, 2)
        row += 1

        for label, attr, setting_key, tip in [
            ('X module:',     'rc_x_mod_edit',  'x_module',
             'DAQ_Move module name for the X linear axis'),
            ('Y module:',     'rc_y_mod_edit',  'y_module',
             'DAQ_Move module name for the Y linear axis'),
            ('Theta module:', 'rc_th_mod_edit', 'theta_module',
             'DAQ_Move module name for the rotation stage'),
        ]:
            grid.addWidget(QtWidgets.QLabel(label), row, 0)
            edit = QtWidgets.QLineEdit()
            edit.setPlaceholderText('module name…')
            edit.setText(self.settings['rc_modules', setting_key])
            edit.setToolTip(tip)
            setattr(self, attr, edit)
            grid.addWidget(edit, row, 1)
            row += 1

        sync_btn = QtWidgets.QPushButton('Sync from Settings')
        sync_btn.setToolTip('Fill module names from the RC Axis Modules group in Settings')
        sync_btn.clicked.connect(self._sync_rc_modules_from_settings)
        grid.addWidget(sync_btn, row, 0, 1, 2)
        row += 1

        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setStyleSheet('color: #444;')
        grid.addWidget(sep, row, 0, 1, 2)
        row += 1

        # ---- Rotation parameters ----
        grid.addWidget(self._bold_label('Rotation parameters'), row, 0, 1, 2)
        row += 1

        # Absolute / relative toggle
        grid.addWidget(QtWidgets.QLabel('Move type:'), row, 0)
        _move_type_w = QtWidgets.QWidget()
        _move_type_lay = QtWidgets.QHBoxLayout()
        _move_type_lay.setContentsMargins(0, 0, 0, 0)
        _move_type_w.setLayout(_move_type_lay)
        self.rc_abs_radio = QtWidgets.QRadioButton('Absolute')
        self.rc_rel_radio = QtWidgets.QRadioButton('Relative')
        self.rc_abs_radio.setChecked(False)
        self.rc_rel_radio.setChecked(True)
        self.rc_abs_radio.setToolTip(
            'Target angle is an absolute position in m-deg.\n'
            'The stage moves to exactly this angle.')
        self.rc_rel_radio.setToolTip(
            'Target angle is a signed offset in m-deg added to the current\n'
            'angle.  Actual target = θ0 + value, resolved at compute time.')
        self.rc_abs_radio.toggled.connect(self._update_rc_angle_tip)
        _move_type_lay.addWidget(self.rc_abs_radio)
        _move_type_lay.addWidget(self.rc_rel_radio)
        _move_type_lay.addStretch()
        grid.addWidget(_move_type_w, row, 1)
        row += 1

        grid.addWidget(QtWidgets.QLabel('Target angle (m-deg):'), row, 0)
        self.rc_theta_end_spin = QtWidgets.QDoubleSpinBox()
        self.rc_theta_end_spin.setRange(-36_000_000.0, 36_000_000.0)
        self.rc_theta_end_spin.setDecimals(1)
        self.rc_theta_end_spin.setSingleStep(1000.0)   # 1 degree steps
        self.rc_theta_end_spin.setValue(500.0)      # default 500 milli-degrees
        self.rc_theta_end_spin.setToolTip(
            'Absolute mode: target angle in milli-degrees.\n'
            'Relative mode: signed offset from current angle in milli-degrees.\n'
            '1 degree = 1000 m-deg.')
        grid.addWidget(self.rc_theta_end_spin, row, 1)
        row += 1

        grid.addWidget(QtWidgets.QLabel('N Frames:'), row, 0)
        self.rc_nframes_spin = QtWidgets.QSpinBox()
        self.rc_nframes_spin.setRange(2, 1_000_000)
        self.rc_nframes_spin.setValue(500)
        self.rc_nframes_spin.setSingleStep(100)
        self.rc_nframes_spin.setToolTip('Duration = N / Stream Rate')
        self.rc_nframes_spin.valueChanged.connect(self._update_rc_duration)
        grid.addWidget(self.rc_nframes_spin, row, 1)
        row += 1

        grid.addWidget(QtWidgets.QLabel('Duration (s):'), row, 0)
        self.rc_duration_label = QtWidgets.QLabel('--')
        self.rc_duration_label.setStyleSheet('font-weight: bold;')
        grid.addWidget(self.rc_duration_label, row, 1)
        row += 1

        # Hook stream_rate changes
        self.settings.child('stream_settings', 'stream_rate'
                             ).sigValueChanged.connect(self._update_rc_duration)

        lay.addLayout(grid)
        self._update_rc_duration()

        # ---- Position read-back ----
        lay.addWidget(self._build_pos_readback_group(
            labels=[
                ('Current X (um):',      'rc_x0_label',  '#7ef'),
                ('Current Y (um):',      'rc_y0_label',  '#7ef'),
                ('Current θ (m-deg):',   'rc_th0_label', '#7ef'),
                ('Target X (um):',       'rc_xt_label',  '#a0e8a0'),
                ('Target Y (um):',       'rc_yt_label',  '#a0e8a0'),
                ('Target θ (m-deg):',    'rc_tht_label', '#a0e8a0'),
            ],
            title='Stage positions (updated on Preview / Generate & Stream)',
            ncols=3,
        ))

        # ---- Trajectory stats ----
        stats_group = QtWidgets.QGroupBox('Trajectory statistics')
        stats_group.setStyleSheet(
            'QGroupBox { font-weight: bold; color: #ccc; '
            'border: 1px solid #555; border-radius: 4px; margin-top: 6px; }'
            'QGroupBox::title { subcontrol-origin: margin; left: 8px; }')
        stats_lay = QtWidgets.QGridLayout()
        stats_group.setLayout(stats_lay)

        for i, (label, attr) in enumerate([
            ('Max |DX| (um):', 'rc_stat_dx'),
            ('Max |DY| (um):', 'rc_stat_dy'),
            ('|xy0| (um):', 'rc_stat_r'),
            ('phi0 (m-deg):', 'rc_stat_phi'),
        ]):
            r, c = divmod(i, 2)
            stats_lay.addWidget(QtWidgets.QLabel(label), r, c * 2)
            lbl = self._make_readback_label('#fbbf24')
            setattr(self, attr, lbl)
            stats_lay.addWidget(lbl, r, c * 2 + 1)

        lay.addWidget(stats_group)

        # ---- Buttons ----
        btn_lay = QtWidgets.QHBoxLayout()

        self.rc_preview_btn = QtWidgets.QPushButton('Preview')
        self.rc_preview_btn.setMinimumHeight(44)
        self.rc_preview_btn.setStyleSheet(self._btn_style('#6f42c1', '#4b2d8f'))
        self.rc_preview_btn.setToolTip(
            'Compute compensation trajectory and show in preview -- no streaming')
        self.rc_preview_btn.clicked.connect(self.preview_rotation_comp)
        btn_lay.addWidget(self.rc_preview_btn)

        self.rc_stream_btn = QtWidgets.QPushButton('Generate && Stream')
        self.rc_stream_btn.setMinimumHeight(44)
        self.rc_stream_btn.setStyleSheet(self._btn_style('#e67e22', '#b85a0a'))
        self.rc_stream_btn.setToolTip(
            'Build rotation-compensation trajectory from current positions and stream it')
        self.rc_stream_btn.clicked.connect(self.generate_and_stream_rotation_comp)
        btn_lay.addWidget(self.rc_stream_btn)

        lay.addLayout(btn_lay)
        lay.addStretch()
        self.docks['rot_comp'].addWidget(w)

    # -----------------------------------------------------------------------
    # UI helpers
    # -----------------------------------------------------------------------
    def _build_pos_readback_group(
            self,
            labels: list,
            title: str,
            ncols: int,
    ) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox(title)
        group.setStyleSheet(
            'QGroupBox { font-weight: bold; color: #ccc; '
            'border: 1px solid #555; border-radius: 4px; margin-top: 6px; }'
            'QGroupBox::title { subcontrol-origin: margin; left: 8px; }')
        grid = QtWidgets.QGridLayout()
        group.setLayout(grid)
        for i, (text, attr, colour) in enumerate(labels):
            row, pair = divmod(i, ncols)
            grid.addWidget(QtWidgets.QLabel(text), row, pair * 2)
            lbl = self._make_readback_label(colour)
            setattr(self, attr, lbl)
            grid.addWidget(lbl, row, pair * 2 + 1)
        return group

    @staticmethod
    def _make_readback_label(colour: str) -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel('--')
        lbl.setStyleSheet(
            f'font-family: monospace; font-size: 11pt; color: {colour}; '
            f'background: #111; padding: 2px 6px; border-radius: 3px;')
        lbl.setMinimumWidth(90)
        return lbl

    @staticmethod
    def _bold_label(text: str) -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel(text)
        lbl.setStyleSheet('font-weight: bold; color: #aaa;')
        return lbl

    @staticmethod
    def _info_style(bg: str, fg: str, border: str) -> str:
        return (f'background: {bg}; color: {fg}; '
                f'border: 1px solid {border}; border-radius: 4px; padding: 8px;')

    def _update_ax_duration(self):
        try:
            n = self.ax_nframes_spin.value()
            rate = self.settings['stream_settings', 'stream_rate']
            self.ax_duration_label.setText(f'{n / rate:.3f} s')
        except Exception:
            pass

    def _update_rc_duration(self):
        try:
            n = self.rc_nframes_spin.value()
            rate = self.settings['stream_settings', 'stream_rate']
            self.rc_duration_label.setText(f'{n / rate:.3f} s')
        except Exception:
            pass

    def _update_rc_angle_tip(self):
        """Update the tooltip on the target-angle spinbox to reflect abs/rel."""
        try:
            if self.rc_abs_radio.isChecked():
                self.rc_theta_end_spin.setToolTip(
                    'Absolute mode: stage moves to this exact angle (m-deg).\n'
                    '1 degree = 1000 m-deg.')
            else:
                self.rc_theta_end_spin.setToolTip(
                    'Relative mode: signed offset from current angle (m-deg).\n'
                    'Actual target = θ0 + this value, resolved at compute time.\n'
                    '1 degree = 1000 m-deg.')
        except Exception:
            pass

    def _sync_rc_modules_from_settings(self):
        """Fill RC dock module name edits from the Settings tree."""
        self.rc_x_mod_edit.setText(self.settings['rc_modules', 'x_module'])
        self.rc_y_mod_edit.setText(self.settings['rc_modules', 'y_module'])
        self.rc_th_mod_edit.setText(self.settings['rc_modules', 'theta_module'])

    def _sync_ax_modules_from_settings(self):
        """Fill Axis Move dock module name edits from the Settings tree."""
        self.ax_x1_mod_edit.setText(self.settings['ax_modules', 'x1_module'])
        self.ax_x2_mod_edit.setText(self.settings['ax_modules', 'x2_module'])
        self.ax_x3_mod_edit.setText(self.settings['ax_modules', 'x3_module'])
        self.ax_rot_mod_edit.setText(self.settings['ax_modules', 'rot_module'])

    def _update_axis_norm_label(self):
        try:
            v = np.array([self.ax_v1_spin.value(),
                           self.ax_v2_spin.value(),
                           self.ax_v3_spin.value()])
            norm = np.linalg.norm(v)
            if norm == 0.0:
                self.ax_unit_label.setText('WARNING: zero vector -- invalid')
                return
            u = v / norm

            if self.ax_body_frame_chk.isChecked():
                # Try to show the lab-frame equivalent using last known theta.
                # This is only a display hint; the actual theta is read from
                # hardware at compute time.
                try:
                    theta_text = self.ax_theta_label.text()
                    theta_deg = float(theta_text)
                    c, s = np.cos(np.deg2rad(theta_deg)), np.sin(np.deg2rad(theta_deg))
                    # Only rotate the XY (v1, v2) components; v3 is unaffected
                    u_lab = np.array([
                        c * u[0] - s * u[1],
                        s * u[0] + c * u[1],
                        u[2],
                    ])
                    n2 = np.linalg.norm(u_lab)
                    if n2 > 0:
                        u_lab /= n2
                    self.ax_unit_label.setText(
                        f'({u_lab[0]:.4f},  {u_lab[1]:.4f},  {u_lab[2]:.4f})'
                        f'  [body: ({u[0]:.3f}, {u[1]:.3f}, {u[2]:.3f})]')
                except (ValueError, AttributeError):
                    self.ax_unit_label.setText(
                        f'body: ({u[0]:.4f},  {u[1]:.4f},  {u[2]:.4f})'
                        f'  -> lab: read theta first')
            else:
                self.ax_unit_label.setText(
                    f'({u[0]:.4f},  {u[1]:.4f},  {u[2]:.4f})')
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Toolbar actions
    # -----------------------------------------------------------------------
    def setup_actions(self):
        self.add_action('quit', 'Quit', 'close2', 'Quit extension')
        self.add_action('load_file', 'Load File', 'load2',
                        'Load trajectory CSV / TXT file')
        self.add_action('refresh_modules', 'Refresh Modules', 'update2',
                        'Re-resolve all DAQ_Move modules from the dashboard')

    def connect_things(self):
        self.connect_action('quit', self.quit_fun)
        self.connect_action('load_file', self.load_trajectory_file)
        self.connect_action('refresh_modules', self.refresh_modules)

    def value_changed(self, param):
        if param.name() in ('master_name',):
            self.refresh_modules()

    # -----------------------------------------------------------------------
    # Module access helpers
    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    # Module access helpers  (multi-module / master-slave architecture)
    # -----------------------------------------------------------------------
    def refresh_modules(self):
        """Resolve all module names from Settings against dashboard modules.

        Populates self._master_module and self._axis_modules.
        Logs a status line for each resolved / missing name.
        """
        mm = self.dashboard.modules_manager

        # -- Master module ----------------------------------------------------
        master_name = self.settings['module_settings', 'master_name']
        try:
            self._master_module = mm.get_mod_from_name(master_name, 'act')
            if self._master_module is not None:
                self.settings.child('module_settings', 'master_status'
                                    ).setValue(f'OK: {master_name}')
                self.log_message(f'Master module: "{master_name}"')
            else:
                self.settings.child('module_settings', 'master_status'
                                    ).setValue('Not found')
                self.log_message(
                    f'Master module "{master_name}" not found.', level='warning')
        except Exception as e:
            self._master_module = None
            self.settings.child('module_settings', 'master_status'
                                ).setValue('Error')
            self.log_message(f'Error resolving master "{master_name}": {e}',
                             level='error')

        # -- Axis modules -----------------------------------------------------
        all_names = set()
        for group, key in [
            ('rc_modules',  'x_module'),
            ('rc_modules',  'y_module'),
            ('rc_modules',  'theta_module'),
            ('ax_modules',  'x1_module'),
            ('ax_modules',  'x2_module'),
            ('ax_modules',  'x3_module'),
            ('ax_modules',  'rot_module'),
            ('axis_mapping', 'col0_module'),
            ('axis_mapping', 'col1_module'),
            ('axis_mapping', 'col2_module'),
        ]:
            n = self.settings[group, key].strip()
            if n:
                all_names.add(n)

        self._axis_modules = {}
        for name in all_names:
            try:
                mod = mm.get_mod_from_name(name, 'act')
                if mod is not None:
                    self._axis_modules[name] = mod
                    self.log_message(f'  axis module "{name}" OK')
                else:
                    self.log_message(
                        f'  axis module "{name}" not found.', level='warning')
            except Exception as e:
                self.log_message(
                    f'  axis module "{name}" error: {e}', level='error')

        self._refresh_stream_btn()

    def _get_module(self, name: str):
        """Return the DAQ_Move for *name*, or None with a log message."""
        name = name.strip()
        if not name:
            return None
        mod = self._axis_modules.get(name)
        if mod is None:
            self.log_message(
                f'Module "{name}" not loaded. Click Refresh Modules.',
                level='warning')
        return mod

    def _get_wrapper(self):
        """Return the MCS2 wrapper from the master module."""
        if self._master_module is None:
            self.log_message('No master module connected. '
                             'Click Refresh Modules first.', level='warning')
            return None
        wrapper = getattr(self._master_module, 'controller', None)
        if wrapper is None:
            self.log_message(
                'Master DAQ_Move has no .controller attribute.', level='error')
        return wrapper

    def _get_bindings(self):
        try:
            from pymodaq_plugins_smaract.hardware.mcs2 import (
                MCS2_bindings as b)
            return b
        except Exception as e:
            self.log_message(f'Cannot import MCS2_bindings: {e}',
                             level='error')
            return None

    def _module_channel(self, module) -> int:
        """Return the MCS2 controller channel index for a DAQ_Move module.

        The MCS2 plugin exposes move_settings/axis as a list of axis name
        strings, one per physical channel, where each entry equals the title
        of the corresponding DAQ_Move module.  The channel index is simply
        the position of this module's title in that list.

        Example: axis = ['Target X', 'Target Y', 'Target Rotation']
                 module.title = 'Target Y'  ->  channel index = 1
        """
        try:
            name = module.title
            axes = module.settings.child('move_settings', 'multiaxes', 'axis').opts['limits']
            # axes is a list of axis-name strings
            if isinstance(axes, list):
                return axes.index(name)
            elif isinstance(axes, dict):
                # some plugin versions use {name: index} dicts
                return list(axes.keys()).index(name)
            else:
                raise ValueError(f"Unexpected axis value type: {type(axes)}")
        except Exception as e:
            name = getattr(module, 'title', repr(module))
            self.log_message(
                f'Cannot determine channel for "{name}": {e}. Falling back to 0.',
                level='warning')
            return 0

    def _read_module_position(self, module) -> Optional[float]:
        """Read the CURRENT hardware position of a DAQ_Move module in plugin units.

        We deliberately bypass module._current_value (a DataActuator updated
        only via the DAQ_Move signal/thread system) because it goes stale after
        a trajectory stream -- streaming moves hardware directly via the MCS2
        API without going through the PyMoDAQ move machinery.

        From the DAQ_Move source:
          module.controller  ->  the SmarActMCS2Wrapper (set by INI_STAGE)
          wrapper.get_position(ch)  ->  live raw integer from hardware

        Unit conversion -- scale 1e-6 applies to both axis types:
          linear:  raw [pm]    * 1e-6  =  value [um]
          rotary:  raw [n-deg] * 1e-6  =  value [m-deg]
                   (1 n-deg = 1e-9 deg; 1 m-deg = 1e-3 deg  =>  ratio = 1e-6)
        """
        try:
            ch      = self._module_channel(module)
            wrapper = module.controller          # SmarActMCS2Wrapper
            raw     = wrapper.get_position(ch)   # live hardware read (int)
            return float(raw) * 1e-6             # pm->um  or  n-deg->m-deg
        except Exception as e:
            name = getattr(module, 'title', repr(module))
            self.log_message(
                f'Failed to read position from "{name}": {e}', level='error')
            return None

    def _plugin_to_raw(self, module, value: float) -> int:
        """Convert a plugin-unit value to the MCS2 raw integer (base units).

        Hardcoded scale 1e6 for both axis types:
          linear:  value [um]    * 1e6  ->  raw [pm]
          rotary:  value [m-deg] * 1e6  ->  raw [n-deg]
        """
        return int(round(value * 1e6))

    # -----------------------------------------------------------------------
    # Rotation Compensation -- computation
    # -----------------------------------------------------------------------
    def _compute_rotation_comp_trajectory(
            self,
    ) -> Optional[Tuple[np.ndarray, List]]:
        """Read current (X, Y, theta) positions and build the compensation traj.

        Physics:
            xy_cmd(theta) = R(θ0 - theta) * (x0, y0)

        When x0 = y0 = 0 the XY correction is identically zero -- pure rotation.
        All rotation angles are in m-deg (native MCS2 rotary unit).
        XY positions are in um.

        The channel_map entries are (col_index, DAQ_Move_module) so that
        _build_frames_from_channel_map can use module.settings to get both
        the controller channel and the base resolution for raw encoding.

        Returns
        -------
        (traj, channel_map) on success, None on failure.
        traj        : (n_frames, 3) array -- cols [X (um), Y (um), θ (m-deg)]
        channel_map : [(col, module), ...]
        Also updates read-back labels and trajectory-statistics labels.
        """
        # Resolve modules from the dock's line-edit widgets
        x_mod  = self._get_module(self.rc_x_mod_edit.text())
        y_mod  = self._get_module(self.rc_y_mod_edit.text())
        th_mod = self._get_module(self.rc_th_mod_edit.text())

        if any(m is None for m in (x_mod, y_mod, th_mod)):
            self.log_message(
                'X, Y, and Theta modules must all be resolved for rotation '
                'compensation. Check module names and click Refresh Modules.',
                level='error')
            return None

        # Read current positions via live hardware query (bypasses stale cache)
        x0  = self._read_module_position(x_mod)
        y0  = self._read_module_position(y_mod)
        th0 = self._read_module_position(th_mod)  # m-deg
        if any(v is None for v in (x0, y0, th0)):
            return None

        n_frames = self.rc_nframes_spin.value()

        # Resolve absolute vs relative target angle (both in m-deg)
        theta_input = self.rc_theta_end_spin.value()
        if self.rc_abs_radio.isChecked():
            theta_end_mdeg = theta_input
        else:
            theta_end_mdeg = th0 + theta_input

        try:
            x_traj, y_traj, theta_traj = build_rotation_compensation_trajectory(
                x0=x0,
                y0=y0,
                θ0_mdeg=th0,
                theta_end_mdeg=theta_end_mdeg,
                n_frames=n_frames,
            )
        except ValueError as e:
            self.log_message(str(e), level='error')
            return None

        # Stack into (n_frames, 3): col0=X, col1=Y, col2=theta
        traj = np.column_stack([x_traj, y_traj, theta_traj])

        # channel_map: (col_index, module) -- module carries ch + base_resolution
        channel_map = [(0, x_mod), (1, y_mod), (2, th_mod)]

        # Update position read-back labels
        self.rc_x0_label.setText(f'{x0:.3f}')
        self.rc_y0_label.setText(f'{y0:.3f}')
        self.rc_th0_label.setText(f'{th0:.1f}')
        self.rc_xt_label.setText(f'{float(x_traj[-1]):.3f}')
        self.rc_yt_label.setText(f'{float(y_traj[-1]):.3f}')
        self.rc_tht_label.setText(f'{theta_end_mdeg:.1f}')

        # Trajectory statistics
        xy0_r = np.sqrt(x0**2 + y0**2)
        phi0_mdeg = np.degrees(np.arctan2(y0, x0)) * 1000.0

        self.rc_stat_dx.setText(f'{float(np.max(np.abs(x_traj - x0))):.3f}')
        self.rc_stat_dy.setText(f'{float(np.max(np.abs(y_traj - y0))):.3f}')
        self.rc_stat_r.setText(f'{xy0_r:.3f}')
        self.rc_stat_phi.setText(f'{phi0_mdeg:.1f}')

        x_name  = self.rc_x_mod_edit.text().strip()
        y_name  = self.rc_y_mod_edit.text().strip()
        th_name = self.rc_th_mod_edit.text().strip()
        self.log_message(
            f'Rotation comp: {x_name} x0={x0:.3f} um, '
            f'{y_name} y0={y0:.3f} um, '
            f'{th_name} theta0={th0:.1f} m-deg -> theta_end={theta_end_mdeg:.1f} m-deg, '
            f'|xy0|={xy0_r:.3f} um, {n_frames} frames.')

        return traj, channel_map

    def preview_rotation_comp(self):
        """Compute the rotation-compensation trajectory and display it in the
        preview table without streaming."""
        result = self._compute_rotation_comp_trajectory()
        if result is None:
            return
        traj, channel_map = result
        self._trajectory = traj
        self._pending_channel_map = channel_map
        self._populate_table(traj)
        self._refresh_stream_btn()
        self.log_message(
            'Rotation-comp trajectory in preview. '
            'Click "Start Streaming" or "Generate & Stream" to execute.')

    def generate_and_stream_rotation_comp(self):
        """Compute and immediately stream the rotation-compensation trajectory."""
        if self._master_module is None:
            self.log_message('No module connected.', level='error')
            return
        if self._stream_thread and self._stream_thread.is_alive():
            self.log_message('Streaming already in progress.', level='warning')
            return
        result = self._compute_rotation_comp_trajectory()
        if result is None:
            return
        traj, channel_map = result
        self._trajectory = traj
        self._populate_table(traj)
        try:
            frames = self._build_frames_from_channel_map(traj, channel_map)
        except ValueError as e:
            self.log_message(str(e), level='error')
            return
        self._start_stream_with_frames(frames)

    # -----------------------------------------------------------------------
    # Arbitrary axis move
    # -----------------------------------------------------------------------
    def _compute_axis_move_trajectory(
            self,
    ) -> Optional[Tuple[np.ndarray, List]]:
        """Read (X1, X2, X3) positions, build axis-move trajectory.

        When the "Direction in XY body frame" checkbox is ticked the user-
        supplied direction vector v_body is expressed in the rotating XY stage
        frame.  The required lab-frame direction is:

            v_lab = R(θ0) * v_body

        where θ0 is the current rotation-stage angle and R is the 2-D
        rotation matrix acting on the (v1, v2) components.  v3 is not
        affected by the rotation (it is orthogonal to the rotation axis).

        The channel_map entries are (col_index, module) so that
        _build_frames_from_channel_map can convert plugin units -> raw.
        """
        # Resolve axis modules from the dock's line-edit widgets
        mod_names = [
            self.ax_x1_mod_edit.text().strip(),
            self.ax_x2_mod_edit.text().strip(),
            self.ax_x3_mod_edit.text().strip(),
        ]
        modules = [self._get_module(n) if n else None for n in mod_names]
        active_mask = np.array([m is not None for m in modules])

        if not active_mask.any():
            self.log_message(
                'All axis module fields are empty. '
                'Enter at least one module name.', level='error')
            return None

        v_user = np.array([self.ax_v1_spin.value(),
                           self.ax_v2_spin.value(),
                           self.ax_v3_spin.value()])

        # --- Body-frame -> lab-frame direction conversion ---
        body_frame = self.ax_body_frame_chk.isChecked()
        if body_frame:
            rot_name = self.ax_rot_mod_edit.text().strip()
            if not rot_name:
                self.log_message(
                    'Body-frame mode requires a rotation module name. '
                    'Fill the Rotation module field or uncheck body-frame mode.',
                    level='error')
                return None
            rot_mod = self._get_module(rot_name)
            if rot_mod is None:
                return None
            th_mdeg = self._read_module_position(rot_mod)
            if th_mdeg is None:
                return None
            θ0_deg = th_mdeg / 1000.0
            self.ax_theta_label.setText(f'{th_mdeg:.1f}')

            c = np.cos(np.deg2rad(θ0_deg))
            s = np.sin(np.deg2rad(θ0_deg))
            v_full = np.array([
                c * v_user[0] - s * v_user[1],
                s * v_user[0] + c * v_user[1],
                v_user[2],
            ])
            self.log_message(
                f'Body-frame mode: theta0={th_mdeg:.1f} m-deg ({θ0_deg:.4f}deg), '
                f'v_body=({v_user[0]:.4f},{v_user[1]:.4f},{v_user[2]:.4f}) -> '
                f'v_lab=({v_full[0]:.4f},{v_full[1]:.4f},{v_full[2]:.4f})')
            self._update_axis_norm_label()
        else:
            v_full = v_user
            self.ax_theta_label.setText('--')

        # --- Read stage positions (live hardware query) ---
        cur_labels = [self.ax_x1_label, self.ax_x2_label, self.ax_x3_label]
        tgt_labels = [self.ax_t1_label,  self.ax_t2_label,  self.ax_t3_label]
        positions_full = np.zeros(3)
        for i, mod in enumerate(modules):
            if mod is not None:
                pos = self._read_module_position(mod)
                if pos is None:
                    return None
                positions_full[i] = pos
                cur_labels[i].setText(f'{pos:.4f}')
            else:
                cur_labels[i].setText('--')

        n_frames = self.ax_nframes_spin.value()
        distance = self.ax_dist_spin.value()

        try:
            traj = build_axis_move_trajectory(
                positions_full, v_full, distance, n_frames, active_mask)
        except ValueError as e:
            self.log_message(str(e), level='error')
            return None

        # Build channel_map as (col_index, module) and update target labels
        channel_map = []
        col = 0
        axis_names = ['X1', 'X2', 'X3']
        summary_parts = []
        for i, mod in enumerate(modules):
            if mod is not None:
                tgt_labels[i].setText(f'{float(traj[-1, col]):.4f}')
                channel_map.append((col, mod))
                summary_parts.append(
                    f'{mod_names[i]}({axis_names[i]}): '
                    f'{positions_full[i]:.3f} -> {float(traj[-1, col]):.3f}')
                col += 1
            else:
                tgt_labels[i].setText('--')

        norm = np.linalg.norm(v_full)
        unit = v_full / norm if norm > 0 else v_full
        frame_label = 'body' if body_frame else 'lab'
        self.log_message(
            f'Axis move ({frame_label} frame): d={distance}, '
            f'unit=({unit[0]:.3f},{unit[1]:.3f},{unit[2]:.3f}), '
            f'{n_frames} frames.  ' + ',  '.join(summary_parts))

        return traj, channel_map

    def preview_axis_move(self):
        result = self._compute_axis_move_trajectory()
        if result is None:
            return
        traj, channel_map = result
        self._trajectory = traj
        self._pending_channel_map = channel_map
        self._populate_table(traj)
        self._refresh_stream_btn()
        self.log_message(
            'Axis-move trajectory in preview. '
            'Click "Start Streaming" or "Generate & Stream" to execute.')

    def generate_and_stream_axis_move(self):
        if self._master_module is None:
            self.log_message('No module connected.', level='error')
            return
        if self._stream_thread and self._stream_thread.is_alive():
            self.log_message('Streaming already in progress.', level='warning')
            return
        result = self._compute_axis_move_trajectory()
        if result is None:
            return
        traj, channel_map = result
        self._trajectory = traj
        self._populate_table(traj)
        try:
            frames = self._build_frames_from_channel_map(traj, channel_map)
        except ValueError as e:
            self.log_message(str(e), level='error')
            return
        self._start_stream_with_frames(frames)

    # -----------------------------------------------------------------------
    # File-based trajectory loading
    # -----------------------------------------------------------------------
    def _delimiter_char(self) -> str:
        return {'comma': ',', 'tab': '\t', 'space': ' ',
                'semicolon': ';'}.get(
            self.settings['traj_file', 'delimiter'], ',')

    def load_trajectory_file(self):
        start_dir = self._get_last_dir()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.mainwindow, 'Load Trajectory File', start_dir,
            'Data Files (*.csv *.txt *.tsv);;All Files (*)')
        if not path:
            return
        self._set_last_dir(path)
        self._parse_trajectory(path)

    def _parse_trajectory(self, path: str):
        try:
            data = np.loadtxt(
                path,
                delimiter=self._delimiter_char(),
                skiprows=self.settings['traj_file', 'skip_rows'],
                ndmin=2)
        except Exception as e:
            self.log_message(f'Failed to parse file: {e}', level='error')
            return

        self._trajectory = data
        self._pending_channel_map = None
        n_frames, n_cols = data.shape
        rate = self.settings['stream_settings', 'stream_rate']

        self.settings.child('traj_file', 'file_path').setValue(
            Path(path).name)
        self.settings.child('traj_file', 'n_frames').setValue(n_frames)
        self.settings.child('traj_file', 'duration').setValue(
            round(n_frames / rate, 4))

        self.log_message(
            f'Loaded "{Path(path).name}": {n_frames} frames x {n_cols} cols.  '
            f'Duration at {rate} Hz: {n_frames / rate:.3f} s')

        self._populate_table(data)
        self._refresh_stream_btn()

    def _populate_table(self, data: np.ndarray):
        max_rows = 500
        n_rows = min(len(data), max_rows)
        n_cols = data.shape[1]
        self.traj_table.setColumnCount(n_cols)
        self.traj_table.setRowCount(n_rows)
        self.traj_table.setHorizontalHeaderLabels(
            [f'Col {i}' for i in range(n_cols)])
        for r in range(n_rows):
            for c in range(n_cols):
                item = QtWidgets.QTableWidgetItem(f'{data[r, c]:.6g}')
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.traj_table.setItem(r, c, item)
        if len(data) > max_rows:
            self.log_message(
                f'Preview capped at {max_rows} of {len(data)} frames.',
                level='warning')

    # -----------------------------------------------------------------------
    # Frame building
    # -----------------------------------------------------------------------
    def _build_frames(self) -> List[bytes]:
        """File-mode frame builder: uses the Axis Mapping settings.

        The channel_map for file mode is [(col_index, module), ...] where
        the module is used to get both the channel number and the base
        resolution for unit->raw conversion.
        """
        data = self._trajectory
        n_frames, n_cols = data.shape
        col_keys = ['col0_module', 'col1_module', 'col2_module']
        mapping = []
        for i in range(min(n_cols, 3)):
            name = self.settings['axis_mapping', col_keys[i]].strip()
            if not name:
                continue
            mod = self._get_module(name)
            if mod is not None:
                mapping.append((i, mod))
        if not mapping:
            raise ValueError(
                'No valid column->module mappings. '
                'Set module names in Axis Mapping and click Refresh Modules.')
        return self._build_frames_from_channel_map(data, mapping)

    def _build_frames_from_channel_map(
            self,
            data: np.ndarray,
            channel_map: List[Tuple],
    ) -> List[bytes]:
        """Build binary frames from a trajectory array and an explicit map.

        Parameters
        ----------
        data        : (n_frames, n_cols) array in plugin units
                      (um for linear, m-deg for rotary)
        channel_map : list of (col_index, DAQ_Move_module) tuples.
                      The module is used to derive the controller channel
                      index and to convert plugin units -> raw base units.

        Each frame entry is (channel_index: int, raw_position: int) packed
        as 1-byte uint8 + 8-byte int64 little-endian by encode_frame().
        """
        if not channel_map:
            raise ValueError('channel_map is empty -- nothing to stream.')
        frames = []
        for row in data:
            ch_pos = []
            for col, mod in channel_map:
                ch  = self._module_channel(mod)
                raw = self._plugin_to_raw(mod, float(row[col]))
                ch_pos.append((ch, raw))
            frames.append(encode_frame(ch_pos))
        return frames

    # -----------------------------------------------------------------------
    # Streaming
    # -----------------------------------------------------------------------
    def _refresh_stream_btn(self):
        ready = (self._master_module is not None and
                 self._trajectory is not None and
                 not (self._stream_thread and
                      self._stream_thread.is_alive()))
        self.stream_btn.setEnabled(ready)

    def start_streaming(self):
        if self._master_module is None:
            self.log_message('No master module connected.', level='error')
            return
        if self._trajectory is None:
            self.log_message('No trajectory loaded.', level='error')
            return
        if self._stream_thread and self._stream_thread.is_alive():
            self.log_message('Streaming already in progress.', level='warning')
            return
        try:
            if self._pending_channel_map is not None:
                frames = self._build_frames_from_channel_map(
                    self._trajectory, self._pending_channel_map)
            else:
                frames = self._build_frames()
        except ValueError as e:
            self.log_message(str(e), level='error')
            return
        self._start_stream_with_frames(frames)

    def _start_stream_with_frames(self, frames: List[bytes]):
        self._abort_event.clear()
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(len(frames))
        self.frame_label.setText(f'Frame: 0 / {len(frames)}')

        for btn in (self.stream_btn, self.load_btn,
                    self.ax_stream_btn, self.ax_preview_btn,
                    self.rc_stream_btn, self.rc_preview_btn):
            btn.setEnabled(False)
        self.abort_btn.setEnabled(True)

        self._stream_thread = threading.Thread(
            target=self._stream_worker, args=(frames,), daemon=True)
        self._stream_thread.start()

    def _stream_worker(self, frames: List[bytes]):
        """Background thread: pre-move to start position, then stream all frames.

        The MCS2 Programmer's Guide (S2.15.2) is explicit: all channels that
        participate in the stream MUST be at the first frame's position before
        SA_CTL_OpenStream is called.  If they aren't, the controller computes
        an impossibly high velocity for that first step and silently enters a
        holding/error state -- this is exactly the "moves to a position then
        does nothing" symptom.

        Sequence implemented here
        -------------------------
        1. Set STREAM_BASE_RATE and STREAM_OPTIONS.
        2. Decode the first frame to extract (channel, raw_position) pairs.
        3. For each channel: set MOVE_MODE = CL_ABSOLUTE and SA_CTL_Move.
        4. Wait for SA_CTL_EVENT_MOVEMENT_FINISHED (0x0001) on every channel.
        5. SA_CTL_OpenStream -> SA_CTL_StreamFrame x N -> SA_CTL_CloseStream.
        6. Wait for SA_CTL_EVENT_STREAM_FINISHED (0x8000) to confirm the
           hardware has actually executed all buffered frames.
        """
        b = self._get_bindings()
        if b is None:
            self._on_stream_done(success=False)
            return
        wrapper = self._get_wrapper()
        if wrapper is None:
            self._on_stream_done(success=False)
            return

        dh = wrapper.controller_index
        n  = len(frames)

        # -- 1. Stream properties --------------------------------------------
        rate = self.settings['stream_settings', 'stream_rate']
        try:
            b.SetProperty_i32(dh, 0, b.Property.STREAM_BASE_RATE, rate)
        except b.Error as e:
            self._log_signal.emit(
                f'Failed to set STREAM_BASE_RATE: {e}', 'error')
            self._on_stream_done(success=False)
            return

        no_interp = self.settings['stream_settings', 'no_interpolation']
        opts = int(b.StreamOption.INTERPOLATION_DIS) if no_interp else 0
        try:
            b.SetProperty_i32(dh, 0, b.Property.STREAM_OPTIONS, opts)
        except b.Error as e:
            self._log_signal.emit(
                f'Could not set STREAM_OPTIONS: {e}', 'warning')

        # -- 2. Decode first frame -> (channel_index, raw_position) ----------
        # Each tuple in a frame = 1-byte channel + 8-byte int64 little-endian.
        TUPLE = 9
        first = frames[0]
        start_positions = []          # list of (ch, raw_pos)
        for t in range(len(first) // TUPLE):
            off = t * TUPLE
            ch  = struct.unpack_from('<B', first, off)[0]
            pos = struct.unpack_from('<q', first, off + 1)[0]
            start_positions.append((ch, pos))

        # -- 3. Move every participating channel to the first-frame position -
        channels_pending = set()
        self._log_signal.emit(
            f'Pre-moving {len(start_positions)} channel(s) to trajectory '
            f'start position...', 'info')
        try:
            for ch, raw_pos in start_positions:
                b.SetProperty_i32(
                    dh, ch, b.Property.MOVE_MODE,
                    int(b.MoveMode.CL_ABSOLUTE))
                b.Move(dh, ch, raw_pos, 0)
                channels_pending.add(ch)
        except b.Error as e:
            self._log_signal.emit(
                f'Pre-move command failed on ch{ch}: {e}', 'error')
            self._on_stream_done(success=False)
            return

        # -- 4. Wait for MOVEMENT_FINISHED on every channel ------------------
        # MOVEMENT_FINISHED = event type 0x0001, idx = channel
        PRE_MOVE_TIMEOUT_MS = 60_000   # 60 s -- increase for slow / long moves
        try:
            while channels_pending:
                if self._abort_event.is_set():
                    for ch in channels_pending:
                        try:
                            b.Stop(dh, ch, 0)
                        except Exception:
                            pass
                    self._log_signal.emit(
                        'Pre-move aborted by user.', 'warning')
                    self._on_stream_done(success=False)
                    return
                evnt = b.WaitForEvent(dh, PRE_MOVE_TIMEOUT_MS)
                if evnt.type == 0x0001:          # MOVEMENT_FINISHED
                    ch_done = evnt.idx
                    if ch_done in channels_pending:
                        if evnt.i32 != 0:
                            self._log_signal.emit(
                                f'Pre-move ch{ch_done} error '
                                f'0x{evnt.i32:08X} -- aborting.', 'error')
                            self._on_stream_done(success=False)
                            return
                        channels_pending.discard(ch_done)
                        self._log_signal.emit(
                            f'  ch{ch_done} at start position.', 'info')
        except b.Error as e:
            self._log_signal.emit(
                f'Error waiting for pre-move: {e}', 'error')
            self._on_stream_done(success=False)
            return

        self._log_signal.emit('All channels at start. Opening stream...', 'info')

        # -- 5. OpenStream -> pump frames -> CloseStream -----------------------
        try:
            sHandle = b.OpenStream(dh, 0)
        except b.Error as e:
            self._log_signal.emit(f'OpenStream failed: {e}', 'error')
            self._on_stream_done(success=False)
            return

        self._log_signal.emit(
            f'Streaming {n} frames at {rate} Hz '
            f'(interpolation {"off" if no_interp else "on"})...', 'info')

        aborted = False
        try:
            for i, frame in enumerate(frames):
                if self._abort_event.is_set():
                    aborted = True
                    break
                b.StreamFrame(dh, sHandle, frame)
                if i % 50 == 0 or i == n - 1:
                    self._qt_update_progress(i + 1, n)
            if aborted:
                b.AbortStream(dh, sHandle)
                self._log_signal.emit('Stream ABORTED by user.', 'warning')
            else:
                b.CloseStream(dh, sHandle)
                self._log_signal.emit(
                    f'All {n} frames sent -- waiting for hardware to finish...',
                    'info')
        except b.Error as e:
            self._log_signal.emit(f'Streaming error: {e}', 'error')
            try:
                b.AbortStream(dh, sHandle)
            except Exception:
                pass
            self._on_stream_done(success=False)
            return

        # -- 6. Wait for STREAM_FINISHED (0x8000) ----------------------------
        # The buffer on the controller still holds frames after CloseStream
        # returns; we must wait for STREAM_FINISHED before re-enabling the UI
        # or allowing a new stream, otherwise the controller is still busy.
        if not aborted:
            # generous timeout: actual duration + 10 s headroom
            STREAM_FINISH_MS = max(30_000, int(n / rate * 1000) + 10_000)
            try:
                while True:
                    if self._abort_event.is_set():
                        break
                    evnt = b.WaitForEvent(dh, STREAM_FINISH_MS)
                    if evnt.type == 0x8000:      # STREAM_FINISHED
                        rc = evnt.i32 & 0xFFFF   # lower 16 bits = result code
                        if rc == 0:
                            self._log_signal.emit(
                                'Stream finished successfully.', 'info')
                        else:
                            self._log_signal.emit(
                                f'Stream finished with error 0x{rc:04X}.',
                                'warning')
                        break
                    # ignore unrelated events (movement finished on other channels
                    # triggered by the stream, etc.)
            except b.Error as e:
                # A timeout or cancel here is non-fatal -- all frames were sent.
                self._log_signal.emit(
                    f'Note: WaitForEvent after CloseStream returned: {e}',
                    'warning')

        self._on_stream_done(success=not aborted)

    def _qt_update_progress(self, current: int, total: int):
        QtCore.QMetaObject.invokeMethod(
            self.progress_bar, 'setValue',
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(int, current))
        QtCore.QMetaObject.invokeMethod(
            self.frame_label, 'setText',
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(str, f'Frame: {current} / {total}'))

    def _on_stream_done(self, success: bool):
        QtCore.QMetaObject.invokeMethod(
            self, '_reset_stream_ui', QtCore.Qt.QueuedConnection)

    @QtCore.Slot()
    def _reset_stream_ui(self):
        self.abort_btn.setEnabled(False)
        for btn in (self.load_btn,
                    self.ax_stream_btn, self.ax_preview_btn,
                    self.rc_stream_btn, self.rc_preview_btn):
            btn.setEnabled(True)
        self._refresh_stream_btn()

    def abort_streaming(self):
        self._abort_event.set()
        self.log_message('Abort requested...', level='warning')

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------
    @QtCore.Slot(str, str)
    def _on_log_signal(self, message: str, level: str):
        self.log_message(message, level=level)

    def log_message(self, message: str, level: str = 'info'):
        import datetime
        ts = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        colour = {'error': '#f87171', 'warning': '#fbbf24'}.get(
            level, '#86efac')
        prefix = {'error': 'ERR', 'warning': 'WRN'}.get(level, 'INF')
        self.status_text.append(
            f'<span style="color:{colour}">'
            f'[{ts}] {prefix}: {message}</span>')
        getattr(logger, level if level in ('error', 'warning') else 'info')(
            message)

    # -----------------------------------------------------------------------
    # Misc helpers
    # -----------------------------------------------------------------------
    @staticmethod
    def _btn_style(col1: str, col2: str) -> str:
        return (
            f'QPushButton {{'
            f'  font-size: 13pt; font-weight: bold;'
            f'  border: 2px solid {col2}; border-radius: 8px;'
            f'  background-color: qlineargradient('
            f'    x1:0,y1:0,x2:0,y2:1,stop:0 {col1},stop:1 {col2});'
            f'  color: white; padding: 8px;'
            f'}}'
            f'QPushButton:hover {{ background-color: {col1}; }}'
            f'QPushButton:pressed {{ background-color: {col2}; }}'
            f'QPushButton:disabled {{'
            f'  background-color: #555; color: #999;'
            f'  border: 2px solid #444; }}'
        )

    def _get_last_dir(self) -> str:
        try:
            d = traj_config[('last_directory',)]
            if d and Path(d).exists():
                return d
        except Exception:
            pass
        return str(Path.home())

    def _set_last_dir(self, filepath: str):
        traj_config[('last_directory',)] = str(Path(filepath).parent)
        traj_config.save()

    def quit_fun(self):
        self.mainwindow.close()


def main():
    from pymodaq_gui.utils.utils import mkQApp
    from pymodaq.utils.gui_utils.loader_utils import load_dashboard_with_preset

    app = mkQApp('MCS2TrajectoryStreaming')
    preset_file_name = config_pymodaq(
        'presets', 'default_preset_for_MCS2_trajectory')
    dashboard, extension, win = load_dashboard_with_preset(
        preset_file_name, EXTENSION_NAME)
    app.exec()
    return dashboard, extension, win


if __name__ == '__main__':
    main()