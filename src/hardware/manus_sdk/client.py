#!/usr/bin/env python3
"""Manus SDK Python ctypes client used by the console Manus runtime.

- 连接 Manus Core（Local / Remote / Integrated）并订阅原始骨骼流，
  坐标系采用 OpenVR 约定（RH、Y-up、-Z-forward、米制）
- 以 UDP JSON 将 manus_hand_skeleton 和 glove status 推送到
  MANUS_OUT_PORT（默认 9001）
- Vive 不进入 MANUS Core；本进程只处理手套骨骼、设置和官方校准
环境变量：
  MANUS_CONNECTION_MODE   Linux 默认 integrated；其他平台默认 local
                          可选 local | remote | integrated
  MANUS_OUT_HOST          127.0.0.1
  MANUS_OUT_PORT          9001
  MANUS_COMMAND_PORT      9003
  MANUS_SEND_RATE         60 (Hz)
  MANUS_SDK_ROOT          SDK include/lib 根目录
  MANUS_SDK_LIBRARY       手动指定 SDK 动态库路径
  MANUS_SDK_DLL           MANUS_SDK_LIBRARY 的旧版兼容别名
"""

import ctypes as C
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import sys
import threading
import time
from ctypes import (
    CFUNCTYPE, POINTER, Structure, byref,
    c_bool, c_char, c_float, c_int32, c_ubyte, c_uint32, c_uint64,
)


# ---------------------------------------------------------------------------
# Constants (mirror ManusSDKTypes.h #defines)
# ---------------------------------------------------------------------------
MAX_NUM_CHARS_IN_HOST_NAME = 256
MAX_NUM_CHARS_IN_IP_ADDRESS = 40
MAX_NUM_CHARS_IN_VERSION = 16
MAX_NUM_CHARS_IN_CALIBRATION_TITLE = 64
MAX_NUM_CHARS_IN_CALIBRATION_DESCRIPTION = 256
MAX_NUM_CHARS_IN_LICENSE_TYPE = 64
MAX_NUMBER_OF_DONGLES = 16
MAX_NUMBER_OF_GLOVES = MAX_NUMBER_OF_DONGLES * 2
MAX_NUM_IMUS_ON_GLOVE = 6


# ---------------------------------------------------------------------------
# Enum values (from ManusSDKTypes.h)
# ---------------------------------------------------------------------------
class SDKReturnCode:
    Success = 0
    NotConnected = 11


class Side:
    Invalid = 0
    Left = 1
    Right = 2
    Center = 3


class ChainType:
    _NAMES = {
        0: 'unknown', 1: 'arm', 2: 'leg', 3: 'neck', 4: 'spine',
        5: 'thumb', 6: 'index', 7: 'middle', 8: 'ring', 9: 'pinky',
        10: 'pelvis', 11: 'head', 12: 'shoulder', 13: 'hand',
        14: 'foot', 15: 'toe',
    }


class FingerJointType:
    _NAMES = {
        0: '', 1: 'metacarpal', 2: 'proximal',
        3: 'intermediate', 4: 'distal', 5: 'tip',
    }


class AxisPolarity:
    PositiveY = 5


class AxisView:
    ZFromViewer = 1


class AxisDirection:
    Backward = 1
    Up = 4
    Right = 5


class HandMotion:
    None_ = 0
    IMU = 1
    Tracker = 2
    TrackerRotationOnly = 3
    Auto = 4


class DeviceFamilyType:
    Unknown = 0
    Prime1 = 1
    Prime2 = 2
    PrimeX = 3
    Metaglove = 4
    Prime3 = 5
    Virtual = 6
    MetaglovePro = 7
    MetagloveProPrecision = 8
    MetagloveProHaptics = 9
    MetagloveProPrecisionHaptics = 10

    _NAMES = {
        Unknown: 'Unknown',
        Prime1: 'Prime 1',
        Prime2: 'Prime 2',
        PrimeX: 'Prime X',
        Metaglove: 'Metaglove / Quantum',
        Prime3: 'Prime 3',
        Virtual: 'Virtual',
        MetaglovePro: 'Metaglove Pro',
        MetagloveProPrecision: 'Metaglove Pro Precision',
        MetagloveProHaptics: 'Metaglove Pro Haptics',
        MetagloveProPrecisionHaptics: 'Metaglove Pro Precision Haptics',
    }


SIDE_STR = {Side.Invalid: 'x', Side.Left: 'l', Side.Right: 'r', Side.Center: 'c'}
SIDE_HAND = {Side.Left: 'left', Side.Right: 'right'}

# ---------------------------------------------------------------------------
# Struct definitions (mirror ManusSDKTypes.h exactly — field order matters!)
# ---------------------------------------------------------------------------
class ManusVec3(Structure):
    _fields_ = [('x', c_float), ('y', c_float), ('z', c_float)]


class ManusQuaternion(Structure):
    # Note: w first (matches C header)
    _fields_ = [('w', c_float), ('x', c_float), ('y', c_float), ('z', c_float)]


class ManusTransform(Structure):
    _fields_ = [
        ('position', ManusVec3),
        ('rotation', ManusQuaternion),
        ('scale', ManusVec3),
    ]


class ManusTimestamp(Structure):
    _fields_ = [('time', c_uint64)]


class SkeletonNode(Structure):
    _fields_ = [
        ('id', c_uint32),
        ('transform', ManusTransform),
    ]


class NodeInfo(Structure):
    _fields_ = [
        ('nodeId', c_uint32),
        ('parentId', c_uint32),
        ('chainType', c_int32),
        ('side', c_int32),
        ('fingerJointType', c_int32),
    ]


class RawSkeletonInfo(Structure):
    _fields_ = [
        ('gloveId', c_uint32),
        ('nodesCount', c_uint32),
        ('publishTime', ManusTimestamp),
    ]


class SkeletonStreamInfo(Structure):
    _fields_ = [
        ('publishTime', ManusTimestamp),
        ('skeletonsCount', c_uint32),
    ]


class CoordinateSystemVUH(Structure):
    _fields_ = [
        ('view', c_int32),
        ('up', c_int32),
        ('handedness', c_int32),
        ('unitScale', c_float),
    ]


class CoordinateSystemDirection(Structure):
    _fields_ = [
        ('x', c_int32),
        ('y', c_int32),
        ('z', c_int32),
        ('unitScale', c_float),
    ]


class Version(Structure):
    _fields_ = [
        ('major', c_uint32),
        ('minor', c_uint32),
        ('patch', c_uint32),
        ('label', c_char * MAX_NUM_CHARS_IN_VERSION),
        ('sha', c_char * MAX_NUM_CHARS_IN_VERSION),
        ('tag', c_char * MAX_NUM_CHARS_IN_VERSION),
    ]


class IMUCalibrationInfo(Structure):
    _fields_ = [
        ('mag', c_uint32),
        ('acc', c_uint32),
        ('gyr', c_uint32),
        ('sys', c_uint32),
    ]


class DongleLandscapeData(Structure):
    _fields_ = [
        ('id', c_uint32),
        ('classType', c_int32),
        ('familyType', c_int32),
        ('isHaptics', c_bool),
        ('hardwareVersion', Version),
        ('firmwareVersion', Version),
        ('firmwareTimestamp', ManusTimestamp),
        ('chargingState', c_uint32),
        ('channel', c_int32),
        ('updateStatus', c_int32),
        ('licenseType', c_char * MAX_NUM_CHARS_IN_LICENSE_TYPE),
        ('lastSeen', ManusTimestamp),
        ('leftGloveID', c_uint32),
        ('rightGloveID', c_uint32),
        ('licenseLevel', c_int32),
        ('licenseExpiration', ManusTimestamp),
        ('licenseMaxNumberOfGlovePairs', c_uint32),
        ('netDeviceID', c_uint32),
    ]


class GloveLandscapeData(Structure):
    _fields_ = [
        ('id', c_uint32),
        ('classType', c_int32),
        ('familyType', c_int32),
        ('side', c_int32),
        ('isHaptics', c_bool),
        ('pairedState', c_int32),
        ('dongleID', c_uint32),
        ('hardwareVersion', Version),
        ('firmwareVersion', Version),
        ('firmwareTimestamp', ManusTimestamp),
        ('updateStatus', c_int32),
        ('batteryPercentage', c_uint32),
        ('transmissionStrength', c_int32),
        ('iMUCalibrationInfo', IMUCalibrationInfo * MAX_NUM_IMUS_ON_GLOVE),
        ('lastSeen', ManusTimestamp),
        ('excluded', c_bool),
        ('netDeviceID', c_uint32),
    ]


class DeviceLandscape(Structure):
    _fields_ = [
        ('dongles', DongleLandscapeData * MAX_NUMBER_OF_DONGLES),
        ('dongleCount', c_uint32),
        ('gloves', GloveLandscapeData * MAX_NUMBER_OF_GLOVES),
        ('gloveCount', c_uint32),
    ]


class LandscapePrefix(Structure):
    """Leading portion of Landscape needed for glove battery state."""

    _fields_ = [('gloveDevices', DeviceLandscape)]


class ManusVersion(Structure):
    _fields_ = [('versionInfo', c_char * MAX_NUM_CHARS_IN_VERSION)]


class ManusHost(Structure):
    _fields_ = [
        ('hostName', c_char * MAX_NUM_CHARS_IN_HOST_NAME),
        ('ipAddress', c_char * MAX_NUM_CHARS_IN_IP_ADDRESS),
        ('manusCoreVersion', Version),
    ]


class GloveCalibrationArgs(Structure):
    _fields_ = [('gloveId', c_uint32)]


class GloveCalibrationStepArgs(Structure):
    _fields_ = [('gloveId', c_uint32), ('stepIndex', c_uint32)]


class GloveCalibrationStepData(Structure):
    _fields_ = [
        ('index', c_uint32),
        ('title', c_char * MAX_NUM_CHARS_IN_CALIBRATION_TITLE),
        ('description', c_char * MAX_NUM_CHARS_IN_CALIBRATION_DESCRIPTION),
        ('time', c_float),
    ]


# Stream callbacks: void (*)(const StreamInfo*)
RawSkeletonStreamCallback = CFUNCTYPE(None, POINTER(SkeletonStreamInfo))
LandscapeStreamCallback = CFUNCTYPE(None, POINTER(LandscapePrefix))
LoggingCallback = CFUNCTYPE(None, c_int32, C.c_char_p, c_uint32)


# ---------------------------------------------------------------------------
# Logging (stdout parsed by Node backend's manus-sdk-service.js)
# ---------------------------------------------------------------------------
def log_info(msg):  print(f'INFO: {msg}',  flush=True)
def log_warn(msg):  print(f'WARN: {msg}',  file=sys.stderr, flush=True)
def log_error(msg): print(f'ERROR: {msg}', file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# SDK library loading
# ---------------------------------------------------------------------------
def find_sdk_library(connection_mode=None):
    env_path = os.environ.get('MANUS_SDK_LIBRARY') or os.environ.get('MANUS_SDK_DLL')
    if env_path and os.path.isfile(env_path):
        return os.path.abspath(env_path)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
    linux_sdk_root = os.path.join(repo_root, 'third-party', 'ManusSDK_v3.1.1_linux')
    windows_sdk_root = os.path.join(repo_root, 'third-party', 'ManusSDK_v3.1.1_win')
    configured_root = os.environ.get('MANUS_SDK_ROOT', '').strip()
    sdk_roots = [
        configured_root,
        linux_sdk_root,
    ]
    mode = (connection_mode or '').strip().lower()
    if sys.platform.startswith('linux'):
        library_name = 'libManusSDK_Integrated.so' if mode == 'integrated' else 'libManusSDK.so'
        candidates = []
        for root in sdk_roots:
            if not root:
                continue
            candidates.extend([
                os.path.join(root, 'lib', library_name),
                os.path.join(root, 'SDKClient_Linux', 'ManusSDK', 'lib', library_name),
            ])
    elif sys.platform == 'win32':
        candidates = []
        for root in [configured_root, windows_sdk_root]:
            if root:
                candidates.append(os.path.join(root, 'lib', 'ManusSDK.dll'))
    else:
        candidates = []

    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    raise FileNotFoundError(
        'MANUS SDK library not found. Set MANUS_SDK_LIBRARY or MANUS_SDK_ROOT. Searched: '
        + ', '.join(candidates)
    )


def load_sdk(connection_mode=None):
    library_path = find_sdk_library(connection_mode)
    sdk_dir = os.path.dirname(library_path)
    if sys.platform == 'win32' and hasattr(os, 'add_dll_directory'):
        try:
            os.add_dll_directory(sdk_dir)
        except OSError:
            pass
    log_info(f'Loading SDK from {library_path}')
    return C.CDLL(library_path), library_path


def bind_signatures(sdk):
    """Attach argtypes / restype so ctypes knows how to marshal each call."""
    # Init / shutdown
    sdk.CoreSdk_InitializeCore.restype = c_int32
    sdk.CoreSdk_InitializeIntegrated.restype = c_int32
    sdk.CoreSdk_ShutDown.restype = c_int32
    sdk.CoreSdk_SetSettingsLocation.argtypes = [C.c_char_p]
    sdk.CoreSdk_SetSettingsLocation.restype = c_int32

    # Coordinate system (by VALUE)
    sdk.CoreSdk_InitializeCoordinateSystemWithVUH.argtypes = [CoordinateSystemVUH, c_bool]
    sdk.CoreSdk_InitializeCoordinateSystemWithVUH.restype = c_int32
    sdk.CoreSdk_InitializeCoordinateSystemWithDirection.argtypes = [CoordinateSystemDirection, c_bool]
    sdk.CoreSdk_InitializeCoordinateSystemWithDirection.restype = c_int32

    # Host discovery
    sdk.CoreSdk_LookForHosts.argtypes = [c_uint32, c_bool]
    sdk.CoreSdk_LookForHosts.restype = c_int32

    sdk.CoreSdk_GetNumberOfAvailableHostsFound.argtypes = [POINTER(c_uint32)]
    sdk.CoreSdk_GetNumberOfAvailableHostsFound.restype = c_int32

    sdk.CoreSdk_GetAvailableHostsFound.argtypes = [POINTER(ManusHost), c_uint32]
    sdk.CoreSdk_GetAvailableHostsFound.restype = c_int32

    # ConnectToHost takes ManusHost BY VALUE
    sdk.CoreSdk_ConnectToHost.argtypes = [ManusHost]
    sdk.CoreSdk_ConnectToHost.restype = c_int32

    # Callback registration
    sdk.CoreSdk_RegisterCallbackForOnLog.argtypes = [LoggingCallback]
    sdk.CoreSdk_RegisterCallbackForOnLog.restype = c_int32

    sdk.CoreSdk_RegisterCallbackForRawSkeletonStream.argtypes = [RawSkeletonStreamCallback]
    sdk.CoreSdk_RegisterCallbackForRawSkeletonStream.restype = c_int32

    sdk.CoreSdk_RegisterCallbackForLandscapeStream.argtypes = [LandscapeStreamCallback]
    sdk.CoreSdk_RegisterCallbackForLandscapeStream.restype = c_int32

    # Skeleton stream API
    sdk.CoreSdk_SetRawSkeletonHandMotion.argtypes = [c_int32]
    sdk.CoreSdk_SetRawSkeletonHandMotion.restype = c_int32
    sdk.CoreSdk_GetRawSkeletonHandMotion.argtypes = [POINTER(c_int32)]
    sdk.CoreSdk_GetRawSkeletonHandMotion.restype = c_int32

    sdk.CoreSdk_SetRawSkeletonPinchCompensation.argtypes = [c_bool]
    sdk.CoreSdk_SetRawSkeletonPinchCompensation.restype = c_int32
    sdk.CoreSdk_GetRawSkeletonPinchCompensation.argtypes = [POINTER(c_bool)]
    sdk.CoreSdk_GetRawSkeletonPinchCompensation.restype = c_int32

    sdk.CoreSdk_SetRawSkeletonCasingCompensation.argtypes = [c_float]
    sdk.CoreSdk_SetRawSkeletonCasingCompensation.restype = c_int32
    sdk.CoreSdk_GetRawSkeletonCasingCompensation.argtypes = [POINTER(c_float)]
    sdk.CoreSdk_GetRawSkeletonCasingCompensation.restype = c_int32

    sdk.CoreSdk_GetVersionsAndCheckCompatibility.argtypes = [
        POINTER(ManusVersion), POINTER(ManusVersion), POINTER(c_bool),
    ]
    sdk.CoreSdk_GetVersionsAndCheckCompatibility.restype = c_int32

    sdk.CoreSdk_GloveCalibrationStart.argtypes = [GloveCalibrationArgs, POINTER(c_bool)]
    sdk.CoreSdk_GloveCalibrationStart.restype = c_int32
    sdk.CoreSdk_GloveCalibrationStop.argtypes = [GloveCalibrationArgs, POINTER(c_bool)]
    sdk.CoreSdk_GloveCalibrationStop.restype = c_int32
    sdk.CoreSdk_GloveCalibrationFinish.argtypes = [GloveCalibrationArgs, POINTER(c_bool)]
    sdk.CoreSdk_GloveCalibrationFinish.restype = c_int32
    sdk.CoreSdk_GloveCalibrationGetNumberOfSteps.argtypes = [GloveCalibrationArgs, POINTER(c_uint32)]
    sdk.CoreSdk_GloveCalibrationGetNumberOfSteps.restype = c_int32
    sdk.CoreSdk_GloveCalibrationGetStepData.argtypes = [
        GloveCalibrationStepArgs, POINTER(GloveCalibrationStepData),
    ]
    sdk.CoreSdk_GloveCalibrationGetStepData.restype = c_int32
    sdk.CoreSdk_GloveCalibrationStartStep.argtypes = [GloveCalibrationStepArgs, POINTER(c_bool)]
    sdk.CoreSdk_GloveCalibrationStartStep.restype = c_int32

    sdk.CoreSdk_GetGloveCalibrationSize.argtypes = [c_uint32, POINTER(c_uint32)]
    sdk.CoreSdk_GetGloveCalibrationSize.restype = c_int32
    sdk.CoreSdk_GetGloveCalibration.argtypes = [POINTER(c_ubyte), c_uint32]
    sdk.CoreSdk_GetGloveCalibration.restype = c_int32
    sdk.CoreSdk_SetGloveCalibration.argtypes = [
        c_uint32, POINTER(c_ubyte), c_uint32, POINTER(c_int32),
    ]
    sdk.CoreSdk_SetGloveCalibration.restype = c_int32

    sdk.CoreSdk_GetRawSkeletonInfo.argtypes = [c_uint32, POINTER(RawSkeletonInfo)]
    sdk.CoreSdk_GetRawSkeletonInfo.restype = c_int32

    sdk.CoreSdk_GetRawSkeletonData.argtypes = [c_uint32, POINTER(SkeletonNode), c_uint32]
    sdk.CoreSdk_GetRawSkeletonData.restype = c_int32

    # The C++ header takes a uint32_t& — map as pointer for ctypes.
    sdk.CoreSdk_GetRawSkeletonNodeCount.argtypes = [c_uint32, POINTER(c_uint32)]
    sdk.CoreSdk_GetRawSkeletonNodeCount.restype = c_int32

    sdk.CoreSdk_GetRawSkeletonNodeInfoArray.argtypes = [c_uint32, POINTER(NodeInfo), c_uint32]
    sdk.CoreSdk_GetRawSkeletonNodeInfoArray.restype = c_int32

    # Optional: glove-side lookup (not strictly needed since skeleton NodeInfo
    # has the side, but kept for parity with C++ client fallback path)
    try:
        sdk.CoreSdk_GetDataForGlove_UsingGloveId.restype = c_int32
        # Skip argtypes — we don't actually call it in the Python client
    except AttributeError:
        pass


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------
class ManusSdkClient:
    def __init__(self):
        self.running = True
        self.sdk = None
        self.config = self._load_config_from_env()

        # Skeleton callback scratch — populated inside the SDK callback thread,
        # drained by the main send loop.
        self._lock = threading.Lock()
        self._new_data = False
        self._next_frames = []         # list[dict] ready to send
        self._new_glove_status = False
        self._next_glove_status = []

        # NodeInfo cache per gloveId (static once glove connects)
        self._node_info_cache = {}     # gloveId -> list[NodeInfo]

        # Sockets
        self._out_sock = None
        self._command_sock = None

        self._frame_counter = 0
        self._restored_calibration_gloves = set()
        self._calibration_restore_attempted = set()
        self._glove_hands = {}

        # The external Core used by MANUS' sample client blocks in StartStep,
        # while Integrated Core can return as soon as sampling is accepted.
        # Keep the call off the command thread and enforce the advertised step
        # duration before reporting completion in either mode.
        self._calibration_lock = threading.Lock()
        self._calibration_thread = None
        self._calibration_preparation_seconds = max(
            0.0, float(os.environ.get('MANUS_CALIBRATION_PREPARATION_SECONDS', '2.0'))
        )
        self._calibration_min_sample_seconds = max(
            0.1, float(os.environ.get('MANUS_CALIBRATION_MIN_SAMPLE_SECONDS', '3.0'))
        )
        self._calibration = {
            'active': False,
            'gloveId': 0,
            'stepCount': 0,
            'stepIndex': -1,
            'completedStepIndex': -1,
            'inProgress': False,
            'stepComplete': False,
            'phase': 'idle',
            'error': '',
        }

        # Hold a reference to the CFUNCTYPE wrapper so Python GC doesn't
        # collect it while Core holds a pointer to it.
        self._log_cb_ref = LoggingCallback(self._on_sdk_log)
        self._raw_skeleton_cb_ref = RawSkeletonStreamCallback(self._on_raw_skeleton_stream)
        self._landscape_cb_ref = LandscapeStreamCallback(self._on_landscape_stream)

    @staticmethod
    def _load_config_from_env():
        default_mode = 'integrated' if sys.platform.startswith('linux') else 'local'
        mode = os.environ.get('MANUS_CONNECTION_MODE', default_mode).strip().lower()
        if mode not in ('local', 'remote', 'integrated'):
            mode = default_mode
        rate = max(1, int(os.environ.get('MANUS_SEND_RATE', '60')))
        return {
            'connection_mode': mode,
            'out_host': os.environ.get('MANUS_OUT_HOST', '127.0.0.1'),
            'out_port': int(os.environ.get('MANUS_OUT_PORT', '9001')),
            'command_port': int(os.environ.get('MANUS_COMMAND_PORT', '9003')),
            'send_interval_ms': max(1, 1000 // rate),
            'calibration_root': os.path.abspath(os.environ.get(
                'MANUS_CALIBRATION_ROOT',
                os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..',
                             'calibration', 'assets', 'hardware_configs', 'manus'),
            )),
        }

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def initialize(self):
        self.sdk, _library_path = load_sdk(self.config['connection_mode'])
        bind_signatures(self.sdk)

        if self.config['connection_mode'] == 'integrated':
            rc = self.sdk.CoreSdk_InitializeIntegrated()
        else:
            rc = self.sdk.CoreSdk_InitializeCore()
        if rc != SDKReturnCode.Success:
            log_error(f'CoreSdk_Initialize failed: {rc}')
            return False

        settings_location = os.path.join(self.config['calibration_root'], 'integrated_sdk_settings')
        os.makedirs(settings_location, exist_ok=True)
        settings_name = 'CoreLite.Settings.3.1.1.json'
        existing_settings = os.path.join(os.path.expanduser('~'), '.config', 'Manus', 'Core 3')
        seed_global_settings = os.environ.get('MANUS_SEED_GLOBAL_SETTINGS', '1') == '1'
        if seed_global_settings and not os.path.isfile(os.path.join(settings_location, settings_name)):
            for name in (settings_name, 'CoreLite.Settings.3.1.1.Devices.json',
                         'CoreLite.Settings.3.1.1.Calibrations.json'):
                source = os.path.join(existing_settings, name)
                if os.path.isfile(source):
                    shutil.copy2(source, os.path.join(settings_location, name))
            log_info(f'Seeded project MANUS settings from existing Linux Integrated profile: {existing_settings}')
        settings_argument = settings_location.replace('\\', '/').rstrip('/') + '/'
        rc = self.sdk.CoreSdk_SetSettingsLocation(settings_argument.encode('utf-8'))
        if rc != SDKReturnCode.Success:
            log_warn(f'CoreSdk_SetSettingsLocation failed: {rc}; calibration persistence may be unavailable')
        else:
            log_info(f'Manus Integrated settings: {settings_location}')

        rc = self.sdk.CoreSdk_RegisterCallbackForOnLog(self._log_cb_ref)
        if rc != SDKReturnCode.Success:
            log_error(f'RegisterCallbackForOnLog failed: {rc}')
            return False

        # Hands-only still needs Landscape updates for battery, device family,
        # calibration capabilities, and calibration blob restoration.
        rc = self.sdk.CoreSdk_RegisterCallbackForLandscapeStream(self._landscape_cb_ref)
        if rc != SDKReturnCode.Success:
            log_error(f'RegisterCallbackForLandscapeStream failed: {rc}')
            return False

        rc = self.sdk.CoreSdk_RegisterCallbackForRawSkeletonStream(self._raw_skeleton_cb_ref)
        if rc != SDKReturnCode.Success:
            log_error(f'RegisterCallbackForRawSkeletonStream failed: {rc}')
            return False
        log_info('Registered Manus Core RawSkeletonStream callback')

        coordinate_system = CoordinateSystemDirection(
            x=AxisDirection.Right,
            y=AxisDirection.Up,
            z=AxisDirection.Backward,
            unitScale=1.0,  # meters
        )
        rc = self.sdk.CoreSdk_InitializeCoordinateSystemWithDirection(coordinate_system, True)
        if rc != SDKReturnCode.Success:
            log_error(f'CoreSdk_InitializeCoordinateSystemWithDirection failed: {rc}')
            return False

        log_info('Manus SDK initialized (world coords, OpenVR-compatible: RH/Y-up/-Z-forward)')

        if not self._init_udp():
            return False
        log_info('Hand-skeleton-only mode enabled; Vive remains outside MANUS Core')
        return True

    def _init_udp(self):
        try:
            self._out_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except OSError as e:
            log_error(f'UDP out socket() failed: {e}')
            return False
        log_info(f'UDP out: {self.config["out_host"]}:{self.config["out_port"]}')

        # Settings and official glove calibration share this local command socket.
        try:
            command_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            command_sock.bind(('127.0.0.1', self.config['command_port']))
            command_sock.setblocking(False)
            self._command_sock = command_sock
            log_info(f'UDP command: 127.0.0.1:{self.config["command_port"]}')
        except OSError as e:
            log_warn(f'Command bind({self.config["command_port"]}) failed: {e}; interactive controls disabled')

        return True

    def connect(self):
        local = (self.config['connection_mode'] == 'local')
        for attempt in range(60):
            if not self.running:
                return False
            rc = self.sdk.CoreSdk_LookForHosts(1, local)
            if rc == SDKReturnCode.Success:
                n = c_uint32(0)
                rc2 = self.sdk.CoreSdk_GetNumberOfAvailableHostsFound(byref(n))
                if rc2 == SDKReturnCode.Success and n.value > 0:
                    hosts = (ManusHost * n.value)()
                    rc3 = self.sdk.CoreSdk_GetAvailableHostsFound(hosts, n.value)
                    if rc3 == SDKReturnCode.Success:
                        host = hosts[0]  # copied out by value
                        rc4 = self.sdk.CoreSdk_ConnectToHost(host)
                        if rc4 == SDKReturnCode.Success:
                            host_name = bytes(host.hostName).split(b'\x00', 1)[0].decode('utf-8', errors='replace')
                            connection_name = (
                                'Manus Integrated'
                                if self.config['connection_mode'] == 'integrated'
                                else 'Manus Core'
                            )
                            log_info(f'Connected to {connection_name}: {host_name}')
                            return True
            log_warn(f'Manus host not found yet — retrying in 1s (attempt {attempt + 1})')
            time.sleep(1)

        log_error('Failed to connect to Manus host')
        return False

    @staticmethod
    def _redact_sdk_log(text):
        return re.sub(
            r'("Key"\s*:\s*")[^"]+(")',
            r'\1<redacted>\2',
            text,
            flags=re.IGNORECASE,
        )

    def _on_sdk_log(self, severity, p_log, length):
        try:
            text = C.string_at(p_log, int(length)).decode('utf-8', errors='replace').strip()
            if not text:
                return
            text = self._redact_sdk_log(text)
            if int(severity) >= 2:
                log_warn(f'[MANUS SDK] {text}')
            else:
                log_info(f'[MANUS SDK] {text}')
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Skeleton stream callback — runs on an SDK-owned thread
    # ------------------------------------------------------------------
    def _on_raw_skeleton_stream(self, p_info):
        try:
            if not p_info:
                return
            info = p_info.contents
            skeleton_count = info.skeletonsCount

            frames = []
            for i in range(skeleton_count):
                raw = RawSkeletonInfo()
                rc = self.sdk.CoreSdk_GetRawSkeletonInfo(i, byref(raw))
                if rc != SDKReturnCode.Success:
                    continue

                nodes_count = raw.nodesCount
                if nodes_count == 0:
                    continue

                nodes = (SkeletonNode * nodes_count)()
                rc = self.sdk.CoreSdk_GetRawSkeletonData(i, nodes, nodes_count)
                if rc != SDKReturnCode.Success:
                    continue

                node_info = self._get_node_info(raw.gloveId, nodes_count)

                # Build JSON-ready dicts now so main thread only does UDP send.
                skeleton_frame = self._encode_raw_frame(raw, nodes, node_info, info.publishTime.time)
                frames.append(skeleton_frame)

            if frames:
                with self._lock:
                    self._next_frames = frames + self._next_frames
                    self._new_data = True
        except Exception as e:
            # Never let an exception propagate back into native code
            log_warn(f'Skeleton callback error: {e}')

    def _on_landscape_stream(self, p_landscape):
        try:
            if not p_landscape:
                return
            devices = p_landscape.contents.gloveDevices
            statuses = []
            for index in range(min(int(devices.gloveCount), MAX_NUMBER_OF_GLOVES)):
                status = self._encode_glove_status(devices.gloves[index])
                if status is not None:
                    statuses.append(status)
            if statuses:
                with self._lock:
                    self._next_glove_status = statuses
                    self._new_glove_status = True
        except Exception as e:
            log_warn(f'Landscape callback error: {e}')

    @staticmethod
    def _encode_glove_status(glove):
        hand = SIDE_HAND.get(int(glove.side))
        if hand is None:
            return None
        battery = int(glove.batteryPercentage)
        family = int(glove.familyType)
        metaglove_families = {
            DeviceFamilyType.Metaglove,
            DeviceFamilyType.MetaglovePro,
            DeviceFamilyType.MetagloveProPrecision,
            DeviceFamilyType.MetagloveProHaptics,
            DeviceFamilyType.MetagloveProPrecisionHaptics,
        }
        metaglove_pro_families = metaglove_families - {DeviceFamilyType.Metaglove}
        return {
            'type': 'manus_glove_status',
            'hand': hand,
            'gloveId': int(glove.id),
            'batteryPercentage': battery if 0 <= battery <= 100 else None,
            'deviceFamily': {
                'id': family,
                'name': DeviceFamilyType._NAMES.get(family, f'Unknown ({family})'),
            },
            'calibrationTunables': {
                'pinchCompensation': family in metaglove_families,
                'casingCompensation': family in metaglove_pro_families,
            },
        }

    def _get_node_info(self, glove_id, expected_count):
        cached = self._node_info_cache.get(glove_id)
        if cached is not None and len(cached) == expected_count:
            return cached

        count = c_uint32(0)
        rc = self.sdk.CoreSdk_GetRawSkeletonNodeCount(glove_id, byref(count))
        if rc != SDKReturnCode.Success or count.value == 0 or count.value != expected_count:
            return None

        info_arr = (NodeInfo * count.value)()
        rc = self.sdk.CoreSdk_GetRawSkeletonNodeInfoArray(glove_id, info_arr, count.value)
        if rc != SDKReturnCode.Success:
            return None

        info_list = list(info_arr)
        self._node_info_cache[glove_id] = info_list
        return info_list

    @staticmethod
    def _bone_name(info):
        side = SIDE_STR.get(info.side, 'x')
        chain = ChainType._NAMES.get(info.chainType, 'unknown')
        joint = FingerJointType._NAMES.get(info.fingerJointType, '')
        name = f'{side}_{chain}'
        if joint:
            name += f'_{joint}'
        return name

    @staticmethod
    def _is_wrist_root_name(name):
        return str(name or '').lower() in {'l_hand', 'r_hand', 'hand_l', 'hand_r'}

    def _encode_raw_frame(self, raw_info, nodes, node_info, publish_time_ns):
        hand = 'unknown'
        if node_info:
            # Hand side comes from any bone on this glove
            for ni in node_info:
                if ni.side == Side.Left:
                    hand = 'left'
                    break
                if ni.side == Side.Right:
                    hand = 'right'
                    break

        bones = []
        for i, node in enumerate(nodes):
            p = node.transform.position
            r = node.transform.rotation
            if node_info and i < len(node_info):
                info = node_info[i]
                name = self._bone_name(info)
            else:
                info = None
                name = f'node_{node.id}'
            is_wrist_root = self._is_wrist_root_name(name)
            bones.append({
                'name': name,
                'nodeId': int(info.nodeId) if info is not None else int(node.id),
                'parentId': int(info.parentId) if info is not None else None,
                'chainType': ChainType._NAMES.get(int(info.chainType), 'unknown') if info is not None else 'unknown',
                'fingerJointType': FingerJointType._NAMES.get(int(info.fingerJointType), '') if info is not None else '',
                'pos': None if is_wrist_root else [p.x, p.y, p.z],
                # Match C++ client output order: [x, y, z, w]
                'rot': None if is_wrist_root else [r.x, r.y, r.z, r.w],
                # The Manus tab needs the complete hierarchy, including wrist/root.
                # Keep legacy pos/rot nulling above for the live-overlay consumer.
                'rawPos': [p.x, p.y, p.z],
                'rawRot': [r.x, r.y, r.z, r.w],
            })

        skeleton_frame = {
            'type': 'manus_hand_skeleton',
            'hand': hand,
            'source': 'manus_sdk_raw',
            'skeletonType': 'hand_without_wrist_pose',
            'coordinateFrame': 'openvr_raw_uncalibrated_meters',
            'gloveId': int(raw_info.gloveId),
            'sourceTimestampNs': str(int(raw_info.publishTime.time)),
            'sourcePublishTimeNs': str(int(publish_time_ns)),
            'bones': bones,
        }
        return skeleton_frame

    # ------------------------------------------------------------------
    # Main send loop
    # ------------------------------------------------------------------
    def _apply_raw_skeleton_defaults(self):
        self._require_sdk_success(
            self.sdk.CoreSdk_SetRawSkeletonHandMotion(HandMotion.Auto),
            'SetRawSkeletonHandMotion(default)',
        )
        self._require_sdk_success(
            self.sdk.CoreSdk_SetRawSkeletonPinchCompensation(False),
            'SetRawSkeletonPinchCompensation(default)',
        )

    def run(self):
        self._apply_raw_skeleton_defaults()

        last_send_ms = 0
        send_interval_ms = self.config['send_interval_ms']

        while self.running:
            self._process_commands()
            self._pump()
            now_ms = int(time.monotonic() * 1000)
            if now_ms - last_send_ms < send_interval_ms:
                time.sleep(0.001)
                continue
            last_send_ms = now_ms

    @staticmethod
    def _decode_c_string(value):
        return bytes(value).split(b'\x00', 1)[0].decode('utf-8', errors='replace')

    @staticmethod
    def _require_sdk_success(rc, operation):
        if rc != SDKReturnCode.Success:
            raise RuntimeError(f'{operation} failed: SDK return code {rc}')

    def _settings(self):
        motion = c_int32(HandMotion.Auto)
        pinch = c_bool(False)
        casing = c_float(0.0)
        self._require_sdk_success(
            self.sdk.CoreSdk_GetRawSkeletonHandMotion(byref(motion)),
            'GetRawSkeletonHandMotion',
        )
        self._require_sdk_success(
            self.sdk.CoreSdk_GetRawSkeletonPinchCompensation(byref(pinch)),
            'GetRawSkeletonPinchCompensation',
        )
        self._require_sdk_success(
            self.sdk.CoreSdk_GetRawSkeletonCasingCompensation(byref(casing)),
            'GetRawSkeletonCasingCompensation',
        )
        sdk_version = ManusVersion()
        core_version = ManusVersion()
        compatible = c_bool(False)
        version_rc = self.sdk.CoreSdk_GetVersionsAndCheckCompatibility(
            byref(sdk_version), byref(core_version), byref(compatible),
        )
        versions = {}
        if version_rc == SDKReturnCode.Success:
            sdk_text = self._decode_c_string(sdk_version.versionInfo)
            core_text = self._decode_c_string(core_version.versionInfo)
            if sdk_text and core_text:
                versions = {
                    'sdkVersion': sdk_text,
                    'coreVersion': core_text,
                    'compatible': bool(compatible.value),
                }
        return {
            'handMotion': int(motion.value),
            'pinchCompensation': bool(pinch.value),
            'casingCompensation': round(float(casing.value), 3),
            **versions,
        }

    def _apply_settings(self, params):
        motion = int(params.get('handMotion', HandMotion.Auto))
        if motion not in range(5):
            raise ValueError('handMotion must be 0 through 4')
        pinch = bool(params.get('pinchCompensation', False))
        casing = float(params.get('casingCompensation', 0.0))
        if not 0.0 <= casing <= 1.0:
            raise ValueError('casingCompensation must be between 0 and 1')
        self._require_sdk_success(
            self.sdk.CoreSdk_SetRawSkeletonHandMotion(motion),
            'SetRawSkeletonHandMotion',
        )
        self._require_sdk_success(
            self.sdk.CoreSdk_SetRawSkeletonCasingCompensation(c_float(casing)),
            'SetRawSkeletonCasingCompensation',
        )
        # Integrated SDK 3.1.1 can couple a casing write back into endpoint
        # prioritization. Apply pinch last so the requested value is final.
        self._require_sdk_success(
            self.sdk.CoreSdk_SetRawSkeletonPinchCompensation(pinch),
            'SetRawSkeletonPinchCompensation',
        )
        return self._settings()

    def _calibration_snapshot(self):
        with self._calibration_lock:
            state = dict(self._calibration)
        phase_started = float(state.pop('_phaseStartedMonotonic', 0.0) or 0.0)
        phase = str(state.get('phase') or 'idle')
        duration = (
            float(state.get('preparationDuration') or 0.0)
            if phase == 'preparing'
            else float(state.get('samplingDuration') or 0.0)
        )
        elapsed = max(0.0, time.monotonic() - phase_started) if phase_started else 0.0
        state['elapsedSeconds'] = round(elapsed, 2)
        state['remainingSeconds'] = round(max(0.0, duration - elapsed), 2) if state.get('inProgress') else 0.0
        return state

    def _calibration_active(self):
        with self._calibration_lock:
            return bool(self._calibration['active'])

    def _execute_calibration_step(self, args, sampling_duration):
        started = c_bool(False)
        error = ''
        preparation_duration = max(0.0, float(self._calibration_preparation_seconds))
        if preparation_duration:
            time.sleep(preparation_duration)
        sampling_started = time.monotonic()
        with self._calibration_lock:
            self._calibration.update({
                'phase': 'sampling',
                '_phaseStartedMonotonic': sampling_started,
                'samplingStartedAtUnixMs': int(time.time() * 1000),
            })
        try:
            self._require_sdk_success(
                self.sdk.CoreSdk_GloveCalibrationStartStep(args, byref(started)),
                'GloveCalibrationStartStep',
            )
            if not started.value:
                raise RuntimeError('Manus Core rejected calibration step')
        except Exception as exc:
            error = str(exc)
        if not error:
            remaining = max(0.0, float(sampling_duration) - (time.monotonic() - sampling_started))
            if remaining:
                time.sleep(remaining)
        with self._calibration_lock:
            self._calibration['inProgress'] = False
            self._calibration['stepComplete'] = not error
            self._calibration['phase'] = 'complete' if not error else 'error'
            self._calibration['_phaseStartedMonotonic'] = time.monotonic()
            self._calibration['error'] = error
            if not error:
                self._calibration['completedStepIndex'] = int(args.stepIndex)

    def _calibration_step(self, glove_id, step_index):
        state = self._calibration_snapshot()
        if not state['active'] or int(state['gloveId']) != glove_id:
            raise RuntimeError('Manus calibration has not been started for this glove')
        if state['inProgress']:
            raise RuntimeError('Manus calibration step is still in progress')
        expected = int(state['completedStepIndex']) + 1
        if step_index != expected:
            raise RuntimeError(f'Manus calibration steps must run sequentially; expected step {expected}')
        if step_index < 0 or step_index >= int(state['stepCount']):
            raise ValueError('calibration step index is out of range')
        args = GloveCalibrationStepArgs(gloveId=glove_id, stepIndex=step_index)
        data = GloveCalibrationStepData()
        self._require_sdk_success(
            self.sdk.CoreSdk_GloveCalibrationGetStepData(args, byref(data)),
            'GloveCalibrationGetStepData',
        )
        step = {
            'index': int(data.index),
            'title': self._decode_c_string(data.title),
            'description': self._decode_c_string(data.description),
            'duration': float(data.time),
        }
        sampling_duration = max(abs(step['duration']), float(self._calibration_min_sample_seconds))
        with self._calibration_lock:
            self._calibration.update({
                'stepIndex': step_index,
                'inProgress': True,
                'stepComplete': False,
                'phase': 'preparing',
                'preparationDuration': float(self._calibration_preparation_seconds),
                'samplingDuration': sampling_duration,
                '_phaseStartedMonotonic': time.monotonic(),
                'error': '',
                **step,
            })
        self._calibration_thread = threading.Thread(
            target=self._execute_calibration_step,
            args=(args, sampling_duration),
            daemon=True,
            name=f'manus-calibration-{glove_id}-{step_index}',
        )
        self._calibration_thread.start()
        return self._calibration_snapshot()

    def _export_calibration_blob(self, glove_id):
        size = c_uint32(0)
        self._require_sdk_success(
            self.sdk.CoreSdk_GetGloveCalibrationSize(glove_id, byref(size)),
            'GetGloveCalibrationSize',
        )
        if size.value <= 0:
            raise RuntimeError('MANUS returned an empty glove calibration')
        buffer = (c_ubyte * size.value)()
        self._require_sdk_success(
            self.sdk.CoreSdk_GetGloveCalibration(buffer, size.value),
            'GetGloveCalibration',
        )
        payload = bytes(buffer)
        root = self.config['calibration_root']
        os.makedirs(root, exist_ok=True)
        hand = self._glove_hands.get(glove_id, 'unknown')
        profile = f'{hand}_{glove_id}' if hand in ('left', 'right') else str(glove_id)
        path = os.path.join(root, f'manus_calibration_{profile}.bin')
        temporary = path + '.tmp'
        with open(temporary, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        self._restored_calibration_gloves.add(glove_id)
        return {
            'path': path,
            'bytes': len(payload),
            'sha256': hashlib.sha256(payload).hexdigest(),
        }

    def _maybe_restore_calibrations(self, statuses):
        if self._calibration_active():
            return
        root = self.config['calibration_root']
        if not os.path.isdir(root):
            return
        for status in statuses:
            glove_id = int(status.get('gloveId') or 0)
            if glove_id <= 0 or glove_id in self._restored_calibration_gloves:
                continue
            if glove_id in self._calibration_restore_attempted:
                continue
            self._calibration_restore_attempted.add(glove_id)
            hand = str(status.get('hand') or self._glove_hands.get(glove_id) or '')
            stable_names = [f'manus_calibration_{glove_id}.bin']
            if hand in ('left', 'right'):
                stable_names.insert(0, f'manus_calibration_{hand}_{glove_id}.bin')
            candidates = [
                os.path.join(root, name)
                for name in os.listdir(root)
                if name in stable_names
                or (name.startswith(f'manus_sdk_calibration_{glove_id}_') and name.endswith('.bin'))
            ]
            if not candidates:
                continue
            path = max(candidates, key=os.path.getmtime)
            try:
                with open(path, 'rb') as stream:
                    payload = stream.read()
                if not payload:
                    raise RuntimeError('calibration blob is empty')
                buffer = (c_ubyte * len(payload)).from_buffer_copy(payload)
                result = c_int32(0)
                self._require_sdk_success(
                    self.sdk.CoreSdk_SetGloveCalibration(glove_id, buffer, len(payload), byref(result)),
                    'SetGloveCalibration',
                )
                if result.value != 1:
                    raise RuntimeError(f'MANUS rejected calibration blob with result {result.value}')
                self._restored_calibration_gloves.add(glove_id)
                log_info(f'Restored MANUS calibration for glove {glove_id}: {os.path.basename(path)}')
            except Exception as exc:
                log_warn(f'Could not restore MANUS calibration for glove {glove_id}: {exc}')

    def _handle_command(self, request):
        action = str(request.get('action') or '')
        params = request.get('params') if isinstance(request.get('params'), dict) else {}
        if action == 'get_settings':
            return self._settings()
        if action == 'apply_settings':
            return self._apply_settings(params)

        glove_id = int(params.get('gloveId') or 0)
        if glove_id <= 0:
            raise ValueError('a connected gloveId is required')
        args = GloveCalibrationArgs(gloveId=glove_id)
        result = c_bool(False)
        if action == 'calibration_start':
            if self._calibration_active():
                raise RuntimeError('A Manus calibration is already active')
            self._require_sdk_success(
                self.sdk.CoreSdk_GloveCalibrationStart(args, byref(result)),
                'GloveCalibrationStart',
            )
            if not result.value:
                raise RuntimeError('Manus Core rejected calibration start')
            count = c_uint32(0)
            self._require_sdk_success(
                self.sdk.CoreSdk_GloveCalibrationGetNumberOfSteps(args, byref(count)),
                'GloveCalibrationGetNumberOfSteps',
            )
            with self._calibration_lock:
                self._calibration = {
                    'active': True,
                    'gloveId': glove_id,
                    'stepCount': int(count.value),
                    'stepIndex': -1,
                    'completedStepIndex': -1,
                    'inProgress': False,
                    'stepComplete': False,
                    'phase': 'ready',
                    'error': '',
                }
            return self._calibration_snapshot()
        if action == 'calibration_step':
            return self._calibration_step(glove_id, int(params.get('stepIndex') or 0))
        if action == 'calibration_status':
            state = self._calibration_snapshot()
            if state['active'] and int(state['gloveId']) != glove_id:
                raise RuntimeError('Another glove calibration is active')
            return state
        state = self._calibration_snapshot()
        if state['inProgress']:
            raise RuntimeError('Wait for the current Manus calibration step to complete')
        if action == 'calibration_finish':
            if int(state['completedStepIndex']) < int(state['stepCount']) - 1:
                raise RuntimeError('Complete every Manus calibration step before saving')
            fn = self.sdk.CoreSdk_GloveCalibrationFinish
        elif action == 'calibration_cancel':
            fn = self.sdk.CoreSdk_GloveCalibrationStop
        else:
            raise ValueError(f'unknown Manus command: {action}')
        self._require_sdk_success(fn(args, byref(result)), action)
        if not result.value:
            raise RuntimeError(f'Manus Core rejected {action}')
        blob = None
        if action == 'calibration_finish':
            try:
                blob = self._export_calibration_blob(glove_id)
            except Exception as exc:
                blob = {'error': str(exc)}
                log_warn(f'MANUS calibration committed but binary export failed: {exc}')
        with self._calibration_lock:
            self._calibration.update({
                'active': False,
                'inProgress': False,
                'phase': 'saved' if action == 'calibration_finish' else 'cancelled',
                'saved': action == 'calibration_finish',
                'cancelled': action == 'calibration_cancel',
                'error': '',
            })
            if blob is not None:
                self._calibration['calibrationBlob'] = blob
        return self._calibration_snapshot()

    def _process_commands(self):
        if self._command_sock is None:
            return
        for _index in range(8):
            try:
                payload, sender = self._command_sock.recvfrom(16384)
            except BlockingIOError:
                return
            except OSError:
                return
            request_id = None
            try:
                request = json.loads(payload.decode('utf-8'))
                if not isinstance(request, dict):
                    raise ValueError('command must be a JSON object')
                request_id = request.get('id')
                response = {
                    'id': request_id,
                    'ok': True,
                    'result': self._handle_command(request),
                }
            except Exception as e:
                response = {'id': request_id, 'ok': False, 'error': str(e)}
            try:
                self._command_sock.sendto(
                    json.dumps(response, separators=(',', ':')).encode('utf-8'),
                    sender,
                )
            except OSError:
                pass

    def _send_packets(self, packets):
        now_ms = int(time.time() * 1000)
        for packet in packets:
            packet['timestamp'] = now_ms
            try:
                payload = json.dumps(packet, separators=(',', ':')).encode('utf-8')
                self._out_sock.sendto(payload, (self.config['out_host'], self.config['out_port']))
            except OSError as e:
                log_warn(f'UDP send failed: {e}')

        self._frame_counter += len(packets)
        if self._frame_counter and self._frame_counter % 300 == 0:
            log_info(f'Forwarded {self._frame_counter} Manus UDP packets')

    def _pump(self):
        frames = []
        glove_status = []
        with self._lock:
            if self._new_data:
                frames = self._next_frames
                self._next_frames = []
                self._new_data = False
            if self._new_glove_status:
                glove_status = self._next_glove_status
                self._next_glove_status = []
                self._new_glove_status = False

        if frames:
            self._send_packets(frames)
        if glove_status:
            for status in glove_status:
                glove_id = int(status.get('gloveId') or 0)
                hand = str(status.get('hand') or '')
                if glove_id > 0 and hand in ('left', 'right'):
                    self._glove_hands[glove_id] = hand
            self._maybe_restore_calibrations(glove_status)
            self._send_packets(glove_status)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    def shutdown(self):
        self.running = False

        if self._out_sock is not None:
            try:
                self._out_sock.close()
            except OSError:
                pass
            self._out_sock = None

        if self._command_sock is not None:
            try:
                self._command_sock.close()
            except OSError:
                pass
            self._command_sock = None

        if self.sdk is not None:
            try:
                self.sdk.CoreSdk_ShutDown()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    log_info('manus_sdk_client (python) starting...')
    client = ManusSdkClient()

    def _handle_signal(_signum, _frame):
        client.running = False

    try:
        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)
    except (ValueError, AttributeError):
        pass

    try:
        if not client.initialize():
            log_error('Initialization failed.')
            return 1
        if not client.connect():
            return 2
        log_info('Streaming Manus skeleton → UDP. Press Ctrl+C to stop.')
        client.run()
        return 0
    finally:
        client.shutdown()


if __name__ == '__main__':
    sys.exit(main())
