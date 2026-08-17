import json
from pathlib import Path
import socket
import threading
import time

import numpy as np
import pytest

from hardware.manus import ManusReceiver, parse_manus_message
from hardware.manus_sdk import client as sdk_client


FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def packet(hand: str = "left", glove_id: int = 17) -> dict:
    bones = [
        {
            "chainType": "hand",
            "fingerJointType": "",
            "rawPos": [0.1, 0.2, 0.3],
        }
    ]
    for finger_index, finger in enumerate(FINGERS):
        for joint_index, joint in enumerate(("distal", "tip")):
            base = float(finger_index * 2 + joint_index)
            bones.append(
                {
                    "chainType": finger,
                    "fingerJointType": joint,
                    "rawPos": [base + 0.1, base + 0.2, base + 0.3],
                }
            )
    return {
        "type": "manus_hand_skeleton",
        "hand": hand,
        "gloveId": glove_id,
        "coordinateFrame": "openvr_raw_uncalibrated_meters",
        "sourceTimestampNs": "123456",
        "bones": bones,
    }


def test_parse_sdk_311_skeleton_and_transform_y_axis():
    sample = parse_manus_message(json.dumps(packet()), {"left": 17}, timestamp=4.0)
    assert sample is not None
    assert sample.glove_id == 17
    assert sample.hand == "left"
    assert sample.timestamp == 4.0
    assert sample.source_timestamp_ns == 123456
    np.testing.assert_allclose(sample.wrist, [0.1, -0.2, 0.3])
    np.testing.assert_allclose(sample.fingers[0], [0.1, -0.2, 0.3])
    np.testing.assert_allclose(sample.fingers[-1], [9.1, -9.2, 9.3])
    assert not sample.fingers.flags.writeable


def test_status_packet_is_ignored():
    assert parse_manus_message({"type": "manus_glove_status", "hand": "left"}) is None


def test_unknown_packet_type_is_rejected():
    with pytest.raises(ValueError, match="packet type"):
        parse_manus_message({"type": "manus_tracker_pose"})


def test_unknown_glove_id_is_rejected():
    with pytest.raises(ValueError, match="Unknown left Manus glove ID"):
        parse_manus_message(packet(glove_id=99), {"left": 17})


@pytest.mark.parametrize(
    "mutation, match",
    [
        (lambda value: value.update(hand="unknown"), "hand side"),
        (lambda value: value.update(coordinateFrame="unity"), "coordinate frame"),
        (lambda value: value["bones"].pop(), "missing bones"),
        (lambda value: value["bones"][1].update(rawPos=[1, 2]), "three finite"),
    ],
)
def test_malformed_skeleton_is_rejected(mutation, match):
    value = packet()
    mutation(value)
    with pytest.raises(ValueError, match=match):
        parse_manus_message(value)


def test_non_object_json_is_rejected():
    with pytest.raises(ValueError, match="JSON object"):
        parse_manus_message("[]")


def test_receiver_latches_first_id_for_each_side():
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    receiver = ManusReceiver(f"udp://127.0.0.1:{port}", start_bridge=False)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sender.sendto(json.dumps(packet(glove_id=23)).encode(), ("127.0.0.1", port))
        sample = receiver.wait_for_sample(timeout=1.0)
        assert sample is not None and sample.glove_id == 23

        sender.sendto(json.dumps(packet(glove_id=24)).encode(), ("127.0.0.1", port))
        deadline = time.monotonic() + 1.0
        while receiver.last_error is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert receiver.last_error is not None
        assert "expected 23" in receiver.last_error
        assert receiver.get_latest().glove_id == 23
    finally:
        sender.close()
        receiver.close()


def test_receiver_reports_egotac_glove_status_and_display_frame():
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    receiver = ManusReceiver(f"udp://127.0.0.1:{port}", start_bridge=False)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        status = {
            "type": "manus_glove_status",
            "hand": "left",
            "gloveId": 17,
            "batteryPercentage": 81,
            "deviceFamily": {"id": 2, "name": "Metaglove"},
            "calibrationTunables": {
                "pinchCompensation": True,
                "casingCompensation": False,
            },
        }
        frame = packet()
        for index, bone in enumerate(frame["bones"]):
            bone["nodeId"] = index
            bone["parentId"] = 0 if index else 0
        sender.sendto(json.dumps(status).encode(), ("127.0.0.1", port))
        sender.sendto(json.dumps(frame).encode(), ("127.0.0.1", port))
        assert receiver.wait_for_sample(timeout=1.0) is not None

        snapshot = receiver.status(include_frames=True)

        left = snapshot["hands"]["left"]
        assert left["connected"]
        assert left["batteryPercentage"] == 81
        assert left["deviceFamily"]["name"] == "Metaglove"
        assert left["boneCount"] == 11
        assert len(left["frame"]["bones"]) == 11
    finally:
        sender.close()
        receiver.close()


def test_receiver_forwards_settings_command_to_sdk_bridge():
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    data_port = probe.getsockname()[1]
    probe.close()
    command_server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    command_server.bind(("127.0.0.1", 0))
    command_port = command_server.getsockname()[1]
    receiver = ManusReceiver(
        f"udp://127.0.0.1:{data_port}",
        start_bridge=False,
        command_port=command_port,
    )

    class RunningProcess:
        @staticmethod
        def poll():
            return None

    receiver._process = RunningProcess()

    def serve_command():
        raw, sender = command_server.recvfrom(65535)
        request = json.loads(raw)
        response = {
            "id": request["id"],
            "ok": True,
            "result": {"sdkVersion": "3.1.1", "handMotion": 4},
        }
        command_server.sendto(json.dumps(response).encode(), sender)

    thread = threading.Thread(target=serve_command)
    thread.start()
    try:
        result = receiver.command("get_settings")
        assert result == {"sdkVersion": "3.1.1", "handMotion": 4}
        assert receiver.status()["settings"] == result
    finally:
        thread.join(timeout=1.0)
        command_server.close()
        receiver._process = None
        receiver.close()


def test_missing_sdk_library_has_actionable_path(tmp_path: Path):
    missing = tmp_path / "libManusSDK_Integrated.so"
    with pytest.raises(FileNotFoundError, match=str(missing)):
        ManusReceiver("udp://127.0.0.1:9001", sdk_library=missing)


def test_egotac_sdk_client_preserves_raw_skeleton_metadata():
    client = sdk_client.ManusSdkClient()
    raw = sdk_client.RawSkeletonInfo(
        gloveId=17, nodesCount=1, publishTime=sdk_client.ManusTimestamp(123)
    )
    node = sdk_client.SkeletonNode()
    node.id = 8
    node.transform.position = sdk_client.ManusVec3(1.0, 2.0, 3.0)
    node.transform.rotation = sdk_client.ManusQuaternion(1.0, 0.1, 0.2, 0.3)
    info = sdk_client.NodeInfo(
        nodeId=8,
        parentId=8,
        chainType=13,
        side=sdk_client.Side.Left,
        fingerJointType=0,
    )

    frame = client._encode_raw_frame(raw, [node], [info], 456)

    assert frame["type"] == "manus_hand_skeleton"
    assert frame["hand"] == "left"
    assert frame["gloveId"] == 17
    assert frame["coordinateFrame"] == "openvr_raw_uncalibrated_meters"
    assert frame["bones"][0]["rawPos"] == [1.0, 2.0, 3.0]


def test_sdk_client_keeps_settings_and_calibration_command_socket(monkeypatch):
    sockets = []

    class FakeSocket:
        def __init__(self):
            self.bound = None
            self.blocking = None
            sockets.append(self)

        def bind(self, address):
            self.bound = address

        def setblocking(self, value):
            self.blocking = value

    monkeypatch.setattr(sdk_client.socket, "socket", lambda *_args, **_kwargs: FakeSocket())
    client = sdk_client.ManusSdkClient()
    client.config.update(command_port=9003)

    assert client._init_udp()
    assert len(sockets) == 2
    assert client._command_sock is sockets[1]
    assert sockets[1].bound == ("127.0.0.1", 9003)
    assert sockets[1].blocking is False


def test_sdk_client_contains_no_tracker_binding_or_alignment_api():
    source = Path(sdk_client.__file__).read_text(encoding="utf-8")
    for removed in (
        "CoreSdk_SendDataForTrackers",
        "CoreSdk_SetTrackerOffset",
        "alignment_apply",
        "alignment_status",
        "MANUS_TRACKER_IN_PORT",
        "MANUS_TRACKER_FEED",
    ):
        assert removed not in source
