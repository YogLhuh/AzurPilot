"""NemuIpc SDK DLL 选择逻辑的单元测试。

覆盖实例版本推断（vms 目录名）与 DLL 候选路径顺序：
实例版本精确匹配的 SDK 必须排在通用路径之前，避免同机多版本
并存时（如 nx_device/12.0 与 15.0）错用旧版 DLL 连新版实例。
"""

import os
import shutil
import tempfile
import unittest

import cv2
import numpy as np

from module.device.method.nemu_ipc import NemuIpcImpl


def build_fake_install(root, versions=('12.0',), instance_names=('MuMuPlayer-12.0-0',)):
    """构造一个仿真的 MuMu 安装目录结构。

    Args:
        root (str): 临时目录。
        versions (tuple): nx_device 下存在的版本目录。
        instance_names (tuple): vms 下存在的实例目录名。

    Returns:
        str: 安装根目录。
    """
    for version in versions:
        folder = os.path.join(root, 'nx_device', version, 'shell', 'sdk')
        os.makedirs(folder)
        with open(os.path.join(folder, 'external_renderer_ipc.dll'), 'w') as f:
            f.write('fake')
    folder = os.path.join(root, 'nx_main', 'sdk')
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'external_renderer_ipc.dll'), 'w') as f:
        f.write('fake')
    for name in instance_names:
        os.makedirs(os.path.join(root, 'vms', name))
    return root


class TestDetectVersion(unittest.TestCase):
    def test_detect_from_vms_name(self):
        with tempfile.TemporaryDirectory() as root:
            build_fake_install(
                root, versions=('12.0', '15.0'),
                instance_names=('MuMuPlayer-12.0-1', 'MuMuPlayer-15.0-0'))
            self.assertEqual(
                NemuIpcImpl.detect_version(root, 0), '15.0')
            self.assertEqual(
                NemuIpcImpl.detect_version(root, 1), '12.0')

    def test_detect_yxarknights_instance(self):
        with tempfile.TemporaryDirectory() as root:
            build_fake_install(
                root, versions=('12.0',), instance_names=('YXArkNights-12.0-1',))
            self.assertEqual(
                NemuIpcImpl.detect_version(root, 1), '12.0')

    def test_detect_missing_vms_returns_none(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertIsNone(NemuIpcImpl.detect_version(root, 0))


class TestDllPathOrder(unittest.TestCase):
    """验证加载的 DLL 来自实例版本对应的 nx_device/<版本> 目录。

    __init__ 会真实加载 DLL，这里用写入假内容的 dll 文件会让
    ctypes.CDLL 抛 OSError 并继续尝试下一个候选；因此只对
    "最终选中路径" 的语义做间接验证：让仅实例版本的路径真实可加载
    是不可能的（假 DLL），改为验证候选列表顺序 via detect + 文件系统。
    """

    def _probe_selected(self, root, instance_id, version=None):
        """拦截 ctypes.CDLL，记录第一个存在的候选路径。"""
        import ctypes as _ctypes

        selected = []

        class _FakeLib:
            pass

        original = _ctypes.CDLL

        def fake_cdll(path, *args, **kwargs):
            if path not in selected:
                selected.append(path)
            return _FakeLib()

        _ctypes.CDLL = fake_cdll
        try:
            NemuIpcImpl(
                nemu_folder=root, instance_id=instance_id,
                version=version)
        finally:
            _ctypes.CDLL = original
        return selected[0]

    def test_explicit_version_wins(self):
        with tempfile.TemporaryDirectory() as root:
            build_fake_install(
                root, versions=('12.0', '15.0'),
                instance_names=('MuMuPlayer-12.0-1', 'MuMuPlayer-15.0-0'))
            selected = self._probe_selected(root, 0)
            self.assertIn('nx_device/15.0', selected.replace('\\', '/'))
            self.assertNotIn('nx_device/12.0', selected.replace('\\', '/'))

    def test_version_inferred_from_vms(self):
        with tempfile.TemporaryDirectory() as root:
            build_fake_install(
                root, versions=('12.0', '15.0'),
                instance_names=('MuMuPlayer-12.0-1', 'MuMuPlayer-15.0-0'))
            # 不传 version，从 vms/MuMuPlayer-15.0-0 自动推断
            selected = self._probe_selected(root, 0)
            self.assertIn('nx_device/15.0', selected.replace('\\', '/'))

    def test_fallback_prefers_newer_version(self):
        with tempfile.TemporaryDirectory() as root:
            # 只有 12.0/15.0 目录，vms 为空（无法推断版本）
            build_fake_install(root, versions=('12.0', '15.0'), instance_names=())
            selected = self._probe_selected(root, 0)
            normalized = selected.replace('\\', '/')
            # 经典 shell/sdk 路径不存在时，兜底应选更高版本 15.0
            self.assertIn('nx_device/15.0', normalized)
            self.assertNotIn('nx_device/12.0', normalized)


class TestBgraLayout(unittest.TestCase):
    """像素通道序按 SDK 来源架构区分：新架构（nx_device/nx_main）为 BGRA，
    经典布局（shell/sdk）为 RGBA。实测 12.0 与 15.0 的 nx_device SDK 均为 BGRA。
    """

    def _build(self, version):
        import ctypes as _ctypes

        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        folder = os.path.join(root, 'nx_device', version, 'shell', 'sdk')
        os.makedirs(folder)
        with open(os.path.join(folder, 'external_renderer_ipc.dll'), 'w') as f:
            f.write('fake')
        original = _ctypes.CDLL

        class _FakeLib:
            pass

        _ctypes.CDLL = lambda path, *a, **k: _FakeLib()
        try:
            impl = NemuIpcImpl(
                nemu_folder=root, instance_id=0, version=version)
        finally:
            _ctypes.CDLL = original
        return impl

    def test_nx_device_layout_is_bgra(self):
        self.assertTrue(self._build('15.0').bgra_layout)
        self.assertTrue(self._build('12.0').bgra_layout)

    def test_classic_and_nx_main_is_not_handled_here(self):
        # 经典 shell/sdk 布局无法用 nx_device 目录构造，直接验证路径判定语义
        impl = self._build('12.0')
        impl.ipc_dll = r'C:\MuMu\shell\sdk\external_renderer_ipc.dll'
        self.assertFalse(impl.bgra_layout)
        impl.ipc_dll = r'C:\MuMu\nx_main\sdk\external_renderer_ipc.dll'
        self.assertTrue(impl.bgra_layout)


class TestDecideChannelOrder(unittest.TestCase):
    """ADB 真值帧通道序判定：与真值比对 RGBA / BGRA 两种解释的误差。"""

    def _make_raw(self, bgra_frame):
        """构造原始捕获帧：真实画面为 bgra_frame（RGB、方向已正），
        nemu_capture_display 返回 BGRA 序、上下颠倒。"""
        raw = cv2.cvtColor(bgra_frame, cv2.COLOR_RGB2BGRA)
        return cv2.flip(raw, 0)

    def test_bgra_frame_detected_as_bgra(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[:, :, 0] = 200  # 蓝色画面
        raw = self._make_raw(frame)
        self.assertEqual(NemuIpcImpl._decide_channel_order(raw, frame), 'bgra')

    def test_rgba_frame_detected_as_rgba(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[:, :, 2] = 200  # 红色画面
        raw = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
        raw = cv2.flip(raw, 0)
        self.assertEqual(NemuIpcImpl._decide_channel_order(raw, frame), 'rgba')

    def test_black_screen_returns_none(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        raw = self._make_raw(frame)
        self.assertIsNone(NemuIpcImpl._decide_channel_order(raw, frame))


class TestTouchCoordinateSpace(unittest.TestCase):
    """触摸坐标的显示缓冲旋转按模拟器版本区分。

    MuMu12 的 IPC 需要本层把标准坐标旋转成显示缓冲坐标；MuMu15 的
    external_renderer_ipc.dll 内部已经按 rotation=90 换算过，本层再旋转会让
    所有点击落到二次旋转后的位置，表现为「想点 A 却触发了 B」。
    """

    def test_version_gate(self):
        self.assertFalse(NemuIpcImpl._resolve_rotate_xy('15.0'))
        self.assertFalse(NemuIpcImpl._resolve_rotate_xy('16.0'))
        self.assertTrue(NemuIpcImpl._resolve_rotate_xy('12.0'))
        self.assertTrue(NemuIpcImpl._resolve_rotate_xy('3.8.13'))
        self.assertTrue(NemuIpcImpl._resolve_rotate_xy(None))
        self.assertTrue(NemuIpcImpl._resolve_rotate_xy('unknown'))

    def _build(self, version):
        import ctypes as _ctypes

        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        folder = os.path.join(root, 'nx_device', version, 'shell', 'sdk')
        os.makedirs(folder)
        with open(os.path.join(folder, 'external_renderer_ipc.dll'), 'w') as f:
            f.write('fake')
        original = _ctypes.CDLL

        class _FakeLib:
            pass

        _ctypes.CDLL = lambda path, *a, **k: _FakeLib()
        try:
            impl = NemuIpcImpl(nemu_folder=root, instance_id=0, version=version)
        finally:
            _ctypes.CDLL = original
        impl.height = 720
        return impl

    def test_mumu15_passes_touch_xy_through(self):
        impl = self._build('15.0')
        self.assertEqual(impl.convert_xy(17, 241), (17, 241))

    def test_mumu12_rotates_touch_xy(self):
        impl = self._build('12.0')
        self.assertEqual(impl.convert_xy(17, 241), (720 - 241, 17))


if __name__ == '__main__':
    unittest.main()
