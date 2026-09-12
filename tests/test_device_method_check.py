"""method_check 的截图/控制方式组合检查单元测试。

覆盖 nemu_ipc 截图与控制的强制配套，以及配套检查位于末尾时
非 MuMu 环境回退 auto 不会让控制方式悬空在 nemu_ipc 上。
"""

import unittest
from types import SimpleNamespace

from module.device.device import Device


def make_stub(screenshot, control, is_mumu=True):
    """构造满足 method_check 属性访问的最小 self 桩。"""
    return SimpleNamespace(
        config=SimpleNamespace(
            Emulator_ScreenshotMethod=screenshot,
            Emulator_ControlMethod=control,
        ),
        is_vmos=False,
        is_emulator=True,
        is_mumu_family=is_mumu,
        is_ldplayer_bluestacks_family=False,
        sdk_ver=31,
    )


class TestMethodCheck(unittest.TestCase):
    def test_nemu_screenshot_forces_nemu_control(self):
        stub = make_stub('nemu_ipc', 'MaaTouch')
        Device.method_check(stub)
        self.assertEqual(stub.config.Emulator_ControlMethod, 'nemu_ipc')

    def test_nemu_control_without_nemu_screenshot_falls_back(self):
        stub = make_stub('DroidCast', 'nemu_ipc')
        Device.method_check(stub)
        self.assertEqual(stub.config.Emulator_ControlMethod, 'minitouch')
        # 截图方式本身合法，不应被改动
        self.assertEqual(stub.config.Emulator_ScreenshotMethod, 'DroidCast')

    def test_nemu_screenshot_on_non_mumu_no_dangling_control(self):
        stub = make_stub('nemu_ipc', 'MaaTouch', is_mumu=False)
        Device.method_check(stub)
        # 截图回退 auto 后，控制不应悬空在 nemu_ipc 上
        self.assertEqual(stub.config.Emulator_ScreenshotMethod, 'auto')
        self.assertEqual(stub.config.Emulator_ControlMethod, 'MaaTouch')

    def test_both_nemu_unchanged(self):
        stub = make_stub('nemu_ipc', 'nemu_ipc')
        Device.method_check(stub)
        self.assertEqual(stub.config.Emulator_ScreenshotMethod, 'nemu_ipc')
        self.assertEqual(stub.config.Emulator_ControlMethod, 'nemu_ipc')


if __name__ == '__main__':
    unittest.main()
