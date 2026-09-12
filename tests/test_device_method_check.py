"""method_check 的截图/控制方式组合检查单元测试。

nemu_ipc 控制不做强制配套：截图与控制通路相互独立，混搭可用，
用户手动选择的控制方式不被静默改写；仅修正非法组合
（Hermit 仅限 VMOS、非对应模拟器的专属截图方式回退 auto）。
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
    def test_nemu_screenshot_keeps_user_control(self):
        """截图 nemu_ipc + 手动选择的 MaaTouch 控制不被强制改写。"""
        stub = make_stub('nemu_ipc', 'MaaTouch')
        Device.method_check(stub)
        self.assertEqual(stub.config.Emulator_ScreenshotMethod, 'nemu_ipc')
        self.assertEqual(stub.config.Emulator_ControlMethod, 'MaaTouch')

    def test_nemu_control_without_nemu_screenshot_respected(self):
        """用户显式选择 nemu_ipc 控制 + 其他截图方式：尊重选择不改写。"""
        stub = make_stub('DroidCast', 'nemu_ipc')
        Device.method_check(stub)
        self.assertEqual(stub.config.Emulator_ControlMethod, 'nemu_ipc')
        self.assertEqual(stub.config.Emulator_ScreenshotMethod, 'DroidCast')

    def test_nemu_screenshot_on_non_mumu_falls_back_to_auto(self):
        stub = make_stub('nemu_ipc', 'MaaTouch', is_mumu=False)
        Device.method_check(stub)
        self.assertEqual(stub.config.Emulator_ScreenshotMethod, 'auto')
        self.assertEqual(stub.config.Emulator_ControlMethod, 'MaaTouch')

    def test_both_nemu_unchanged(self):
        stub = make_stub('nemu_ipc', 'nemu_ipc')
        Device.method_check(stub)
        self.assertEqual(stub.config.Emulator_ScreenshotMethod, 'nemu_ipc')
        self.assertEqual(stub.config.Emulator_ControlMethod, 'nemu_ipc')


if __name__ == '__main__':
    unittest.main()
