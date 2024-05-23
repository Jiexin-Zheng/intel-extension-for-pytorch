from torch.testing._internal.common_utils import TestCase


class TestTorchXPUMethod(TestCase):
    def test_event_record(self):
        import torch
        import intel_extension_for_pytorch  # noqa F401

        ev = torch.xpu.Event(enable_timing=True)
        # before the fix,
        # AttributeError: module 'intel_extension_for_pytorch #noqa' has no attribute 'current_stream'
        ev.record()
        ev.wait()

    def test_event_elapsed_time(self):
        import torch
        import intel_extension_for_pytorch  # noqa F401

        t1 = torch.rand(1024, 1024).to("xpu")
        t2 = torch.rand(1024, 1024).to("xpu")
        torch.xpu.synchronize()
        start_event = torch.xpu.Event(enable_timing=True)
        start_event.record()
        t2 = t1 * t2
        t1 = t1 + t2
        end_event = torch.xpu.Event(enable_timing=True)
        end_event.record()
        end_event.synchronize()
        with self.assertRaisesRegex(
            NotImplementedError, "elapsed_time is not supported by XPUEvent."
        ):
            start_event.elapsed_time(end_event)
