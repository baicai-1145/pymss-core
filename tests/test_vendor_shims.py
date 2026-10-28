"""The shims that route around missing vendor kernels."""
import pytest
import torch

from pymss_core.vendor_shims import (
    complex_planes_product,
    install_vendor_shims,
    installed_shims,
    reset_shims,
)


class TestComplexPlanesProduct:
    def test_real_times_complex_matches_reference(self):
        torch.manual_seed(0)
        mask = torch.rand(2, 5, 7)
        spec = torch.randn(2, 5, 7, dtype=torch.complex64)
        got = complex_planes_product(mask, spec)
        assert got.dtype == torch.complex64
        assert torch.equal(got, mask.to(torch.complex64) * spec)

    def test_complex_times_real_is_commutative(self):
        torch.manual_seed(1)
        mask = torch.rand(2, 5, 7)
        spec = torch.randn(2, 5, 7, dtype=torch.complex64)
        assert torch.equal(complex_planes_product(spec, mask), complex_planes_product(mask, spec))

    def test_complex_times_complex_matches_reference(self):
        torch.manual_seed(2)
        a = torch.randn(2, 4, 6, dtype=torch.complex64)
        b = torch.randn(2, 4, 6, dtype=torch.complex64)
        assert torch.allclose(complex_planes_product(a, b), a * b, atol=1e-5)

    def test_broadcast_mask_over_channels(self):
        mask = torch.rand(1, 5, 7)
        spec = torch.randn(3, 5, 7, dtype=torch.complex64)
        assert complex_planes_product(mask, spec).shape == (3, 5, 7)


class TestShimInstallation:
    def test_cpu_installs_nothing(self):
        reset_shims()
        assert install_vendor_shims("cpu") == []
        assert installed_shims() == frozenset()

    def test_profile_flags_choose_the_shims(self):
        reset_shims()
        installed = install_vendor_shims("supa", complex_mul=False, window_on_device=False, glu=True)
        assert set(installed) == {"complex_mul", "window", "glu"}
        assert install_vendor_shims("supa", complex_mul=False, window_on_device=False, glu=True) == []
        reset_shims()

    def test_window_shim_builds_on_host_for_device_targets(self):
        reset_shims()
        install_vendor_shims("supa", complex_mul=False, window_on_device=False, glu=False)
        # "meta" keeps this test device-independent while exercising the shim path.
        window = torch.hann_window(8, device="meta")
        assert window.device.type == "meta"
        reset_shims()


class TestGluShim:
    def test_equivalent_matches_the_fused_kernel(self):
        import torch.nn.functional as F

        from pymss_core.vendor_shims import glu_equivalent

        torch.manual_seed(4)
        x = torch.randn(2, 8, 5)
        assert torch.allclose(glu_equivalent(x, 1), F.glu(x, 1), atol=1e-6)

    def test_shim_leaves_cpu_calls_alone(self):
        import torch.nn.functional as F

        from pymss_core.vendor_shims import install_glu_shim, reset_shims

        reset_shims()
        original = F.glu
        install_glu_shim()
        x = torch.randn(2, 8, 5)
        assert torch.allclose(F.glu(x, 1), original(x, 1), atol=1e-6)
        reset_shims()
