"""Build a platform wheel with a Rust C-ABI library, without importing torch."""
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py
from setuptools.command.bdist_wheel import bdist_wheel

ROOT = Path(__file__).resolve().parent


class NativeDistribution(Distribution):
    def has_ext_modules(self):
        return True


class BuildPy(build_py):
    def run(self):
        super().run()
        target = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "build/cargo")).resolve()
        environment = dict(os.environ, CARGO_TARGET_DIR=str(target))
        flags = (environment.get("CARGO_ENCODED_RUSTFLAGS", "").split("\x1f")
                 if environment.get("CARGO_ENCODED_RUSTFLAGS")
                 else shlex.split(environment.get("RUSTFLAGS", "")))
        flags += ["--remap-path-prefix", f"{ROOT}=vitni-tensor"]
        if environment.get("CARGO_HOME"):
            flags += ["--remap-path-prefix", f"{Path(environment['CARGO_HOME']).resolve()}=cargo-registry"]
        if sys.platform == "darwin":
            flags += ["-C", "link-arg=-Wl,-install_name,@rpath/libvitni_torch_native.dylib"]
        environment["CARGO_ENCODED_RUSTFLAGS"] = "\x1f".join(flags)
        subprocess.run(["cargo", "build", "--release", "--locked", "--manifest-path",
                        str(ROOT / "bindings/torch/native/Cargo.toml")],
                       cwd=ROOT, env=environment, check=True)
        name = ("vitni_torch_native.dll" if sys.platform=="win32" else
                "libvitni_torch_native.dylib" if sys.platform=="darwin" else
                "libvitni_torch_native.so")
        destination = Path(self.build_lib) / "vitni_torch"
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target / "release" / name, destination / name)
        shaders = destination / "_shaders"
        shaders.mkdir(exist_ok=True)
        for source in ["canonical_matmul.metal", "q4k.metal"]:
            shutil.copy2(ROOT / "kernels" / source, shaders / source)


class PlatformWheel(bdist_wheel):
    def get_tag(self):
        _, _, platform = super().get_tag()
        return "py3", "none", platform


setup(distclass=NativeDistribution,
      cmdclass={"build_py": BuildPy, "bdist_wheel": PlatformWheel})
