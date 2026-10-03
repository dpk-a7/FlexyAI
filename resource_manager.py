"""Host and GPU safety checks for the isolated native inference runner.

The manager intentionally knows nothing about Flexy Apps.  A runner manager
registers its one child process, asks for an admission decision before loading,
and supplies a callback that stops that known child on an emergency.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import os
import time
from typing import Callable, Mapping, Sequence


GIB = 1024**3


class ResourceState(str, Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    HARD_STOP = "HARD_STOP"
    RECOVERING = "RECOVERING"


@dataclass(frozen=True)
class ResourceSettings:
    """Centralized resource thresholds, with conservative defaults."""

    ram_warning_percent: float = 80
    ram_critical_percent: float = 90
    ram_hard_stop_percent: float = 95
    ram_recovery_percent: float = 85
    swap_warning_percent: float = 25
    swap_critical_percent: float = 50
    swap_hard_stop_percent: float = 90
    swap_recovery_percent: float = 20
    offload_min_available_percent: float = 10
    offload_hard_stop_available_percent: float = 5
    offload_min_available_gb: float = 8
    offload_hard_stop_available_gb: float = 4
    offload_recovery_available_percent: float = 15
    offload_recovery_available_gb: float = 8
    gpu_temp_warning_c: float = 80
    gpu_temp_critical_c: float = 85
    gpu_temp_hard_stop_c: float = 90
    gpu_temp_recovery_c: float = 80
    gpu_temp_hard_stop_duration_seconds: float = 10
    resource_poll_seconds: float = 1

    @classmethod
    def from_environment(cls, environment: Mapping[str, object] | None = None) -> "ResourceSettings":
        return cls.from_mapping(os.environ if environment is None else environment)

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "ResourceSettings":
        defaults = cls()

        def number(name: str, default: float) -> float:
            raw = values.get(name, default)
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a number") from exc
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            return value

        settings = cls(
            ram_warning_percent=number("RAM_WARNING_PERCENT", defaults.ram_warning_percent),
            ram_critical_percent=number("RAM_CRITICAL_PERCENT", defaults.ram_critical_percent),
            ram_hard_stop_percent=number("RAM_HARD_STOP_PERCENT", defaults.ram_hard_stop_percent),
            ram_recovery_percent=number("RAM_RECOVERY_PERCENT", defaults.ram_recovery_percent),
            swap_warning_percent=number("SWAP_WARNING_PERCENT", defaults.swap_warning_percent),
            swap_critical_percent=number("SWAP_CRITICAL_PERCENT", defaults.swap_critical_percent),
            swap_hard_stop_percent=number("SWAP_HARD_STOP_PERCENT", defaults.swap_hard_stop_percent),
            swap_recovery_percent=number("SWAP_RECOVERY_PERCENT", defaults.swap_recovery_percent),
            offload_min_available_percent=number(
                "OFFLOAD_MIN_AVAILABLE_PERCENT", defaults.offload_min_available_percent),
            offload_hard_stop_available_percent=number(
                "OFFLOAD_HARD_STOP_AVAILABLE_PERCENT",
                defaults.offload_hard_stop_available_percent,
            ),
            offload_min_available_gb=number(
                "OFFLOAD_MIN_AVAILABLE_GB", defaults.offload_min_available_gb),
            offload_hard_stop_available_gb=number(
                "OFFLOAD_HARD_STOP_AVAILABLE_GB", defaults.offload_hard_stop_available_gb),
            offload_recovery_available_percent=number(
                "OFFLOAD_RECOVERY_AVAILABLE_PERCENT",
                defaults.offload_recovery_available_percent,
            ),
            offload_recovery_available_gb=number(
                "OFFLOAD_RECOVERY_AVAILABLE_GB", defaults.offload_recovery_available_gb),
            gpu_temp_warning_c=number("GPU_TEMP_WARNING", defaults.gpu_temp_warning_c),
            gpu_temp_critical_c=number("GPU_TEMP_CRITICAL", defaults.gpu_temp_critical_c),
            gpu_temp_hard_stop_c=number("GPU_TEMP_HARD_STOP", defaults.gpu_temp_hard_stop_c),
            gpu_temp_recovery_c=number("GPU_TEMP_RECOVERY", defaults.gpu_temp_recovery_c),
            gpu_temp_hard_stop_duration_seconds=number(
                "GPU_TEMP_HARD_STOP_DURATION_SECONDS",
                defaults.gpu_temp_hard_stop_duration_seconds,
            ),
            resource_poll_seconds=number("RESOURCE_POLL_SECONDS", defaults.resource_poll_seconds),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not 0 <= self.ram_warning_percent < self.ram_critical_percent < self.ram_hard_stop_percent <= 100:
            raise ValueError("RAM warning, critical, and hard-stop percentages must increase up to 100")
        if not 0 <= self.ram_recovery_percent < self.ram_critical_percent:
            raise ValueError("RAM_RECOVERY_PERCENT must be below RAM_CRITICAL_PERCENT")
        if not 0 <= self.swap_recovery_percent <= self.swap_warning_percent < self.swap_critical_percent < self.swap_hard_stop_percent <= 100:
            raise ValueError("Swap warning, critical, and hard-stop percentages must increase up to 100")
        if not 0 <= self.offload_hard_stop_available_percent <= self.offload_min_available_percent < self.offload_recovery_available_percent <= 100:
            raise ValueError("Offload available-RAM percentages must increase from hard stop to recovery")
        if not 0 < self.offload_hard_stop_available_gb <= self.offload_min_available_gb <= self.offload_recovery_available_gb:
            raise ValueError("Offload available-RAM GiB thresholds must increase from hard stop to recovery")
        if not 0 <= self.gpu_temp_recovery_c <= self.gpu_temp_warning_c < self.gpu_temp_critical_c < self.gpu_temp_hard_stop_c:
            raise ValueError("GPU temperature thresholds must increase from recovery to hard stop")
        if self.gpu_temp_hard_stop_duration_seconds < 0:
            raise ValueError("GPU_TEMP_HARD_STOP_DURATION_SECONDS cannot be negative")
        if self.resource_poll_seconds <= 0:
            raise ValueError("RESOURCE_POLL_SECONDS must be positive")


@dataclass(frozen=True)
class HostMetrics:
    total_ram_bytes: int
    available_ram_bytes: int
    used_ram_percent: float
    swap_usage_percent: float


@dataclass(frozen=True)
class GPUMetrics:
    id: int
    temperature_c: float
    vram_used_bytes: int
    vram_free_bytes: int
    vram_total_bytes: int
    utilization_percent: float | None = None
    power_usage_watts: float | None = None
    hardware_slowdown_temperature_c: float | None = None
    hardware_shutdown_temperature_c: float | None = None
    thermal_emergency: bool = False


@dataclass(frozen=True)
class RunnerInfo:
    """The only process the resource manager may ask to stop."""

    runner_id: str
    pid: int
    gpu_id: int
    cpu_offload: bool
    active: bool


@dataclass(frozen=True)
class HostStatus:
    total_ram_bytes: int
    available_ram_bytes: int
    available_ram_percent: float
    used_ram_percent: float
    swap_usage_percent: float
    swap_state: ResourceState
    state: ResourceState
    offload_state: ResourceState

    def to_dict(self) -> dict[str, float | str]:
        return {
            "ram_total_gb": _gib(self.total_ram_bytes),
            "ram_available_gb": _gib(self.available_ram_bytes),
            "ram_available_percent": _rounded(self.available_ram_percent),
            "ram_used_percent": _rounded(self.used_ram_percent),
            "swap_usage_percent": _rounded(self.swap_usage_percent),
            "swap_state": self.swap_state.value,
            "state": self.state.value,
            "offload_state": self.offload_state.value,
        }


@dataclass(frozen=True)
class GPUStatus:
    id: int
    temperature_c: float
    vram_used_bytes: int
    vram_free_bytes: int
    vram_total_bytes: int
    utilization_percent: float | None
    power_usage_watts: float | None
    temperature_hard_limit_c: float
    thermal_emergency: bool
    state: ResourceState

    def to_dict(self) -> dict[str, float | int | bool | str | None]:
        return {
            "id": self.id,
            "temperature_c": _rounded(self.temperature_c),
            "vram_used_gb": _gib(self.vram_used_bytes),
            "vram_free_gb": _gib(self.vram_free_bytes),
            "vram_total_gb": _gib(self.vram_total_bytes),
            "utilization_percent": _optional_rounded(self.utilization_percent),
            "power_usage_watts": _optional_rounded(self.power_usage_watts),
            "temperature_hard_limit_c": _rounded(self.temperature_hard_limit_c),
            "thermal_emergency": self.thermal_emergency,
            "state": self.state.value,
        }


@dataclass(frozen=True)
class ResourceSnapshot:
    host: HostStatus | None
    gpus: tuple[GPUStatus, ...]
    host_monitor_available: bool
    gpu_monitor_available: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "host": self.host.to_dict() if self.host else {"monitor_available": False},
            "gpus": [gpu.to_dict() for gpu in self.gpus],
            "host_monitor_available": self.host_monitor_available,
            "gpu_monitor_available": self.gpu_monitor_available,
        }


@dataclass(frozen=True)
class AdmissionDecision:
    allowed: bool
    code: str | None
    state: ResourceState | None

    def to_dict(self) -> dict[str, bool | str | None]:
        return {
            "allowed": self.allowed,
            "code": self.code,
            "state": self.state.value if self.state else None,
        }


HostSampler = Callable[[], HostMetrics]
GPUSampler = Callable[[], Sequence[GPUMetrics] | None]
StopRunner = Callable[[RunnerInfo, str], None]


class ResourceManager:
    """Centralizes admission, hysteresis, and emergency-stop decisions."""

    def __init__(
        self,
        settings: ResourceSettings | None = None,
        *,
        host_sampler: HostSampler | None = None,
        gpu_sampler: GPUSampler | None = None,
        monotonic_clock: Callable[[], float] | None = None,
        stop_runner: StopRunner | None = None,
    ) -> None:
        self.settings = settings or ResourceSettings.from_environment()
        self.settings.validate()
        self._host_sampler = host_sampler or default_host_sampler
        self._gpu_sampler = gpu_sampler or default_gpu_sampler
        self._clock = monotonic_clock or time.monotonic
        self._stop_runner = stop_runner
        self._runner: RunnerInfo | None = None
        self._ram_load_blocked = False
        self._swap_load_blocked = False
        self._offload_load_blocked = False
        self._gpu_blocked: dict[int, bool] = {}
        self._gpu_hard_started_at: dict[int, float] = {}
        self._hard_stop_notified = False
        self._last_snapshot: ResourceSnapshot | None = None

    @property
    def last_snapshot(self) -> ResourceSnapshot | None:
        return self._last_snapshot

    def set_runner(self, runner: RunnerInfo | None) -> None:
        self._runner = runner
        if runner is None:
            self._hard_stop_notified = False

    def poll(self) -> ResourceSnapshot:
        """Read monitoring data once and notify the registered runner on hard stop."""
        host = self._sample_host()
        gpus, gpu_monitor_available = self._sample_gpus()
        snapshot = ResourceSnapshot(
            host=host,
            gpus=tuple(self._gpu_status(metric) for metric in gpus),
            host_monitor_available=host is not None,
            gpu_monitor_available=gpu_monitor_available,
        )
        self._last_snapshot = snapshot
        self._notify_hard_stop_if_needed(snapshot)
        return snapshot

    def admission_decision(self, *, gpu_id: int, cpu_offload: bool) -> AdmissionDecision:
        """Return whether a new native runner may load on one physical GPU."""
        snapshot = self.poll()
        if snapshot.host is None:
            return AdmissionDecision(False, "HOST_MONITOR_UNAVAILABLE", None)
        if snapshot.host.state is ResourceState.HARD_STOP:
            return AdmissionDecision(False, "HOST_RAM_HARD_STOP", snapshot.host.state)
        if snapshot.host.state in (ResourceState.CRITICAL, ResourceState.RECOVERING):
            return AdmissionDecision(False, "RAM_LOADS_BLOCKED", snapshot.host.state)
        if snapshot.host.swap_state is ResourceState.HARD_STOP:
            return AdmissionDecision(False, "SWAP_HARD_STOP", snapshot.host.swap_state)
        if snapshot.host.swap_state in (ResourceState.CRITICAL, ResourceState.RECOVERING):
            return AdmissionDecision(False, "SWAP_LOADS_BLOCKED", snapshot.host.swap_state)
        if cpu_offload:
            if snapshot.host.offload_state is ResourceState.HARD_STOP:
                return AdmissionDecision(False, "OFFLOAD_RAM_HARD_STOP", snapshot.host.offload_state)
            if snapshot.host.offload_state in (ResourceState.CRITICAL, ResourceState.RECOVERING):
                return AdmissionDecision(False, "OFFLOAD_RAM_TOO_LOW", snapshot.host.offload_state)
        if not snapshot.gpu_monitor_available:
            return AdmissionDecision(False, "GPU_MONITOR_UNAVAILABLE", None)
        gpu = next((item for item in snapshot.gpus if item.id == gpu_id), None)
        if gpu is None:
            return AdmissionDecision(False, "GPU_UNAVAILABLE", None)
        if gpu.state is ResourceState.HARD_STOP:
            return AdmissionDecision(False, "GPU_HARD_STOP", gpu.state)
        if gpu.state in (ResourceState.CRITICAL, ResourceState.RECOVERING):
            return AdmissionDecision(False, "GPU_THERMAL_BLOCKED", gpu.state)
        return AdmissionDecision(True, None, gpu.state)

    def _sample_host(self) -> HostStatus | None:
        try:
            metrics = self._host_sampler()
            if metrics.total_ram_bytes <= 0:
                raise ValueError("total RAM must be positive")
            if metrics.available_ram_bytes < 0:
                raise ValueError("available RAM cannot be negative")
        except Exception:
            return None

        available_percent = metrics.available_ram_bytes * 100 / metrics.total_ram_bytes
        if metrics.used_ram_percent >= self.settings.ram_critical_percent:
            self._ram_load_blocked = True
        elif self._ram_load_blocked and metrics.used_ram_percent < self.settings.ram_recovery_percent:
            self._ram_load_blocked = False

        if metrics.swap_usage_percent >= self.settings.swap_critical_percent:
            self._swap_load_blocked = True
        elif self._swap_load_blocked and metrics.swap_usage_percent < self.settings.swap_recovery_percent:
            self._swap_load_blocked = False

        admission_limit = self._offload_limit(
            self.settings.offload_min_available_percent,
            self.settings.offload_min_available_gb,
            metrics.total_ram_bytes,
        )
        recovery_limit = self._offload_limit(
            self.settings.offload_recovery_available_percent,
            self.settings.offload_recovery_available_gb,
            metrics.total_ram_bytes,
        )
        hard_stop_limit = self._offload_limit(
            self.settings.offload_hard_stop_available_percent,
            self.settings.offload_hard_stop_available_gb,
            metrics.total_ram_bytes,
        )
        if metrics.available_ram_bytes < admission_limit:
            self._offload_load_blocked = True
        elif self._offload_load_blocked and metrics.available_ram_bytes > recovery_limit:
            self._offload_load_blocked = False

        if metrics.used_ram_percent >= self.settings.ram_hard_stop_percent:
            ram_state = ResourceState.HARD_STOP
        elif self._ram_load_blocked:
            ram_state = (
                ResourceState.CRITICAL
                if metrics.used_ram_percent >= self.settings.ram_critical_percent
                else ResourceState.RECOVERING
            )
        elif metrics.used_ram_percent >= self.settings.ram_warning_percent:
            ram_state = ResourceState.WARNING
        else:
            ram_state = ResourceState.NORMAL

        if metrics.available_ram_bytes < hard_stop_limit:
            offload_state = ResourceState.HARD_STOP
        elif self._offload_load_blocked:
            offload_state = (
                ResourceState.CRITICAL
                if metrics.available_ram_bytes < admission_limit
                else ResourceState.RECOVERING
            )
        else:
            offload_state = ResourceState.NORMAL

        if metrics.swap_usage_percent >= self.settings.swap_hard_stop_percent:
            swap_state = ResourceState.HARD_STOP
        elif self._swap_load_blocked:
            swap_state = (
                ResourceState.CRITICAL
                if metrics.swap_usage_percent >= self.settings.swap_critical_percent
                else ResourceState.RECOVERING
            )
        elif metrics.swap_usage_percent >= self.settings.swap_warning_percent:
            swap_state = ResourceState.WARNING
        else:
            swap_state = ResourceState.NORMAL

        return HostStatus(
            total_ram_bytes=metrics.total_ram_bytes,
            available_ram_bytes=metrics.available_ram_bytes,
            available_ram_percent=available_percent,
            used_ram_percent=metrics.used_ram_percent,
            swap_usage_percent=metrics.swap_usage_percent,
            swap_state=swap_state,
            state=ram_state,
            offload_state=offload_state,
        )

    def _sample_gpus(self) -> tuple[tuple[GPUMetrics, ...], bool]:
        try:
            metrics = self._gpu_sampler()
        except Exception:
            return (), False
        if metrics is None:
            return (), False
        return tuple(metrics), True

    def _gpu_status(self, metrics: GPUMetrics) -> GPUStatus:
        hard_limit = self._gpu_hard_limit(metrics)
        now = self._clock()
        is_hard = metrics.thermal_emergency or metrics.temperature_c >= hard_limit
        if is_hard:
            started_at = self._gpu_hard_started_at.setdefault(metrics.id, now)
            hard_stop = metrics.thermal_emergency or (
                now - started_at >= self.settings.gpu_temp_hard_stop_duration_seconds
            )
        else:
            self._gpu_hard_started_at.pop(metrics.id, None)
            hard_stop = False

        if metrics.temperature_c >= self.settings.gpu_temp_critical_c:
            self._gpu_blocked[metrics.id] = True
        elif self._gpu_blocked.get(metrics.id) and metrics.temperature_c < self.settings.gpu_temp_recovery_c:
            self._gpu_blocked[metrics.id] = False

        if hard_stop:
            state = ResourceState.HARD_STOP
        elif self._gpu_blocked.get(metrics.id):
            state = (
                ResourceState.CRITICAL
                if metrics.temperature_c >= self.settings.gpu_temp_critical_c
                else ResourceState.RECOVERING
            )
        elif metrics.temperature_c >= self.settings.gpu_temp_warning_c:
            state = ResourceState.WARNING
        else:
            state = ResourceState.NORMAL
        return GPUStatus(
            id=metrics.id,
            temperature_c=metrics.temperature_c,
            vram_used_bytes=metrics.vram_used_bytes,
            vram_free_bytes=metrics.vram_free_bytes,
            vram_total_bytes=metrics.vram_total_bytes,
            utilization_percent=metrics.utilization_percent,
            power_usage_watts=metrics.power_usage_watts,
            temperature_hard_limit_c=hard_limit,
            thermal_emergency=metrics.thermal_emergency,
            state=state,
        )

    def _notify_hard_stop_if_needed(self, snapshot: ResourceSnapshot) -> None:
        reason = self._hard_stop_reason(snapshot)
        if reason is None:
            self._hard_stop_notified = False
            return
        if self._hard_stop_notified or self._runner is None or self._stop_runner is None:
            return
        self._hard_stop_notified = True
        self._stop_runner(self._runner, reason)

    def _hard_stop_reason(self, snapshot: ResourceSnapshot) -> str | None:
        if self._runner is None or snapshot.host is None:
            return None
        if snapshot.host.state is ResourceState.HARD_STOP:
            return "HOST_RAM_HARD_STOP"
        if snapshot.host.swap_state is ResourceState.HARD_STOP:
            return "SWAP_HARD_STOP"
        if self._runner.cpu_offload and snapshot.host.offload_state is ResourceState.HARD_STOP:
            return "OFFLOAD_RAM_HARD_STOP"
        gpu = next((item for item in snapshot.gpus if item.id == self._runner.gpu_id), None)
        if gpu and gpu.state is ResourceState.HARD_STOP:
            return "GPU_TEMPERATURE_HARD_STOP"
        return None

    def _offload_limit(self, percent: float, gib: float, total_ram_bytes: int) -> float:
        return max(total_ram_bytes * percent / 100, gib * GIB)

    def _gpu_hard_limit(self, metrics: GPUMetrics) -> float:
        hardware_limits = [
            limit
            for limit in (
                metrics.hardware_slowdown_temperature_c,
                metrics.hardware_shutdown_temperature_c,
            )
            if limit is not None and limit > 0
        ]
        return min([self.settings.gpu_temp_hard_stop_c, *hardware_limits])


def default_host_sampler() -> HostMetrics:
    """Use psutil's available-RAM definition, which accounts for file cache."""
    import psutil

    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return HostMetrics(
        total_ram_bytes=int(memory.total),
        available_ram_bytes=int(memory.available),
        used_ram_percent=float(memory.percent),
        swap_usage_percent=float(swap.percent),
    )


_nvml_module = None
_nvml_initialized = False


def default_gpu_sampler() -> tuple[GPUMetrics, ...]:
    """Read NVIDIA telemetry through NVML without exposing it to callers."""
    global _nvml_initialized, _nvml_module
    if _nvml_module is None:
        import pynvml

        _nvml_module = pynvml
    nvml = _nvml_module
    if not _nvml_initialized:
        nvml.nvmlInit()
        _nvml_initialized = True
    metrics = []
    for gpu_id in range(nvml.nvmlDeviceGetCount()):
        handle = nvml.nvmlDeviceGetHandleByIndex(gpu_id)
        memory = nvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = _nvml_optional(lambda: nvml.nvmlDeviceGetUtilizationRates(handle))
        power_milliwatts = _nvml_optional(lambda: nvml.nvmlDeviceGetPowerUsage(handle))
        throttle_reasons = _nvml_optional(lambda: nvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle), 0)
        thermal_flag = getattr(nvml, "nvmlClocksThrottleReasonHwThermalSlowdown", 0)
        metrics.append(GPUMetrics(
            id=gpu_id,
            temperature_c=float(nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)),
            vram_used_bytes=int(memory.used),
            vram_free_bytes=int(memory.free),
            vram_total_bytes=int(memory.total),
            utilization_percent=float(utilization.gpu) if utilization is not None else None,
            power_usage_watts=float(power_milliwatts) / 1000 if power_milliwatts is not None else None,
            hardware_slowdown_temperature_c=_nvml_temperature_threshold(
                nvml, handle, "NVML_TEMPERATURE_THRESHOLD_SLOWDOWN"),
            hardware_shutdown_temperature_c=_nvml_temperature_threshold(
                nvml, handle, "NVML_TEMPERATURE_THRESHOLD_SHUTDOWN"),
            thermal_emergency=bool(throttle_reasons & thermal_flag),
        ))
    return tuple(metrics)


def shutdown_gpu_monitor() -> None:
    """Release NVML when the application lifespan ends."""
    global _nvml_initialized
    if _nvml_module is not None and _nvml_initialized:
        _nvml_module.nvmlShutdown()
        _nvml_initialized = False


def _nvml_optional(call: Callable[[], object], default: object = None) -> object:
    try:
        return call()
    except Exception:
        return default


def _nvml_temperature_threshold(nvml: object, handle: object, name: str) -> float | None:
    threshold = getattr(nvml, name, None)
    if threshold is None:
        return None
    result = _nvml_optional(lambda: nvml.nvmlDeviceGetTemperatureThreshold(handle, threshold))
    return float(result) if result is not None else None


def _gib(value: int) -> float:
    return _rounded(value / GIB)


def _rounded(value: float) -> float:
    return round(value, 2)


def _optional_rounded(value: float | None) -> float | None:
    return _rounded(value) if value is not None else None
