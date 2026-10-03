import unittest

from resource_manager import (
    GIB,
    GPUMetrics,
    HostMetrics,
    ResourceManager,
    ResourceSettings,
    ResourceState,
    RunnerInfo,
)


class SequenceSampler:
    def __init__(self, values):
        self.values = list(values)
        self.index = 0

    def __call__(self):
        value = self.values[min(self.index, len(self.values) - 1)]
        self.index += 1
        return value


class Clock:
    def __init__(self, values):
        self.values = list(values)
        self.index = 0

    def __call__(self):
        value = self.values[min(self.index, len(self.values) - 1)]
        self.index += 1
        return value


def host(*, total_gb=128, available_gb=40, used_percent=70, swap_percent=0):
    return HostMetrics(
        total_ram_bytes=total_gb * GIB,
        available_ram_bytes=available_gb * GIB,
        used_ram_percent=used_percent,
        swap_usage_percent=swap_percent,
    )


def gpu(*, gpu_id=0, temperature=70, used_gb=8, total_gb=24, emergency=False):
    return GPUMetrics(
        id=gpu_id,
        temperature_c=temperature,
        vram_used_bytes=used_gb * GIB,
        vram_free_bytes=(total_gb - used_gb) * GIB,
        vram_total_bytes=total_gb * GIB,
        utilization_percent=50,
        power_usage_watts=150,
        thermal_emergency=emergency,
    )


class ResourceManagerTests(unittest.TestCase):
    def manager(self, hosts, gpus, *, clock=None, stopped=None, settings=None):
        return ResourceManager(
            settings=settings or ResourceSettings(),
            host_sampler=SequenceSampler(hosts),
            gpu_sampler=SequenceSampler(gpus),
            monotonic_clock=clock or Clock([0]),
            stop_runner=stopped,
        )

    def test_settings_are_loaded_from_mapping_and_invalid_order_is_rejected(self):
        settings = ResourceSettings.from_mapping({
            "RAM_WARNING_PERCENT": "75",
            "SWAP_HARD_STOP_PERCENT": "85",
            "GPU_TEMP_HARD_STOP_DURATION_SECONDS": "5",
        })
        self.assertEqual(settings.ram_warning_percent, 75)
        self.assertEqual(settings.swap_hard_stop_percent, 85)
        self.assertEqual(settings.gpu_temp_hard_stop_duration_seconds, 5)
        with self.assertRaisesRegex(ValueError, "RAM warning"):
            ResourceSettings.from_mapping({
                "RAM_WARNING_PERCENT": "95",
                "RAM_CRITICAL_PERCENT": "90",
            })

    def test_swap_pressure_blocks_then_recovers_and_hard_stops_the_runner(self):
        stopped = []
        manager = self.manager(
            [
                host(swap_percent=55),
                host(swap_percent=95),
                host(swap_percent=40),
                host(swap_percent=10),
            ],
            [[gpu()], [gpu()], [gpu()], [gpu()]],
            stopped=lambda runner, reason: stopped.append(reason),
        )
        manager.set_runner(RunnerInfo("runner-1", 42, 0, cpu_offload=True, active=True))

        decision = manager.admission_decision(gpu_id=0, cpu_offload=True)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.code, "SWAP_LOADS_BLOCKED")
        self.assertEqual(manager.poll().host.swap_state, ResourceState.HARD_STOP)
        self.assertEqual(manager.poll().host.swap_state, ResourceState.RECOVERING)
        self.assertEqual(manager.poll().host.swap_state, ResourceState.NORMAL)
        self.assertEqual(stopped, ["SWAP_HARD_STOP"])

    def test_ram_load_block_uses_hysteresis(self):
        manager = self.manager(
            [
                host(used_percent=91),
                host(used_percent=88),
                host(used_percent=84),
            ],
            [[gpu()], [gpu()], [gpu()]],
        )
        self.assertEqual(manager.poll().host.state, ResourceState.CRITICAL)
        self.assertEqual(manager.poll().host.state, ResourceState.RECOVERING)
        self.assertEqual(manager.poll().host.state, ResourceState.WARNING)

    def test_offload_ram_limits_block_then_recover_and_hard_stop_runner(self):
        stopped = []
        manager = self.manager(
            [
                host(available_gb=12),
                host(available_gb=6),
                host(available_gb=20),
            ],
            [[gpu()], [gpu()], [gpu()]],
            stopped=lambda runner, reason: stopped.append((runner.runner_id, reason)),
        )
        manager.set_runner(RunnerInfo("runner-1", 42, 0, cpu_offload=True, active=True))

        self.assertEqual(manager.poll().host.offload_state, ResourceState.CRITICAL)
        self.assertEqual(manager.poll().host.offload_state, ResourceState.HARD_STOP)
        self.assertEqual(stopped, [("runner-1", "OFFLOAD_RAM_HARD_STOP")])
        self.assertEqual(manager.poll().host.offload_state, ResourceState.NORMAL)

    def test_vram_exhaustion_without_ram_pressure_does_not_stop_an_offloading_runner(self):
        stopped = []
        manager = self.manager(
            [host(available_gb=40, used_percent=70)],
            [[gpu(used_gb=24, total_gb=24)]],
            stopped=lambda runner, reason: stopped.append(reason),
        )
        manager.set_runner(RunnerInfo("runner-1", 42, 0, cpu_offload=True, active=True))

        snapshot = manager.poll()
        self.assertEqual(snapshot.host.state, ResourceState.NORMAL)
        self.assertEqual(snapshot.host.offload_state, ResourceState.NORMAL)
        self.assertEqual(snapshot.gpus[0].vram_free_bytes, 0)
        self.assertEqual(stopped, [])

    def test_temperature_requires_sustained_hard_limit_and_recovers_after_cooling(self):
        stopped = []
        manager = self.manager(
            [host()] * 5,
            [[gpu(temperature=90)], [gpu(temperature=90)], [gpu(temperature=90)],
             [gpu(temperature=82)], [gpu(temperature=79)]],
            clock=Clock([0, 9, 10, 11, 12]),
            stopped=lambda runner, reason: stopped.append(reason),
        )
        manager.set_runner(RunnerInfo("runner-1", 42, 0, cpu_offload=True, active=True))

        self.assertEqual(manager.poll().gpus[0].state, ResourceState.CRITICAL)
        self.assertEqual(manager.poll().gpus[0].state, ResourceState.CRITICAL)
        self.assertEqual(manager.poll().gpus[0].state, ResourceState.HARD_STOP)
        self.assertEqual(manager.poll().gpus[0].state, ResourceState.RECOVERING)
        self.assertEqual(manager.poll().gpus[0].state, ResourceState.NORMAL)
        self.assertEqual(stopped, ["GPU_TEMPERATURE_HARD_STOP"])

    def test_hardware_thermal_emergency_stops_immediately(self):
        stopped = []
        manager = self.manager(
            [host()],
            [[gpu(temperature=70, emergency=True)]],
            stopped=lambda runner, reason: stopped.append(reason),
        )
        manager.set_runner(RunnerInfo("runner-1", 42, 0, cpu_offload=True, active=True))

        self.assertEqual(manager.poll().gpus[0].state, ResourceState.HARD_STOP)
        self.assertEqual(stopped, ["GPU_TEMPERATURE_HARD_STOP"])

    def test_hot_unused_gpu_does_not_stop_runner_on_another_gpu(self):
        stopped = []
        immediate = ResourceSettings(gpu_temp_hard_stop_duration_seconds=0)
        manager = self.manager(
            [host()],
            [[gpu(gpu_id=0, temperature=70), gpu(gpu_id=1, temperature=95)]],
            stopped=lambda runner, reason: stopped.append(reason),
            settings=immediate,
        )
        manager.set_runner(RunnerInfo("runner-1", 42, 0, cpu_offload=True, active=True))

        snapshot = manager.poll()
        self.assertEqual(snapshot.gpus[1].state, ResourceState.HARD_STOP)
        self.assertEqual(stopped, [])

    def test_missing_gpu_monitor_blocks_admission_and_snapshot_hides_runner_details(self):
        manager = ResourceManager(
            host_sampler=lambda: host(),
            gpu_sampler=lambda: None,
            monotonic_clock=lambda: 0,
        )
        manager.set_runner(RunnerInfo("private-runner", 42, 0, cpu_offload=True, active=True))

        decision = manager.admission_decision(gpu_id=0, cpu_offload=True)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.code, "GPU_MONITOR_UNAVAILABLE")
        serialized = manager.last_snapshot.to_dict()
        self.assertNotIn("runner", str(serialized).lower())
        self.assertNotIn("pid", str(serialized).lower())

    def test_host_monitor_failure_blocks_admission_without_killing_runner(self):
        stopped = []
        manager = ResourceManager(
            host_sampler=lambda: (_ for _ in ()).throw(RuntimeError("not available")),
            gpu_sampler=lambda: [gpu()],
            monotonic_clock=lambda: 0,
            stop_runner=lambda runner, reason: stopped.append(reason),
        )
        manager.set_runner(RunnerInfo("runner-1", 42, 0, cpu_offload=True, active=True))

        decision = manager.admission_decision(gpu_id=0, cpu_offload=True)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.code, "HOST_MONITOR_UNAVAILABLE")
        self.assertEqual(stopped, [])


if __name__ == "__main__":
    unittest.main()
