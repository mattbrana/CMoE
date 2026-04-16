# energy_tracker.py
import threading
import time
import wandb
import pynvml


class EnergyTracker:
    """Polls GPU power and integrates to get energy in Joules."""

    def __init__(self, gpu_index=0, poll_interval=1.0):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self.poll_interval = poll_interval
        self.total_energy_j = 0.0
        self._running = False
        self._thread = None

    def _poll(self):
        last_time = time.time()
        while self._running:
            now = time.time()
            dt = now - last_time
            last_time = now
            power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
            power_w = power_mw / 1000.0
            self.total_energy_j += power_w * dt
            wandb.log({
                "energy/gpu_power_w": power_w,
                "energy/total_energy_j": self.total_energy_j,
                "energy/total_energy_kwh": self.total_energy_j / 3_600_000,
            })
            time.sleep(self.poll_interval)

    def start(self):
        self.total_energy_j = 0.0  # reset on each start
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self, phase_name=None):
        self._running = False
        self._thread.join()
        if phase_name:
            wandb.log({f"phase/{phase_name}_joules": self.total_energy_j})
            wandb.run.summary[f"phase/{phase_name}_joules"] = self.total_energy_j
        return self.total_energy_j

    def shutdown(self):
        pynvml.nvmlShutdown()
        wandb.run.summary["energy/total_joules"] = self.total_energy_j
        wandb.run.summary["energy/total_kwh"] = self.total_energy_j / 3_600_000
