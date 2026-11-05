"""Run blocking OpenPI inference in a killable, single-owner process."""

import multiprocessing as mp
import time
import traceback

import numpy as np
from settings import CONTRACT


def _worker(connection, host, port, mock, delay, fps):
    try:
        if not mock:
            from openpi_client.websocket_client_policy import WebsocketClientPolicy

            policy = WebsocketClientPolicy(host=host, port=port)
            metadata = policy.get_server_metadata()
            if metadata.get("ur5_dagger_contract") != CONTRACT:
                raise ValueError("Server has a different action contract; use this folder's serve.py")
            if metadata.get("fps") != fps:
                raise ValueError("Server training FPS differs from the collector's control FPS")
        connection.send(("ready", None))
        while True:
            request = connection.recv()
            if request is None:
                break
            generation, observed_at, observation = request
            if mock:
                time.sleep(delay)
                actions = np.zeros((10, 7), dtype=np.float32)
                actions[:, 0] = 0.0002
                actions[:, 6] = observation["grasp_command"]
            else:
                actions = np.asarray(policy.infer(observation)["actions"], dtype=np.float32)
            connection.send(("result", (generation, observed_at, actions)))
    except (EOFError, BrokenPipeError):
        pass
    except BaseException:
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


class PolicyWorker:
    def __init__(self, cfg, mock=False, fps=30):
        self.cfg = cfg
        ctx = mp.get_context("spawn")
        self.connection, child = ctx.Pipe()
        self.process = ctx.Process(
            target=_worker,
            args=(child, cfg.host, cfg.port, mock, cfg.mock_delay_s, fps),
            daemon=True,
        )
        self.pending_at = None
        self.generation = 0
        self.actions = []
        self.error = None
        self.ready = False
        self.process.start()
        child.close()

    def reset(self):
        # A late reply from the previous control mode must never be executed.
        self.generation += 1
        self.actions = []
        self.error = None

    def poll(self):
        while self.connection.poll():
            try:
                kind, payload = self.connection.recv()
            except (EOFError, OSError):
                self.error = self.error or "Inference process connection closed"
                break
            if kind == "ready":
                self.ready = True
            elif kind == "error":
                self.error = payload
                self.pending_at = None
            else:
                self.pending_at = None
                generation, observed_at, actions = payload
                if generation != self.generation:
                    continue
                if time.monotonic() - observed_at > self.cfg.max_result_age_s:
                    self.error = "Inference result exceeded max_result_age_s"
                    continue
                if actions.ndim != 2 or actions.shape[1] != 7 or not len(actions):
                    self.error = f"Expected actions [T,7], got {actions.shape}"
                    continue
                if not np.isfinite(actions).all():
                    self.error = "Non-finite model action"
                    continue
                self.actions = list(actions[: self.cfg.execution_horizon].copy())
        if not self.process.is_alive() and self.error is None:
            self.error = "Inference process exited"
        if self.pending_at is not None and time.monotonic() - self.pending_at > self.cfg.timeout_s:
            self.error = "Inference request timed out"

    def action(self, observation):
        self.poll()
        if self.error:
            raise RuntimeError(self.error)
        if self.actions:
            return self.actions.pop(0)
        if self.ready and self.pending_at is None:
            self.pending_at = time.monotonic()
            try:
                self.connection.send((self.generation, self.pending_at, observation))
            except (BrokenPipeError, EOFError, OSError) as exc:
                raise RuntimeError("Inference process connection closed") from exc
        return None

    def close(self):
        # infer() may block indefinitely in recv(). Termination is bounded.
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(3)
        self.connection.close()
