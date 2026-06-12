"""
Camera interface for remote cameras shared via `HEROS <https://gitlab.com/atomiq-project/heros>`_.

This module wraps a remote camera HERO -- i.e. a device representation from
`herosdevices <https://gitlab.com/atomiq-project/herosdevices>`_ implementing
:class:`herosdevices.core.templates.camera.CameraTemplate` -- such that it can be used
as a drop-in :class:`~slmsuite.hardware.cameras.camera.Camera`, e.g. inside a
:class:`~slmsuite.hardware.cameraslms.FourierSLM`.

Acquisition model
~~~~~~~~~~~~~~~~~
``herosdevices`` cameras are event-driven: a device-side acquisition thread emits each
frame through the ``acquisition_data`` event. :mod:`slmsuite`, in contrast, expects a
blocking, synchronous :meth:`~HerosCamera._get_image_hw`. This class bridges the two
worlds by connecting a local callback to the remote event and pushing every received
``(frame, metadata)`` payload into a thread-safe queue. A frame grab then amounts to
(optionally) firing the remote software trigger ``start()`` and performing a blocking
``Queue.get(timeout=...)``.

The wrapper additionally tracks the remote acquisition state through the
``acquisition_started`` / ``acquisition_stopped`` events. Device-side acquisition
threads commonly terminate after a configured number of frames (``frame_count``); with
``auto_rearm=True`` (default) the wrapper transparently re-arms before the next grab,
which is essential for the long ``set_phase``/``get_image`` feedback loops used by the
:mod:`slmsuite` calibration and optimization routines.

Configuration model
~~~~~~~~~~~~~~~~~~~
Device configuration (exposure, gain, ROI, trigger mode, ...) in ``herosdevices`` is
intentionally device-specific and handled through named configuration dicts on the
*device side* (``update_configuration()`` / ``configure()``). The :class:`CameraTemplate`
properties ``exposure_time``, ``roi_coordinates``, and ``binning`` are used by
:meth:`set_exposure`, :meth:`set_woi`, and :meth:`set_binning` respectively. All three
setters require acquisition to be stopped before the device accepts configuration; this
wrapper automatically calls ``stop()`` and re-arms afterwards so callers need not manage
acquisition state manually. The remote interface remains accessible via :attr:`HerosCamera.cam`.

Important
~~~~~~~~~
HEROS events are delivered ``BEST_EFFORT`` by default; frames may be dropped on a
congested network. For measurement data, decorate the device-side
``acquisition_data`` event with :func:`heros.event.reliable`, e.g. via the BOSS
``extra_decorators`` mechanism::

    "extra_decorators": [["acquisition_data", "heros.event.reliable"]]
"""

import queue
import time
import warnings
from typing import Any, Optional, Tuple

import numpy as np

from slmsuite.hardware.cameras.camera import Camera

try:
    from heros import RemoteHERO
except ImportError:
    # No module-level warning: a missing heros only matters if the user passes a HERO
    # name instead of a proxy object, which raises an ImportError in __init__.
    RemoteHERO = None


class HerosCamera(Camera):
    """
    Wraps a remote ``herosdevices`` camera HERO as a :mod:`slmsuite` camera.

    Attributes
    ----------
    cam : heros.RemoteHERO
        The remote camera proxy. All device-specific functionality
        (``configure()``, ``update_configuration()``, ``get_status()``, ...)
        remains directly accessible through this attribute.
    last_metadata : dict
        Metadata dict attached to the most recently grabbed frame
        (see ``payload_metadata`` in ``herosdevices``).
    acquisition_metadata : dict
        Metadata dict from the most recent ``acquisition_started`` /
        ``acquisition_stopped`` event (key ``"event"`` records which one).
    """

    ### Initialization and termination

    def __init__(
        self,
        camera,
        software_trigger: bool = True,
        auto_rearm: bool = True,
        rearm_grace_s: float = 0.5,
        arm: bool = True,
        configure: Optional[str] = None,
        resolution: Optional[Tuple[int, int]] = None,
        bitdepth: Optional[int] = None,
        pitch_um: Optional[Tuple[float, float]] = None,
        queue_size: int = 64,
        probe_timeout_s: float = 10,
        verbose: bool = True,
        **kwargs,
    ):
        """
        Initializes a wrapper around a remote camera HERO.

        Parameters
        ----------
        camera : heros.RemoteHERO OR str
            Either an already-constructed ``RemoteHERO`` proxy of a camera implementing
            the ``herosdevices`` ``CameraTemplate``, or the HERO name as a string, in
            which case the proxy is constructed here.
        software_trigger : bool
            If ``True``, every grab discards stale frames and fires the remote software
            trigger ``start()`` before waiting, guaranteeing frame freshness (the
            returned frame postdates the request). Free-running devices
            (``auto_trigger``-style acquisition loops) may reject the redundant
            trigger; this is tolerated as long as a frame still arrives.
            If ``False`` (e.g. hardware-triggered or free-running acquisition), no
            trigger is sent and no stale-frame flush is performed -- grabs return
            queued frames in arrival order, so externally triggered exposures fired
            *before* the grab call are not lost.
        auto_rearm : bool
            If ``True``, the wrapper re-arms the remote camera whenever the device-side
            acquisition has stopped (tracked via the ``acquisition_stopped`` event)
            before the next grab. Required for devices whose acquisition thread
            terminates after ``frame_count`` frames when multiple grabs are performed,
            e.g. in closed-loop calibrations. If ``False``, a grab in stopped state
            fails fast with a descriptive ``RuntimeError`` instead of burning the
            full timeout.
        rearm_grace_s : float
            After an automatic re-arm, how long to wait for an arm-induced frame
            (auto-triggering devices) before firing an explicit software trigger.
        arm : bool
            Whether to arm the remote camera (with its currently active device-side
            configuration) during construction. Skipped if already acquiring.
        configure : str OR None
            If given, the named device-side configuration is activated via the remote
            ``configure()`` before arming. The configuration *content* remains
            device-side; this merely selects it.
        resolution : (int, int) OR None
            ``(width, height)`` of the camera. If ``None``, inferred from a probe
            frame (which requires a trigger source; with ``software_trigger=False``
            an external trigger must arrive within ``probe_timeout_s``).
        bitdepth : int OR None
            Bit depth of the camera. If ``None``, inferred from the dtype of a probe
            frame.

            Caution
            ~~~~~~~
            dtype is an upper bound: a 12-bit sensor commonly returns ``uint16``
            frames, which would be inferred as 16 bits and skew
            :meth:`.Camera.autoexposure`. Pass ``bitdepth`` explicitly if known.
        pitch_um : (float, float) OR None
            Pixel pitch in microns, if known. Not exposed by the generic
            ``CameraTemplate``, hence not auto-detected.
        queue_size : int
            Maximum number of frames buffered locally. When the queue is full, the
            *oldest* frame is dropped (relevant for free-running devices without a
            consumer). Use ``0`` for an unbounded queue.
        probe_timeout_s : float
            Timeout for the probe frame used to infer ``resolution``/``bitdepth``.
        verbose : bool
            Whether to print extra information.
        **kwargs
            See :meth:`.Camera.__init__` for permissible options.

        Raises
        ------
        RuntimeError
            If the probe frame needed to infer ``resolution`` could not be acquired.
        """
        # Resolve the remote proxy.
        if isinstance(camera, str):
            if RemoteHERO is None:
                raise ImportError(
                    "heros is required to construct a RemoteHERO from a name. "
                    "Install via `pip install heros`, or pass a proxy object directly."
                )
            if verbose:
                print(f"Connecting to remote camera HERO '{camera}'...", end="")
            camera = RemoteHERO(camera)
            if verbose:
                print("success")
        self.cam = camera

        self.software_trigger = bool(software_trigger)
        self.auto_rearm = bool(auto_rearm)
        self.rearm_grace_s = float(rearm_grace_s)
        self.last_metadata = {}
        self.acquisition_metadata = {}

        # Thread-safe frame buffer filled by the heros event callback. The callback
        # executes in the heros networking thread; the Queue decouples it from the
        # (blocking) consumer in _get_image_hw(). Bounded with drop-oldest semantics
        # to limit memory for free-running acquisition without a consumer.
        self._frames: "queue.Queue[Tuple[np.ndarray, dict]]" = queue.Queue(
            maxsize=int(queue_size)
        )

        # Local mirror of the device acquisition state, maintained via events to avoid
        # a remote attribute query on every grab. None = unknown (no event seen yet).
        self._acquisition_running: Optional[bool] = None

        # Subscribe to the remote events (RemoteEventHandler.connect).
        self.cam.acquisition_data.connect(self._frame_callback)
        self.cam.acquisition_started.connect(self._started_callback)
        self.cam.acquisition_stopped.connect(self._stopped_callback)

        # Seed the state mirror with one (potentially slow) remote attribute read.
        try:
            self._acquisition_running = bool(self.cam.acquisition_running)
        except Exception:
            pass

        # Activate a named device-side configuration if requested.
        if configure is not None:
            try:
                if not self.cam.configure(configure):
                    warnings.warn(f"Remote configure('{configure}') returned False.")
            except Exception as exc:
                warnings.warn(f"Remote configure('{configure}') failed: {exc}")

        # Arm the device with its currently active (device-side) configuration.
        if arm and not self._acquisition_running:
            self._arm(verbose=verbose)

        # Infer geometry/depth from a probe frame if not provided.
        if resolution is None or bitdepth is None:
            if verbose:
                print("Acquiring probe frame to infer camera parameters...", end="")
            frame = self._grab(timeout_s=probe_timeout_s)
            if frame is None:
                raise RuntimeError(
                    "HerosCamera could not acquire a probe frame to infer resolution/"
                    "bitdepth. Pass resolution=(width, height) and bitdepth= "
                    "explicitly, verify that the camera is armed and configured "
                    "(camera.cam.get_status()), and -- for hardware-triggered setups "
                    "-- that a trigger arrives within probe_timeout_s."
                )
            if verbose:
                print("success")
            self._probe_dtype = frame.dtype
            if resolution is None:
                # herosdevices emits (height, width); Camera expects (width, height)
                resolution = (frame.shape[1], frame.shape[0])
            if bitdepth is None:
                bitdepth = 8 * frame.dtype.itemsize
                warnings.warn(
                    f"HerosCamera inferred bitdepth={bitdepth} from frame dtype "
                    f"{frame.dtype}. This is an upper bound (e.g. 12-bit sensors "
                    "return uint16); pass bitdepth= explicitly for correct "
                    "autoexposure scaling."
                )

        # Sensor dimensions in herosdevices row-major convention: (height, width).
        # Stored before super().__init__ because set_woi() is called from there.
        # resolution here is always (width, height) per Camera convention.
        self._sensor_shape: Tuple[int, int] = (int(resolution[1]), int(resolution[0]))

        if self._acquisition_running is False:
            self._arm(verbose=False)

        super().__init__(
            resolution,
            bitdepth=int(bitdepth),
            pitch_um=pitch_um,
            name=kwargs.pop(
                "name", str(getattr(self.cam, "_hero_name", "heros_camera"))
            ),
            **kwargs,
        )

    def close(self):
        """
        Disconnects the event callbacks from the remote events.

        Note
        ~~~~
        The remote device itself is *not* stopped or torn down: a HERO is a shared
        network resource and other clients may be subscribed to it. Use
        ``camera.cam.stop()`` / ``camera.cam.teardown()`` explicitly to control the
        device lifecycle.
        """
        for event_name, callback in (
            ("acquisition_data", self._frame_callback),
            ("acquisition_started", self._started_callback),
            ("acquisition_stopped", self._stopped_callback),
        ):
            try:
                getattr(self.cam, event_name).disconnect(callback)
            except Exception as exc:
                warnings.warn(f"Could not disconnect {event_name} callback: {exc}")
        self.flush()

    ### Event callbacks (run in the heros networking thread)

    def _frame_callback(self, *payload: Any) -> None:
        """
        Callback connected to the remote ``acquisition_data`` event.

        Executed with the return value of the device-side event, which for
        ``CameraTemplate`` is an ``(frame, metadata)`` tuple. Defensively also accepts
        a list (serdes round-trip) or a bare frame.
        """
        frame, metadata = None, {}
        if len(payload) == 2:
            frame, metadata = payload
        elif len(payload) == 1:
            data = payload[0]
            if isinstance(data, (tuple, list)) and len(data) == 2:
                frame, metadata = data
            else:
                frame = data
        if frame is None:
            warnings.warn(
                f"HerosCamera received an unexpected event payload: {payload!r}"
            )
            return

        item = (np.asarray(frame), dict(metadata) if metadata else {})
        try:
            self._frames.put_nowait(item)
        except queue.Full:
            # Drop-oldest: discard the stalest frame to make room for the new one.
            try:
                self._frames.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frames.put_nowait(item)
            except queue.Full:
                pass  # Racing producers; losing one frame here is acceptable.

    def _started_callback(self, *payload: Any) -> None:
        """Callback for ``acquisition_started``; mirrors the device state locally."""
        self._acquisition_running = True
        if payload and isinstance(payload[0], dict):
            self.acquisition_metadata = {"event": "started", **payload[0]}

    def _stopped_callback(self, *payload: Any) -> None:
        """Callback for ``acquisition_stopped``; mirrors the device state locally."""
        self._acquisition_running = False
        if payload and isinstance(payload[0], dict):
            self.acquisition_metadata = {"event": "stopped", **payload[0]}

    ### Acquisition

    def _arm(self, verbose: bool = False) -> bool:
        """Arm the remote camera with its active device-side configuration."""
        try:
            if verbose:
                print("Arming remote camera...", end="")
            armed = bool(self.cam.arm(kill_running=True))
            if verbose:
                print("success" if armed else "failed")
        except Exception as exc:
            warnings.warn(f"Could not arm remote camera: {exc}")
            return False
        if armed:
            # arm() starting the device acquisition thread implies running; the
            # acquisition_started event will confirm asynchronously.
            self._acquisition_running = True
        else:
            warnings.warn(
                "Remote camera arm() returned False; frame grabs may time out. "
                "Check the device-side configuration (camera.cam.get_configuration())."
            )
        return armed

    def _ensure_armed(self) -> bool:
        """
        Re-arm if the device-side acquisition has stopped (e.g. ``frame_count``
        exhausted), or fail fast if ``auto_rearm`` is disabled.

        Returns
        -------
        bool
            Whether a re-arm was performed.
        """
        if self._acquisition_running is not False:
            return False  # Running, or unknown: proceed optimistically.
        if self.auto_rearm:
            return self._arm()
        raise RuntimeError(
            "The remote camera acquisition has stopped "
            f"(acquisition_stopped metadata: {self.acquisition_metadata}) and "
            "auto_rearm=False. Re-arm manually via camera.cam.arm() or construct "
            "HerosCamera with auto_rearm=True."
        )

    def _try_start(self) -> None:
        """Fire the remote software trigger, tolerating rejection."""
        try:
            self.cam.start()
        except Exception as exc:
            # Non-fatal: free-running devices may reject a redundant trigger
            # (e.g. full device buffer) while still delivering frames.
            warnings.warn(
                f"Remote software trigger start() failed ({exc}); "
                "waiting for a frame regardless."
            )

    def _pop(self, timeout_s: float) -> Optional[np.ndarray]:
        """Blocking queue pop; stores metadata and returns the frame, or ``None``."""
        try:
            frame, metadata = self._frames.get(timeout=max(0.0, timeout_s))
        except queue.Empty:
            return None
        self.last_metadata = metadata
        return frame

    def _grab(self, timeout_s: float) -> Optional[np.ndarray]:
        """
        Trigger (if configured to) and block until the next frame arrives.

        Returns ``None`` on timeout; retry logic is left to
        :attr:`.Camera.capture_attempts` in the superclass.

        Note
        ~~~~
        Ordering matters: stale frames are flushed *before* (re-)arming, because
        auto-triggering devices begin emitting immediately upon ``arm()`` -- a flush
        afterwards would discard the fresh frame.

        The wait is staged to be robust against two realities of the event transport:
        (1) auto-triggering devices need no explicit ``start()`` after a re-arm, while
        software-trigger-only devices do -- resolved by granting :attr:`rearm_grace_s`
        before firing; (2) ``acquisition_started``/``acquisition_stopped`` travel on
        separate event endpoints, so the local state mirror can be transiently wrong --
        resolved by verifying the remote ``acquisition_running`` (one RPC) and
        re-arming mid-grab if nothing arrived within the grace period, instead of
        burning the full timeout on a stopped device.
        """
        deadline = time.monotonic() + timeout_s

        if self.software_trigger:
            # Discard stale frames: only data acquired *after* this point counts.
            self.flush()

        rearmed = self._ensure_armed()

        if self.software_trigger and not rearmed:
            self._try_start()

        # Phase 1: short wait. Catches triggered or arm-induced frames quickly.
        frame = self._pop(min(self.rearm_grace_s, deadline - time.monotonic()))
        if frame is not None:
            return frame

        # Phase 2: nothing arrived within the grace period -- resolve the cause.
        if rearmed:
            # The fresh arm did not produce frames by itself: the device is not
            # auto-triggering, so fire the explicit trigger now.
            if self.software_trigger:
                self._try_start()
        else:
            # The state mirror may be stale-True (out-of-order started/stopped
            # events): verify once against the device and re-arm if stopped.
            try:
                running = bool(self.cam.acquisition_running)
            except Exception:
                running = None
            self._acquisition_running = running
            if running is False:
                if not self.auto_rearm:
                    raise RuntimeError(
                        "The remote camera acquisition has stopped "
                        f"(metadata: {self.acquisition_metadata}) and "
                        "auto_rearm=False. Re-arm manually via camera.cam.arm() "
                        "or construct HerosCamera with auto_rearm=True."
                    )
                if self._arm():
                    frame = self._pop(
                        min(self.rearm_grace_s, deadline - time.monotonic())
                    )
                    if frame is not None:
                        return frame
                    if self.software_trigger:
                        self._try_start()

        # Phase 3: wait out the remaining timeout.
        frame = self._pop(deadline - time.monotonic())
        if frame is None:
            # Heal the state mirror for the superclass retry (capture_attempts).
            try:
                self._acquisition_running = bool(self.cam.acquisition_running)
            except Exception:
                self._acquisition_running = None
        return frame

    ### Required Camera subclass implementations

    def _get_dtype(self, get_image_function=None):
        """
        See :meth:`.Camera._get_dtype`.

        Uses the probe frame dtype when available to avoid an extra frame grab during
        construction. Without a probe frame, forces the bitdepth-based fallback directly
        rather than trying (and timing out) a live grab.
        """
        if get_image_function is None:
            probe = getattr(self, "_probe_dtype", None)
            if probe is not None:

                def get_image_function(probe=probe):
                    return np.empty(0, dtype=probe)
            else:

                def get_image_function():
                    raise RuntimeError("no probe frame; using bitdepth fallback")

        return super()._get_dtype(get_image_function)

    def _get_image_hw(self, timeout_s: float) -> np.ndarray:
        """
        See :meth:`.Camera._get_image_hw`.

        The frame metadata emitted alongside the image by the remote device is stored
        in :attr:`last_metadata`.
        """
        frame = self._grab(timeout_s)
        if frame is None:
            raise TimeoutError(
                f"HerosCamera did not receive a frame within {timeout_s} s. "
                "Check that the remote camera is armed and configured "
                "(camera.cam.get_status()), that the trigger mode matches "
                f"software_trigger={self.software_trigger}, and that the network "
                "delivers the acquisition_data event (consider the "
                "heros.event.reliable decorator on the device side)."
            )
        return frame

    def _get_images_hw(
        self, image_count: int, timeout_s: float, out: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        See :meth:`.Camera._get_images_hw`.

        With ``software_trigger=False``, frames are collected as they stream in
        (one external trigger per frame). With ``software_trigger=True``, one
        ``start()`` is fired per frame, since the generic ``CameraTemplate`` does not
        expose a burst length.
        """
        first = self._get_image_hw(timeout_s)
        if out is None:
            out = np.empty((image_count, *first.shape), dtype=first.dtype)
        out[0] = first
        for i in range(1, image_count):
            if self.software_trigger:
                out[i] = self._get_image_hw(timeout_s)
            else:
                self._ensure_armed()
                try:
                    frame, metadata = self._frames.get(timeout=timeout_s)
                except queue.Empty as exc:
                    raise TimeoutError(
                        f"HerosCamera received {i}/{image_count} frames before timing "
                        f"out after {timeout_s} s waiting for the next trigger."
                    ) from exc
                self.last_metadata = metadata
                out[i] = frame
        return out

    def flush(self, *_args, **_kwargs):
        """
        See :meth:`.Camera.flush`. Discards all locally queued frames.

        Note
        ~~~~
        This only empties the client-side queue; buffers on the remote device are
        managed by the device itself (see ``stop()``/``reset()`` on :attr:`cam`).
        """
        try:
            while True:
                self._frames.get_nowait()
        except queue.Empty:
            pass

    ### Configuration

    def _configure_hw(self, fn) -> None:
        """
        Stop acquisition if running, call fn(), then re-arm.

        Used by all hardware configuration setters. ``CameraTemplate.configure()``
        refuses to apply settings while acquisition is running, so we stop the device
        first and restore the running state after.
        """
        was_running = self._acquisition_running
        if was_running:
            try:
                self.cam.stop()
            except Exception as exc:
                warnings.warn(f"Could not stop acquisition before reconfiguring: {exc}")
        fn()
        if was_running:
            self._arm(verbose=False)

    def _get_exposure_hw(self) -> float:
        """
        See :meth:`.Camera._get_exposure_hw`.

        Reads ``exposure_time`` from the remote ``CameraTemplate`` property.
        Returns ``NaN`` (with a warning) if the property returns ``None`` (not set in
        the active configuration) -- in that case :meth:`.Camera.autoexposure` and HDR
        features will not work.
        """
        try:
            val = self.cam.exposure_time
        except Exception as exc:
            warnings.warn(f"HerosCamera could not read exposure_time: {exc}")
            return float("nan")
        if val is None:
            warnings.warn(
                "HerosCamera: exposure_time is None (not set in the active config). "
                "Exposure-dependent features (autoexposure, HDR) are unavailable; "
                "add an exposure_time entry via camera.cam.update_configuration()."
            )
            return float("nan")
        return float(val)

    def _set_exposure_hw(self, exposure_s: float) -> None:
        """
        See :meth:`.Camera._set_exposure_hw`.

        Sets ``exposure_time`` on the remote device. Automatically stops and re-arms
        the acquisition if it is currently running.
        """

        def _apply():
            self.cam.exposure_time = float(exposure_s)

        self._configure_hw(_apply)

    def set_woi(self, woi=None):
        """
        Set the hardware region of interest on the remote device.

        Translates the slmsuite ``(x_min, x_max, y_min, y_max)`` WOI format into the
        herosdevices ``(x_offset, y_offset, width, height)`` ``roi_coordinates`` format
        and pushes it to the device. The device will emit frames sized to the ROI,
        reducing network traffic proportionally.

        When ``woi=None`` (full-frame reset), the hardware call is skipped: the device
        already emits full frames by default, and this path is also taken during
        construction (called from ``Camera.__init__``). To explicitly push a full-frame
        ROI to the device, pass ``woi=(0, width, 0, height)`` directly.

        Automatically stops and re-arms the acquisition if it is currently running.

        Parameters
        ----------
        woi : tuple OR None
            Window of interest ``(x_min, x_max, y_min, y_max)`` in sensor pixels.
            x is the column axis (width direction), y is the row axis (height direction).
            All values are clipped to the sensor bounds.
            If ``None``, reset local bookkeeping to the full sensor frame without a
            hardware call.

        Returns
        -------
        tuple
            The resulting WOI ``(x_min, x_max, y_min, y_max)``.
        """
        # _sensor_shape is (height, width) matching herosdevices row-major frames.
        sensor_h, sensor_w = self._sensor_shape
        if woi is None:
            self.woi = (0, sensor_w, 0, sensor_h)
            self.shape = (sensor_h, sensor_w)
            return self.woi
        woi = (
            max(0, int(woi[0])),
            min(sensor_w, int(woi[1])),
            max(0, int(woi[2])),
            min(sensor_h, int(woi[3])),
        )
        # herosdevices roi_coordinates: (x_offset, y_offset, width, height)
        roi = (woi[0], woi[2], woi[1] - woi[0], woi[3] - woi[2])

        def _apply():
            self.cam.roi_coordinates = roi

        self._configure_hw(_apply)
        self.woi = woi
        self.shape = (woi[3] - woi[2], woi[1] - woi[0])
        return self.woi

    def set_binning(self, binning=None):
        """
        Set the hardware binning on the remote device.

        Automatically stops and re-arms the acquisition if it is currently running.

        Parameters
        ----------
        binning : (int, int) OR None
            ``(horizontal, vertical)`` binning factors. ``None`` resets to ``(1, 1)``.
        """
        if binning is None:
            binning = (1, 1)
        h, v = int(binning[0]), int(binning[1])

        def _apply():
            self.cam.binning = (h, v)

        self._configure_hw(_apply)
