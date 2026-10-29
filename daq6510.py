"""
Reusable library for Keithley/Tektronix DAQ6510 data acquisition over LAN.

The library is deliberately layered so simple uses stay simple:

    Channels   -> helpers for building SCPI channel lists
    ScanPlan   -> *what* to scan (a flat list, or groups at differing rates)
    DAQ6510    -> *how* to talk to the instrument (connect, configure, read)

A caller who just wants N channels at one rate never has to know that
tiered scanning exists::

    from daq6510 import DAQ6510, ScanPlan

    with DAQ6510("169.254.160.157") as daq:
        daq.start_scan(ScanPlan.flat(range(101, 121)))
        for cycle, readings in daq.iter_cycles():
            print(readings[101])

while a caller who wants different rates per group builds a different
plan and uses the same machinery::

    plan = ScanPlan.tiered([(range(101, 111), 4),
                            (range(111, 121), 2),
                            (range(121, 131), 1)])

Performance notes measured on a DAQ6510 + 7702 multiplexer:

* A scan step costs ~12 ms, dominated by relay actuation and settling.
  The DMM itself measures in ~0.1 ms, so channel count sets the rate.
* Autoranging roughly triples sweep time; a fixed range is much faster.
* Reading results out over LAN while scanning costs <0.1% - the
  instrument scans autonomously once started.
"""

from __future__ import annotations

import atexit
import signal
import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Iterable, Iterator, Mapping, Sequence

import pyvisa

__all__ = [
    "Channels",
    "ScanPlan",
    "DAQ6510",
    "Sample",
    "GroupRate",
    "RateReport",
    "RateMonitor",
    "NotReadyError",
    "ScanDataError",
    "DEFAULT_RANGE_V",
    "DEFAULT_NPLC",
]


class ScanDataError(RuntimeError):
    """
    A scan produced data that does not match the active plan.

    Raised for genuine faults - missing channels, a buffer that wrapped
    past the read cursor - as opposed to simply not having data yet.
    Distinct from :class:`NotReadyError` so callers can retry the benign
    case while surfacing real problems instead of silently looping.
    """


class NotReadyError(RuntimeError):
    """
    The requested data has not been acquired yet.

    A transient, expected condition: the caller should wait and retry.
    """

SOCKET_PORT = 5025          # DAQ6510 raw SCPI socket port
DEFAULT_BUFFER = "defbuffer1"

# A fixed range avoids per-channel autorange hunting (~3x faster). NPLC
# is integration time in power-line cycles; below ~0.06 relay settling
# dominates and lowering it further gains almost nothing.
DEFAULT_RANGE_V = 10.0
DEFAULT_NPLC = 0.06


# --------------------------------------------------------------------------
# Channel helpers
# --------------------------------------------------------------------------
class Channels:
    """Helpers for converting channel numbers to SCPI channel lists."""

    @staticmethod
    def to_scpi(channels: Sequence[int]) -> str:
        """
        Render channel numbers as a SCPI channel list.

        Contiguous runs collapse into ``start:end`` ranges to keep the
        command short. Repeats and non-monotonic order are preserved,
        since tiered scan lists rely on both.

        Parameters
        ----------
        channels : Sequence[int]
            Channel numbers in scan order.

        Returns
        -------
        str
            SCPI channel list, e.g. ``"(@101:110,121:130)"``.

        Raises
        ------
        ValueError
            If no channels are supplied.
        """
        if not channels:
            raise ValueError("channel list must not be empty")

        parts: list[str] = []
        run_start = prev = channels[0]

        for ch in channels[1:]:
            if ch == prev + 1:                 # extend the contiguous run
                prev = ch
                continue
            parts.append(str(run_start) if run_start == prev
                         else f"{run_start}:{prev}")
            run_start = prev = ch
        parts.append(str(run_start) if run_start == prev
                     else f"{run_start}:{prev}")

        return f"(@{','.join(parts)})"


# --------------------------------------------------------------------------
# Scan planning
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ScanPlan:
    """
    Describes what to scan and how often each channel is sampled.

    A plan flattens to a repeating sequence of channels (``sequence``)
    that the instrument walks continuously. Sampling a channel more than
    once per pass simply means listing it more than once.

    Attributes
    ----------
    sequence : tuple[int, ...]
        Channels in scan order for exactly one cycle; may contain repeats.
    channels : tuple[int, ...]
        Distinct channels, ascending.
    groups : dict[str, tuple[int, ...]]
        Named groups, for labelling output. A flat plan has one group.
    """

    sequence: tuple[int, ...]
    channels: tuple[int, ...]
    # Stored as a tuple of pairs so the dataclass is genuinely immutable
    # and hashable; exposed as a read-only mapping via `groups`.
    group_items: tuple[tuple[str, tuple[int, ...]], ...] = field(default=())

    @property
    def groups(self) -> Mapping[str, tuple[int, ...]]:
        """Named channel groups, as a read-only mapping."""
        return MappingProxyType(dict(self.group_items))

    @property
    def cycle_length(self) -> int:
        """Number of readings in one full cycle."""
        return len(self.sequence)

    def samples_per_cycle(self, channel: int) -> int:
        """How many times ``channel`` is sampled per cycle."""
        return self.sequence.count(channel)

    def group_of(self, channel: int) -> str:
        """Name of the group containing ``channel``, or ``""`` if none."""
        for name, members in self.group_items:
            if channel in members:
                return name
        return ""

    @staticmethod
    def describe_channels(channels: Sequence[int]) -> str:
        """
        Render channel numbers compactly and *accurately*.

        Collapses only genuinely contiguous runs, so a sparse selection
        reads ``101,105,110`` rather than being misreported as the range
        ``101-110``.

        Parameters
        ----------
        channels : Sequence[int]
            Channel numbers to describe.

        Returns
        -------
        str
            Compact description, e.g. ``"101-110"`` or ``"101,105,110"``.
        """
        if not channels:
            return "(none)"
        ordered = sorted(set(channels))
        parts: list[str] = []
        run_start = prev = ordered[0]
        for ch in ordered[1:]:
            if ch == prev + 1:
                prev = ch
                continue
            parts.append(str(run_start) if run_start == prev
                         else f"{run_start}-{prev}")
            run_start = prev = ch
        parts.append(str(run_start) if run_start == prev
                     else f"{run_start}-{prev}")
        return ",".join(parts)

    @classmethod
    def flat(cls, channels: Iterable[int]) -> "ScanPlan":
        """
        Scan every channel once per cycle, at a single uniform rate.

        This is the common case: N channels, all equally important.

        Parameters
        ----------
        channels : Iterable[int]
            Channel numbers to scan.

        Returns
        -------
        ScanPlan
        """
        seq = tuple(channels)
        if not seq:
            raise ValueError("a scan plan needs at least one channel")
        return cls(sequence=seq,
                   channels=tuple(sorted(set(seq))),
                   group_items=(("all", tuple(sorted(set(seq)))),))

    @classmethod
    def tiered(cls, groups: Sequence[tuple[Iterable[int], int]],
               names: Sequence[str] | None = None) -> "ScanPlan":
        """
        Scan groups of channels at different relative rates.

        Each group is given a weight: a group with weight 4 is sampled
        four times per cycle, one with weight 1 just once. Repeated
        blocks are interleaved rather than clumped, so a group's samples
        are spread evenly through the cycle instead of bunching together.

        Parameters
        ----------
        groups : Sequence[tuple[Iterable[int], int]]
            ``(channels, weight)`` pairs. Weights are relative, so
            ``4:2:1`` and ``8:4:2`` describe the same ratio.
        names : Sequence[str] | None
            Optional group labels; defaults to ``A``, ``B``, ``C``, ...

        Returns
        -------
        ScanPlan

        Examples
        --------
        >>> plan = ScanPlan.tiered([(range(101, 111), 4),
        ...                         (range(111, 121), 2),
        ...                         (range(121, 131), 1)])
        >>> plan.cycle_length
        70
        """
        if not groups:
            raise ValueError("a tiered plan needs at least one group")

        members = [tuple(chans) for chans, _ in groups]
        weights = [int(w) for _, w in groups]

        if any(w < 1 for w in weights):
            raise ValueError("group weights must be >= 1")
        if any(not m for m in members):
            raise ValueError("every group needs at least one channel")

        # Overlapping groups would give a channel the sum of both weights
        # rather than either one, and group_of() could only report the
        # first match - so reject rather than silently mismeasure.
        seen: dict[int, int] = {}
        for gi, chans in enumerate(members):
            for ch in chans:
                if ch in seen:
                    raise ValueError(
                        f"channel {ch} appears in group {seen[ch] + 1} and "
                        f"group {gi + 1}; groups must not overlap"
                    )
                seen[ch] = gi

        labels = list(names) if names else [
            chr(ord("A") + i) for i in range(len(members))
        ]
        if len(labels) != len(members):
            raise ValueError("names must match the number of groups")
        if len(set(labels)) != len(labels):
            raise ValueError("group names must be unique")

        # Interleave by spacing each group's blocks evenly across the
        # cycle, then ordering by position. Ties break toward the faster
        # group so the highest-rate channels keep the most even spacing.
        slots: list[tuple[float, int, int]] = []
        total = max(weights)
        for gi, w in enumerate(weights):
            for k in range(w):
                slots.append(((k + 0.5) / w * total, -w, gi))
        slots.sort()

        sequence: list[int] = []
        for _, _, gi in slots:
            sequence.extend(members[gi])

        return cls(
            sequence=tuple(sequence),
            channels=tuple(sorted({c for m in members for c in m})),
            group_items=tuple((labels[i], members[i])
                              for i in range(len(members))),
        )


@dataclass(frozen=True)
class Sample:
    """
    One individual reading retrieved from the instrument buffer.

    Attributes
    ----------
    index : int
        1-based position in the instrument's reading buffer.
    channel : int
        Channel the reading came from.
    value : float
        The measured value, in volts.
    rel_time : float
        Seconds since the scan started, timestamped by the instrument.
    """

    index: int
    channel: int
    value: float
    rel_time: float


# --------------------------------------------------------------------------
# Rate monitoring
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class GroupRate:
    """
    Achieved sampling rate for one channel group over a time window.

    Attributes
    ----------
    name : str
        Group name, or ``""`` for channels outside any group.
    channels : tuple[int, ...]
        Distinct channels of this group seen in the window.
    samples : int
        Total readings from this group in the window.
    hz_per_channel : float
        Measured sampling rate of each individual channel, in Hz.
    expected_hz : float | None
        Rate predicted by the scan plan, or ``None`` if unknown.
    """

    name: str
    channels: tuple[int, ...]
    samples: int
    hz_per_channel: float
    expected_hz: float | None = None

    @property
    def drift_percent(self) -> float | None:
        """Deviation from the expected rate, as a percentage."""
        if not self.expected_hz:
            return None
        return (self.hz_per_channel - self.expected_hz) / self.expected_hz * 100.0


@dataclass(frozen=True)
class RateReport:
    """
    Measured acquisition performance over one window.

    Derived from the instrument's own timestamps, so it reflects real
    acquisition timing rather than host-side polling behaviour. Intended
    to be consumed programmatically - a console printer, a dashboard
    widget, or a logger can all render the same report.

    Attributes
    ----------
    window_seconds : float
        Duration covered, by instrument timestamps.
    samples : int
        Total readings in the window.
    readings_per_second : float
        Overall throughput across all channels.
    ms_per_step : float
        Average time per scan step: relay switch, settle and measure.
        Dominated by multiplexer switching on relay-based cards.
    groups : tuple[GroupRate, ...]
        Per-group breakdown.
    cycles : float
        Number of full scan cycles completed in the window.
    """

    window_seconds: float
    samples: int
    readings_per_second: float
    ms_per_step: float
    groups: tuple[GroupRate, ...]
    cycles: float


class RateMonitor:
    """
    Accumulates samples and reports achieved rates over fixed windows.

    Separates *measuring* the rate from *presenting* it: feed samples in
    with :meth:`add`, and take a :class:`RateReport` whenever one is due.
    A console script, a dashboard, or a file logger can each render the
    same report however they wish.

    Examples
    --------
    >>> monitor = RateMonitor(plan, window_seconds=30.0)
    >>> for sample in daq.iter_samples():             # doctest: +SKIP
    ...     monitor.add(sample)
    ...     report = monitor.take_report()
    ...     if report is not None:
    ...         print(format_rate_report(report))
    """

    def __init__(self, plan: ScanPlan | None = None,
                 window_seconds: float = 30.0,
                 step_ms: float | None = None):
        """
        Parameters
        ----------
        plan : ScanPlan | None
            Active scan plan, used to predict expected rates. Without it
            the report still measures actual rates, but cannot show drift.
        window_seconds : float
            Length of each reporting window, in seconds of instrument time.
        step_ms : float | None
            Assumed time per scan step for predicting expected rates. If
            ``None``, the previous window's measured value is used, so
            predictions self-calibrate to the hardware.
        """
        self.plan = plan
        self.window_seconds = window_seconds
        self._step_ms = step_ms
        self._counts: dict[int, int] = {}
        self._first_rel: float | None = None
        self._last_rel: float | None = None
        self._samples = 0

    def add(self, sample: Sample):
        """
        Record one reading.

        Parameters
        ----------
        sample : Sample
            Reading to include in the current window.
        """
        if self._first_rel is None:
            self._first_rel = sample.rel_time
        self._last_rel = sample.rel_time
        self._counts[sample.channel] = self._counts.get(sample.channel, 0) + 1
        self._samples += 1

    @property
    def elapsed(self) -> float:
        """Instrument-time span of the current window, in seconds."""
        if self._first_rel is None or self._last_rel is None:
            return 0.0
        return self._last_rel - self._first_rel

    def is_due(self) -> bool:
        """Whether the current window has reached ``window_seconds``."""
        return self.elapsed >= self.window_seconds and self._samples > 1

    def take_report(self, force: bool = False) -> RateReport | None:
        """
        Produce a report and start a new window, if one is due.

        Parameters
        ----------
        force : bool
            Report on the partial window even if it is not yet complete,
            e.g. to emit a final summary when acquisition stops.

        Returns
        -------
        RateReport | None
            The completed window, or ``None`` if no report is due or too
            little data was collected to compute meaningful rates.
        """
        if not (force or self.is_due()):
            return None
        report = self.snapshot()
        if report is None:
            return None
        self.reset()
        return report

    def snapshot(self) -> RateReport | None:
        """
        Compute a report for the current window without resetting it.

        Returns
        -------
        RateReport | None
            ``None`` if the window is too short to measure a rate.
        """
        elapsed = self.elapsed
        if elapsed <= 0.0 or self._samples < 2:
            return None

        # The span covers samples-1 intervals, so use that for per-step time.
        ms_per_step = elapsed / (self._samples - 1) * 1000.0
        cycles = (self._samples / self.plan.cycle_length) if self.plan else 0.0

        # Group the observed channels, keeping any not covered by the plan
        # under an empty-named group rather than dropping them.
        buckets: dict[str, list[int]] = {}
        for channel in self._counts:
            name = self.plan.group_of(channel) if self.plan else ""
            buckets.setdefault(name, []).append(channel)

        step_ms = self._step_ms if self._step_ms else ms_per_step

        groups = []
        for name, channels in buckets.items():
            channels.sort()
            total = sum(self._counts[c] for c in channels)
            # Rate of an individual channel, averaged across the group.
            hz = total / len(channels) / elapsed

            expected = None
            if self.plan:
                per_cycle = self.plan.samples_per_cycle(channels[0])
                cycle_s = self.plan.cycle_length * step_ms / 1000.0
                if cycle_s > 0:
                    expected = per_cycle / cycle_s

            groups.append(GroupRate(
                name=name,
                channels=tuple(channels),
                samples=total,
                hz_per_channel=hz,
                expected_hz=expected,
            ))

        groups.sort(key=lambda g: -g.hz_per_channel)

        return RateReport(
            window_seconds=elapsed,
            samples=self._samples,
            readings_per_second=self._samples / elapsed,
            ms_per_step=ms_per_step,
            groups=tuple(groups),
            cycles=cycles,
        )

    def reset(self):
        """Discard the current window and begin a new one."""
        self._counts.clear()
        self._first_rel = None
        self._last_rel = None
        self._samples = 0


def format_rate_report(report: RateReport, plan: ScanPlan | None = None) -> str:
    """
    Render a :class:`RateReport` as human-readable console text.

    A presentation helper kept separate from :class:`RateMonitor`, so
    other front ends (dashboards, structured logs) can consume the same
    report without inheriting this formatting.

    Parameters
    ----------
    report : RateReport
        Report to render.
    plan : ScanPlan | None
        Scan plan, used to describe each group's channels accurately.

    Returns
    -------
    str
        Multi-line summary.
    """
    lines = [
        f"[rate] {report.window_seconds:.1f}s window: {report.samples} readings, "
        f"{report.readings_per_second:.1f} rdg/s, "
        f"{report.ms_per_step:.2f} ms/step, {report.cycles:.1f} cycles"
    ]
    for group in report.groups:
        label = group.name or "ungrouped"
        chans = ScanPlan.describe_channels(group.channels)
        drift = group.drift_percent
        drift_text = f"  ({drift:+.1f}% vs expected)" if drift is not None else ""
        lines.append(
            f"[rate]   {label:>9}: {chans}  "
            f"{group.hz_per_channel:6.2f} Hz/channel  "
            f"{group.samples} readings{drift_text}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Instrument
# --------------------------------------------------------------------------
class DAQ6510:
    """
    Connection to a DAQ6510, with scan configuration and buffered readout.

    Usable as a context manager, which guarantees the scan is stopped and
    the session closed on exit::

        with DAQ6510("169.254.160.157") as daq:
            daq.start_scan(ScanPlan.flat(range(101, 111)))
            cycle, readings = daq.read_latest_cycle()
    """

    def __init__(self, ip: str, timeout_ms: int = 10000,
                 buffer_name: str = DEFAULT_BUFFER,
                 install_exit_handlers: bool = True):
        """
        Open a VISA TCPIP socket session to the instrument.

        Parameters
        ----------
        ip : str
            Instrument IP address.
        timeout_ms : int
            VISA I/O timeout in milliseconds.
        buffer_name : str
            Instrument reading buffer to use.
        install_exit_handlers : bool
            Register atexit/signal handlers that stop the scan on exit.
            A hard kill cannot be intercepted, so this is best-effort.
        """
        self.ip = ip
        self.buffer_name = buffer_name
        self._plan: ScanPlan | None = None
        self._stopped = False
        self._next_index = 1
        self._atexit_registered = False

        rm = pyvisa.ResourceManager("@py")
        self._inst = rm.open_resource(f"TCPIP0::{ip}::{SOCKET_PORT}::SOCKET")
        self._inst.timeout = timeout_ms
        self._inst.read_termination = "\n"
        self._inst.write_termination = "\n"

        self.identity = self._inst.query("*IDN?").strip()

        if install_exit_handlers:
            self._install_exit_handlers()

    # -- plumbing ---------------------------------------------------------
    def _install_exit_handlers(self):
        """
        Best-effort: stop scanning on interpreter exit or termination.

        A hard kill (task manager, power loss) cannot be caught and will
        leave the instrument scanning until it is next reset.
        """
        atexit.register(self.abort)
        self._atexit_registered = True

        def handle_signal(signum, frame):
            raise KeyboardInterrupt

        for name in ("SIGTERM", "SIGBREAK"):
            sig = getattr(signal, name, None)
            if sig is not None:
                try:
                    signal.signal(sig, handle_signal)
                except (ValueError, OSError):
                    pass        # not on the main thread; skip silently

    def write(self, cmd: str):
        """Send a SCPI command."""
        self._inst.write(cmd)

    def query(self, cmd: str) -> str:
        """Send a SCPI query and return the stripped response."""
        return self._inst.query(cmd).strip()

    def check_errors(self, context: str = "") -> list[str]:
        """
        Drain the instrument's error queue.

        Parameters
        ----------
        context : str
            Label describing the operation being checked, used in the
            raised message.

        Returns
        -------
        list[str]
            Any errors found, oldest first. If the queue exceeded the
            drain limit, a final entry says so explicitly rather than
            letting a truncated list look complete.
        """
        limit = 20
        errors = []
        while True:
            err = self.query("SYST:ERR?")
            if err.startswith("0,"):
                return errors
            errors.append(f"{context}: {err}" if context else err)
            if len(errors) >= limit:    # guard against a stuck queue
                errors.append(
                    f"...error queue still not empty after {limit} entries"
                )
                return errors

    # -- configuration ----------------------------------------------------
    def configure_dcv(self, channels: Sequence[int],
                      range_v: float | None = DEFAULT_RANGE_V,
                      nplc: float = DEFAULT_NPLC,
                      autozero: bool = False,
                      line_sync: bool = False,
                      averaging: int = 0,
                      auto_delay: bool = True,
                      input_impedance_auto: bool = False):
        """
        Configure channels for DC voltage measurement.

        This is the single place every speed/accuracy trade-off is set.
        All figures below were measured on a DAQ6510 with a 7702
        multiplexer, scanning 35 channels, and are quoted as time per
        scan step (relay switch + settle + measure).

        Speed/accuracy trade-offs, roughly in order of impact:

        ============  ==========================  =========================
        Setting       Faster                      More accurate
        ============  ==========================  =========================
        ``range_v``   fixed range (~12 ms/step)   ``None`` = autorange
                                                  (~35 ms/step, 3x slower)
        ``nplc``      0.0005 (min integration)    1.0+ (mains rejection)
        ``autozero``  ``False``                   ``True`` (drift removal)
        ``line_sync`` ``False``                   ``True`` (hum rejection)
        ``averaging`` 0 (off)                     N>1 (N x slower)
        ============  ==========================  =========================

        Measured NPLC trade-off on a stable 1.5 V source (35 channels):

        =========  ==============  =====================================
        NPLC       Sweep time      Noise on the reading
        =========  ==============  =====================================
        0.0005     0.41 s          highest
        0.06       0.44 s          ~69 uV std dev  (default)
        0.2        0.60 s          ~47 uV std dev
        1.0        1.28 s          ~4 uV std dev
        =========  ==============  =====================================

        Note that below NPLC ~0.06 the relay's ~12 ms switching time
        dominates, so reducing integration further buys almost nothing.

        Parameters
        ----------
        channels : Sequence[int]
            Channels to configure.
        range_v : float | None
            Fixed measurement range in volts, which must exceed the
            largest expected signal (e.g. ``10`` for a 0-5 V signal).
            Pass ``None`` to enable autoranging, which adapts to unknown
            signal sizes but roughly triples sweep time because the
            instrument re-hunts the range on every channel.
        nplc : float
            Integration time in power-line cycles; one cycle is 20 ms at
            50 Hz. Higher values average over more of the mains waveform,
            rejecting hum and reducing noise, at proportionally more time
            per reading. Valid range is 0.0005 to 15.
        autozero : bool
            Periodically re-measure and subtract the instrument's own
            internal offset, correcting thermal drift over long runs. It
            roughly doubles time per reading, since each measurement is
            paired with a zero measurement.
        line_sync : bool
            Trigger each measurement at the same point in the mains
            cycle, so any residual hum contributes a consistent offset
            rather than random noise. Costs up to one line period
            (20 ms at 50 Hz) of waiting per reading.
        averaging : int
            Number of readings the instrument averages internally into
            each result. 0 or 1 disables it. Reduces random noise by
            about sqrt(N), but multiplies time per reading by N - usually
            a worse deal than raising ``nplc`` for the same time budget.
        auto_delay : bool
            Insert the instrument's automatic settling delay before each
            measurement, letting the input settle after the relay closes.
            Disabling it can cause readings smeared from the previous
            channel, especially with high source impedance.
        input_impedance_auto : bool
            Use >10 GOhm input impedance on the lowest ranges instead of
            a fixed 10 MOhm. Essential for high-impedance sources, which
            a 10 MOhm input would otherwise load down and read low; on
            low-impedance sources it makes no difference.

        Raises
        ------
        RuntimeError
            If the instrument rejects any setting.
        """
        chans = Channels.to_scpi(channels)
        self.write(f'SENSE:FUNCTION "VOLT:DC", {chans}')
        self.write(f"SENSE:VOLT:DC:NPLC {nplc}, {chans}")

        # A fixed range skips autorange hunting on every channel; the
        # measured difference is roughly 3x on sweep time.
        if range_v is None:
            self.write(f"SENSE:VOLT:DC:RANGE:AUTO ON, {chans}")
        else:
            self.write(f"SENSE:VOLT:DC:RANGE {range_v}, {chans}")

        self.write(f"SENSE:VOLT:DC:AZER:STATE {'ON' if autozero else 'OFF'}, {chans}")
        self.write(f"SENSE:VOLT:DC:LINE:SYNC {'ON' if line_sync else 'OFF'}, {chans}")
        self.write(f"SENSE:VOLT:DC:DELAY:AUTO {'ON' if auto_delay else 'OFF'}, {chans}")
        self.write(
            f"SENSE:VOLT:DC:INPUTIMPEDANCE {'AUTO' if input_impedance_auto else 'MOHM10'}"
            f", {chans}"
        )

        if averaging and averaging > 1:
            self.write(f"SENSE:VOLT:DC:AVERAGE:COUNT {averaging}, {chans}")
            self.write(f"SENSE:VOLT:DC:AVERAGE:TCONTROL REPEAT, {chans}")
            self.write(f"SENSE:VOLT:DC:AVERAGE:STATE ON, {chans}")
        else:
            self.write(f"SENSE:VOLT:DC:AVERAGE:STATE OFF, {chans}")

        errors = self.check_errors("configuring channels")
        if errors:
            raise RuntimeError("; ".join(errors))

    def start_scan(self, plan: ScanPlan,
                   range_v: float | None = DEFAULT_RANGE_V,
                   nplc: float = DEFAULT_NPLC,
                   autozero: bool = False,
                   line_sync: bool = False,
                   averaging: int = 0,
                   auto_delay: bool = True,
                   input_impedance_auto: bool = False):
        """
        Configure the channels in ``plan`` and start scanning continuously.

        The instrument free-runs the scan autonomously once started,
        storing every reading into its buffer; reading results out does
        not slow it down.

        Measurement parameters mirror :meth:`configure_dcv`, which
        documents the full speed/accuracy trade-off for each. They are
        named explicitly rather than forwarded as ``**kwargs`` so a
        mistyped option fails here instead of deep inside configuration.

        Parameters
        ----------
        plan : ScanPlan
            What to scan, and how often each channel is sampled.
        range_v : float | None
            Fixed DC voltage range in volts; ``None`` enables autoranging.
        nplc : float
            Integration time in power-line cycles.
        autozero : bool
            Periodically cancel internal offset drift.
        line_sync : bool
            Synchronise readings to the mains cycle.
        averaging : int
            Readings averaged into each result; 0 or 1 disables.
        auto_delay : bool
            Insert the automatic per-channel settling delay.
        input_impedance_auto : bool
            Use >10 GOhm input impedance instead of a fixed 10 MOhm.

        Raises
        ------
        RuntimeError
            If the instrument rejects the configuration, or builds a scan
            list of unexpected length (which would desynchronise cycle
            boundary calculations).
        """
        self.abort()
        self.write("*RST")
        self.write("*CLS")
        self._stopped = False

        self.configure_dcv(
            plan.channels,
            range_v=range_v,
            nplc=nplc,
            autozero=autozero,
            line_sync=line_sync,
            averaging=averaging,
            auto_delay=auto_delay,
            input_impedance_auto=input_impedance_auto,
        )

        self.write(f"ROUTE:SCAN:CREATE {Channels.to_scpi(plan.sequence)}")
        errors = self.check_errors("creating scan list")
        if errors:
            raise RuntimeError("; ".join(errors))

        # A silently truncated scan list would corrupt cycle alignment,
        # so verify the instrument built what we asked for.
        steps = int(self.query("ROUTE:SCAN:COUNT:STEP?"))
        if steps != plan.cycle_length:
            raise RuntimeError(
                f"scan list has {steps} steps, expected {plan.cycle_length}"
            )

        self.write("ROUTE:SCAN:COUNT:SCAN 0")        # repeat forever
        self.write(f'TRACE:CLEAR "{self.buffer_name}"')
        self.check_errors("preparing scan")

        self._plan = plan
        self._next_index = 1
        self.write("INITIATE")

    # -- readout ----------------------------------------------------------
    @property
    def plan(self) -> ScanPlan:
        """The active scan plan."""
        if self._plan is None:
            raise RuntimeError("no scan started; call start_scan() first")
        return self._plan

    def buffered_count(self) -> int:
        """Number of readings currently stored in the instrument buffer."""
        return int(self.query(f'TRACE:ACTUAL? "{self.buffer_name}"'))

    def read_latest_cycle(self) -> tuple[int, dict[int, float]]:
        """
        Return the most recent *complete* cycle, one value per channel.

        Aligning to cycle boundaries ensures every channel in the result
        comes from the same pass. Without it, the final N buffer entries
        straddle a boundary and different channels end up from different
        cycles, which shows up as staleness rolling down the channel list.

        Channels sampled several times per cycle resolve to their last
        sample within that cycle; use :meth:`iter_samples` to keep all of
        them.

        Returns
        -------
        tuple[int, dict[int, float]]
            Completed cycle number, and channel -> value.

        Raises
        ------
        NotReadyError
            If the first cycle has not finished yet; retry later.
        ScanDataError
            If a completed cycle is missing channels, which indicates a
            real fault rather than a timing race.
        """
        plan = self.plan
        n = self.buffered_count()
        if n < plan.cycle_length:
            raise NotReadyError(
                f"only {n}/{plan.cycle_length} readings buffered, "
                "waiting for the first cycle"
            )

        cycles_done = n // plan.cycle_length
        end = cycles_done * plan.cycle_length
        start = end - plan.cycle_length + 1

        readings = {s.channel: s.value for s in self._fetch(start, end)}

        missing = [c for c in plan.channels if c not in readings]
        if missing:
            raise ScanDataError(
                f"cycle {cycles_done} is missing channels: "
                f"{ScanPlan.describe_channels(missing)}"
            )
        return cycles_done, readings

    def iter_cycles(self, poll: float = 0.05,
                    max_cycles: int | None = None
                    ) -> Iterator[tuple[int, dict[int, float]]]:
        """
        Yield each completed cycle as it becomes available.

        Polls faster than a cycle completes but only yields when a new
        cycle has actually finished, so nothing is duplicated and latency
        stays low.

        Parameters
        ----------
        poll : float
            Seconds between checks for a newly completed cycle.
        max_cycles : int | None
            Stop after this many cycles; ``None`` runs until interrupted.

        Yields
        ------
        tuple[int, dict[int, float]]
            Cycle number, and channel -> value.
        """
        last = 0
        produced = 0
        while max_cycles is None or produced < max_cycles:
            try:
                cycle, readings = self.read_latest_cycle()
            except NotReadyError:
                # Only the benign "no data yet" case is retried; a
                # ScanDataError propagates so real faults are visible
                # instead of becoming a silent infinite loop.
                time.sleep(poll)
                continue

            if cycle != last:
                last = cycle
                produced += 1
                yield cycle, readings
            else:
                time.sleep(poll)

    def iter_samples(self, poll: float = 0.05, max_chunk: int = 500
                     ) -> Iterator[Sample]:
        """
        Yield every individual reading, losing nothing to aggregation.

        Where :meth:`iter_cycles` collapses a cycle to one value per
        channel, this yields each reading with the instrument's own
        timestamp - so a channel sampled four times per cycle produces
        four samples.

        Parameters
        ----------
        poll : float
            Seconds between checks when no new readings are available.
        max_chunk : int
            Maximum readings to transfer per query, bounding the size of
            any single readout.

        Yields
        ------
        Sample
        """
        while True:
            n = self.buffered_count()
            if n < self._next_index - 1:
                # The buffer shrank below our cursor, so it was cleared or
                # wrapped and the readings in between are gone for good.
                raise ScanDataError(
                    f"buffer went backwards ({n} readings < cursor "
                    f"{self._next_index}); readings were lost"
                )
            if n < self._next_index:
                time.sleep(poll)        # nothing new since the last drain
                continue

            end = min(n, self._next_index + max_chunk - 1)
            samples = self._fetch(self._next_index, end)

            # Advance past the whole requested span, not past the last
            # parsed sample: a trailing malformed triple would otherwise
            # leave the cursor stuck, re-reading it forever.
            self._next_index = end + 1

            if not samples:
                time.sleep(poll)
                continue
            yield from samples

    def _fetch(self, start: int, end: int) -> list[Sample]:
        """
        Retrieve a contiguous span of buffer entries.

        Parameters
        ----------
        start, end : int
            Inclusive 1-based buffer indices.

        Returns
        -------
        list[Sample]
        """
        resp = self.query(
            f'TRACE:DATA? {start}, {end}, "{self.buffer_name}", READ, CHAN, REL'
        )
        fields = resp.split(",")

        samples = []
        # Elements arrive as "<reading>,<channel>,<rel_seconds>" triples.
        # The buffer index is derived from the triple's ordinal position,
        # never from a running counter: a malformed triple must still
        # consume its index, or every later sample would be mislabelled
        # and the read cursor would rewind onto consumed data.
        for ordinal, i in enumerate(range(0, len(fields) - 2, 3)):
            try:
                value = float(fields[i])
                channel = int(fields[i + 1].strip())
                rel = float(fields[i + 2])
            except ValueError:
                continue            # skip a malformed triple, keep the rest
            samples.append(Sample(start + ordinal, channel, value, rel))
        return samples

    # -- lifecycle --------------------------------------------------------
    def abort(self):
        """Stop any scan in progress. Safe to call repeatedly."""
        if self._stopped:
            return
        self._stopped = True
        try:
            self._inst.write("ABORT")
        except Exception:
            pass                    # session may already be closed

    def close(self):
        """Stop scanning and close the VISA session."""
        self.abort()
        # Drop the exit handler, or a long-lived process that opens many
        # connections would accumulate handlers pinning dead sessions.
        if self._atexit_registered:
            atexit.unregister(self.abort)
            self._atexit_registered = False
        try:
            self._inst.close()
        except Exception:
            pass

    def __enter__(self) -> "DAQ6510":
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
