"""
Command-line acquisition tool for the Keithley DAQ6510, built on the
reusable :mod:`daq6510` library.

Supports two scan layouts:

* **flat**   - every channel sampled at the same rate (the common case)
* **tiered** - channel groups sampled at different relative rates

and two logging formats:

* **wide** - one CSV row per cycle, one column per channel
* **long** - one CSV row per individual reading, discarding nothing

Examples
--------
Read 20 channels at a uniform rate::

    python daq6510_reader.py --channels 101-120

Tiered 4:2:1 scan (fast/medium/slow groups)::

    python daq6510_reader.py --tiers 101-110:4 111-120:2 121-130:1

Log every individual sample to CSV::

    python daq6510_reader.py --channels 101-110 --log out.csv --format long

Trade speed for lower noise::

    python daq6510_reader.py --channels 101-110 --nplc 1 --autozero
"""

import argparse
import csv
import datetime as dt
import sys

from daq6510 import (
    DAQ6510,
    DEFAULT_NPLC,
    DEFAULT_RANGE_V,
    RateMonitor,
    ScanPlan,
    format_rate_report,
)

DEFAULT_IP = "169.254.160.157"


def parse_channel_spec(spec: str) -> list[int]:
    """
    Parse a channel specification into channel numbers.

    Accepts comma-separated numbers and inclusive ``start-end`` ranges,
    e.g. ``"101-110"``, ``"101,103,105"`` or ``"101-105,110"``.

    Parameters
    ----------
    spec : str
        Channel specification.

    Returns
    -------
    list[int]
        Channel numbers in the order written.

    Raises
    ------
    ValueError
        If the specification is malformed or describes a reversed range.
    """
    channels: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, _, hi_s = part.partition("-")
            lo, hi = int(lo_s), int(hi_s)
            if hi < lo:
                raise ValueError(f"reversed channel range: {part!r}")
            channels.extend(range(lo, hi + 1))
        else:
            channels.append(int(part))
    if not channels:
        raise ValueError(f"no channels in specification: {spec!r}")
    return channels


def parse_tier_spec(specs: list[str]) -> ScanPlan:
    """
    Build a tiered scan plan from ``channels:weight`` arguments.

    Parameters
    ----------
    specs : list[str]
        Specifications such as ``["101-110:4", "111-120:2", "121-130:1"]``,
        where the weight is how many times per cycle that group is read.

    Returns
    -------
    ScanPlan
    """
    groups = []
    for spec in specs:
        chan_part, sep, weight_part = spec.rpartition(":")
        if not sep:
            raise ValueError(
                f"tier {spec!r} must be written as CHANNELS:WEIGHT, e.g. 101-110:4"
            )
        groups.append((parse_channel_spec(chan_part), int(weight_part)))
    return ScanPlan.tiered(groups)


def describe_plan(plan: ScanPlan, step_ms: float = 12.2) -> str:
    """
    Summarise a scan plan and its expected per-group sampling rates.

    Parameters
    ----------
    plan : ScanPlan
        The plan to describe.
    step_ms : float
        Measured time per scan step, used to estimate rates. The default
        reflects a DAQ6510 with a 7702 multiplexer.

    Returns
    -------
    str
        Multi-line human-readable summary.
    """
    cycle_s = plan.cycle_length * step_ms / 1000.0
    lines = [
        f"  {plan.cycle_length} readings per cycle "
        f"(~{cycle_s:.2f} s at {step_ms:.1f} ms/step)"
    ]
    for name, members in plan.groups.items():
        per_cycle = plan.samples_per_cycle(members[0])
        # Describe channels exactly: a sparse group must not be rendered
        # as a contiguous range it does not actually cover.
        lines.append(
            f"  group {name}: channels {ScanPlan.describe_channels(members)} "
            f"({len(members)} ch) x{per_cycle} per cycle "
            f"-> ~{per_cycle / cycle_s:.2f} Hz each"
        )
    return "\n".join(lines)


def report_rates(monitor: RateMonitor, plan: ScanPlan, force: bool = False):
    """
    Print a rate report to stderr, if one is due.

    Rate output goes to stderr so it never contaminates data piped from
    stdout. The measurement itself lives in :class:`RateMonitor`; this
    only renders it, so a dashboard can consume the same report object
    without inheriting console formatting.

    Parameters
    ----------
    monitor : RateMonitor
        Monitor accumulating samples.
    plan : ScanPlan
        Active plan, used to describe channels.
    force : bool
        Emit a final partial-window report even if not yet due.
    """
    report = monitor.take_report(force=force)
    if report is not None:
        print(format_rate_report(report, plan), file=sys.stderr)


def log_wide(daq: DAQ6510, csv_file, quiet: bool, monitor: RateMonitor | None):
    """
    Emit one row per completed cycle, one column per channel.

    Channels sampled several times per cycle contribute their last
    sample in that cycle; use long format to keep every sample.

    Rate reporting is not available here: cycle reads do not retrieve
    per-sample instrument timestamps, which the rate measurement is
    derived from. ``monitor`` is accepted for a uniform signature and
    reported on only at the end, from cycle timing.

    Parameters
    ----------
    daq : DAQ6510
        Instrument with an active scan.
    csv_file : file | None
        Open file to write CSV rows to, or ``None`` for console only.
    quiet : bool
        Suppress console output.
    monitor : RateMonitor | None
        Unused in wide mode; see above.
    """
    plan = daq.plan
    writer = csv.writer(csv_file) if csv_file else None
    if writer:
        writer.writerow(["timestamp", "cycle"] +
                        [f"CH{c}" for c in plan.channels])

    for cycle, readings in daq.iter_cycles():
        stamp = dt.datetime.now().isoformat(timespec="milliseconds")
        if writer:
            writer.writerow([stamp, cycle] +
                            [f"{readings[c]:.6f}" for c in plan.channels])
            csv_file.flush()
        if not quiet:
            preview = "  ".join(f"{c}={readings[c]:+8.4f}"
                                for c in plan.channels[:8])
            more = " ..." if len(plan.channels) > 8 else ""
            print(f"{stamp}  cycle {cycle:<6d} {preview}{more}")


def log_long(daq: DAQ6510, csv_file, quiet: bool, monitor: RateMonitor | None):
    """
    Emit one row per individual reading, discarding nothing.

    Uses the instrument's own relative timestamps, which reflect when
    each reading was actually taken rather than when it was read out.

    Parameters
    ----------
    daq : DAQ6510
        Instrument with an active scan.
    csv_file : file | None
        Open file to write CSV rows to, or ``None`` for console only.
    quiet : bool
        Suppress console output.
    monitor : RateMonitor | None
        Rate monitor to feed, or ``None`` to skip rate reporting.
    """
    plan = daq.plan
    writer = csv.writer(csv_file) if csv_file else None
    if writer:
        writer.writerow(["timestamp", "index", "channel", "group",
                         "rel_seconds", "volts"])

    written = 0
    for sample in daq.iter_samples():
        group = plan.group_of(sample.channel)
        if writer:
            stamp = dt.datetime.now().isoformat(timespec="milliseconds")
            writer.writerow([stamp, sample.index, sample.channel, group,
                             f"{sample.rel_time:.6f}", f"{sample.value:.6f}"])
            written += 1
            if written % 50 == 0:       # bound data loss without thrashing disk
                csv_file.flush()
        if not quiet:
            print(f"{sample.rel_time:10.4f}s  CH{sample.channel} "
                  f"[{group}]  {sample.value:+10.6f} V")

        if monitor:
            monitor.add(sample)
            report_rates(monitor, plan)


def main():
    """Parse arguments, start the scan, and log until interrupted."""
    parser = argparse.ArgumentParser(
        description="Continuous DAQ6510 acquisition with optional tiered scan rates.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples\n--------\n", 1)[-1],
    )
    parser.add_argument("--ip", default=DEFAULT_IP, help="Instrument IP address")

    scan = parser.add_mutually_exclusive_group()
    scan.add_argument("--channels", default="101-130",
                      help="Channels to scan at a uniform rate, e.g. 101-120 "
                           "or 101,103,105 (default: 101-130)")
    scan.add_argument("--tiers", nargs="+", metavar="CH:WEIGHT",
                      help="Channel groups at different rates, e.g. "
                           "--tiers 101-110:4 111-120:2 121-130:1")

    parser.add_argument("--log", help="CSV file to append readings to")
    parser.add_argument("--format", choices=("wide", "long"), default="wide",
                        help="wide: one row per cycle, one column per channel. "
                             "long: one row per individual reading (default: wide)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress console output (useful when logging)")
    parser.add_argument("--stats-interval", type=float, default=30.0,
                        metavar="SECONDS",
                        help="Report achieved sampling rate per group every N "
                             "seconds, to stderr (default: 30; 0 disables). "
                             "Requires --format long, which carries the "
                             "instrument timestamps the rate is measured from.")

    # Measurement settings. See DAQ6510.configure_dcv for the full
    # speed/accuracy trade-off table behind each of these.
    meas = parser.add_argument_group(
        "measurement settings",
        "Speed/accuracy trade-offs. Defaults favour speed; a scan step "
        "costs ~12 ms and is dominated by multiplexer relay switching."
    )
    meas.add_argument("--range", type=float, default=DEFAULT_RANGE_V,
                      dest="range_v",
                      help=f"Fixed voltage range in volts (default: {DEFAULT_RANGE_V:g}). "
                           "Must exceed the largest expected signal.")
    meas.add_argument("--autorange", action="store_true",
                      help="Autorange instead of a fixed range: adapts to unknown "
                           "signals but roughly triples sweep time.")
    meas.add_argument("--nplc", type=float, default=DEFAULT_NPLC,
                      help=f"Integration time in power-line cycles (default: {DEFAULT_NPLC:g}). "
                           "Higher rejects mains hum and lowers noise, but is slower.")
    meas.add_argument("--autozero", action="store_true",
                      help="Periodically cancel internal offset drift; ~2x slower.")
    meas.add_argument("--line-sync", action="store_true",
                      help="Synchronise readings to the mains cycle to reject hum; "
                           "costs up to one line period per reading.")
    meas.add_argument("--averaging", type=int, default=0, metavar="N",
                      help="Average N readings into each result; N times slower (default: off).")
    meas.add_argument("--high-impedance", action="store_true",
                      help="Use >10 GOhm input instead of 10 MOhm, for high-impedance "
                           "sources that a 10 MOhm input would load down.")
    args = parser.parse_args()

    try:
        plan = (parse_tier_spec(args.tiers) if args.tiers
                else ScanPlan.flat(parse_channel_spec(args.channels)))
    except ValueError as exc:
        parser.error(str(exc))

    if args.stats_interval and args.format != "long":
        parser.error("--stats-interval requires --format long, which carries "
                     "the instrument timestamps rates are measured from")

    csv_file = None
    monitor = (RateMonitor(plan, window_seconds=args.stats_interval)
               if args.stats_interval else None)
    try:
        with DAQ6510(args.ip) as daq:
            print(f"Connected: {daq.identity}")

            daq.start_scan(
                plan,
                range_v=None if args.autorange else args.range_v,
                nplc=args.nplc,
                autozero=args.autozero,
                line_sync=args.line_sync,
                averaging=args.averaging,
                input_impedance_auto=args.high_impedance,
            )
            print("Scan started:")
            print(describe_plan(plan))

            if args.log:
                csv_file = open(args.log, "a", newline="")
                print(f"Logging {args.format} format to {args.log}")

            print("\nAcquiring (Ctrl+C to stop)...\n")
            emit = log_long if args.format == "long" else log_wide
            try:
                emit(daq, csv_file, args.quiet, monitor)
            finally:
                # Summarise the final partial window, so short runs still
                # report a rate rather than nothing at all.
                if monitor:
                    report_rates(monitor, plan, force=True)

    except KeyboardInterrupt:
        print("\nStopping acquisition.")
    except (RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if csv_file:
            csv_file.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
