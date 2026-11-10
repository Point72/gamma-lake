"""Compare the pre-Rust Python engine with the current one, by thread count and storage.

Every measurement runs in a fresh subprocess pinned to a CPU set, so "1 thread" means
one core for both engines however many threads their libraries start. The old engine
is imported from a separate source tree::

    git archive <old-commit> gammalake | tar -x -C /path/to/old
    python benchmarks/compare_engines.py run --old-src /path/to/old \\
        --storage local=/tmp/bench --storage s3=s3://bucket/prefix \\
        --threads 1 8 --scale small --out results.jsonl
    python benchmarks/compare_engines.py report results.jsonl

S3 credentials come from the standard ``AWS_*`` environment variables
(``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``, ``AWS_ENDPOINT_URL``; add
``AWS_ALLOW_HTTP=true`` for plain-HTTP endpoints). Both engines read them.
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import median
from urllib.parse import urlparse

REPO = Path(__file__).resolve().parent.parent
SCALES = {"small": (200, 30), "large": (2000, 250)}
N_GROUPS = 3
N_FEATURES = 50
READS = ("full", "window", "symbol", "narrow")
WRITES = ("create", "append", "overlap")
ORIGIN = datetime(2024, 1, 1, tzinfo=UTC)


# Worker side: runs in a subprocess with the engine, affinity and thread variables set.


def _frame(group: int, scale: str):
    import numpy as np
    import polars as pl

    n_symbols, n_days = SCALES[scale]
    rng = np.random.default_rng(group)
    days = [ORIGIN + timedelta(days=d) for d in range(n_days)]
    symbols = [f"SYM_{i:05d}" for i in range(n_symbols)]
    rows = n_days * n_symbols
    data = {
        "timestamp": np.repeat(np.array(days, dtype="datetime64[us]"), n_symbols),
        "symbol": symbols * n_days,
        **{f"g{group}_f{i:02d}": rng.standard_normal(rows) for i in range(N_FEATURES)},
    }
    return pl.DataFrame(data).with_columns(pl.col("timestamp").dt.replace_time_zone("UTC"))


def _checksum(frame) -> list:
    """Row count and an order-independent hash of a read, with canonical column order."""
    frame = frame.select(sorted(frame.columns)).sort("timestamp", "symbol")
    return [frame.height, int(frame.hash_rows().sum())]


def _lake(path: str):
    from gammalake import GammaFeatureLake

    return GammaFeatureLake(base_path=path).initialize()


def _features(groups: int) -> list[str]:
    return [f"g{g}_f{i:02d}" for g in range(groups) for i in range(N_FEATURES)]


def _read(lake, workload: str, scale: str):
    import polars as pl

    _, n_days = SCALES[scale]
    names = _features(N_GROUPS)
    if workload == "full":
        return lake.read(names)
    if workload == "window":
        start = ORIGIN + timedelta(days=n_days // 2)
        return lake.read(names, start=start, end=start + timedelta(days=6))
    if workload == "symbol":
        return lake.read(names, materialized=False).filter(pl.col("symbol") == "SYM_00007").collect()
    return lake.read(names[:5])


def _write(path: str, workload: str, scale: str) -> tuple[float, list, list]:
    """Prepare a lake, time one ``add_features`` call and return it with checksums."""
    import polars as pl

    _, n_days = SCALES[scale]
    frame = _frame(0, scale)
    lake = _lake(path)
    expected = frame
    if workload == "create":
        timed = frame
    elif workload == "append":
        half = ORIGIN + timedelta(days=n_days // 2)
        lake.add_features(frame.filter(pl.col("timestamp") < half), owner="bench")
        timed = frame.filter(pl.col("timestamp") >= half)
    else:
        cut = ORIGIN + timedelta(days=n_days // 3)
        lake.add_features(frame, owner="bench")
        names = _features(1)
        timed = frame.filter(pl.col("timestamp") >= cut).with_columns(pl.col(n) + 1 for n in names)
        expected = frame.with_columns(pl.when(pl.col("timestamp") >= cut).then(pl.col(n) + 1).otherwise(pl.col(n)).alias(n) for n in names)
    start = time.perf_counter()
    lake.add_features(timed, owner="bench", overlap_mode="copy")
    elapsed = time.perf_counter() - start
    return elapsed, _checksum(_lake(path).read(_features(1))), _checksum(expected)


def worker(args) -> None:
    import gammalake

    origin = Path(gammalake.__file__).resolve()
    expected = Path(args.old_src if args.engine == "old" else REPO).resolve()
    if expected not in origin.parents:
        raise RuntimeError(f"{args.engine} engine imported from {origin}, expected under {expected}")
    record = {"load": os.getloadavg()[0], "cpus": len(os.sched_getaffinity(0))}
    if args.task == "build":
        lake = _lake(args.path)
        start = time.perf_counter()
        for group in range(N_GROUPS):
            lake.add_features(_frame(group, args.scale), owner="bench")
        record["seconds"] = time.perf_counter() - start
    elif args.task == "read":
        lake = _lake(args.path)
        record["samples"] = {}
        for workload in READS:
            start = time.perf_counter()
            frame = _read(lake, workload, args.scale)
            first = time.perf_counter() - start
            times = []
            for _ in range(args.repeat):
                start = time.perf_counter()
                _read(lake, workload, args.scale)
                times.append(time.perf_counter() - start)
            record["samples"][workload] = {"first": first, "times": times, "checksum": _checksum(frame)}
    else:
        seconds, got, want = _write(args.path, args.task, args.scale)
        if got != want:
            raise RuntimeError(f"{args.task} wrote {got}, expected {want}")
        record["seconds"] = seconds
    print(json.dumps(record))


# Driver side.


def _cpu_set(threads: int) -> list[int]:
    allowed = sorted(os.sched_getaffinity(0))
    if threads > len(allowed):
        raise ValueError(f"{threads} threads requested but only {len(allowed)} CPUs are available")
    return allowed[:threads]


def _spawn(args, engine: str, task: str, path: str, scale: str, threads: int) -> dict:
    env = dict(os.environ)
    env.update(POLARS_MAX_THREADS=str(threads), GAMMA_LAKE_THREADS=str(threads), TOKIO_WORKER_THREADS=str(threads))
    env.setdefault("AWS_S3_ALLOW_UNSAFE_RENAME", "true")
    pythonpath = [args.old_src] if engine == "old" else [str(REPO)]
    env["PYTHONPATH"] = os.pathsep.join(pythonpath + [p for p in [env.get("PYTHONPATH")] if p])
    # taskset sets the affinity before the interpreter starts any thread pool.
    cpus = ",".join(map(str, _cpu_set(threads)))
    command = ["taskset", "--cpu-list", cpus, sys.executable, "-W", "ignore", __file__, "worker", engine, task, path, scale]
    command += ["--repeat", str(args.repeat), "--old-src", args.old_src]
    run = subprocess.run(command, check=False, env=env, capture_output=True, text=True, timeout=args.timeout)
    if run.returncode != 0:
        raise RuntimeError(f"{engine} {task} {path} failed:\n{run.stderr[-3000:]}")
    return json.loads(run.stdout.strip().splitlines()[-1])


def _join(base: str, *parts: str) -> str:
    return "/".join([base.rstrip("/"), *parts])


def _layout(path: str) -> dict:
    """Parquet file count and bytes under a lake."""
    files = size = 0
    if path.startswith("s3://"):
        client, parsed = _s3(), urlparse(path)
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=parsed.netloc, Prefix=parsed.path.strip("/") + "/"):
            for item in page.get("Contents", []):
                if item["Key"].endswith(".parquet"):
                    files, size = files + 1, size + item["Size"]
    else:
        for file in Path(path).rglob("*.parquet"):
            files, size = files + 1, size + file.stat().st_size
    return {"parquet_files": files, "parquet_bytes": size}


def _s3():
    import boto3

    return boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL"))


def _remove(path: str) -> None:
    if path.startswith("s3://"):
        client, parsed = _s3(), urlparse(path)
        # One key at a time: some S3-compatible stores reject DeleteObjects without Content-MD5.
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=parsed.netloc, Prefix=parsed.path.strip("/") + "/"):
            for item in page.get("Contents", []):
                client.delete_object(Bucket=parsed.netloc, Key=item["Key"])
    else:
        shutil.rmtree(path, ignore_errors=True)


def _identity(args) -> dict:
    import deltalake
    import polars
    import pyarrow

    def git(*cmd: str, cwd: Path = REPO) -> str:
        return subprocess.run(["git", *cmd], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()

    extension = next((REPO / "gammalake").glob("gammalake*.so"), None)
    delta_lib = next(Path(deltalake.__file__).parent.glob("_internal*.so"), None)
    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "old_src": args.old_src,
        "old_label": args.old_label,
        "extension": None if extension is None else {"path": str(extension), "bytes": extension.stat().st_size, "mtime": extension.stat().st_mtime},
        "python": platform.python_version(),
        "polars": polars.__version__,
        "deltalake": deltalake.__version__,
        "deltalake_library": None if delta_lib is None else {"path": str(delta_lib), "bytes": delta_lib.stat().st_size},
        "pyarrow": pyarrow.__version__,
        "host_cpus": os.cpu_count(),
        "started": datetime.now(UTC).isoformat(),
    }


def run(args) -> None:
    storages = dict(item.split("=", 1) for item in args.storage)
    max_threads = max(args.threads)
    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:6]}"
    out = open(args.out, "a")  # noqa: SIM115 - closed in the finally block below

    def emit(**record):
        out.write(json.dumps({"run": run_id, **record}) + "\n")
        out.flush()
        print({k: v for k, v in record.items() if k != "samples"}, file=sys.stderr)

    emit(kind="identity", **_identity(args))
    roots = {name: _join(base, "gamma-lake-compare", run_id) for name, base in storages.items()}
    try:
        for scale in args.scale:
            fixtures = {}
            for storage, root in roots.items():
                for engine in ("old", "new"):
                    path = _join(root, scale, f"fixture-{engine}")
                    if not path.startswith("s3://"):
                        Path(path).mkdir(parents=True)
                    built = _spawn(args, engine, "build", path, scale, max_threads)
                    fixtures[storage, engine] = path
                    emit(kind="fixture", storage=storage, scale=scale, writer=engine, seconds=built["seconds"], **_layout(path))
            # Rounds interleave every configuration so drift on a shared host spreads evenly.
            for round_ in range(args.rounds):
                for storage, root in roots.items():
                    for threads in args.threads:
                        for engine, writer in (("old", "old"), ("new", "old"), ("new", "new")):
                            result = _spawn(args, engine, "read", fixtures[storage, writer], scale, threads)
                            emit(kind="read", round=round_, storage=storage, scale=scale, threads=threads, engine=engine, writer=writer, **result)
                        for workload in WRITES:
                            for engine in ("old", "new"):
                                path = _join(root, scale, f"write-{engine}-{workload}-{threads}-{round_}")
                                if not path.startswith("s3://"):
                                    Path(path).mkdir(parents=True)
                                result = _spawn(args, engine, workload, path, scale, threads)
                                emit(
                                    kind="write",
                                    round=round_,
                                    storage=storage,
                                    scale=scale,
                                    threads=threads,
                                    engine=engine,
                                    workload=workload,
                                    **result,
                                )
                                _remove(path)
    finally:
        if not args.keep:
            for root in roots.values():
                _remove(root)
        out.close()


def _ms(values) -> str:
    return f"{median(values) * 1e3:,.0f}"


def _report_run(records: list[dict]) -> bool:
    """Print one run's tables; return whether old and new read the same data."""
    identity = next(r for r in records if r["kind"] == "identity")
    dirty = " (dirty)" if identity["dirty"] else ""
    print(f"## Run {identity['run']}\n")
    print(
        f"Commit {identity['commit'][:9]}{dirty}, old engine {identity['old_label']}, Python {identity['python']}, polars {identity['polars']}, deltalake {identity['deltalake']}."
    )
    if library := identity.get("deltalake_library"):
        print(f"deltalake library {library['path']}, {library['bytes'] / 1e6:,.0f} MB.")
    print("\n| storage | scale | writer | parquet files | MB |\n|---|---|---|---|---|")
    for r in (r for r in records if r["kind"] == "fixture"):
        print(f"| {r['storage']} | {r['scale']} | {r['writer']} | {r['parquet_files']} | {r['parquet_bytes'] / 1e6:.0f} |")

    reads: dict = {}
    for r in (r for r in records if r["kind"] == "read"):
        for workload, sample in r["samples"].items():
            entry = reads.setdefault(
                (r["storage"], r["scale"], workload, r["threads"], r["engine"], r["writer"]), {"times": [], "first": [], "checksums": set()}
            )
            entry["times"] += sample["times"]
            entry["first"].append(sample["first"])
            entry["checksums"].add(tuple(sample["checksum"]))
    by_query: dict = {}
    for (storage, scale, workload, *_), entry in reads.items():
        by_query.setdefault((storage, scale, workload), set()).update(entry["checksums"])
    mismatched = sorted(key for key, sums in by_query.items() if len(sums) > 1)
    if mismatched:
        print(f"\nThe engines read different data for {mismatched}; no timings are reported for this run.\n")
        return False

    print("\nReads, median ms, with the first read in each process in brackets. Writer is the engine that wrote the lake.\n")
    print("| storage | scale | query | threads | old | new, same lake | new, own lake | old / new |\n|---|---|---|---|---|---|---|---|")
    for storage, scale, workload, threads in sorted({k[:4] for k in reads}, key=lambda k: (k[0], k[1], READS.index(k[2]), k[3])):
        cells = [reads.get((storage, scale, workload, threads, e, w)) for e, w in (("old", "old"), ("new", "old"), ("new", "new"))]
        text = [f"{_ms(c['times'])} ({_ms(c['first'])})" if c else "-" for c in cells]
        ratio = f"{median(cells[0]['times']) / median(cells[1]['times']):.1f}x" if cells[0] and cells[1] else "-"
        print(f"| {storage} | {scale} | {workload} | {threads} | {' | '.join(text)} | {ratio} |")

    writes: dict = {}
    for r in (r for r in records if r["kind"] == "write"):
        writes.setdefault((r["storage"], r["scale"], r["workload"], r["threads"], r["engine"]), []).append(r["seconds"])
    print("\nWrites, one add_features call on a prepared lake, median ms\n")
    print("| storage | scale | write | threads | old | new | old / new |\n|---|---|---|---|---|---|---|")
    for storage, scale, workload, threads in sorted({k[:4] for k in writes}, key=lambda k: (k[0], k[1], WRITES.index(k[2]), k[3])):
        old, new = writes.get((storage, scale, workload, threads, "old")), writes.get((storage, scale, workload, threads, "new"))
        ratio = f"{median(old) / median(new):.1f}x" if old and new else "-"
        print(f"| {storage} | {scale} | {workload} | {threads} | {_ms(old) if old else '-'} | {_ms(new) if new else '-'} | {ratio} |")
    print()
    return True


def report(args) -> None:
    with open(args.results) as results:
        records = [json.loads(line) for line in results]
    runs: dict = {}
    for record in records:
        runs.setdefault(record["run"], []).append(record)
    if args.run:
        runs = {args.run: runs[args.run]}
    # Each run is reported on its own, so results from different builds are never pooled.
    ok = [_report_run(run) for run in runs.values()]
    if not all(ok):
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run", help="run the comparison")
    runner.add_argument("--old-src", required=True, help="directory holding the old gammalake package")
    runner.add_argument("--old-label", default="old", help="label for the old engine, such as its commit")
    runner.add_argument("--storage", action="append", required=True, help="NAME=PATH or NAME=s3://bucket/prefix; repeatable")
    runner.add_argument("--threads", type=int, nargs="+", default=[1, os.cpu_count()])
    runner.add_argument("--scale", nargs="+", choices=sorted(SCALES), default=["small"])
    runner.add_argument("--rounds", type=int, default=3)
    runner.add_argument("--repeat", type=int, default=3, help="timed reads per workload per process")
    runner.add_argument("--timeout", type=float, default=3600)
    runner.add_argument("--keep", action="store_true", help="keep the generated lakes")
    runner.add_argument("--out", required=True)
    reporter = commands.add_parser("report", help="print markdown tables from a results file")
    reporter.add_argument("results")
    reporter.add_argument("--run", help="only this run id")
    work = commands.add_parser("worker", help=argparse.SUPPRESS)
    work.add_argument("engine", choices=["old", "new"])
    work.add_argument("task", choices=["build", "read", *WRITES])
    work.add_argument("path")
    work.add_argument("scale", choices=sorted(SCALES))
    work.add_argument("--repeat", type=int, default=3)
    work.add_argument("--old-src", required=True)
    args = parser.parse_args()
    {"run": run, "report": report, "worker": worker}[args.command](args)


if __name__ == "__main__":
    main()
