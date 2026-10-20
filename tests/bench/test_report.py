from flux_endpoint.bench.report import render_cold_start_report, render_report
from flux_endpoint.bench.runner import Sample

L40S = "NVIDIA L40S"

SAMPLES = [
    Sample("job-1", "COMPLETED", 75_000, 60_000, 14_000, gpu=L40S),
    Sample("job-2", "COMPLETED", 15_000, 1_000, 14_000, gpu=L40S),
    Sample("job-3", "FAILED", 2_000, 500, None, gpu=L40S, error="Invalid input"),
    Sample("job-4", "COMPLETED", 4_500, 100, 4_000, cost_usd=0.021),
    Sample(None, "CLIENT_TIMEOUT", 600_000, error="Job job-5 is still IN_QUEUE"),
]


def test_latency_table_groups_jobs_by_target() -> None:
    """Given mixed samples, When rendered, Then each target shows job counts and timings."""
    report = render_report(SAMPLES, gpu_prices={L40S: 1.75}, public_usd_per_image=0.021)

    assert "| NVIDIA L40S | 3 | 2 | 30.5 | 60.0 | 14.0 | 14.0 | 72.0 |" in report
    assert "| Public Endpoint | 1 | 1 | 0.1 | 0.1 | 4.0 | 4.0 | 4.5 |" in report
    assert "| No output | 1 | 0 | - | - | - | - | - |" in report


def test_cost_table_compares_gpu_with_public_endpoint() -> None:
    """Given a priced GPU, When rendered, Then cost per image and break-even are shown."""
    report = render_report(SAMPLES, gpu_prices={L40S: 1.75}, public_usd_per_image=0.021)

    assert "| NVIDIA L40S | $1.75 | 14.0 | $0.0068 | 257 | 83 | 32% |" in report
    assert "| Public Endpoint | - | 4.0 | $0.0210 | - | - | - |" in report


def test_unpriced_gpu_is_left_out_of_cost_table() -> None:
    """Given no price for the GPU, When rendered, Then the cost table explains the gap."""
    report = render_report(SAMPLES[:2], gpu_prices={}, public_usd_per_image=0.021)

    assert "| NVIDIA L40S | ? | 14.0 | - | 257 | - | - |" in report


def test_cold_start_table_groups_jobs_by_variant() -> None:
    """Given cold starts of three variants, When rendered, Then each row shows its timings."""
    samples = [
        Sample("job-1", "COMPLETED", 50_000, 45_000, 5_000, variant="baked"),
        Sample("job-2", "COMPLETED", 70_000, 65_000, 5_000, variant="baked"),
        Sample("job-3", "COMPLETED", 20_000, 15_000, 5_000, variant="cached-model"),
        Sample(None, "CLIENT_TIMEOUT", 900_000, error="still queued", variant="global-volume"),
    ]

    report = render_cold_start_report(samples)

    assert "| baked | 2 | 2 | 55.0 | 65.0 | 5.0 |" in report
    assert "| cached-model | 1 | 1 | 15.0 | 15.0 | 5.0 |" in report
    assert "| global-volume | 1 | 0 | - | - | - |" in report


def test_cold_starts_without_a_variant_are_grouped_as_unnamed() -> None:
    """Given a cold start saved without a variant, When rendered, Then its row is unnamed."""
    report = render_cold_start_report([Sample("job-1", "COMPLETED", 50_000, 45_000, 5_000)])

    assert "| unnamed | 1 | 1 | 45.0 | 45.0 | 5.0 |" in report
