# Fixed-batch and residency results

Completed on 2026-10-25 at 03:30 UTC on GPU0, NVIDIA RTX PRO 4000 Blackwell, 145 W power limit. The GPU worker exited and the shared lock was released before the next serving experiment. These are diagnostic fixed-batch timing and static residency results; production scheduling and SLO conclusions require the separate serving experiments.

The fixed campaign completed three independently loaded rounds with rotated format order: 18 engine runs, 36 condition blocks, and 99 measured full-generation trials. Each request used 256 output tokens. Three warmup trials per batch preceded measurement; all last-two-warmup, minimum-duration, output-count, exact-batch, and zero-preemption checks passed. The independent raw auditor reports no failed audits. The experimental unit is an engine round; cycles within a block are not independent repetitions.

| Input tokens | Batch | BF16 output tok/s | MXFP6 output tok/s | SSNF output tok/s | SSNF/BF16 paired ratio, mean [range] |
|---:|---:|---:|---:|---:|---:|
| 2048 | 1 | 37.60 | 39.37 | 47.54 | 1.264 [1.184, 1.309] |
| 2048 | 4 | 137.17 | 138.36 | 164.98 | 1.203 [1.133, 1.240] |
| 2048 | 8 | 255.84 | 242.52 | 305.36 | 1.194 [1.130, 1.228] |
| 16384 | 1 | 33.66 | 31.91 | 40.49 | 1.203 [1.201, 1.207] |

Timing starts at the first released, nonempty scheduler-output timestamp and ends after all request outputs and final CUDA synchronization. The numerator is 255×B: every request has already produced its first token before barrier release. The first scheduler call is excluded; later scheduler, observer, asynchronous drain, and completion overhead are included. The stored pure engine-step sum excludes the last empty drain and is retained only as a diagnostic. This table does not measure network latency or production TTFT.

SSNF 2K round variation is material and retained. Independently loaded engines selected different measured split factors: the two affected bucket-16 weight shapes used 1/1 in round 1, 3/4 in round 2, and 3/3 in round 3. All source, binary, prompt, and sampled B=1 output hashes match across these rounds. Step-time differences account for the observed variation; the drain is only about 20–29 ms. Board power stays near 145 W, and the faster second round has lower SM clocks. The dispatch difference is an observed plausible contributor, not a controlled causal attribution. See fixed_variation_audit.json and fixed_variation_telemetry.json.

| Input tokens | Format | Actual pool pages ×16 tokens | Verified resident B | Adjacent rejected B | Evidence at adjacent B |
|---:|---|---:|---:|---:|---|
| 2048 | bf16 | 2015 | 13 | 14 | One preemption; eventual output completion |
| 2048 | mxfp6w_wkv | 15308 | 106 | 107 | One preemption; eventual output completion |
| 2048 | ssnf_wkv | 19309 | 134 | 135 | One preemption; eventual output completion |
| 16384 | bf16 | 2015 | 1 | 2 | Insufficient pages to admit all prefills |
| 16384 | mxfp6w_wkv | 15308 | 14 | 15 | Insufficient pages to admit all prefills |
| 16384 | ssnf_wkv | 19308 | 18 | 19 | Insufficient pages to admit all prefills |

Each capacity cell first passed a separate high-batch full-length pilot. Final probes used the current engine allocation and B0−1/B0/B0+1, with B0=floor(pool pages / ceil((input+256)/16)). The successful boundary completed every request, had no preemption, and retained all B requests at the final decode length. The adjacent 2K candidates eventually completed after one preemption, which rejects simultaneous full-length residency. The adjacent 16K candidates retained waiting requests while free pages were below the full-prompt requirement for three empty scheduler steps; those trials were aborted and are explicitly incomplete resource-failure observations. There were no inconclusive timeouts or arbitrary scheduler errors. One final trial per boundary is a capacity check, not a statistical estimate of deployment reliability.

SSNF allocation was 19,309 pages at 2K and 19,308 at 16K; all fixed rounds preserve this observed context difference. These allocations differ slightly from the pools of other experiments, so each run uses its own allocation. No unverified cause is assigned.

The initial BF16 smoke with incorrect external/internal request-ID matching remains preserved and excluded. The corrected smoke and all subsequent pilots passed. A CPU controller-path typo occurred before any child GPU launch and is also recorded. No core, energy, accuracy, or installed vLLM source was modified.

Reproducibility records: fixed_completion.json, capacity_completion.json, campaign_spec.json, fixed_frozen.json, campaign_commands.jsonl, per-run source snapshots and manifests, scheduler/request/workload/bounds records, and ../analysis/fixed_capacity_audit.json. The parent hardware monitor spans all main runs. Kernel and SLO experiments may continue while these CPU records are finalized.

All three measured engine-round values are retained below. Each entry is synchronized output tokens/s; the adjacent SSNF row gives the paired SSNF/BF16 ratios for the same rounds.

| Input tokens | Batch | Format | Round 1 | Round 2 | Round 3 |
|---:|---:|---|---:|---:|---:|
| 2048 | 1 | bf16 | 37.6383 | 37.5935 | 37.5577 |
| 2048 | 1 | mxfp6w_wkv | 39.4068 | 39.4167 | 39.2815 |
| 2048 | 1 | ssnf_wkv | 44.5482 | 49.2024 | 48.8644 |
| 2048 | 1 | SSNF/BF16 | 1.183587 | 1.308799 | 1.301047 |
| 2048 | 4 | bf16 | 137.2243 | 137.2105 | 137.0732 |
| 2048 | 4 | mxfp6w_wkv | 138.4681 | 138.4861 | 138.1375 |
| 2048 | 4 | ssnf_wkv | 155.5429 | 170.0796 | 169.3219 |
| 2048 | 4 | SSNF/BF16 | 1.133493 | 1.239552 | 1.235266 |
| 2048 | 8 | bf16 | 255.8471 | 255.8854 | 255.7983 |
| 2048 | 8 | mxfp6w_wkv | 242.8607 | 242.1759 | 242.5322 |
| 2048 | 8 | ssnf_wkv | 289.0846 | 314.2237 | 312.7676 |
| 2048 | 8 | SSNF/BF16 | 1.129911 | 1.227986 | 1.222712 |
| 16384 | 1 | bf16 | 33.6757 | 33.6276 | 33.6630 |
| 16384 | 1 | mxfp6w_wkv | 31.9224 | 31.9044 | 31.9048 |
| 16384 | 1 | ssnf_wkv | 40.4355 | 40.5935 | 40.4453 |
| 16384 | 1 | SSNF/BF16 | 1.200733 | 1.207148 | 1.201474 |
