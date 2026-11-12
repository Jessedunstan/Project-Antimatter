# Project Antimatter — Autonomous Bug-Repair Benchmark Report

## 1. Aggregate Model Scoreboard

| Model | Tasks Passed | Pass Rate | Avg Steps | Total Tool Calls | Tool Call Errors | Avg Wall Time (s) | Total Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `qwen2.5-coder:7b` | **10 / 10** | **100.0%** | 4.70 | 50 | 0 | 3.98s | 39.8s |
| `llama3.2:1b` | **0 / 10** | **0.0%** | 4.10 | 49 | 31 | 5.24s | 52.4s |

## 2. Per-Task Breakdown

| Task ID | Bug Description | Model | Result | Steps | Tool Calls | Tool Errors | Time (s) | Tools Used |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `01_pagination_offset` | Off-by-one pagination slice calculation | `qwen2.5-coder:7b` | **PASS** | 4 | 4 | 0 | 8.15s | `read_file, read_file, edit_file, run_command` |
| `01_pagination_offset` | Off-by-one pagination slice calculation | `llama3.2:1b` | **FAIL** | 6 | 10 | 9 | 7.63s | `read_file, edit_file, run_command, list_directory, read_file, edit_file, run_command, edit_file, run_command, run_command` |
| `02_discount_clamp` | Inverted min/max clamp on discount price | `qwen2.5-coder:7b` | **PASS** | 4 | 4 | 0 | 2.69s | `read_file, read_file, edit_file, run_command` |
| `02_discount_clamp` | Inverted min/max clamp on discount price | `llama3.2:1b` | **FAIL** | 3 | 2 | 1 | 2.68s | `read_file, edit_file` |
| `03_lru_cache_eviction` | LRU cache evicts most-recently-used instead of least-recently-used | `qwen2.5-coder:7b` | **PASS** | 4 | 3 | 0 | 2.23s | `read_file, edit_file, run_command` |
| `03_lru_cache_eviction` | LRU cache evicts most-recently-used instead of least-recently-used | `llama3.2:1b` | **FAIL** | 5 | 6 | 4 | 8.72s | `read_file, write_file, edit_file, run_command, read_file, run_command` |
| `04_slugify_normalization` | Slug generator forgets lowercase conversion and hyphen collapsing | `qwen2.5-coder:7b` | **PASS** | 6 | 7 | 0 | 3.63s | `read_file, read_file, edit_file, run_command, run_command, run_command, run_command` |
| `04_slugify_normalization` | Slug generator forgets lowercase conversion and hyphen collapsing | `llama3.2:1b` | **FAIL** | 4 | 2 | 2 | 2.51s | `read_file, read_file` |
| `05_safe_mean_empty` | Weighted mean crashes on zero total weight and divides by wrong denominator | `qwen2.5-coder:7b` | **PASS** | 4 | 4 | 0 | 2.70s | `read_file, read_file, edit_file, run_command` |
| `05_safe_mean_empty` | Weighted mean crashes on zero total weight and divides by wrong denominator | `llama3.2:1b` | **FAIL** | 4 | 3 | 0 | 3.46s | `read_file, edit_file, write_file` |
| `06_moving_average_window` | Moving average slices wrong trailing window length | `qwen2.5-coder:7b` | **PASS** | 5 | 5 | 0 | 5.52s | `read_file, read_file, edit_file, run_command, write_file` |
| `06_moving_average_window` | Moving average slices wrong trailing window length | `llama3.2:1b` | **FAIL** | 3 | 4 | 0 | 6.36s | `read_file, edit_file, write_file, run_command` |
| `07_deep_merge_override` | Config merge keeps base value instead of override value | `qwen2.5-coder:7b` | **PASS** | 4 | 4 | 0 | 2.83s | `read_file, read_file, edit_file, run_command` |
| `07_deep_merge_override` | Config merge keeps base value instead of override value | `llama3.2:1b` | **FAIL** | 4 | 12 | 10 | 8.25s | `read_file, write_file, edit_file, run_command, read_file, write_file, edit_file, run_command, edit_file, write_file, run_command, run_command` |
| `08_token_bucket_cap` | Token bucket refill exceeds maximum capacity | `qwen2.5-coder:7b` | **PASS** | 6 | 7 | 0 | 4.09s | `read_file, read_file, edit_file, run_command, run_command, run_command, run_command` |
| `08_token_bucket_cap` | Token bucket refill exceeds maximum capacity | `llama3.2:1b` | **FAIL** | 4 | 3 | 3 | 2.95s | `edit_file, write_file, edit_file` |
| `09_env_parser_equals` | Key-value parser breaks when value contains '=' character | `qwen2.5-coder:7b` | **PASS** | 4 | 4 | 0 | 2.85s | `read_file, read_file, edit_file, run_command` |
| `09_env_parser_equals` | Key-value parser breaks when value contains '=' character | `llama3.2:1b` | **FAIL** | 4 | 4 | 2 | 6.73s | `edit_file, run_command, edit_file, run_command` |
| `10_retry_backoff_exponent` | Exponential backoff multiplies instead of exponentiating base factor | `qwen2.5-coder:7b` | **PASS** | 6 | 8 | 0 | 5.09s | `read_file, read_file, edit_file, run_command, run_command, list_directory, run_command, list_directory` |
| `10_retry_backoff_exponent` | Exponential backoff multiplies instead of exponentiating base factor | `llama3.2:1b` | **FAIL** | 4 | 3 | 0 | 3.09s | `read_file, edit_file, run_command` |
