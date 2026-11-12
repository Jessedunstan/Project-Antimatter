"""Seeded bug-repair tasks for benchmarking Project Antimatter's autonomous ReAct loop.

Each task contains:
- A realistic Python source file with a genuine logic bug.
- A pytest test file that fails on the buggy code and passes only when the bug is fixed.
- A task prompt instructing the autonomous agent to inspect, repair, and verify the code.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class BugRepairTask:
    """Specification of a single autonomous bug-repair benchmark task."""
    task_id: str
    title: str
    source_filename: str
    buggy_code: str
    test_filename: str
    test_code: str
    prompt: str


def _make_prompt(source_filename: str, test_filename: str, description: str) -> str:
    return (
        f"There is a bug in `{source_filename}` ({description}).\n"
        f"1. Read `{source_filename}` and `{test_filename}` using `read_file`.\n"
        f"2. Fix the bug in `{source_filename}` using `edit_file` or `write_file` (do NOT modify `{test_filename}`).\n"
        f"3. Once fixed, finish your response."
    )


def get_benchmark_tasks() -> List[BugRepairTask]:
    """Return the 10 seeded bug-repair benchmark tasks."""
    return [
        BugRepairTask(
            task_id="01_pagination_offset",
            title="Off-by-one pagination slice calculation",
            source_filename="paginator.py",
            buggy_code=(
                "from typing import List, TypeVar\n\n"
                "T = TypeVar('T')\n\n"
                "def paginate(items: List[T], page: int, page_size: int) -> List[T]:\n"
                "    \"\"\"Return 1-indexed page of items. Raise ValueError if page < 1 or page_size < 1.\"\"\"\n"
                "    if page < 1 or page_size < 1:\n"
                "        raise ValueError('page and page_size must be >= 1')\n"
                "    # BUG: uses page * page_size instead of (page - 1) * page_size\n"
                "    start = page * page_size\n"
                "    end = start + page_size\n"
                "    return items[start:end]\n"
            ),
            test_filename="test_paginator.py",
            test_code=(
                "import pytest\n"
                "from paginator import paginate\n\n"
                "def test_paginate_first_and_second_page():\n"
                "    data = [10, 20, 30, 40, 50]\n"
                "    assert paginate(data, page=1, page_size=2) == [10, 20]\n"
                "    assert paginate(data, page=2, page_size=2) == [30, 40]\n"
                "    assert paginate(data, page=3, page_size=2) == [50]\n\n"
                "def test_paginate_invalid_args():\n"
                "    with pytest.raises(ValueError):\n"
                "        paginate([1, 2], page=0, page_size=2)\n"
            ),
            prompt=_make_prompt(
                "paginator.py",
                "test_paginator.py",
                "1-based page slicing skips the first page of items",
            ),
        ),
        BugRepairTask(
            task_id="02_discount_clamp",
            title="Inverted min/max clamp on discount price",
            source_filename="pricing.py",
            buggy_code=(
                "def apply_discount(price: float, percent: float, min_price: float = 0.0) -> float:\n"
                "    \"\"\"Apply percentage discount (0..100) and ensure result is never below min_price.\"\"\"\n"
                "    if not (0.0 <= percent <= 100.0):\n"
                "        raise ValueError('percent must be between 0 and 100')\n"
                "    discounted = round(price * (1.0 - percent / 100.0), 2)\n"
                "    # BUG: min() used instead of max() when enforcing floor min_price\n"
                "    return min(discounted, min_price)\n"
            ),
            test_filename="test_pricing.py",
            test_code=(
                "import pytest\n"
                "from pricing import apply_discount\n\n"
                "def test_apply_discount_normal_and_floor():\n"
                "    assert apply_discount(100.0, 20.0, min_price=10.0) == 80.0\n"
                "    assert apply_discount(50.0, 90.0, min_price=15.0) == 15.0\n"
                "    assert apply_discount(40.0, 0.0, min_price=5.0) == 40.0\n"
            ),
            prompt=_make_prompt(
                "pricing.py",
                "test_pricing.py",
                "apply_discount returns min_price instead of enforcing min_price as a lower bound",
            ),
        ),
        BugRepairTask(
            task_id="03_lru_cache_eviction",
            title="LRU cache evicts most-recently-used instead of least-recently-used",
            source_filename="lru.py",
            buggy_code=(
                "from collections import OrderedDict\n"
                "from typing import Any, Optional\n\n"
                "class SimpleLRU:\n"
                "    def __init__(self, capacity: int = 3):\n"
                "        self.capacity = capacity\n"
                "        self.store: OrderedDict[str, Any] = OrderedDict()\n\n"
                "    def get(self, key: str) -> Optional[Any]:\n"
                "        if key not in self.store:\n"
                "            return None\n"
                "        self.store.move_to_end(key)\n"
                "        return self.store[key]\n\n"
                "    def put(self, key: str, value: Any) -> None:\n"
                "        if key in self.store:\n"
                "            self.store.move_to_end(key)\n"
                "        self.store[key] = value\n"
                "        if len(self.store) > self.capacity:\n"
                "            # BUG: last=True pops most recently used instead of oldest (last=False)\n"
                "            self.store.popitem(last=True)\n"
            ),
            test_filename="test_lru.py",
            test_code=(
                "from lru import SimpleLRU\n\n"
                "def test_lru_evicts_oldest_entry():\n"
                "    cache = SimpleLRU(capacity=2)\n"
                "    cache.put('a', 1)\n"
                "    cache.put('b', 2)\n"
                "    assert cache.get('a') == 1  # 'a' is now freshest, 'b' is oldest\n"
                "    cache.put('c', 3)           # should evict 'b'\n"
                "    assert cache.get('b') is None\n"
                "    assert cache.get('a') == 1\n"
                "    assert cache.get('c') == 3\n"
            ),
            prompt=_make_prompt(
                "lru.py",
                "test_lru.py",
                "SimpleLRU.put evicts the newest item instead of the oldest item when capacity is exceeded",
            ),
        ),
        BugRepairTask(
            task_id="04_slugify_normalization",
            title="Slug generator forgets lowercase conversion and hyphen collapsing",
            source_filename="slugify.py",
            buggy_code=(
                "import re\n\n"
                "def make_slug(title: str) -> str:\n"
                "    \"\"\"Convert title to lowercase URL slug with single hyphens and no outer hyphens.\"\"\"\n"
                "    cleaned = re.sub(r'[^a-zA-Z0-9]+', '-', title)\n"
                "    # BUG: missing .lower() and .strip('-')\n"
                "    return cleaned\n"
            ),
            test_filename="test_slugify.py",
            test_code=(
                "from slugify import make_slug\n\n"
                "def test_make_slug():\n"
                "    assert make_slug('  Hello, World!  ') == 'hello-world'\n"
                "    assert make_slug('Project---Antimatter 2.0') == 'project-antimatter-2-0'\n"
            ),
            prompt=_make_prompt(
                "slugify.py",
                "test_slugify.py",
                "make_slug does not lowercase the string or strip leading/trailing hyphens",
            ),
        ),
        BugRepairTask(
            task_id="05_safe_mean_empty",
            title="Weighted mean crashes on zero total weight and divides by wrong denominator",
            source_filename="stats.py",
            buggy_code=(
                "from typing import Sequence\n\n"
                "def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:\n"
                "    \"\"\"Return weighted average, or 0.0 if lists are empty or total weight is 0.\"\"\"\n"
                "    if len(values) != len(weights):\n"
                "        raise ValueError('Length mismatch')\n"
                "    if not values:\n"
                "        return 0.0\n"
                "    total_weight = sum(weights)\n"
                "    if total_weight == 0:\n"
                "        return 0.0\n"
                "    weighted_sum = sum(v * w for v, w in zip(values, weights))\n"
                "    # BUG: divides by len(values) instead of total_weight\n"
                "    return weighted_sum / len(values)\n"
            ),
            test_filename="test_stats.py",
            test_code=(
                "import pytest\n"
                "from stats import weighted_mean\n\n"
                "def test_weighted_mean_calculation():\n"
                "    assert weighted_mean([10.0, 20.0], [1.0, 3.0]) == 17.5\n"
                "    assert weighted_mean([], []) == 0.0\n"
                "    assert weighted_mean([5.0, 10.0], [0.0, 0.0]) == 0.0\n"
            ),
            prompt=_make_prompt(
                "stats.py",
                "test_stats.py",
                "weighted_mean divides by len(values) instead of total_weight",
            ),
        ),
        BugRepairTask(
            task_id="06_moving_average_window",
            title="Moving average slices wrong trailing window length",
            source_filename="moving_avg.py",
            buggy_code=(
                "from typing import List\n\n"
                "def moving_average(series: List[float], window: int) -> List[float]:\n"
                "    \"\"\"Compute simple moving average for each position where a full window is available.\"\"\"\n"
                "    if window <= 0:\n"
                "        raise ValueError('window must be positive')\n"
                "    if len(series) < window:\n"
                "        return []\n"
                "    result: List[float] = []\n"
                "    for i in range(window - 1, len(series)):\n"
                "        # BUG: slices series[i - window : i] instead of series[i - window + 1 : i + 1]\n"
                "        chunk = series[i - window : i]\n"
                "        result.append(sum(chunk) / window)\n"
                "    return result\n"
            ),
            test_filename="test_moving_avg.py",
            test_code=(
                "from moving_avg import moving_average\n\n"
                "def test_moving_average_values():\n"
                "    assert moving_average([2.0, 4.0, 6.0, 8.0], window=2) == [3.0, 5.0, 7.0]\n"
                "    assert moving_average([1.0, 2.0, 3.0], window=3) == [2.0]\n"
                "    assert moving_average([1.0], window=3) == []\n"
            ),
            prompt=_make_prompt(
                "moving_avg.py",
                "test_moving_avg.py",
                "moving_average slices `series[i - window : i]` instead of `series[i - window + 1 : i + 1]`",
            ),
        ),
        BugRepairTask(
            task_id="07_deep_merge_override",
            title="Config merge keeps base value instead of override value",
            source_filename="config_merger.py",
            buggy_code=(
                "from typing import Any, Dict\n\n"
                "def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:\n"
                "    \"\"\"Recursively merge override into base without mutating inputs. Override wins on conflict.\"\"\"\n"
                "    merged: Dict[str, Any] = dict(base)\n"
                "    for key, val in override.items():\n"
                "        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):\n"
                "            merged[key] = merge_configs(merged[key], val)\n"
                "        else:\n"
                "            # BUG: keeps merged.get(key, val) instead of assigning val\n"
                "            merged[key] = merged.get(key, val)\n"
                "    return merged\n"
            ),
            test_filename="test_config_merger.py",
            test_code=(
                "from config_merger import merge_configs\n\n"
                "def test_merge_configs_nested_and_scalar():\n"
                "    base = {'host': 'localhost', 'db': {'port': 5432, 'pool': 5}}\n"
                "    over = {'host': '0.0.0.0', 'db': {'port': 6432}, 'debug': True}\n"
                "    res = merge_configs(base, over)\n"
                "    assert res == {'host': '0.0.0.0', 'db': {'port': 6432, 'pool': 5}, 'debug': True}\n"
                "    assert base['host'] == 'localhost'\n"
            ),
            prompt=_make_prompt(
                "config_merger.py",
                "test_config_merger.py",
                "merge_configs keeps the old base scalar value instead of overwriting with `val` from override",
            ),
        ),
        BugRepairTask(
            task_id="08_token_bucket_cap",
            title="Token bucket refill exceeds maximum capacity",
            source_filename="rate_limiter.py",
            buggy_code=(
                "class TokenBucket:\n"
                "    def __init__(self, capacity: float, refill_rate: float):\n"
                "        self.capacity = float(capacity)\n"
                "        self.refill_rate = float(refill_rate)\n"
                "        self.tokens = float(capacity)\n\n"
                "    def refill(self, elapsed_seconds: float) -> float:\n"
                "        if elapsed_seconds > 0:\n"
                "            # BUG: does not cap self.tokens at self.capacity\n"
                "            self.tokens = self.tokens + elapsed_seconds * self.refill_rate\n"
                "        return self.tokens\n\n"
                "    def consume(self, amount: float = 1.0) -> bool:\n"
                "        if self.tokens >= amount:\n"
                "            self.tokens -= amount\n"
                "            return True\n"
                "        return False\n"
            ),
            test_filename="test_rate_limiter.py",
            test_code=(
                "from rate_limiter import TokenBucket\n\n"
                "def test_token_bucket_refill_and_consume():\n"
                "    bucket = TokenBucket(capacity=10.0, refill_rate=2.0)\n"
                "    assert bucket.consume(6.0) is True\n"
                "    assert bucket.tokens == 4.0\n"
                "    assert bucket.refill(10.0) == 10.0  # capped at capacity 10.0\n"
                "    assert bucket.consume(11.0) is False\n"
            ),
            prompt=_make_prompt(
                "rate_limiter.py",
                "test_rate_limiter.py",
                "TokenBucket.refill allows self.tokens to exceed self.capacity",
            ),
        ),
        BugRepairTask(
            task_id="09_env_parser_equals",
            title="Key-value parser breaks when value contains '=' character",
            source_filename="env_parser.py",
            buggy_code=(
                "from typing import Dict\n\n"
                "def parse_env_lines(raw_text: str) -> Dict[str, str]:\n"
                "    \"\"\"Parse KEY=VALUE lines, ignoring comments (#) and blank lines. Values may contain '='.\"\"\"\n"
                "    result: Dict[str, str] = {}\n"
                "    for raw_line in raw_text.splitlines():\n"
                "        line = raw_line.strip()\n"
                "        if not line or line.startswith('#'):\n"
                "            continue\n"
                "        if '=' not in line:\n"
                "            continue\n"
                "        # BUG: line.split('=') without maxsplit=1 fails when value has '='\n"
                "        key, value = line.split('=')\n"
                "        result[key.strip()] = value.strip()\n"
                "    return result\n"
            ),
            test_filename="test_env_parser.py",
            test_code=(
                "from env_parser import parse_env_lines\n\n"
                "def test_parse_env_with_equals_in_value():\n"
                "    sample = '''\n"
                "    # Database settings\n"
                "    HOST = 127.0.0.1\n"
                "    DSN = postgres://user:pass@localhost/db?sslmode=require&pool=5\n"
                "    '''\n"
                "    parsed = parse_env_lines(sample)\n"
                "    assert parsed['HOST'] == '127.0.0.1'\n"
                "    assert parsed['DSN'] == 'postgres://user:pass@localhost/db?sslmode=require&pool=5'\n"
            ),
            prompt=_make_prompt(
                "env_parser.py",
                "test_env_parser.py",
                "parse_env_lines uses `line.split('=')` instead of `line.split('=', 1)`",
            ),
        ),
        BugRepairTask(
            task_id="10_retry_backoff_exponent",
            title="Exponential backoff multiplies instead of exponentiating base factor",
            source_filename="backoff.py",
            buggy_code=(
                "def compute_backoff_delay(attempt: int, base_delay: float = 0.5, max_delay: float = 30.0) -> float:\n"
                "    \"\"\"Compute exponential backoff `base_delay * (2 ** attempt)` capped at `max_delay` for 0-based attempt.\"\"\"\n"
                "    if attempt < 0:\n"
                "        raise ValueError('attempt must be >= 0')\n"
                "    # BUG: uses (2 * attempt) instead of (2 ** attempt)\n"
                "    delay = base_delay * (2 * attempt)\n"
                "    return min(delay, max_delay)\n"
            ),
            test_filename="test_backoff.py",
            test_code=(
                "from backoff import compute_backoff_delay\n\n"
                "def test_exponential_backoff_progression():\n"
                "    assert compute_backoff_delay(0, base_delay=1.0, max_delay=10.0) == 1.0\n"
                "    assert compute_backoff_delay(1, base_delay=1.0, max_delay=10.0) == 2.0\n"
                "    assert compute_backoff_delay(3, base_delay=1.0, max_delay=10.0) == 8.0\n"
                "    assert compute_backoff_delay(5, base_delay=1.0, max_delay=10.0) == 10.0\n"
            ),
            prompt=_make_prompt(
                "backoff.py",
                "test_backoff.py",
                "compute_backoff_delay computes `2 * attempt` instead of `2 ** attempt`",
            ),
        ),
    ]
