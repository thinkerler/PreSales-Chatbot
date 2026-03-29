from __future__ import annotations

import asyncio
import statistics
import time
from typing import List

import httpx


URL = "http://127.0.0.1:8000/ask"
PAYLOAD = {"query": "618可以叠加店铺券吗？", "user_profile": {"budget": "3000", "need": "活动优惠"}}


async def one_call(client: httpx.AsyncClient, idx: int) -> float:
    start = time.perf_counter()
    trace_id = f"load-{idx}"
    resp = await client.post(URL, json=PAYLOAD, headers={"x-trace-id": trace_id})
    resp.raise_for_status()
    return (time.perf_counter() - start) * 1000


async def run(concurrency: int = 20, total: int = 100) -> None:
    async with httpx.AsyncClient(timeout=30) as client:
        tasks: List[asyncio.Task] = []
        for i in range(total):
            tasks.append(asyncio.create_task(one_call(client, i)))
            if len(tasks) >= concurrency:
                await asyncio.gather(*tasks)
                tasks = []
        if tasks:
            await asyncio.gather(*tasks)


async def run_and_report(concurrency: int = 20, total: int = 100) -> None:
    async with httpx.AsyncClient(timeout=30) as client:
        latencies = await asyncio.gather(*[one_call(client, i) for i in range(total)])
    p95 = sorted(latencies)[int(total * 0.95) - 1]
    print(
        {
            "count": total,
            "avg_ms": round(statistics.mean(latencies), 2),
            "p95_ms": round(p95, 2),
            "max_ms": round(max(latencies), 2),
        }
    )


if __name__ == "__main__":
    asyncio.run(run_and_report())
