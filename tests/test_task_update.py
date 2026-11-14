"""任务编辑的并发丢失更新修复：HTTP 层语义与真实并发测试。

运行: python -m unittest discover tests -v
"""
import os
import tempfile
import threading
import time
import unittest

# 必须在导入 app 之前指定独立临时数据库，避免污染开发库
_fd, _DB = tempfile.mkstemp(prefix="test_task_update_", suffix=".db")
os.close(_fd)
os.environ["DB_PATH"] = _DB

import httpx  # noqa: E402
import uvicorn  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.db import get_conn, init_db  # noqa: E402
from app.main import app  # noqa: E402

PORT = 18031
BASE = f"http://127.0.0.1:{PORT}"


def _clear_tables():
    with get_conn() as conn:
        for t in ("run_records", "dependencies", "plans", "tasks"):
            conn.execute(f"DELETE FROM {t}")


class TaskUpdateSemanticsTest(unittest.TestCase):
    """编辑语义：安全合并 / 按实际变化递增版本 / 可重试冲突。"""

    @classmethod
    def setUpClass(cls):
        init_db()
        cls.client = TestClient(app)

    def setUp(self):
        _clear_tables()

    def mk_task(self, name, gran="daily", desc=""):
        r = self.client.post("/api/tasks", json={
            "name": name, "granularity": gran, "description": desc})
        assert r.status_code == 201, r.text
        return r.json()

    def get_task(self, tid):
        st = self.client.get("/api/state").json()
        return next(t for t in st["tasks"] if t["id"] == tid)

    def add_dep(self, down, up, off=0):
        r = self.client.post("/api/dependencies", json={
            "downstream_id": down, "upstream_id": up, "offset_days": off})
        assert r.status_code == 201, r.text
        return r.json()

    def test_desc_only_edit_preserves_committed_config(self):
        """一人新增依赖（版本 1->2），另一人基于 v1 只改说明：
        安全合并——说明生效，已提交的粒度与版本原样保留（不写回旧值）。"""
        t = self.mk_task("t_desc", "daily", "旧说明")
        up = self.mk_task("t_desc_up")
        self.add_dep(t["id"], up["id"])
        self.assertEqual(self.get_task(t["id"])["config_version"], 2)

        r = self.client.put(f"/api/tasks/{t['id']}", json={
            "description": "新说明", "base_version": 1})
        self.assertEqual(r.status_code, 200, r.text)
        got = r.json()
        self.assertEqual(got["description"], "新说明")
        self.assertEqual(got["granularity"], "daily")
        self.assertEqual(got["config_version"], 2)  # 未被写回旧值 1

    def test_granularity_switch_bumps_on_actual_change_only(self):
        """粒度实际变化才递增版本；提交相同粒度不递增。"""
        t = self.mk_task("t_gran", "daily")
        r = self.client.put(f"/api/tasks/{t['id']}", json={
            "granularity": "hourly", "base_version": 1})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["granularity"], "hourly")
        self.assertEqual(r.json()["config_version"], 2)

        r = self.client.put(f"/api/tasks/{t['id']}", json={
            "granularity": "hourly", "base_version": 2})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["config_version"], 2)  # 无实际变化，不递增

    def test_stale_basis_granularity_change_returns_retryable_conflict(self):
        """编辑依据的配置已被改动且本次要改粒度：409 + 当前状态，
        不覆盖已提交修改、不留部分状态；依据最新版本重试后成功。"""
        t = self.mk_task("t_conflict", "daily", "原说明")
        up = self.mk_task("t_conflict_up")
        self.add_dep(t["id"], up["id"])  # 他人先把版本提到 2

        r = self.client.put(f"/api/tasks/{t['id']}", json={
            "granularity": "hourly", "description": "被冲突的说明",
            "base_version": 1})
        self.assertEqual(r.status_code, 409, r.text)
        detail = r.json()["detail"]
        self.assertIn("已被他人修改", detail["reason"])
        self.assertEqual(detail["task"]["config_version"], 2)

        got = self.get_task(t["id"])  # 无部分状态：粒度/版本/说明均未变
        self.assertEqual(got["granularity"], "daily")
        self.assertEqual(got["config_version"], 2)
        self.assertEqual(got["description"], "原说明")

        r = self.client.put(f"/api/tasks/{t['id']}", json={  # 刷新后重试
            "granularity": "hourly", "description": "被冲突的说明",
            "base_version": 2})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["config_version"], 3)
        self.assertEqual(r.json()["granularity"], "hourly")
        self.assertEqual(r.json()["description"], "被冲突的说明")

    def test_stale_success_record_not_misjudged_ready(self):
        """任务场景复现：新增依赖与说明编辑交错后，旧版成功分区
        在回填计划中必须判为「版本过期」而不是「已就绪」。"""
        t = self.mk_task("t_plan", "daily")
        up = self.mk_task("t_plan_up")
        self.add_dep(t["id"], up["id"])  # 版本 1 -> 2
        r = self.client.post("/api/records", json={  # 旧版本的成功记录
            "task_id": t["id"], "partition": "2026-11-13",
            "status": "success", "config_version": 1})
        assert r.status_code == 201, r.text

        r = self.client.put(f"/api/tasks/{t['id']}", json={  # 并发的说明编辑
            "description": "值班备注", "base_version": 1})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["config_version"], 2)

        r = self.client.post("/api/plans", json={
            "target_task_id": t["id"], "biz_date": "2026-11-13"})
        self.assertEqual(r.status_code, 201, r.text)
        item = next(i for i in r.json()["items"] if i["task_id"] == t["id"])
        self.assertEqual(item["status"], "stale")       # 不是 ok
        self.assertEqual(item["action"], "backfill")    # 不是 skip

    def test_legacy_request_without_base_version_still_safe(self):
        """不带 base_version 的旧客户端：说明编辑同样不得回退版本。"""
        t = self.mk_task("t_legacy", "daily")
        up = self.mk_task("t_legacy_up")
        self.add_dep(t["id"], up["id"])  # v2
        r = self.client.put(f"/api/tasks/{t['id']}", json={"description": "x"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["config_version"], 2)


class ConcurrencyTest(unittest.TestCase):
    """真实并发：uvicorn 子线程 + 多线程客户端，验证无丢失更新。"""

    @classmethod
    def setUpClass(cls):
        init_db()
        config = uvicorn.Config(app, host="127.0.0.1", port=PORT,
                                log_level="error")
        cls.server = uvicorn.Server(config)
        cls.thread = threading.Thread(target=cls.server.run, daemon=True)
        cls.thread.start()
        for _ in range(200):  # 等待服务就绪
            try:
                httpx.get(f"{BASE}/api/state", timeout=0.5)
                break
            except Exception:
                time.sleep(0.05)
        else:
            raise RuntimeError("测试服务未能启动")

    @classmethod
    def tearDownClass(cls):
        cls.server.should_exit = True
        cls.thread.join(timeout=10)

    def setUp(self):
        _clear_tables()
        self.http = httpx.Client(base_url=BASE, timeout=10)

    def tearDown(self):
        self.http.close()

    def mk_task(self, name, gran="daily"):
        r = self.http.post("/api/tasks", json={"name": name, "granularity": gran})
        assert r.status_code == 201, r.text
        return r.json()

    def get_task(self, tid):
        st = self.http.get("/api/state").json()
        return next(t for t in st["tasks"] if t["id"] == tid)

    def _run_concurrently(self, funcs):
        barrier = threading.Barrier(len(funcs))
        results = [None] * len(funcs)

        def wrap(i, fn):
            barrier.wait()  # 同时放行，最大化交错
            results[i] = fn()

        threads = [threading.Thread(target=wrap, args=(i, f))
                   for i, f in enumerate(funcs)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        return results

    def test_desc_edits_vs_dep_adds_no_lost_update(self):
        """N 个说明编辑（基于 v1）与 M 个依赖新增并发：
        最终版本必须恰好 1+M（说明编辑不得覆盖已提交的版本递增）。"""
        M = 6   # 依赖新增次数
        N = 8   # 说明编辑次数
        down = self.mk_task("c_down")
        ups = [self.mk_task(f"c_up{i}") for i in range(M)]

        def edit_desc(i):
            return self.http.put(f"/api/tasks/{down['id']}", json={
                "description": f"说明{i}", "base_version": 1})

        def add_dep(i):
            return self.http.post("/api/dependencies", json={
                "downstream_id": down["id"], "upstream_id": ups[i]["id"],
                "offset_days": 0})

        funcs = []
        for i in range(max(M, N)):
            if i < N:
                funcs.append(lambda i=i: edit_desc(i))
            if i < M:
                funcs.append(lambda i=i: add_dep(i))
        results = self._run_concurrently(funcs)

        for r in results:
            self.assertIn(r.status_code, (200, 201), r.text)
        final = self.get_task(down["id"])
        # 关键断言：每次依赖新增恰好 +1，没有被说明编辑写回旧值
        self.assertEqual(final["config_version"], 1 + M)
        self.assertIn(final["description"], {f"说明{i}" for i in range(N)})
        deps = self.http.get("/api/state").json()["dependencies"]
        self.assertEqual(
            len([d for d in deps if d["downstream_id"] == down["id"]]), M)

    def test_concurrent_granularity_switches_no_double_bump(self):
        """两个基于 v1 的粒度切换并发：串行后先到者切换并 +1，
        后到者发现目标粒度已达成 → 安全合并（不重复递增、不覆盖）；
        最终版本恰好 2，状态完整一致。"""
        t = self.mk_task("c_gran", "daily")

        def switch():
            return self.http.put(f"/api/tasks/{t['id']}", json={
                "granularity": "hourly", "base_version": 1})

        r1, r2 = self._run_concurrently([switch, switch])
        self.assertEqual(r1.status_code, 200, r1.text)
        self.assertEqual(r2.status_code, 200, r2.text)

        final = self.get_task(t["id"])
        self.assertEqual(final["granularity"], "hourly")
        self.assertEqual(final["config_version"], 2)  # 恰好递增一次

    def test_granularity_switch_vs_dep_add_serialized(self):
        """粒度切换（基于 v1）与依赖新增并发：两种串行顺序都合法——
        切换先：成功且最终 v3(hourly)；依赖先：切换 409（依据过期），
        状态无部分修改，按最新版本重试后成功。最终版本与历史一致。"""
        t = self.mk_task("c_mix", "daily")
        up = self.mk_task("c_mix_up")

        def switch():
            return self.http.put(f"/api/tasks/{t['id']}", json={
                "granularity": "hourly", "base_version": 1})

        def add_dep():
            return self.http.post("/api/dependencies", json={
                "downstream_id": t["id"], "upstream_id": up["id"],
                "offset_days": 0})

        r_switch, r_dep = self._run_concurrently([switch, add_dep])
        self.assertEqual(r_dep.status_code, 201, r_dep.text)  # 依赖新增必成功
        final = self.get_task(t["id"])

        if r_switch.status_code == 409:
            # 依赖先提交（v2, daily）：切换因依据过期被拒绝，无部分状态
            self.assertEqual(final["granularity"], "daily")
            self.assertEqual(final["config_version"], 2)
            cur = r_switch.json()["detail"]["task"]["config_version"]
            r = self.http.put(f"/api/tasks/{t['id']}", json={  # 重试
                "granularity": "hourly", "base_version": cur})
            self.assertEqual(r.status_code, 200, r.text)
            final = self.get_task(t["id"])
            self.assertEqual(final["granularity"], "hourly")
            self.assertEqual(final["config_version"], 3)
        else:
            # 切换先提交（v2, hourly），依赖随后 +1
            self.assertEqual(r_switch.status_code, 200, r_switch.text)
            self.assertEqual(final["granularity"], "hourly")
            self.assertEqual(final["config_version"], 3)


if __name__ == "__main__":
    unittest.main()
