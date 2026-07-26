"""Unit-test the async spec-decode placeholder repair without torch or a GPU.

The repair is pure list manipulation, so its full contract is testable here:
replace the trailing run of -1 placeholders with the real sampled ids, shrinking
when the optimistic count overshot the acceptance count, growing never, and
never touching a request that has no placeholders pending (which must also not
force a GPU sync).
"""
import json
import sys
import types

PAYLOAD = sys.argv[1]
sys.path.insert(0, PAYLOAD)
import lfm25_patches  # noqa: E402

results = {}
logs = []


# --- stand-ins ------------------------------------------------------------
class FakeEvent:
    def __init__(self):
        self.syncs = 0

    def synchronize(self):
        self.syncs += 1


class FakeCPUTensor:
    def __init__(self, rows):
        self.rows = rows

    def tolist(self):
        return [list(r) for r in self.rows]


class FakeSamplingMetadata:
    def __init__(self, output_token_ids):
        self.output_token_ids = output_token_ids


class FakeInputBatch:
    """Only the attributes the repair touches."""

    stock_calls = 0

    def __init__(self, req_output_token_ids, sampled_rows, needs_output_ids=False):
        self.req_ids = ["r%d" % i for i in range(len(req_output_token_ids))]
        self.prev_req_id_to_index = {r: i for i, r in enumerate(self.req_ids)}
        self.req_output_token_ids = req_output_token_ids
        self.sampled_token_ids_cpu = FakeCPUTensor(sampled_rows)
        self.async_copy_ready_event = FakeEvent()
        self.sampling_metadata = FakeSamplingMetadata(
            req_output_token_ids if needs_output_ids else []
        )

    def update_async_output_token_ids(self):  # replaced by the patch
        FakeInputBatch.stock_calls += 1

    def set_async_sampled_token_ids(self, cpu, event):  # replaced by the patch
        raise AssertionError("stock setter should have been replaced")


module = types.ModuleType("vllm.v1.worker.gpu_input_batch")
module.InputBatch = FakeInputBatch
sys.modules["vllm.v1.worker.gpu_input_batch"] = module

lfm25_patches._install_async_spec_fix(logs.append)
results["armed"] = any("ACTIVE" in m for m in logs)
results["idempotent"] = (
    lfm25_patches._install_async_spec_fix(logs.append) is None
    and sum("ACTIVE" in m for m in logs) == 1
)

# --- A. the bug case: optimistic overshoot must SHRINK the list -----------
# 2 drafts were assumed accepted, only 1 token really came back.
b = FakeInputBatch([[10, 11, -1, -1]], [[42, -1, -1]])
b.update_async_output_token_ids()
results["A_overshoot_repaired"] = b.req_output_token_ids[0] == [10, 11, 42]
results["A_synced_once"] = b.async_copy_ready_event.syncs == 1

# --- B. exact match: all placeholders filled ------------------------------
b = FakeInputBatch([[10, -1, -1]], [[42, 43]])
b.update_async_output_token_ids()
results["B_exact_repaired"] = b.req_output_token_ids[0] == [10, 42, 43]

# --- C. undershoot: more sampled ids than placeholders --------------------
b = FakeInputBatch([[10, -1]], [[42, 43, 44]])
b.update_async_output_token_ids()
results["C_undershoot_clamped"] = b.req_output_token_ids[0] == [10, 42]

# --- D. no placeholders: untouched AND no GPU sync (hot path) -------------
b = FakeInputBatch([[10, 11], [7]], [[42], [43]])
b.update_async_output_token_ids()
results["D_untouched"] = b.req_output_token_ids == [[10, 11], [7]]
results["D_no_sync"] = b.async_copy_ready_event.syncs == 0

# --- E. stock path still delegates ----------------------------------------
before = FakeInputBatch.stock_calls
b = FakeInputBatch([[10, -1]], [[42]], needs_output_ids=True)
b.update_async_output_token_ids()
results["E_delegates_to_stock"] = FakeInputBatch.stock_calls == before + 1

# --- F. mixed batch, and a request missing from the previous step ---------
b = FakeInputBatch([[1, -1, -1], [2, 3], [4, -1]], [[9, 9], [0], [8, -1]])
b.prev_req_id_to_index.pop("r1")
b.update_async_output_token_ids()
results["F_mixed"] = b.req_output_token_ids == [[1, 9, 9], [2, 3], [4, 8]]

# --- G. empty / None rows must not raise ----------------------------------
b = FakeInputBatch([[], [5, -1]], [[], [7]])
b.req_output_token_ids[0] = None  # removed slot
b.update_async_output_token_ids()
results["G_none_slot_safe"] = b.req_output_token_ids[1] == [5, 7]

# --- H. setter always stores (stock drops them) ---------------------------
b = FakeInputBatch([[1]], [[2]])
b.set_async_sampled_token_ids("TENSOR", "EVENT")
results["H_setter_stores"] = (
    b.sampled_token_ids_cpu == "TENSOR" and b.async_copy_ready_event == "EVENT"
)

# --- I. internal failure is loud, not silent ------------------------------
logs.clear()
b = FakeInputBatch([[1, -1]], [[2]])
b.sampled_token_ids_cpu = object()  # no .tolist -> raises inside the repair
b.update_async_output_token_ids()
results["I_loud_on_failure"] = any("OUTPUT MAY BE CORRUPT" in m for m in logs)

print(json.dumps(results, indent=2))
bad = [k for k, v in results.items() if not v]
print("ALL PASS" if not bad else "FAILURES: %s" % bad)
sys.exit(1 if bad else 0)
