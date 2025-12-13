# FAISS HNSW-only 查询过程与优化原理（基于 `faiss/impl/HNSW.cpp` 与 `faiss/IndexHNSW.cpp`）

本文聚焦 **HNSW-only（纯 HNSW 图）** 的查询流程：从 `IndexHNSW::search` 入口到 `HNSW::search`、`search_from_candidates` 等核心函数的调用链与逐行语义，并解释 FAISS 在该路径上的主要优化与其原理。

> 说明：FAISS 的 HNSW 结构与原论文一致：多层小世界图，上层稀疏、下层稠密。FAISS 的 `HNSW` 对象只存图结构（邻接表）；`IndexHNSW*` 系列在其上叠加真实向量存储（Flat/PQ/SQ等）。

---

## 1. “HNSW-only” 在 FAISS 中指什么

通常指 **搜索只依赖 HNSW 图** 来做近邻探索，而不是 IVF/量化粗排等结构。典型索引为：

- `IndexHNSWFlat`：存储是 `IndexFlat`，距离是精确 L2/IP，但候选扩展完全由 HNSW 控制。
- `IndexHNSWPQ / IndexHNSWSQ`：存储为 PQ/SQ，距离为近似；同样候选扩展由 HNSW 控制。

与之对比：`IndexHNSW2Level` 会先用 IVF（quantizer）找入口，再在 level0 上搜索；不是“纯 HNSW-only”的典型路径。

---

## 2. 查询入口与高层调用链

### 2.1 `IndexHNSW::search` 入口

查询从 `IndexHNSW::search` 进入（`faiss/IndexHNSW.cpp:295`）：

1. 构造结果处理器 `HeapBlockResultHandler`（小顶堆形式，收集 top-k）。
2. 调用内部模板函数 `hnsw_search`。
3. 若为相似度度量（IP/Cosine），对距离做取负/还原。

对应代码（`faiss/IndexHNSW.cpp:295-314`）：

```cpp
using RH = HeapBlockResultHandler<HNSW::C>;
RH bres(n, distances, labels, k);
hnsw_search(this, n, x, bres, params);
if (is_similarity_metric(this->metric_type)) {
    for (size_t i = 0; i < k * n; i++) distances[i] = -distances[i];
}
```

### 2.2 `hnsw_search`：批量并行与 DistanceComputer

`hnsw_search`（`faiss/IndexHNSW.cpp:239-291`）做了三件关键事：

1. **解析搜索参数**：从 `HNSW::efSearch` 读取默认值，若 `params` 为 `SearchParametersHNSW` 则覆盖。
2. **分块 + OpenMP 并行**：对 query 按 `check_period` 分块；块内并行。
3. **为每线程创建 `VisitedTable` 与 `DistanceComputer`**，然后对每个 query 调用 `HNSW::search`。

核心代码（`faiss/IndexHNSW.cpp:251-285`）：

```cpp
int efSearch = hnsw.efSearch;
if (params) if (auto hnsw_params = dynamic_cast<const SearchParametersHNSW*>(params))
    efSearch = hnsw_params->efSearch;

#pragma omp parallel
{
    VisitedTable vt(index->ntotal);
    SingleResultHandler res(bres);
    std::unique_ptr<DistanceComputer> dis(storage_distance_computer(index->storage));

    #pragma omp for
    for (idx_t i = i0; i < i1; i++) {
        res.begin(i);
        dis->set_query(x + i * index->d);
        HNSWStats stats = hnsw.search(*dis, res, vt, params);
        res.end();
    }
}
```

> `DistanceComputer` 把“存储层距离计算”抽象化：对 Flat 是精确向量距离；对 PQ/SQ 是量化距离。HNSW 图只依赖它提供的 `operator()(id)` 和 `distances_batch_4(...)`。

---

## 3. HNSW-only 的核心搜索流程（`HNSW::search`）

入口：`HNSW::search`（`faiss/impl/HNSW.cpp:937-997`）。

整体分两段：

1. **上层（level=max_level..1）贪心下降**：每层用 greedy best-first 把入口更新为当前层的局部最近点。
2. **level 0 扩展搜索**：从上一步得到的入口出发，用 efSearch 控制的候选集进行图 BFS/Best-first 扩张，产出 top-k。

### 3.1 参数解析与入口检查

（`faiss/impl/HNSW.cpp:943-955`）

```cpp
if (entry_point == -1) return stats;
int k = extract_k_from_ResultHandler(res);
bool bounded_queue = this->search_bounded_queue;
int efSearch = this->efSearch;
if (params) if (auto hnsw_params = dynamic_cast<const SearchParametersHNSW*>(params)) {
    bounded_queue = hnsw_params->bounded_queue;
    efSearch = hnsw_params->efSearch;
}
```

- `entry_point==-1` 表示图为空直接返回。
- `k` 用于确保 `ef >= k`，避免 efSearch 太小无法得到 k 个结果。
- `bounded_queue` 选择两套 level0 搜索实现（见 3.3/3.4）。

### 3.2 上层贪心下降：`greedy_update_nearest`

初始化 `nearest = entry_point`，计算其距离（`faiss/impl/HNSW.cpp:958-961`）：

```cpp
storage_idx_t nearest = entry_point;
float d_nearest = qdis(nearest);
```

然后从高层往低层循环（`faiss/impl/HNSW.cpp:962-966`）：

```cpp
for (int level = max_level; level >= 1; level--) {
    greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
}
```

`greedy_update_nearest`（`faiss/impl/HNSW.cpp:846-918`）逻辑：

1. 取当前 `nearest` 在 `level` 的邻居区间 `neighbor_range(nearest, level, &begin, &end)`。
2. 扫描邻居，如果发现更近的点就更新 `nearest/d_nearest`。
3. 如果一轮扫描后 `nearest` 不再变化，则在该层收敛并返回；否则继续下一轮（相当于 hill-climbing）。

这对应论文中的“从入口向目标点做贪心爬山，逐层下降到 level0 的入口”。

### 3.3 level0 有界队列搜索（默认分支）

`ef = max(efSearch, k)`（`faiss/impl/HNSW.cpp:968`）。

若 `bounded_queue=true`（最常用，`faiss/impl/HNSW.cpp:969-975`）：

```cpp
MinimaxHeap candidates(ef);
candidates.push(nearest, d_nearest);
search_from_candidates(*this, qdis, res, candidates, vt, stats, 0, 0, params);
```

这里的 `MinimaxHeap` 是一个 **固定容量 ef 的“最小-最大堆”**：

- `pop_min()` 取当前候选中距离最小的点（用于扩展）。
- `max()`/`count_below()` 用来判断停止条件。
- 固定容量可以控制探索宽度与内存/CPU开销。

### 3.4 level0 无界队列搜索（可选）

若 `bounded_queue=false`，调用 `search_from_candidate_unbounded`（`faiss/impl/HNSW.cpp:977-991`）：

```cpp
auto top_candidates = search_from_candidate_unbounded(
    *this, Node(d_nearest, nearest), qdis, ef, &vt, stats);
while (top_candidates.size() > k) top_candidates.pop();
while (!top_candidates.empty()) { res.add_result(...); top_candidates.pop(); }
```

无界版使用两个 `priority_queue`（一个扩展队列 candidates、一个 top_candidates），可得到略不同的探索行为，但开销更高，因此默认不用。

---

## 4. level0 具体扩展：`search_from_candidates`

函数：`search_from_candidates`（`faiss/impl/HNSW.cpp:592-735`），FAISS 的 **高性能 HNSW-only 核心**。

### 4.1 初始化：把初始候选加入结果并标记 visited

（`faiss/impl/HNSW.cpp:606-631`）

```cpp
bool do_dis_check = hnsw.check_relative_distance;
int efSearch = hnsw.efSearch;
const IDSelector* sel = params ? params->sel : nullptr;

for (int i = 0; i < candidates.size(); i++) {
    idx_t v1 = candidates.ids[i];
    float d = candidates.dis[i];
    if (!sel || sel->is_member(v1)) {
        if (d < threshold) res.add_result(d, v1);
    }
    vt.set(v1);
}
```

要点：

- `VisitedTable vt` 是一个位图/打点表，用于 O(1) 判断节点是否已访问过。
- `IDSelector` 可过滤不允许的 id（可选）。
- 把初始候选（通常只有入口一个）直接尝试加入 top-k 结果。

### 4.2 主循环：Best-first 扩展

核心循环（`faiss/impl/HNSW.cpp:635-724`）：

1. `pop_min` 取当前最小距离候选 `v0`。
2. 依据停止条件决定是否终止。
3. 枚举 `v0` 的 level0 邻居 `neighbors[begin:end)`。
4. 对未访问过的邻居计算距离、尝试加入结果、并 push 回 candidates。

#### 4.2.1 停止条件

有两种：

- **相对距离停止（默认）**：`do_dis_check=true` 时（`faiss/impl/HNSW.cpp:639-647`）

```cpp
int n_dis_below = candidates.count_below(d0);
if (n_dis_below >= efSearch) break;
```

含义：如果候选堆中已经有 ≥efSearch 个点比当前要扩展的 `d0` 更近，那么继续扩展 `v0` 很难改进 top-k，可以停止。这是论文中“efSearch 个更好候选存在时停止”的变体。

- **步数停止（非相对距离）**：`do_dis_check=false` 时（`faiss/impl/HNSW.cpp:720-723`）

```cpp
if (!do_dis_check && nstep > efSearch) break;
```

更简单但通常 recall 稍差。

#### 4.2.2 邻居枚举与 SIMD 计算

枚举邻居前先预取 visited 表（`faiss/impl/HNSW.cpp:653-664`）：

```cpp
prefetch_L2(vt.visited.data() + v1);
```

接着把未访问邻居收集到 `saved_j[4]`，每 4 个用批量距离算子计算（`faiss/impl/HNSW.cpp:666-711`）：

```cpp
bool vget = vt.get(v1);
vt.set(v1);
saved_j[counter] = v1;
counter += vget ? 0 : 1;
if (counter == 4) {
    float dis[4];
    qdis.distances_batch_4(saved_j[0], saved_j[1], saved_j[2], saved_j[3], ...);
    for (id4=0..3) add_to_heap(saved_j[id4], dis[id4]);
    counter = 0;
}
```

`add_to_heap`（lambda）会：

1. 若距离比当前 top-k 阈值更小则尝试加入结果。
2. 不论是否进 top-k，都 push 到 candidates（用于后续扩展）。

---

## 5. HNSW-only 查询路径上的优化点与原理

下面把上述代码中体现的优化逐一解释其目的与原理。

### 5.1 有界候选堆 `MinimaxHeap`（默认）

对应：`HNSW::search` 中 `bounded_queue` 分支（`faiss/impl/HNSW.cpp:969-975`）。

原理：

- 固定容量 ef 的堆保证 **探索宽度上界**，使复杂度近似 `O(ef * log ef)`。
- `MinimaxHeap` 支持 `pop_min`（扩展最近点）和 `count_below`（停止条件计算），减少对 `std::priority_queue` 的频繁 push/pop 开销。

效果：在相同 efSearch 下提升吞吐、降低 tail latency。

### 5.2 `check_relative_distance` 的早停

对应：`search_from_candidates` 的 `count_below` 判断（`faiss/impl/HNSW.cpp:639-647`）。

原理：

- HNSW 的候选扩展是 best-first；当堆里已经有很多比当前更好的候选时，当前候选的邻居大概率也不优。
- 用“堆内比 d0 更小的候选数量”近似上界，提前终止扩展，减少无效距离计算。

效果：在保持 recall 的同时显著减少 `ndis` 和 `nhops`（见 stats）。

### 5.3 4 路批量距离 `distances_batch_4`

对应：

- 上层贪心（`greedy_update_nearest`，`faiss/impl/HNSW.cpp:872-899`）
- level0 扩展（`search_from_candidates`，`faiss/impl/HNSW.cpp:691-705`）
- 无界版本同理（`faiss/impl/HNSW.cpp:804-818`）

原理：

- `DistanceComputer` 对 Flat/PQ/SQ 都提供 4 路批量接口，底层往往可用 SIMD（SSE/AVX/NEON）或流水线优化。
- 批量计算减少函数调用与分支开销，提高向量化机会。

效果：单次 hop 的距离评估更快，是 FAISS HNSW 高吞吐的关键来源之一。

### 5.4 `prefetch_L2` 预取 visited 表

对应：邻居扫描时对 `vt.visited[v1]` 预取（`faiss/impl/HNSW.cpp:662-664, 777-779`）。

原理：

- `vt.get(v1)` 是频繁的随机访存；预取能把即将访问的 cache line 提前拉入 L2。
- 对高维大图搜索，visited 表往往比向量数据更小，预取收益明显。

效果：减少 cache miss，提升 hop 内循环效率。

### 5.5 `VisitedTable` 线程内复用与 `advance()`

对应：

- `hnsw_search` 每线程创建一次 vt，并对每个 query 使用（`faiss/IndexHNSW.cpp:268-284`）
- `HNSW::search` 结束后 `vt.advance()`（`faiss/impl/HNSW.cpp:994`）

原理：

- `VisitedTable` 通常用“访问标号 + 当前 epoch”的方式实现：不清零整个表，只递增 epoch 即可让旧标记失效。
- 省去 `O(ntotal)` 的清理成本，避免每 query 触发大内存写。

效果：对多 query 批量搜索时极大降低 overhead。

### 5.6 Query 分块与中断检查

对应：`hnsw_search` 的 `check_period` 与 `InterruptCallback::check()`（`faiss/IndexHNSW.cpp:260-288`）。

原理：

- 把 query 分块可以让长批量搜索定期响应中断/超时。
- `check_period` 与 `hnsw.max_level * d * efSearch` 成比例，避免过于频繁的检查影响吞吐。

效果：提升可控性与工程鲁棒性。

### 5.7 相似度度量的“负距离技巧”

对应：`IndexHNSW::search` 末尾翻转距离（`faiss/IndexHNSW.cpp:309-313`）。

原理：

- HNSW 搜索与堆处理统一假设“越小越好”（L2）。
- 对 IP/Cosine 等相似度，FAISS 在 `DistanceComputer` 中用 `-similarity` 表示“伪距离”，搜索结束再取负还原。

效果：复用同一套 HNSW 算法与堆逻辑，避免重复实现。

---

## 6. 小结：HNSW-only 查询的端到端路径

按调用顺序总结：

1. 用户调用 `IndexHNSWFlat::search` → 实际进入 `IndexHNSW::search`（`faiss/IndexHNSW.cpp:295`）。
2. `IndexHNSW::search` 构造结果 handler → 调用 `hnsw_search`（`faiss/IndexHNSW.cpp:239`）。
3. `hnsw_search` 为每线程准备 `VisitedTable`/`DistanceComputer`，逐 query 调用 `HNSW::search`（`faiss/IndexHNSW.cpp:279`）。
4. `HNSW::search`：
   - 上层贪心下降：`greedy_update_nearest`（`faiss/impl/HNSW.cpp:962-966` + `846`）。
   - level0 搜索：
     - 默认 `bounded_queue`：`MinimaxHeap` + `search_from_candidates`（`faiss/impl/HNSW.cpp:969-975` + `592`）。
     - 可选无界：`search_from_candidate_unbounded`（`faiss/impl/HNSW.cpp:977`）。
5. `search_from_candidates` 进行 best-first 扩展，受 `efSearch` 与早停控制，并用 batch4+prefetch 等优化减少距离与访存成本。
6. 返回 top-k（必要时对相似度距离翻转）。

