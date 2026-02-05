# simple-ANNS-problem

C++ 实现的课程作业 **Simple ANNS Problem**：

- **Part 1：基于图索引的近似最近邻搜索（ANNS Search）**
- **Part 2：索引构建（kNN 图 / 图索引文件）**

评测指标：**QPS** 与 **Recall@K**。

---

## 编译

```bash
cd ANNS-Problem-new/src
make
```

## 运行

### 1) 搜索（Part 1）

```bash
./search <base.fvecs> <graph> <query.fvecs> <groundtruth.ivecs> <K>
```

**示例（siftsmall + 10k.graph）：**

```bash
./search \
  ../res/siftsmall/siftsmall_base.fvecs \
  ../res/10k.graph \
  ../res/siftsmall/siftsmall_query.fvecs \
  ../res/siftsmall/siftsmall_groundtruth.ivecs \
  100
```

**示例（siftbig + 1m.graph）：**

```bash
./search \
  ../res/siftbig/siftbig_base.fvecs \
  ../res/1m.graph \
  ../res/siftbig/siftbig_query.fvecs \
  ../res/siftbig/siftbig_groundtruth_10.ivecs \
  10
```

### 2) 构建图索引（Part 2）

```bash
./build_knn <base.fvecs> <out.graph> <K>
```

**示例（生成 10k 图）：**

```bash
./build_knn \
  ../res/siftsmall/siftsmall_base.fvecs \
  ../res/10k.graph \
  10
```

## 数据集

数据位于 `ANNS-Problem-new/res/`：

- **siftsmall/**：`siftsmall_base.fvecs` / `siftsmall_query.fvecs` / `siftsmall_groundtruth.ivecs`
- **siftbig/**：`siftbig_base.fvecs` / `siftbig_query.fvecs` / `siftbig_groundtruth_10.ivecs`（top10）
- **根目录**：`10k.graph`（siftsmall）/ `1m.graph`（siftbig）

数据下载与整理：

```bash
cd ANNS-Problem-new/res && \
wget -c ftp://ftp.irisa.fr/local/texmex/corpus/{siftsmall,sift}.tar.gz && \
tar -xzf siftsmall.tar.gz && tar -xzf sift.tar.gz && mkdir -p siftbig && cp -n sift/sift_{base,query}.fvecs siftbig/ && \
python - <<'PY'
import numpy as np, struct
a=np.fromfile("sift/sift_groundtruth.ivecs",dtype=np.int32); d=int(a[0])
gt=a.reshape(-1,d+1)[:,1:11].astype(np.int32)
with open("siftbig/siftbig_groundtruth_10.ivecs","wb") as f:
    for row in gt: f.write(struct.pack("i", row.size)); f.write(row.tobytes())
PY
```

**数据来源**：TexMex / BIGANN（IRISA - corpus-texmex.irisa.fr）

## 指标说明

- **QPS** = 查询数 / 搜索总耗时(秒)
- **Recall@K**：返回 TopK 与 groundtruth TopK 的交集比例（对所有 query 取平均）

## 致谢

作业 starter code：ANNS-Problem-new（原始仓库：LTTMG/ANNS-Problem-new）
