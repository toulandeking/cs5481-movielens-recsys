"""
cs5481 MovieLens-1M recommender
==============================

src/adapters/base.py  —  适配层基类 + 模型注册表
------------------------------------------------
所有算法适配器都必须继承 `AdapterBase`。这个文件存在的意义是：

1. **统一评估入口**：`evaluate.py` 只认 `predict_scores(uid, iid)` 或 `score_matrix()`。
2. **共享与算法无关的逻辑**：`export_top10()`、`score_matrix()` 兜底、形状校验。
   三个模型各写一遍必然出现口径不一致（例如漏排除 test），且极难发现。
3. **让上层代码不用写 if/elif**：演示程序与对比表脚本通过 `load_adapter("itemcf")`
   拿到实例，不需要知道具体类名。

新增一个算法只需要三步
~~~~~~~~~~~~~~~~~~~~~~
    class MyAdapter(AdapterBase):
        name = "MyModel"
        def fit(self) -> None: ...
        def predict_scores(self, uid, iid) -> np.ndarray: ...

然后在本文件 `_BUILTIN` 里登记一行即可（延迟导入，不会产生循环依赖）。

职责边界
~~~~~~~~
`fit()` 只能使用 `split == "train"` 的数据；排除已交互物品**不在这里做**，
统一由 `src/evaluate.py`（评估口径）和 `export_top10()`（部署口径）负责。
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import pandas as pd
from scipy import sparse

from src.contract import K, N_ITEMS, N_USERS, SPLIT_TRAIN, read_canon_table

__all__ = [
    "AdapterBase", "build_binary_matrix", "all_rated_mask", "top_k_indices",
    "register_adapter", "load_adapter", "list_adapters",
]


# =============================================================================
# 共享工具
# =============================================================================
def build_binary_matrix(split: str = SPLIT_TRAIN) -> sparse.csr_matrix:
    """从 canon 构造二值交互矩阵 (N_USERS, N_ITEMS)，所有观测交互权重为 1。"""
    r = read_canon_table("ratings")
    part = r[r["split"] == split]
    data = np.ones(len(part), dtype=np.float32)
    return sparse.csr_matrix(
        (data, (part["uid"].to_numpy(), part["iid"].to_numpy())),
        shape=(N_USERS, N_ITEMS),
        dtype=np.float32,
    )


def all_rated_mask() -> np.ndarray:
    """(N_USERS, N_ITEMS) bool：该用户是否评过这部电影（含 train/val/test 全部）。"""
    r = read_canon_table("ratings")
    mask = np.zeros((N_USERS, N_ITEMS), dtype=bool)
    mask[r["uid"].to_numpy(), r["iid"].to_numpy()] = True
    return mask


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """每行取分数最高的 k 个下标，返回 (n_rows, k)。

    排序规则：**分数降序；同分按列下标升序**（`np.lexsort` 是稳定排序）。
    与 `evaluate.py` 的名次定义（同分下标小者在前）保持一致，避免两处口径不同。
    """
    n_rows, n_cols = scores.shape
    if k > n_cols:
        raise ValueError(f"k={k} 超过物品数 {n_cols}")
    rows = np.arange(n_rows)[:, None]
    idx = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.lexsort((idx, -scores[rows, idx]), axis=1)
    return np.take_along_axis(idx, order, axis=1)


def _check_matrix(S: np.ndarray) -> np.ndarray:
    S = np.asarray(S)
    if S.shape != (N_USERS, N_ITEMS):
        raise ValueError(f"分数矩阵形状应为 {(N_USERS, N_ITEMS)}，实际 {S.shape}")
    return S


# =============================================================================
# 基类
# =============================================================================
class AdapterBase(ABC):
    """所有推荐算法的统一接口。评估只认 `predict_scores` / `score_matrix`。"""

    name: str = "base"

    @abstractmethod
    def fit(self) -> None:
        """只用 split == "train" 的数据训练。"""

    @abstractmethod
    def predict_scores(self, uid: np.ndarray, iid: np.ndarray) -> np.ndarray:
        """返回与 (uid, iid) 等长的分数，越大越推荐。支持任意成对查询。"""

    # -- 以下是所有子类共享的实现，通常不需要覆盖 -------------------------
    def score_matrix(self) -> np.ndarray:
        """(N_USERS, N_ITEMS) 完整分数矩阵。

        子类若能用一次矩阵乘法算出全部分数，应当覆盖此方法（比逐对查询快得多）；
        否则退回通用实现：按用户分批调 `predict_scores` 铺满整个物品空间。
        """
        from src.evaluate import score_matrix_from_adapter
        return _check_matrix(score_matrix_from_adapter(self))

    def export_top10(self, k: int = K) -> pd.DataFrame:
        """导出**部署口径**的 top-k：排除该用户评过的所有片（train ∪ val ∪ test）。

        为什么连 test 也排除：作业要求"recommend movies that the user has not rated"，
        而 test 目标只是被我们人为留出的、用户**实际评过**的一部片。
        模型效果评估另由 evaluate.py 负责（那里只排除 train ∪ val）。
        """
        S = _check_matrix(self.score_matrix()).astype(np.float32, copy=True)
        S[all_rated_mask()] = -np.inf
        idx = top_k_indices(S, k)
        return pd.DataFrame({
            "model": self.name,
            "uid": np.repeat(np.arange(N_USERS, dtype=np.int32), k),
            "rank": np.tile(np.arange(1, k + 1, dtype=np.int8), N_USERS),
            "iid": idx.astype(np.int32).ravel(),
            "score": np.take_along_axis(S, idx, axis=1).astype(np.float32).ravel(),
        })

    def __repr__(self) -> str:
        params = {
            k: v for k, v in vars(self).items()
            if not k.startswith("_") and isinstance(v, (int, float, str, type(None), tuple))
        }
        kv = ", ".join(f"{k}={v!r}" for k, v in params.items())
        return f"{type(self).__name__}({kv})"


# =============================================================================
# 注册表（延迟导入，避免与各 adapter 之间产生循环依赖）
# =============================================================================
_BUILTIN: dict[str, tuple[str, str]] = {
    "itemcf": ("src.adapters.itemcf", "ItemCFAdapter"),
    "lightfm": ("src.adapters.lightfm", "LightFMAdapter"),
    # 后续接入时在此登记：
    # "lrurec":  ("src.adapters.lrurec", "LRURecAdapter"),
}
_CUSTOM: dict[str, Callable[..., AdapterBase]] = {}


def register_adapter(name: str, factory: Callable[..., AdapterBase]) -> None:
    """运行时注册自定义适配器工厂（例如测试用假模型）。"""
    _CUSTOM[name] = factory


def list_adapters() -> list[str]:
    """所有已登记的适配器名（按字母序）。"""
    return sorted(set(_BUILTIN) | set(_CUSTOM))


def load_adapter(name: str, **kwargs) -> AdapterBase:
    """按名字实例化适配器。`kwargs` 会透传给构造函数。

        >>> load_adapter("itemcf", k_neighbors=20).fit()
    """
    key = name.lower()
    if key in _CUSTOM:
        return _CUSTOM[key](**kwargs)
    if key not in _BUILTIN:
        raise KeyError(f"未知适配器 {name!r}，已登记：{list_adapters()}")
    module_path, class_name = _BUILTIN[key]
    cls = getattr(importlib.import_module(module_path), class_name)
    return cls(**kwargs)
