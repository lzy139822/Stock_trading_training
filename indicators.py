"""
盘感训练器 - 技术指标计算模块

封装 MyTT 指标函数，提供统一的计算和查询接口。
所有指标在训练开始时全量预计算，推演时只绘制可见切片。
"""

import numpy as np
from easy_tdx.MyTT import (
    MA, EMA,
    MACD, KDJ, RSI, CCI, OBV, CR, TRIX, DMI, BIAS, WR,
    EXPMA, BBI, BOLL,
    ZHUOYAO, BIAS_SIGNAL,
)

import pandas as pd


# ============================================================
# 副图指标注册表
# 每个指标定义了：MyTT函数引用、输入列、默认参数、输出名、颜色、柱状输出
# ============================================================
SUB_INDICATOR_REGISTRY = {
    "MACD": {
        "func": MACD,
        "input_cols": ["CLOSE"],
        "defaults": {"SHORT": 12, "LONG": 26, "M": 9},
        "outputs": ["DIF", "DEA", "MACD"],
        "colors": ["#FFFFFF", "#FFFF00", "#FF4444"],
        "bar_output": "MACD",  # MACD 柱需要用柱状图绘制
        "zero_line": True,     # 显示零线
    },
    "KDJ": {
        "func": KDJ,
        "input_cols": ["CLOSE", "HIGH", "LOW"],
        "defaults": {"N": 9, "M1": 3, "M2": 3},
        "outputs": ["K", "D", "J"],
        "colors": ["#FFFFFF", "#FFFF00", "#FF44FF"],
        "bar_output": None,
        "zero_line": False,
    },
    "RSI": {
        "func": RSI,
        "input_cols": ["CLOSE"],
        "defaults": {"N": 6},
        "outputs": ["RSI6", "RSI12", "RSI24"],
        "colors": ["#FFFFFF", "#FFFF00", "#FF69B4"],
        "bar_output": None,
        "zero_line": False,
        "ref_lines": [30, 70],  # RSI 参考线
        "multi_period": [6, 12, 24],  # 多周期：分别计算 RSI(6/12/24)
    },
    "CCI": {
        "func": CCI,
        "input_cols": ["CLOSE", "HIGH", "LOW"],
        "defaults": {"N": 14},
        "outputs": ["CCI"],
        "colors": ["#FFD700"],
        "bar_output": None,
        "zero_line": True,
        "ref_lines": [-100, 100],
    },
    "OBV": {
        "func": OBV,
        "input_cols": ["CLOSE", "VOL"],
        "defaults": {},
        "outputs": ["OBV"],
        "colors": ["#FFD700"],
        "bar_output": None,
        "zero_line": False,
        "extra_lines": [
            {"name": "OBV_MA30", "period": 30, "color": "#FF69B4"},
        ],
    },
    "CR": {
        "func": CR,
        "input_cols": ["CLOSE", "HIGH", "LOW"],
        "defaults": {"N": 26},
        "outputs": ["CR"],
        "colors": ["#FFD700"],
        "bar_output": None,
        "zero_line": False,
        "ref_lines": [100],
        "extra_lines": [
            {"name": "CR_MA10", "period": 10, "color": "#FFFFFF"},
            {"name": "CR_MA20", "period": 20, "color": "#FFFF00"},
            {"name": "CR_MA40", "period": 40, "color": "#00BFFF"},
            {"name": "CR_MA62", "period": 62, "color": "#FF69B4"},
        ],
    },
    "TRIX": {
        "func": TRIX,
        "input_cols": ["CLOSE"],
        "defaults": {"M1": 12, "M2": 20},
        "outputs": ["TRIX", "TRMA"],
        "colors": ["#FFFFFF", "#FFFF00"],
        "bar_output": None,
        "zero_line": True,
    },
    "DMI": {
        "func": DMI,
        "input_cols": ["CLOSE", "HIGH", "LOW"],
        "defaults": {"M1": 14, "M2": 6},
        "outputs": ["PDI", "MDI", "ADX", "ADXR"],
        "colors": ["#FF4444", "#00CC00", "#FFD700", "#FF44FF"],
        "bar_output": None,
        "zero_line": False,
    },
    "BIAS": {
        "func": BIAS,
        "input_cols": ["CLOSE"],
        "defaults": {"L1": 6, "L2": 12, "L3": 24},
        "outputs": ["BIAS1", "BIAS2", "BIAS3"],
        "colors": ["#FFFFFF", "#FFFF00", "#FF44FF"],
        "bar_output": None,
        "zero_line": True,
    },
    "WR": {
        "func": WR,
        "input_cols": ["CLOSE", "HIGH", "LOW"],
        "defaults": {"N": 10, "N1": 6},
        "outputs": ["WR1", "WR2"],
        "colors": ["#FFFFFF", "#FFFF00"],
        "bar_output": None,
        "zero_line": False,
        "ref_lines": [20, 80],  # WR 超买超卖线
    },
    "捉妖大师": {
        "func": ZHUOYAO,
        "input_cols": ["CLOSE"],
        "defaults": {"N1": 120, "N2": 60, "N3": 20, "M": 10},
        "outputs": ["LONG", "MID", "SHORT", "TREND"],
        "colors": ["#FF4444", "#FFD700", "#00CC00", "#00BFFF"],
        "bar_output": None,
        "zero_line": True,
    },
    "MA30乖离": {
        "func": BIAS_SIGNAL,
        "input_cols": ["CLOSE"],
        "defaults": {"P": 10, "M": 30},
        "outputs": ["X", "S_SMA", "X_LMA"],
        "colors": ["#FFD700", "#FFFFFF", "#FF69B4"],
        "bar_output": None,
        "zero_line": True,
    },
}

# 副图指标名称列表（用于下拉菜单）
SUB_INDICATOR_NAMES = list(SUB_INDICATOR_REGISTRY.keys())


# ============================================================
# 自编主图叠加：常驻叠加均线（EMA_STACK）
#
# 通达信原始公式（用户提供）：
#   短期1:  EMA(C,10)        COLORFFA6FF
#   短期2:  EMA(短期1,3)     COLORFFA6FF
#   短期3:  EMA(短期2,3)     COLORFFA6FF
#   短期4:  EMA(短期3,3)     COLORFFA6FF
#   短期5:= EMA(短期4,3)     COLORFFA6FF   <- 通达信 ":=" 表示不输出（不画线）
#   长期1:  EMA(C,45)        COLOR8F8F47
#   长期2~5: EMA(前一项,3)   COLOR8F8F47
#
# 等价性说明：MyTT.EMA(S,N) = pandas ewm(span=N, adjust=False)，
# 权重 alpha = 2/(N+1)、首值取序列首值，与通达信 EMA(X,N) 定义一致，
# 因此递归调用可直接复现通达信结果。
# ============================================================
EMA_STACK_DEFAULTS = {
    "short_n": 10,          # 短期链基准周期 EMA(C, 10)
    "long_n": 45,           # 长期链基准周期 EMA(C, 45)
    "smooth_n": 3,          # 递进平滑周期 EMA(前值, 3)
    "layers": 5,            # 每条链递推层数（通达信公式里是 5 层）
    "short_draw": 4,        # 短期链实际画到第几层（通达信里 短期5 用 := 隐藏）
    "long_draw": 5,         # 长期链实际画到第几层
    "short_color_tdx": "FFA6FF",   # 通达信 COLORFFA6FF
    "long_color_tdx": "8F8F47",    # 通达信 COLOR8F8F47
    # 通达信 COLOR 后六位是 BGR 顺序（例：COLOR0000FF 在通达信里显示为红色），
    # 与常规 RGB 相反。若您想按 RGB 直接解读，把这里改成 "RGB" 即可。
    "color_order": "BGR",
    "linewidth": 1.0,
}


def tdx_color_to_hex(code: str, order: str = "BGR") -> str:
    """
    把通达信颜色代码转成 matplotlib 可用的 '#RRGGBB'。
    Args:
        code: 通达信颜色码，形如 'FFA6FF'（可带或不带 COLOR 前缀、'#'）
        order: 'BGR'（通达信惯例，默认）或 'RGB'
    Returns:
        str: '#RRGGBB'；解析失败返回白色
    """
    text = str(code).strip().upper()
    if text.startswith("COLOR"):
        text = text[5:]
    text = text.lstrip("#")
    if len(text) != 6 or any(c not in "0123456789ABCDEF" for c in text):
        return "#FFFFFF"
    c0, c1, c2 = text[0:2], text[2:4], text[4:6]
    if str(order).strip().upper() == "BGR":
        r, g, b = c2, c1, c0      # 交换红蓝通道
    else:
        r, g, b = c0, c1, c2
    return f"#{r}{g}{b}"


class IndicatorHub:
    """
    技术指标计算中心。

    在训练开始时全量预计算所有指标，后续只查询结果。
    这样推演时无需重算，切换指标也只更换绘图数据。
    """

    # 均线默认配色（最多 6 条）
    MA_COLORS = ["#FFD700", "#FF69B4", "#00BFFF", "#FFA500", "#FF4500", "#ADFF2F"]

    def __init__(self, df: pd.DataFrame, config: dict):
        """
        初始化指标计算中心。

        Args:
            df: K线数据 DataFrame（含 open/high/low/close/volume 列）
            config: 指标参数配置（来自 config.yaml）
        """
        self.df = df
        self.config = config.get("indicators", {})
        self.app_config = config                    # 保留完整配置（含 ma_periods）
        self.ma_periods: list[int] = config.get("ma_periods", [5, 10, 20, 30, 60, 120])
        self.results: dict[str, dict] = {}       # 副图指标缓存
        self.main_overlays: dict[str, np.ndarray] = {}  # 主图叠加缓存

    def calculate_all(self) -> None:
        """
        一次性预计算所有副图指标和主图叠加指标。

        从 DataFrame 提取价格序列，调用 MyTT 函数，
        结果以 numpy 数组形式缓存。
        """
        # 提取基础序列
        close = self.df["close"].values.astype(float)
        high = self.df["high"].values.astype(float)
        low = self.df["low"].values.astype(float)
        vol = self.df["volume"].values.astype(float)

        # 序列映射表（供指标函数查表取输入）
        col_map = {"CLOSE": close, "HIGH": high, "LOW": low, "VOL": vol}

        # ---- 计算所有副图指标 ----
        for name, reg in SUB_INDICATOR_REGISTRY.items():
            try:
                user_params = self.config.get(name, reg["defaults"])
                args = [col_map[col] for col in reg["input_cols"]]
                kwargs = {k: v for k, v in user_params.items()}

                self.results[name] = {}

                # ---- RSI 多周期特殊处理 ----
                if "multi_period" in reg:
                    periods = reg["multi_period"]
                    for i, period in enumerate(periods):
                        rsi_arr = reg["func"](*args, N=period)
                        if not isinstance(rsi_arr, np.ndarray):
                            rsi_arr = np.array(rsi_arr, dtype=float)
                        self.results[name][reg["outputs"][i]] = rsi_arr
                else:
                    # ---- 常规指标计算 ----
                    result_arrays = reg["func"](*args, **kwargs)
                    if not isinstance(result_arrays, (tuple, list)):
                        result_arrays = (result_arrays,)
                    for out_name, arr in zip(reg["outputs"], result_arrays):
                        self.results[name][out_name] = np.array(arr, dtype=float)

                # ---- extra_lines 均线计算（OBV_MA30, CR_MA10 等） ----
                if "extra_lines" in reg:
                    # 取第一个 output 作为均线源数据
                    first_output = reg["outputs"][0]
                    source_arr = self.results[name].get(first_output)
                    if source_arr is not None:
                        for el in reg["extra_lines"]:
                            self.results[name][el["name"]] = MA(source_arr, el["period"])

            except Exception as e:
                print(f"[警告] 指标 {name} 计算失败: {e}")
                self.results[name] = {}

        # ---- 计算主图叠加指标 ----
        # 清除旧的动态均线数据，避免周期变更后残留
        old_ma_keys = [k for k in self.main_overlays if k.startswith("MA") and k[2:].isdigit()]
        for k in old_ma_keys:
            del self.main_overlays[k]
        # 动态均线：根据配置的 ma_periods 计算每条 MA
        for p in self.ma_periods:
            try:
                self.main_overlays[f"MA{p}"] = MA(close, p)
            except Exception:
                pass

        try:
            ema1, ema2 = EXPMA(close, **self.config.get("EXPMA", {"N1": 12, "N2": 50}))
            self.main_overlays["EXPMA12"] = ema1
            self.main_overlays["EXPMA50"] = ema2
        except Exception:
            pass

        try:
            self.main_overlays["BBI"] = BBI(close, **self.config.get("BBI", {"M1": 3, "M2": 6, "M3": 12, "M4": 20}))
        except Exception:
            pass

        try:
            upper, mid, lower = BOLL(close, **self.config.get("BOLL", {"N": 20, "P": 2}))
            self.main_overlays["BOLL_UPPER"] = upper
            self.main_overlays["BOLL_MID"] = mid
            self.main_overlays["BOLL_LOWER"] = lower
        except Exception:
            pass

        # ---- 自编常驻叠加均线（EMA_STACK）----
        try:
            self.main_overlays.update(self._calc_ema_stack(close))
        except Exception as e:
            print(f"[警告] EMA_STACK 计算失败: {e}")

    # ============================================================
    # 自编主图叠加：常驻叠加均线
    # ============================================================
    def _ema_stack_params(self) -> dict:
        """读取 EMA_STACK 参数（用户配置优先，缺失项用默认值兜底）。"""
        params = dict(EMA_STACK_DEFAULTS)
        for k, v in (self.config.get("EMA_STACK") or {}).items():
            if v is not None:
                params[k] = v
        return params

    def _calc_ema_stack(self, close: np.ndarray) -> dict:
        """
        递推计算短期链 / 长期链 EMA。

        短期链: S1=EMA(C,short_n), S(i)=EMA(S(i-1), smooth_n)
        长期链: L1=EMA(C,long_n),  L(i)=EMA(L(i-1), smooth_n)

        Returns:
            dict: {'ESTK_S1'..'ESTK_S5', 'ESTK_L1'..'ESTK_L5'}
        """
        params = self._ema_stack_params()
        layers = max(1, int(params["layers"]))
        smooth_n = max(1, int(params["smooth_n"]))
        out = {}
        for prefix, base_n in (
            ("ESTK_S", int(params["short_n"])),
            ("ESTK_L", int(params["long_n"])),
        ):
            series = EMA(close, base_n)
            out[f"{prefix}1"] = np.array(series, dtype=float)
            for i in range(2, layers + 1):
                series = EMA(np.array(series, dtype=float), smooth_n)
                out[f"{prefix}{i}"] = np.array(series, dtype=float)
        return out

    def get_sub_indicator(self, name: str) -> dict:
        """
        获取副图指标的绘图数据。

        Args:
            name: 指标名称（如 "MACD"）

        Returns:
            dict: 含 registry 元信息 + results 数组数据
        """
        reg = SUB_INDICATOR_REGISTRY.get(name, {})
        data = self.results.get(name, {})
        return {
            "registry": reg,
            "data": data,
        }

    def get_main_overlay_lines(self, enabled: list[str]) -> list[dict]:
        """
        获取主图叠加线的绘图数据。

        Args:
            enabled: 启用的叠加指标名称列表（如 ["MA", "BOLL"]）

        Returns:
            list[dict]: 每条线的 {name, array, color, linewidth}
        """
        lines = []

        for name in enabled:
            if name == "MA":
                # 动态均线：遍历所有配置的周期
                for i, p in enumerate(self.ma_periods):
                    key = f"MA{p}"
                    arr = self.main_overlays.get(key)
                    if arr is not None:
                        color = self.MA_COLORS[i % len(self.MA_COLORS)]
                        lines.append({
                            "name": key,
                            "array": arr,
                            "color": color,
                            "linewidth": 1.0,
                        })
            elif name == "EXPMA":
                arr = self.main_overlays.get("EXPMA12")
                if arr is not None:
                    lines.append({
                        "name": "EXPMA12",
                        "array": arr,
                        "color": "#00BFFF",
                        "linewidth": 1.0,
                    })
                arr2 = self.main_overlays.get("EXPMA50")
                if arr2 is not None:
                    lines.append({
                        "name": "EXPMA50",
                        "array": arr2,
                        "color": "#FF4500",
                        "linewidth": 1.0,
                    })
            elif name == "BBI":
                arr = self.main_overlays.get("BBI")
                if arr is not None:
                    lines.append({
                        "name": "BBI",
                        "array": arr,
                        "color": "#FFA500",
                        "linewidth": 1.0,
                    })
            elif name == "BOLL":
                for key, lbl, clr in [
                    ("BOLL_UPPER", "BOLL_UPPER", "#888888"),
                    ("BOLL_MID", "BOLL_MID", "#888888"),
                    ("BOLL_LOWER", "BOLL_LOWER", "#888888"),
                ]:
                    arr = self.main_overlays.get(key)
                    if arr is not None:
                        lines.append({
                            "name": lbl,
                            "array": arr,
                            "color": clr,
                            "linewidth": 0.8,
                        })
            elif name == "EMA_STACK":
                lines.extend(self._ema_stack_lines())

        return lines

    def _ema_stack_lines(self) -> list[dict]:
        """
        组装自编常驻叠加均线的绘图数据。

        同一条链内每层同色，越深的层透明度略低，便于区分递推顺序。
        """
        params = self._ema_stack_params()
        chains = [
            ("ESTK_S", "短", int(params["short_draw"]),
             tdx_color_to_hex(params["short_color_tdx"], params["color_order"])),
            ("ESTK_L", "长", int(params["long_draw"]),
             tdx_color_to_hex(params["long_color_tdx"], params["color_order"])),
        ]
        lw = float(params["linewidth"])
        lines = []
        for prefix, label, draw_n, color in chains:
            for i in range(1, draw_n + 1):
                arr = self.main_overlays.get(f"{prefix}{i}")
                if arr is None:
                    continue
                lines.append({
                    "name": f"{label}{i}",
                    "array": arr,
                    "color": color,
                    "linewidth": lw,
                    "alpha": round(max(0.35, 1.0 - 0.12 * (i - 1)), 2),
                })
        return lines

    def get_min_warmup(self) -> int:
        """
        计算指标预热所需的最小 K 线根数。

        确保 EMA/SMA 类指标在第一个可见位置已有有效值。
        取最长均线周期、MACD LONG=26、BOLL N=20 等的最大值。

        Returns:
            int: 预热根数（cursor 初始值）
        """
        max_ma = max(self.ma_periods) if self.ma_periods else 30
        # MACD LONG=26, BOLL N=20, BBI M4=20 都不超过 max_ma
        return max(30, max_ma)
