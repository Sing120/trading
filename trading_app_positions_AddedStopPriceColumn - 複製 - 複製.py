# -*- coding: utf-8 -*-
# Auto-generated single-file integration
# Modified:
# 1. Added "Stop Price" column to Open Orders table.
# 2. Added Excess Liq, MKT Val, Buying Power to Account Summary.
# 3. FIXED: "too many values to unpack" error in all functions (K-line fetch, Watch loop, Update info).
#    Changed all `netliq, _ = ...` to handle 5 return values.

import os
import math
import time
import threading
from datetime import datetime, timedelta, timezone, time as dtime

import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import scrolledtext
import tkinter.font as tkfont

from dateutil import tz
from ib_insync import IB, Stock, MarketOrder, Order, util, PriceCondition, OrderStatus

# ===== Positions Treeview sorting (module-level & persistent) =====
def _pos_is_numeric_col(col):
    return col in ("position","mkt_px","avg_cost","unrl","unrl_pct","pct_netliq","idx")

def _pos_parse_value(col, val):
    s = "" if val is None else str(val).strip()
    if col == "symbol":
        return s.upper()
    s = s.replace(",", "")
    if s.endswith("%"):
        s = s[:-1]
    if s.startswith("$"):
        s = s[1:]
    try:
        return float(s)
    except Exception:
        return float("-inf")

def _pos_apply_sort(tv):
    col = getattr(tv, "_pos_sort_key", None)
    if not col:
        return
    default_reverse = True if _pos_is_numeric_col(col) and col != "symbol" else False
    reverse = getattr(tv, "_pos_sort_reverse", default_reverse)
    try:
        items = list(tv.get_children(""))
        parsed = []
        for iid in items:
            val = tv.set(iid, col)
            key = _pos_parse_value(col, val)
            parsed.append((key, iid))
        parsed.sort(reverse=reverse, key=lambda x: (x[0],))
        for idx, (_k, iid) in enumerate(parsed):
            tv.move(iid, "", idx)
    except Exception:
        pass

def _pos_on_header(tv, col):
    default_reverse = True if _pos_is_numeric_col(col) and col != "symbol" else False
    if getattr(tv, "_pos_sort_key", None) == col:
        tv._pos_sort_reverse = not getattr(tv, "_pos_sort_reverse", default_reverse)
    else:
        tv._pos_sort_key = col
        tv._pos_sort_reverse = default_reverse
    _pos_apply_sort(tv)
# ================================================================

# ===== 時區 =====
US_EASTERN = tz.gettz('America/New_York')
HK_TZ      = tz.gettz('Asia/Hong_Kong')

POLL_SECONDS = 1
CHECK_STREAM_TIMEOUT = 3.0
HIST_MAX_RETRY = 3
HIST_RETRY_SLEEP_SEC = 4

def set_et_mode(mode: str):
    global US_EASTERN
    if mode == 'edt':
        US_EASTERN = tz.tzoffset('EDT', -4*3600)
    elif mode == 'est':
        US_EASTERN = tz.tzoffset('EST', -5*3600)
    else:
        US_EASTERN = tz.gettz('America/New_York')

def now_et():
    return datetime.now(timezone.utc).astimezone(US_EASTERN)

def today_et():
    return now_et().date()

def us_rth_window(date_):
    start = datetime(date_.year, date_.month, date_.day, 9, 30, tzinfo=US_EASTERN)
    end   = datetime(date_.year, date_.month, date_.day, 16, 0, tzinfo=US_EASTERN)
    return start, end

def first_candle_window(date_, tf_min: int):
    s, _ = us_rth_window(date_)
    return s, s + timedelta(minutes=tf_min)

def format_rth_for_ui(date_):
    s_et, e_et = us_rth_window(date_)
    s_hk = s_et.astimezone(HK_TZ); e_hk = e_et.astimezone(HK_TZ)
    return f"RTH：{s_et.strftime('%H:%M')}–{e_et.strftime('%H:%M')} ET（{s_hk.strftime('%H:%M')}–{e_hk.strftime('%H:%M')} HKT）"

def human_usd(n):
    try:
        if n is None or (isinstance(n, float) and (n != n)):
            return "—"
        return f"${float(n):,.2f}"
    except Exception:
        return str(n) if n is not None else "—"

# ============ IB 包裝 ============
class IBWrapper:
    def __init__(self, on_status, on_log):
        self.ib = IB()
        try:
            self._init_streaming_handlers()
        except Exception:
            pass
        self._on_status = on_status
        self._on_log = on_log
        self.last_error = ""
        self.account = None
        self.pnl_sub = None

        self.ib.errorEvent += self._on_ib_error
        self.ib.disconnectedEvent += self._on_disconnected

    def _on_disconnected(self):
        self._status("IB 已斷線。")

    def _status(self, text):
        try: self._on_status(text)
        except: pass
        self._log(text)

    def _log(self, text):
        if self._on_log:
            try: self._on_log(text)
            except: pass
        print("[LOG]", text, flush=True)

    def _on_ib_error(self, reqId, errorCode, errorString, contract):
        msg = f"IB Error [{errorCode}] {errorString}"
        self.last_error = msg
        self._status(msg)

    # --- 連線/斷線 ---
    def connect_manual(self, host, port, client_id):
        try:
            if self.ib.isConnected():
                self.ib.disconnect()
        except: pass
        try:
            self._status(f"嘗試連線 {host}:{port}（ClientID={client_id}）…")
            self.ib.connect(host, port, clientId=client_id, timeout=20)
            if not self.ib.isConnected():
                raise RuntimeError("連線未建立")
            accts = self.ib.managedAccounts()
            if not accts: raise RuntimeError("未取得帳戶列表")
            self.account = accts[0]
            self._status(f"已連線，帳戶：{self.account}")
            self.subscribe_pnl()
            try:
                self.start_account_updates()
            except Exception:
                pass
            try:
                for _p in self.ib.portfolio():
                    try:
                        self.ensure_ticker_sub(getattr(_p, 'contract', None))
                    except Exception:
                        pass
            except Exception:
                pass
            return True
        except Exception as e:
            self.last_error = f"連線失敗：{e}"
            self._status(self.last_error)
            try:
                if self.ib.isConnected(): self.ib.disconnect()
            except: pass
            return False

    def disconnect_manual(self):
        try:
            self.stop_account_updates()
        except Exception:
            pass
        try:
            self.clear_stream_caches()
        except Exception:
            pass
        try:
            if self.pnl_sub:
                self.ib.cancelPnL(self.pnl_sub)
        except: pass
        self.pnl_sub = None
        try:
            if self.ib.isConnected(): self.ib.disconnect()
        except: pass
        self._status("已手動斷線。")

    # ---- 市場數據檢查 ----
    def check_market_data(self, symbol: str = "AAPL", timeout: float = CHECK_STREAM_TIMEOUT):
        if not self.ib.isConnected():
            return "未連線 IB。", False, "—"
        try:
            symbol = symbol.upper()
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)

            self._log(f"[MDCHK] 試LIVE串流：{symbol}")
            self.ib.reqMarketDataType(1)
            ticker = self.ib.reqMktData(contract, "", False, False)
            self.ib.sleep(timeout)

            def has_values(t):
                return any([
                    (t.bid is not None and not util.isNan(t.bid)),
                    (t.ask is not None and not util.isNan(t.ask)),
                    (t.last is not None and not util.isNan(t.last)),
                    (t.close is not None and not util.isNan(t.close)),
                    (t.volume is not None and not util.isNan(t.volume)),
                ])
            ttype = getattr(ticker, "marketDataType", None)
            type_name = {1:"LIVE", 2:"FROZEN", 3:"DELAYED", 4:"DELAYED_FROZEN"}.get(ttype, str(ttype))

            if has_values(ticker) and ttype == 1:
                try: self.ib.cancelMktData(ticker)
                except: pass
                return f"✅ {symbol}：LIVE 串流正常。", True, type_name

            try: self.ib.cancelMktData(ticker)
            except: pass
            self._log(f"[MDCHK] 試DELAYED串流：{symbol}")
            self.ib.reqMarketDataType(3)
            ticker2 = self.ib.reqMktData(contract, "", False, False)
            self.ib.sleep(timeout)
            ttype2 = getattr(ticker2, "marketDataType", None)
            type_name2 = {1:"LIVE", 2:"FROZEN", 3:"DELAYED", 4:"DELAYED_FROZEN"}.get(ttype2, str(ttype2))
            ok2 = has_values(ticker2) and ttype2 in (3, 4)
            try: self.ib.cancelMktData(ticker2)
            except: pass

            if ok2:
                return f"⚠️ {symbol}：只有延遲數據（{type_name2}）。", False, type_name2

            msg = self.last_error or "未收到任何市場數據。可能未訂閱或交易所未授權。"
            return f"❌ {symbol}：{msg}", False, type_name2
        except Exception as e:
            return f"檢查失敗：{e}", False, "—"

    # ---- RT Bars ----
    def start_rt_bars(self, symbol: str):
        symbol = symbol.upper()
        contract = Stock(symbol, 'SMART', 'USD')
        self.ib.qualifyContracts(contract)
        self.ib.reqMarketDataType(1)
        self._log(f"[RT] start reqRealTimeBars {symbol} (5s TRADES useRTH=True)")
        bars = self.ib.reqRealTimeBars(contract, barSize=5, whatToShow='TRADES', useRTH=True)
        return bars

    def cancel_rt_bars(self, bars):
        try:
            self._log("[RT] cancelRealTimeBars")
            self.ib.cancelRealTimeBars(bars)
        except Exception as e:
            self._log(f"[RT] cancel失敗：{e}")

    def _rth_0930_1600_for_day(self, y, m, d):
        o_dt = datetime(y, m, d, 9, 30, tzinfo=US_EASTERN)
        c_dt = datetime(y, m, d,16,  0, tzinfo=US_EASTERN)
        return o_dt, c_dt

    def next_rth_open_close(self, symbol: str, from_dt_et: datetime):
        try:
            contract = Stock(symbol.upper(), 'SMART', 'USD')
            cds = self.ib.reqContractDetails(contract)
            if not cds:
                raise RuntimeError("無合約細節")
            th = cds[0].tradingHours or ""
            parts = [p for p in th.split(';') if p]

            for p in parts:
                if 'CLOSED' in p:
                    continue
                date_str, _times_str = p.split(':', 1)
                y = int(date_str[0:4]); m = int(date_str[4:6]); d = int(date_str[6:8])
                o_dt, c_dt = self._rth_0930_1600_for_day(y, m, d)
                if from_dt_et <= c_dt:
                    if from_dt_et.time() > dtime(16, 0) and from_dt_et.date() == o_dt.date():
                        continue
                    return (o_dt, c_dt)
        except Exception as e:
            self._log(f"[HOURS] next 讀取/解析 tradingHours 失敗：{e}；使用固定 RTH fallback")

        d = from_dt_et.date()
        if from_dt_et.time() >= dtime(16,0):
            d = d + timedelta(days=1)
        while d.weekday() >= 5:
            d += timedelta(days=1)
        return self._rth_0930_1600_for_day(d.year, d.month, d.day)

    
    def prev_rth_open_close(self, symbol: str, from_dt_et: datetime):
        try:
            contract = Stock(symbol.upper(), 'SMART', 'USD')
            cds = self.ib.reqContractDetails(contract)
            if not cds:
                raise RuntimeError("無合約細節")
            th = cds[0].tradingHours or ""
            parts = [p for p in th.split(';') if p]

            in_session = None
            candidate_prev = None
            for p in parts:
                if 'CLOSED' in p:
                    continue
                date_str, _times_str = p.split(':', 1)
                y = int(date_str[0:4]); m = int(date_str[4:6]); d = int(date_str[6:8])
                o_dt, c_dt = self._rth_0930_1600_for_day(y, m, d)

                if o_dt <= from_dt_et <= c_dt:
                    in_session = (o_dt, c_dt)
                    break

                if c_dt <= from_dt_et:
                    candidate_prev = (o_dt, c_dt)

            if in_session:
                return in_session
            if candidate_prev:
                return candidate_prev
        except Exception as e:
            self._log(f"[HOURS] prev 讀取/解析 tradingHours 失敗：{e}；使用固定 RTH fallback")

        d = from_dt_et.date() - timedelta(days=1)
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        return self._rth_0930_1600_for_day(d.year, d.month, d.day)

    def get_min_tick(self, contract_or_symbol) -> float:
        try:
            if isinstance(contract_or_symbol, str):
                c = Stock(contract_or_symbol.upper(), 'SMART', 'USD')
                self.ib.qualifyContracts(c)
            else:
                c = contract_or_symbol
            cds = self.ib.reqContractDetails(c)
            if cds and getattr(cds[0], "minTick", None):
                mt = float(cds[0].minTick)
                self._log(f"[TICK] {getattr(c,'symbol','?')} minTick = {mt}")
                return mt
        except Exception as e:
            self._log(f"[TICK] 讀取 minTick 失敗：{e}")
        return 0.01

    def hist_first_candle(self, symbol: str, tf_min: int, date_):
        if not self.ib.isConnected():
            return None, None, "IB 未連線"
        symbol = symbol.upper()
        contract = Stock(symbol, 'SMART', 'USD')
        self.ib.qualifyContracts(contract)

        start_et, _ = first_candle_window(date_, tf_min)
        end_et = start_et + timedelta(minutes=30)
        first_start = start_et
        first_end = start_et + timedelta(minutes=tf_min)

        for attempt in range(1, HIST_MAX_RETRY + 1):
            t0 = time.monotonic()
            try:
                self._log(f"[HIST] 第{attempt}次請求 {symbol} tf={tf_min}m end={end_et.strftime('%Y-%m-%d %H:%M:%S ET')}")
                bars = self.ib.reqHistoricalData(
                    contract,
                    endDateTime=end_et,
                    durationStr="1800 S",
                    barSizeSetting="1 min" if tf_min == 1 else "5 mins",
                    whatToShow="TRADES",
                    useRTH=True,
                    formatDate=2,
                    keepUpToDate=False,
                )
                cost = time.monotonic() - t0
                self._log(f"[HIST] 收到 {len(bars)} 根bar，用時 {cost:.2f}s")
                if not bars:
                    raise RuntimeError("HMDS 無回應或沒有bar")

                hhi = None; hlo = None
                for b in bars:
                    bt = b.date if isinstance(b.date, datetime) else datetime.strptime(b.date, "%Y%m%d %H:%M:%S")
                    bt = bt.replace(tzinfo=US_EASTERN) if bt.tzinfo is None else bt.astimezone(US_EASTERN)
                    if first_start <= bt < first_end:
                        hhi = b.high if hhi is None else max(hhi, b.high)
                        hlo = b.low  if hlo is None else min(hlo,  b.low)

                if (hhi is not None) and (hlo is not None):
                    self._log(f"[HIST] 完成：H={hhi:.4f} L={hlo:.4f}")
                    return float(hhi), float(hlo), "OK (IB Hist)"

                for b in bars:
                    bt = b.date if isinstance(b.date, datetime) else datetime.strptime(b.date, "%Y%m%d %H:%M:%S")
                    bt = bt.replace(tzinfo=US_EASTERN) if bt.tzinfo is None else bt.astimezone(US_EASTERN)
                    if bt >= first_start:
                        self._log(f"[HIST] 後備匹配：H={b.high:.4f} L={b.low:.4f} @ {bt.strftime('%H:%M:%S')}")
                        return float(b.high), float(b.low), "OK (IB Hist ~ fallback)"
            except Exception as e:
                cost = time.monotonic() - t0
                self._log(f"[HIST] 失敗（{e}）用時 {cost:.2f}s；{('準備重試' if attempt < HIST_MAX_RETRY else '已達上限')}")
            if attempt < HIST_MAX_RETRY:
                time.sleep(HIST_RETRY_SLEEP_SEC)

        return None, None, "IB 歷史錯誤或HMDS未回應"

    def first_candle_any(self, symbol: str, tf_min: int, date_):
        first_start_et, first_end_et = first_candle_window(date_, tf_min)
        if now_et() >= first_end_et:
            fh, fl, msg = self.hist_first_candle(symbol, tf_min, date_)
            return fh, fl, "IB 歷史", msg

        bars = None
        try:
            bars = self.start_rt_bars(symbol)
            hi = lo = None
            deadline = first_end_et
            while now_et() < deadline:
                time.sleep(0.25)
                for b in list(bars):
                    t_et = b.time.replace(tzinfo=timezone.utc).astimezone(US_EASTERN)
                    if first_start_et <= t_et < first_end_et:
                        hi = b.high if hi is None else max(hi, b.high)
                        lo = b.low  if lo is None else min(lo,  b.low)
            self.cancel_rt_bars(bars)
            if hi is not None and lo is not None:
                return float(hi), float(lo), "IB 串流", "OK (IB RT Bars)"
            fh, fl, msg = self.hist_first_candle(symbol, tf_min, date_)
            return fh, fl, "IB 歷史", f"RT 未齊；回補：{msg}"
        except Exception as e:
            try:
                if bars: self.cancel_rt_bars(bars)
            except: pass
            fh, fl, msg = self.hist_first_candle(symbol, tf_min, date_)
            return fh, fl, "IB 歷史", f"RT 例外：{e}；回補：{msg}"

    def subscribe_pnl(self):
        if not self.ib.isConnected() or not self.account:
            return
        try:
            self.pnl_sub = self.ib.reqPnL(self.account, "")
            self._status("已訂閱 P&L（Daily / Unrealized / Realized）")
        except Exception as e:
            self._status(f"訂閱 P&L 失敗：{e}")

    def pnl_values(self):
        daily = unreal = realized = None
        try:
            if self.pnl_sub:
                daily = self.pnl_sub.dailyPnL
                unreal = self.pnl_sub.unrealizedPnL
                realized = self.pnl_sub.realizedPnL
        except Exception as e:
            self._status(f"P&L 讀取失敗：{e}")
        return daily, unreal, realized

    # --- MODIFIED: Return extra account fields ---
    def account_values(self):
        netliq = cash = excess = mktval = buypower = None
        try:
            summary = {v.tag: v.value for v in self.ib.accountSummary()}
            netliq = float(summary.get("NetLiquidation", "0") or 0)
            cash = float(summary.get("TotalCashValue", "0") or 0)
            excess = float(summary.get("ExcessLiquidity", "0") or 0)
            mktval = float(summary.get("GrossPositionValue", "0") or 0)
            buypower = float(summary.get("BuyingPower", "0") or 0)
        except Exception as e:
            self._status(f"讀取帳戶摘要失敗：{e}")
        return netliq, cash, excess, mktval, buypower

    def portfolio_rows(self):
        rows = []
        try:
            src_iter = list(getattr(self, 'portfolio_cache', {}).values()) or list(self.ib.portfolio())
            for p in src_iter:
                contract = (p.get("contract") if isinstance(p, dict) else getattr(p, "contract", None))
                sym = (getattr(contract, "symbol", None) if contract is not None else (p.get("symbol") if isinstance(p, dict) else None)) or "—"
                position = float((p.get("position") if isinstance(p, dict) else getattr(p, "position", getattr(p, "pos", 0))) or 0)
                mkt = (p.get("marketPrice") if isinstance(p, dict) else getattr(p, "marketPrice", None))
                avg = (p.get("averageCost") if isinstance(p, dict) else getattr(p, "averageCost", getattr(p, "avgCost", None)))
                unrl = (p.get("unrealizedPNL") if isinstance(p, dict) else getattr(p, "unrealizedPNL", None))

                try:
                    cid = int(getattr(contract, "conId", 0) or 0)
                    tc = getattr(self, "ticker_cache", {}).get(cid) if cid else None
                    if tc and (tc.get("price") is not None):
                        mkt = tc["price"]
                except Exception:
                    pass

                try:
                    unrl_val = float(unrl) if unrl is not None else 0.0
                except Exception:
                    unrl_val = 0.0
                try:
                    denom = (abs(avg or 0.0) * abs(position))
                    pct = (unrl_val / denom * 100.0) if denom else 0.0
                except Exception:
                    pct = 0.0

                mv = abs(position) * (float(mkt) if mkt is not None else (float(avg) if avg is not None else 0.0))
                rows.append({
                    "symbol": sym,
                    "position": position,
                    "mkt_px": mkt,
                    "avg_cost": avg,
                    "unrealized": unrl_val,
                    "unrealized_pct": pct,
                    "market_value": mv
                })
        except Exception as e:
            self._status(f"讀取持倉失敗：{e}")
        return rows

    def get_position_qty(self, symbol: str):
        qty = 0
        try:
            for p in self.ib.portfolio():
                if getattr(p.contract, "symbol", "") == symbol.upper():
                    qty = int(round(float(getattr(p, "position", 0) or 0)))
                    break
        except Exception:
            qty = 0
        return qty

    def open_orders_all(self):
        try:
            self.ib.reqAllOpenOrders()
            self.ib.sleep(0.5)
            trades = list(self.ib.trades())
            open_like = []
            for t in trades:
                st = t.orderStatus
                status_txt = st.status if isinstance(st, OrderStatus) else getattr(st, "status", "")
                if status_txt and status_txt.lower() in ("filled","cancelled"):
                    continue
                open_like.append(t)
            return open_like
        except Exception as e:
            self._status(f"讀取 Open Orders 失敗：{e}")
            return []

    def cancel_order_by_id(self, order_id: int):
        try:
            trades = list(self.ib.trades())
            target = None
            for t in trades:
                if t.order and t.order.orderId == order_id:
                    target = t
                    break
            if not target:
                return False, f"找不到 OrderId={order_id} 的訂單"
            self.ib.cancelOrder(target.order)
            self._status(f"已送出取消指令：OrderId={order_id}")
            return True, f"已嘗試取消 OrderId={order_id}"
        except Exception as e:
            return False, f"取消失敗：{e}"

    # --- NEW: Native Stop Order Function (with Price Rounding) ---
    def place_stop_order(self, symbol: str, base_price: float, qty: int, action: str = 'BUY', strict: bool = False):
        """
        使用交易所原生 Stop Order (STP)。
        strict=True 時：BUY 單價格 +1 minTick；SELL 單價格 -1 minTick。
        修復：針對股價 > $1 的美股，強制四捨五入至 2 位小數，避免 Error 110。
        """
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            qualified = self.ib.qualifyContracts(contract)
            if not qualified: return False, "合約資格化失敗"
            q = qualified[0]

            adj_price = float(base_price)
            
            if strict:
                mt = self.get_min_tick(q)
                # === 修正邏輯開始 ===
                # 如果 minTick 小於 0.01 但股價大於 1.00，強制改用 0.01 以符合美股規則
                if mt < 0.01 and adj_price > 1.00:
                    mt = 0.01
                
                if action.upper() == 'BUY':
                    adj_price += mt
                elif action.upper() == 'SELL':
                    adj_price -= mt
            
            # === 安全過濾：再次強制 Round 到 2 位小數 (如果股價 > 1) ===
            if adj_price > 1.00:
                adj_price = round(adj_price, 2)
            # ====================

            order = Order()
            order.action = action.upper()
            order.orderType = 'STP'      # Stop Order
            order.auxPrice = adj_price   # Trigger Price
            order.totalQuantity = qty
            order.tif = 'GTC'            # Good-Til-Canceled

            self.ib.placeOrder(q, order)
            self.ib.sleep(0.2)
            
            strict_msg = " (Strict)" if strict else ""
            return True, f"已提交 {action} STOP 單{strict_msg}（{symbol} 觸發價 {adj_price}, 數量 {qty}, GTC）"
        except Exception as e:
            return False, f"Stop 單建立失敗：{e}"

    # (保留舊方法以防萬一，但不會被呼叫)
    def place_conditional_mkt_order(self, symbol: str, trigger_price: float, qty: int,
                                    action: str = 'BUY', is_more: bool = True,
                                    strict: bool = False):
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            qualified = self.ib.qualifyContracts(contract)
            if not qualified: return False, "合約資格化失敗"
            q = qualified[0]

            adj_price = float(trigger_price)
            if strict:
                mt = self.get_min_tick(q)
                if is_more:  # 嚴格「大於」
                    adj_price = float(trigger_price) + mt
                else:        # 嚴格「小於」
                    adj_price = float(trigger_price) - mt

            order = MarketOrder(action, qty)
            
            # 修改處：將 triggerMethod=0 改為 triggerMethod=3 (Last Price)
            # 0=Default, 1=Double Last, 2=Double Bid/Ask, 3=Last, 4=Bid/Ask, 7=Last or Bid/Ask
            cond = PriceCondition(isMore=is_more, price=adj_price, conId=q.conId, exch='SMART', triggerMethod=0)
            order.conditions = [cond]
            order.conditionsIgnoreRth = False
            order.conditionsCancelOrder = False
            self.ib.placeOrder(q, order)
            self.ib.sleep(0.2)
            dir_txt = "嚴格向上突破 >" if (is_more and strict) else ("向上突破 >=" if is_more else ("嚴格跌破 <" if strict else "跌破 <="))
            return True, f"已提交{action}條件市價單（{symbol} {dir_txt}{adj_price:.4f}，數量 {qty}）"
        except Exception as e:
            return False, f"條件單建立失敗：{e}"

# ============ 監察項目 ============

class WatchItem:
    def __init__(self, symbol, tf_min, source, open_dt_et=None, close_dt_et=None):
        self.symbol = symbol.upper()
        self.tf_min = tf_min
        self.source = source
        self.status = "等待中"
        self.first_high = None
        self.first_low = None   # ← 修正縮排
        self.qty = 0
        self.active = True
        self.open_dt_et  = open_dt_et
        self.close_dt_et = close_dt_et

# ============ 主 GUI ============
class App(tk.Tk):

    def sell_half_now(self):
        """
        立即以市價賣出當前選中/偵測到的持倉數量的一半（向下取整）。
        需要：已連線 IB、該 Symbol 有持倉。
        """
        try:
            if not self.ibw.ib.isConnected():
                messagebox.showwarning("提示","請先連線 IB。")
                return

            # 以輸入框為主；如無，從投資組合取第一個
            symbol = (self.ent_symbol.get().strip().upper() if hasattr(self, "ent_symbol") else "") or None
            if not symbol and hasattr(self.ibw, "portfolio_rows"):
                rows = self.ibw.portfolio_rows()
                if rows:
                    symbol = rows[0].get("symbol")
            if not symbol:
                self._append_log("[SELL HALF] 未能判定 Symbol。")
                try:
                    messagebox.showinfo("提示", "請先輸入/選擇要賣出的股票代號。")
                except Exception:
                    pass
                return

            # 讀持倉數量
            qty_pos = 0
            if hasattr(self.ibw, "get_position_qty"):
                qty_pos = int(self.ibw.get_position_qty(symbol) or 0)
            else:
                # 後備：從 portfolio_rows 搜索
                rows = self.ibw.portfolio_rows()
                for r in rows:
                    if r.get("symbol") == symbol:
                        qty_pos = int(r.get("position") or 0)
                        break

            if qty_pos <= 0:
                self._append_log(f"[SELL HALF] {symbol} 無持倉。")
                try:
                    messagebox.showinfo("提示", f"{symbol}：目前持倉=0，未執行賣出。")
                except Exception:
                    pass
                return

            half = qty_pos // 2
            if half <= 0:
                self._append_log(f"[SELL HALF] {symbol} 持倉過少，無法賣出一半。")
                try:
                    messagebox.showinfo("提示", f"{symbol}：持倉過少，無法賣出一半。")
                except Exception:
                    pass
                return

            # 發送市價單
            try:
                contract = Stock(symbol, 'SMART', 'USD')
                self.ibw.ib.qualifyContracts(contract)
                order = MarketOrder('SELL', half)
                self.ibw.ib.placeOrder(contract, order)
                self._append_log(f"[SELL HALF] 已送出市價單：SELL {half} 股 {symbol}")
                self._set_status(f"{symbol}：已送出市價單 SELL {half}")
            except Exception as e:
                self._append_log(f"[ERR] 市價賣半倉失敗：{e}")
                try:
                    messagebox.showwarning("錯誤", f"市價賣半倉失敗：{e}")
                except Exception:
                    pass
        except Exception as e:
            try:
                self._append_log(f"[ERR] sell_half_now 例外：{e}")
            except Exception:
                pass
    def sell_third_now(self):
        """
        立即以市價賣出當前選中/偵測到的持倉數量的三分之一（向下取整）。
        """
        try:
            if not self.ibw.ib.isConnected():
                messagebox.showwarning("提示","請先連線 IB。")
                return
            symbol = (self.ent_symbol.get().strip().upper() if hasattr(self, "ent_symbol") else "") or None
            if not symbol and hasattr(self.ibw, "portfolio_rows"):
                rows = self.ibw.portfolio_rows()
                if rows:
                    symbol = rows[0].get("symbol")
            if not symbol:
                self._append_log("[SELL 1/3] 未能判定 Symbol。")
                try: messagebox.showinfo("提示", "請先輸入/選擇要賣出的股票代號。")
                except Exception: pass
                return
            qty_pos = 0
            if hasattr(self.ibw, "get_position_qty"):
                qty_pos = int(self.ibw.get_position_qty(symbol) or 0)
            else:
                rows = self.ibw.portfolio_rows()
                for r in rows:
                    if r.get("symbol") == symbol:
                        qty_pos = int(r.get("position") or 0); break
            if qty_pos <= 0:
                self._append_log(f"[SELL 1/3] {symbol} 無持倉。")
                try: messagebox.showinfo("提示", f"{symbol}：目前持倉=0，未執行賣出。")
                except Exception: pass
                return
            q = qty_pos // 3
            if q <= 0:
                self._append_log(f"[SELL 1/3] {symbol} 持倉過少，無法賣出三分之一。")
                try: messagebox.showinfo("提示", f"{symbol}：持倉過少，無法賣出三分之一。")
                except Exception: pass
                return
            try:
                contract = Stock(symbol, 'SMART', 'USD')
                self.ibw.ib.qualifyContracts(contract)
                order = MarketOrder('SELL', q)
                self.ibw.ib.placeOrder(contract, order)
                self._append_log(f"[SELL 1/3] 已送出市價單：SELL {q} 股 {symbol}")
                self._set_status(f"{symbol}：已送出市價單 SELL {q}")
            except Exception as e:
                self._append_log(f"[ERR] 市價賣出三分一失敗：{e}")
                try: messagebox.showwarning("錯誤", f"市價賣出三分一失敗：{e}")
                except Exception: pass
        except Exception as e:
            try: self._append_log(f"[ERR] sell_third_now 例外：{e}")
            except Exception: pass

    def sell_quarter_now(self):
        """
        立即以市價賣出當前選中/偵測到的持倉數量的四分之一（向下取整）。
        """
        try:
            if not self.ibw.ib.isConnected():
                messagebox.showwarning("提示","請先連線 IB。")
                return
            symbol = (self.ent_symbol.get().strip().upper() if hasattr(self, "ent_symbol") else "") or None
            if not symbol and hasattr(self.ibw, "portfolio_rows"):
                rows = self.ibw.portfolio_rows()
                if rows:
                    symbol = rows[0].get("symbol")
            if not symbol:
                self._append_log("[SELL 1/4] 未能判定 Symbol。")
                try: messagebox.showinfo("提示", "請先輸入/選擇要賣出的股票代號。")
                except Exception: pass
                return
            qty_pos = 0
            if hasattr(self.ibw, "get_position_qty"):
                qty_pos = int(self.ibw.get_position_qty(symbol) or 0)
            else:
                rows = self.ibw.portfolio_rows()
                for r in rows:
                    if r.get("symbol") == symbol:
                        qty_pos = int(r.get("position") or 0); break
            if qty_pos <= 0:
                self._append_log(f"[SELL 1/4] {symbol} 無持倉。")
                try: messagebox.showinfo("提示", f"{symbol}：目前持倉=0，未執行賣出。")
                except Exception: pass
                return
            q = qty_pos // 4
            if q <= 0:
                self._append_log(f"[SELL 1/4] {symbol} 持倉過少，無法賣出四分之一。")
                try: messagebox.showinfo("提示", f"{symbol}：持倉過少，無法賣出四分之一。")
                except Exception: pass
                return
            try:
                contract = Stock(symbol, 'SMART', 'USD')
                self.ibw.ib.qualifyContracts(contract)
                order = MarketOrder('SELL', q)
                self.ibw.ib.placeOrder(contract, order)
                self._append_log(f"[SELL 1/4] 已送出市價單：SELL {q} 股 {symbol}")
                self._set_status(f"{symbol}：已送出市價單 SELL {q}")
            except Exception as e:
                self._append_log(f"[ERR] 市價賣出四分一失敗：{e}")
                try: messagebox.showwarning("錯誤", f"市價賣出四分一失敗：{e}")
                except Exception: pass
        except Exception as e:
            try: self._append_log(f"[ERR] sell_quarter_now 例外：{e}")
            except Exception: pass

    def __init__(self):
        super().__init__()
        self.title("IBKR Bars + IBKR Trading GUI")
        self.geometry("1280x1000")
        self._closing = False
        self._after_ids = set()
        # 將 safe_after / cancel_all_after 以閉包方式掛到實例，避免解析時序問題
        def _safe_after(ms, func, *args, **kwargs):
            if getattr(self, "_closing", False) or not self.winfo_exists():
                return None
            def _wrap():
                if not getattr(self, "_closing", False) and self.winfo_exists():
                    func(*args, **kwargs)
            aid = self.after(ms, _wrap)
            self._after_ids.add(aid)
            return aid
        def _cancel_all_after():
            for aid in list(self._after_ids):
                try:
                    self.after_cancel(aid)
                except Exception:
                    pass
            self._after_ids.clear()
        self.safe_after = _safe_after
        self.cancel_all_after = _cancel_all_after

        # 綁定關窗：改用內部閉包，避免初始化階段屬性解析問題
        def _on_close_inline():
            self._closing = True
            try: self.cancel_all_after()
            except Exception: pass
            try:
                if getattr(self, "ibw", None) and self.ibw.ib.isConnected():
                    self.ibw.ib.disconnect()
            except Exception:
                pass
            self.destroy()
        self.protocol("WM_DELETE_WINDOW", _on_close_inline)

        self.status_msg   = tk.StringVar(value="")
        self.ib_connected = tk.StringVar(value="未連線")

        self.stream_status = tk.StringVar(value="未檢查")
        self.market_data_type = tk.StringVar(value="—")

        self.ibw = IBWrapper(on_status=self._set_status, on_log=self._append_log)
        self.watch_items = {}
        self.watch_threads = {}
        self.open_order_map = {}

        self.nb = ttk.Notebook(self)
        self.page_account = ttk.Frame(self.nb)
        self.page_trade   = ttk.Frame(self.nb)
        self.nb.add(self.page_account, text="賬戶 / 持倉 / 訂單 / P&L")
        self.nb.add(self.page_trade,   text="交易 / 監察（IB 串流/歷史 自動）")
        self.nb.pack(fill="both", expand=True, padx=10, pady=10)

        self._build_account_page()
        self._build_trade_page()

        self.safe_after(1000, self.refresh_account_ui_loop)
        self.safe_after(1000, self.refresh_open_orders_loop)
                # bridge to avoid Tk attribute lookup issues on bound method during __init__
        def _refresh_positions_bridge():
            try:
                type(self).refresh_portfolio_loop(self)
            except Exception as e:
                try: self._append_log(f"[ERR] bridge refresh: {e}")
                except Exception: pass
        self.safe_after(1000, _refresh_positions_bridge)

    # ---- 主線程執行器 ----
    def run_on_main(self, func, *args, **kwargs):
        if threading.current_thread() is threading.main_thread():
            return func(*args, **kwargs)
        res = {}
        ev = threading.Event()
        def wrapper():
            try:
                res['value'] = func(*args, **kwargs)
            except Exception as e:
                res['error'] = e
            finally:
                ev.set()
        self.after(0, wrapper)
        ev.wait()
        if 'error' in res:
            raise res['error']
        return res.get('value')

    # ---- Debug/Log 視窗 ----
    def _append_log(self, text):
        ts = datetime.now(timezone.utc).astimezone(HK_TZ).strftime("%H:%M:%S")
        line = f"[{ts}] {text}\n"
        def _do():
            self.txt_log.configure(state='normal')
            self.txt_log.insert('end', line)
            self.txt_log.see('end')
            self.txt_log.configure(state='disabled')
        self.after(0, _do)

    def log(self, text):
        print("[UI-LOG]", text, flush=True)
        self._append_log(text)

    # ---- 帳戶頁（UI 不變）----
    def _build_account_page(self):
        top = ttk.Frame(self.page_account); top.pack(fill="x", pady=6)
        ttk.Label(top, text="IB：").pack(side="left")
        ttk.Label(top, textvariable=self.ib_connected, foreground="green").pack(side="left", padx=4)

        conn = ttk.LabelFrame(self.page_account, text="手動連線 IB Gateway")
        conn.pack(fill="x", pady=6)
        ttk.Label(conn, text="Host：").grid(row=0, column=0, padx=6, pady=6, sticky="e")
        self.ent_host = ttk.Entry(conn, width=15); self.ent_host.insert(0, "127.0.0.1")
        self.ent_host.grid(row=0, column=1, padx=6, pady=6, sticky="w")
        ttk.Label(conn, text="Port：").grid(row=0, column=2, padx=6, pady=6, sticky="e")
        self.cmb_port = ttk.Combobox(conn, values=["4001 (Live)", "4002 (Paper)", "自訂"], width=15, state="readonly")
        self.cmb_port.current(0); self.cmb_port.grid(row=0, column=3, padx=6, pady=6, sticky="w")
        self.ent_port_custom = ttk.Entry(conn, width=8, state="disabled"); self.ent_port_custom.grid(row=0, column=4, padx=6, pady=6, sticky="w")
        def on_port_change(_):
            if self.cmb_port.get()=="自訂": self.ent_port_custom.config(state="normal")
            else:
                self.ent_port_custom.delete(0, tk.END); self.ent_port_custom.config(state="disabled")
        self.cmb_port.bind("<<ComboboxSelected>>", on_port_change)
        ttk.Label(conn, text="Client ID：").grid(row=0, column=5, padx=6, pady=6, sticky="e")
        self.ent_client_id = ttk.Entry(conn, width=8); self.ent_client_id.insert(0, "101")
        self.ent_client_id.grid(row=0, column=6, padx=6, pady=6, sticky="w")
        ttk.Button(conn, text="連線 IB（並訂閱P&L）", command=self.connect_ib_manual).grid(row=0, column=7, padx=10)
        ttk.Button(conn, text="斷線 IB", command=self.disconnect_ib_manual).grid(row=0, column=8, padx=4)

        frame = ttk.LabelFrame(self.page_account, text="帳戶摘要（USD）")
        frame.pack(fill="x", pady=6)
        self.lbl_netliq = ttk.Label(frame, text="NetLiq：—")
        self.lbl_cash   = ttk.Label(frame, text="現金餘額：—")
        # --- MODIFIED: Added labels for Excess, MKT Val, Buying Power ---
        self.lbl_excess = ttk.Label(frame, text="Excess Liq：—")
        self.lbl_mktval = ttk.Label(frame, text="MKT Val：—")
        self.lbl_buypwr = ttk.Label(frame, text="Buying Power：—")
        self.lbl_daily  = ttk.Label(frame, text="今日 P&L（Daily）：—")
        self.lbl_unrl   = ttk.Label(frame, text="未實現 P&L（Unrealized）：—")
        self.lbl_real   = ttk.Label(frame, text="已實現 P&L（Realized 今日）：—")
        
        self.lbl_netliq.pack(anchor="w", padx=10, pady=2)
        self.lbl_cash.pack(anchor="w", padx=10, pady=2)
        self.lbl_excess.pack(anchor="w", padx=10, pady=2) # New
        self.lbl_mktval.pack(anchor="w", padx=10, pady=2) # New
        self.lbl_buypwr.pack(anchor="w", padx=10, pady=2) # New
        self.lbl_daily.pack(anchor="w", padx=10, pady=2)
        self.lbl_unrl.pack(anchor="w", padx=10, pady=2)
        self.lbl_real.pack(anchor="w", padx=10, pady=2)

        table = ttk.LabelFrame(self.page_account, text="持倉（未實現損益/比例、%NetLiq）")
        table.pack(fill="both", expand=True, pady=6)
        cols=("idx","symbol","position","mkt_px","avg_cost","unrl","unrl_pct","pct_netliq")
        self.tree = ttk.Treeview(table, columns=cols, show="headings", height=12)
        heads={"idx":"#","symbol":"Symbol","position":"Qty","mkt_px":"Mkt Px","avg_cost":"Avg Cost","unrl":"Unrealized $","unrl_pct":"Unrealized %","pct_netliq":"%NetLiq"}
        for k,v in heads.items():
            base_w = 28 if k=="idx" else 90
            self.tree.heading(k, text=v, command=lambda c=k, tv=self.tree: _pos_on_header(tv, c)); self.tree.column(k, width=base_w, anchor="center", stretch=False)
        self.tree.pack(fill="both", expand=True)

        oo = ttk.LabelFrame(self.page_account, text="未執行訂單（Open Orders｜全帳戶）")
        oo.pack(fill="both", expand=True, pady=6)
        self.lf_oo = oo  # for live count title update

        ops = ttk.Frame(oo); ops.pack(fill="x", padx=6, pady=4)
        ttk.Button(ops, text="取消選中訂單", command=self.cancel_selected_open_order).pack(side="left")

        # === MODIFIED: Added "stop_px" column ===
        self.tree_oo = ttk.Treeview(oo, selectmode="browse", columns=("idx","id","symbol","action","qty","type","status","tif","aux","stop_px"), show="headings", height=8)
        for k, v in [("idx","#"),("id","OrderId"),("symbol","Symbol"),("action","Action"),("qty","Qty"),("type","Type"),("status","Status"),("tif","TIF"),("aux","Trigger"),("stop_px","Stop Price")]:
            base_w = 28 if k=="idx" else 80
            self.tree_oo.heading(k, text=v); self.tree_oo.column(k, width=base_w, anchor="center", stretch=False)
        self.tree_oo.pack(fill="both", expand=True)
        # 保留未執行訂單的選取狀態
        self._oo_selected_id = None
        self.tree_oo.bind("<<TreeviewSelect>>", self._on_oo_select)


    # ---- 交易頁（UI 不變）----
    
    # ---- Tree column auto-fit (minimal width to show content) ----
    def _autosize_tree_columns(self, tree, min_widths=None, max_widths=None, pad=14):
        try:
            fnt = tkfont.nametofont('TkDefaultFont')
        except Exception:
            fnt = None
        cols = tree['columns']
        # build per-column min/max map
        min_widths = min_widths or {}
        max_widths = max_widths or {}
        # include heading text width as baseline
        for c in cols:
            heading = tree.heading(c).get('text', c)
            w = fnt.measure(heading) if fnt else max(40, len(str(heading))*7)
            width = max(min_widths.get(c, 28), w + pad)
            # scan cell values
            for item in tree.get_children(''):
                try:
                    vals = tree.item(item, 'values')
                    # align by index in columns
                    if c in cols:
                        idx = cols.index(c)
                        if idx < len(vals):
                            s = str(vals[idx])
                            wv = fnt.measure(s) if fnt else max(24, len(s)*7)
                            width = max(width, wv + pad)
                except Exception:
                    pass
            # clamp
            width = max(width, min_widths.get(c, 28))
            if c in max_widths:
                width = min(width, max_widths[c])
            tree.column(c, width=int(width), stretch=False)

    def _build_trade_page(self):
        form = ttk.LabelFrame(self.page_trade, text="下單條件設定（IB 串流/歷史｜自動等開市，首支 K 線觸發）")
        form.pack(fill="x", pady=6)

        ttk.Label(form, text="股票代號：").grid(row=0, column=0, padx=8, pady=6, sticky="e")
        self.ent_symbol = ttk.Entry(form, width=12); self.ent_symbol.grid(row=0, column=1, padx=4, pady=6, sticky="w")

        ttk.Label(form, text="時間框：").grid(row=0, column=2, padx=8, pady=6, sticky="e")
        self.tf_choice = tk.StringVar(value="1")
        ttk.Radiobutton(form, text="1 分鐘", variable=self.tf_choice, value="1").grid(row=0, column=3, padx=4)
        ttk.Radiobutton(form, text="5 分鐘", variable=self.tf_choice, value="5").grid(row=0, column=4, padx=4)
        ttk.Radiobutton(form, text="30 分鐘", variable=self.tf_choice, value="30").grid(row=0, column=5, padx=4)

        ttk.Button(form, text="檢查串流（IB）", command=self.check_streaming_status).grid(row=0, column=10, padx=12)
        ttk.Button(form, text="獲取 K 線（IB 串流/歷史）", command=self.fetch_first_candle_stream).grid(row=1, column=10, padx=12)
        # === 新增：收市前2分鐘監察（SMA10 不含今日）＋ 半倉賣出 ===
        ttk.Button(form, text="開始監察（收市前2分鐘｜SMA10不含今日）", command=lambda s=self: s.start_close_monitor_sma10()).grid(row=0, column=11, padx=8, pady=6, sticky="w")
        ttk.Button(form, text="市價賣出一半倉位", command=self.sell_half_now).grid(row=1, column=11, padx=12, pady=6, sticky="w")
        ttk.Button(form, text="市價賣出三分一倉位", command=self.sell_third_now).grid(row=2, column=11, padx=8, pady=6, sticky="w")
        ttk.Button(form, text="市價賣出四分一倉位", command=self.sell_quarter_now).grid(row=3, column=11, padx=8, pady=6, sticky="w")
        ttk.Button(form, text="開始監察（IB 串流/歷史）", command=self.start_watch_auto).grid(row=2, column=10, padx=12)

        ttk.Button(form, text="放置 賣出條件單（首支低點『嚴格跌破』）", command=self.place_sell_break_selected_tf).grid(row=3, column=10, padx=12, pady=6, sticky="w")

        info = ttk.LabelFrame(self.page_trade, text="首支 K 線 / 下單資訊（最新）")
        info.pack(fill="x", pady=6)
        self.lbl_date  = ttk.Label(info, text="交易日：—")
        self.lbl_hilo  = ttk.Label(info, text="首支 K 線 High/Low：—")
        self.lbl_range = ttk.Label(info, text="Range：—")
        self.lbl_qty   = ttk.Label(info, text="建議股數（0.5% 資金）：—")
        self.lbl_cap_pct = ttk.Label(info, text="預估資金佔用（%NetLiq）：—")
        self.lbl_source= ttk.Label(info, text="資料來源：—")
        self.lbl_note  = ttk.Label(info, text="", foreground="orange")
        for r, w in enumerate([self.lbl_date, self.lbl_hilo, self.lbl_range, self.lbl_qty, self.lbl_cap_pct, self.lbl_source, self.lbl_note]):
            w.grid(row=r, column=0, columnspan=10, sticky="w", padx=10, pady=2)

        md = ttk.LabelFrame(self.page_trade, text="市場數據串流狀態（IB）")
        md.pack(fill="x", pady=6)
        ttk.Label(md, text="狀態：").grid(row=0, column=0, padx=6, pady=6, sticky="e")
        ttk.Label(md, textvariable=self.stream_status, foreground="blue").grid(row=0, column=1, padx=6, pady=6, sticky="w")
        ttk.Label(md, text="Market Data Type：").grid(row=0, column=2, padx=6, pady=6, sticky="e")
        ttk.Label(md, textvariable=self.market_data_type).grid(row=0, column=3, padx=6, pady=6, sticky="w")

        #（保留：DST 手動切換）
        tzbox = ttk.LabelFrame(self.page_trade, text="時區模式（ET）")
        tzbox.pack(fill="x", pady=6)
        self.et_mode = tk.StringVar(value="auto")
        def on_tz_change():
            set_et_mode(self.et_mode.get())
            self.log(f"[TZ] 切換 ET 模式為：{self.et_mode.get()}（auto/edt/est）")
            messagebox.showinfo("時區模式", f"已切換 ET 模式為 {self.et_mode.get()}。\nRTH 與時間計算將按新模式。")
        ttk.Radiobutton(tzbox, text="自動（America/New_York）", variable=self.et_mode, value="auto", command=on_tz_change).grid(row=0, column=0, padx=8, pady=4, sticky="w")
        ttk.Radiobutton(tzbox, text="夏令（UTC-4）",           variable=self.et_mode, value="edt",  command=on_tz_change).grid(row=0, column=1, padx=8, pady=4, sticky="w")
        ttk.Radiobutton(tzbox, text="冬令（UTC-5）",           variable=self.et_mode, value="est",  command=on_tz_change).grid(row=0, column=2, padx=8, pady=4, sticky="w")

        watch = ttk.LabelFrame(self.page_trade, text="監察中（可取消）")
        watch.pack(fill="both", expand=True, pady=6)
        self.watch_list = ttk.Treeview(watch, columns=("symbol","tf","src","status","trigger","qty"), show="headings", height=10)
        for k, v in [("symbol","Symbol"),("tf","TF"),("src","來源"),("status","狀態"),("trigger","Trigger ≥"),("qty","Qty")]:
            self.watch_list.heading(k, text=v); self.watch_list.column(k, width=140 if k!="status" else 260, anchor="center")
        self.watch_list.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(watch); right.pack(side="right", fill="y")
        ttk.Button(right, text="取消選中", command=self.cancel_selected).pack(padx=8, pady=8)

        dbg = ttk.LabelFrame(self.page_trade, text="執行日誌 / Debug Console（只讀）")
        dbg.pack(fill="both", expand=True, padx=2, pady=6)
        self.txt_log = scrolledtext.ScrolledText(dbg, height=10, wrap='none', state='disabled')
        self.txt_log.pack(fill="both", expand=True, padx=6, pady=6)
        btns = ttk.Frame(dbg); btns.pack(fill="x", padx=6, pady=2)
        ttk.Button(btns, text="清除日誌", command=self._clear_log).pack(side="left")
        ttk.Button(btns, text="複製全部", command=self._copy_log).pack(side="left", padx=6)

        status_bar = ttk.Frame(self.page_trade); status_bar.pack(fill="x", pady=6)
        ttk.Label(status_bar, textvariable=self.status_msg, foreground="blue").pack(anchor="w", padx=8)

    def _clear_log(self):
        self.txt_log.configure(state='normal'); self.txt_log.delete('1.0', 'end'); self.txt_log.configure(state='disabled')

    def _copy_log(self):
        try:
            self.clipboard_clear()
            self.clipboard_append(self.txt_log.get('1.0', 'end'))
        except:
            pass

    # ---- 狀態/連線 ----
    def _set_status(self, text):
        self.after(0, lambda: self.status_msg.set(text))
        if self.ibw.ib.isConnected():
            self.ib_connected.set("已連線")

    def connect_ib_manual(self):
        host = self.ent_host.get().strip() or "127.0.0.1"
        sel = self.cmb_port.get()
        if sel.startswith("4001"): port=4001
        elif sel.startswith("4002"): port=4002
        else:
            ptxt=self.ent_port_custom.get().strip()
            if not ptxt.isdigit():
                messagebox.showwarning("提示","請輸入自訂 Port（數字）"); return
            port=int(ptxt)
        cid=self.ent_client_id.get().strip()
        if not cid.isdigit():
            messagebox.showwarning("提示","Client ID 需為數字"); return
        self.log(f"[BTN] 連線 IB host={host} port={port} cid={cid}")
        ok = self.ibw.connect_manual(host, port, int(cid))
        self.ib_connected.set("已連線" if ok else "連線失敗")
        if not ok and self.ibw.last_error:
            messagebox.showwarning("連線失敗", self.ibw.last_error + "\n\n檢查：Port、Trusted IP=127.0.0.1、Read-Only API 關閉、Client ID 唔好撞。")

    def disconnect_ib_manual(self):
        self.log("[BTN] 斷線 IB")
        self.ibw.disconnect_manual(); self.ib_connected.set("已斷線")

    # ---- 串流檢查 ----
    def check_streaming_status(self):
        sym = (self.ent_symbol.get().strip().upper() or "AAPL")
        self.log(f"[BTN] 檢查串流：{sym}")
        threading.Thread(target=self._check_stream_worker, args=(sym,), daemon=True).start()

    def _check_stream_worker(self, sym):
        txt, ok, dtype = self.run_on_main(self.ibw.check_market_data, sym, CHECK_STREAM_TIMEOUT)
        self.stream_status.set(txt); self.market_data_type.set(dtype)
        if not ok: self._set_status(f"市場數據檢查：{txt}")
        self.log(f"[MDCHK] 結果：{txt} type={dtype}")

    # ====== 獲取首支 K 線（未開市/週末/假期 → 讀上一交易日） ======
    def fetch_first_candle_stream(self):
        sym = self.ent_symbol.get().strip().upper()
        if not sym:
            messagebox.showwarning("提示","請輸入股票代號"); return
        if not self.ibw.ib.isConnected():
            messagebox.showwarning("提示","請先連線 IB。"); return

        tf = int(self.tf_choice.get())
        nowt = now_et()
        self.log(f"[BTN] 獲取K線：{sym} tf={tf}m now(ET)={nowt.strftime('%Y-%m-%d %H:%M:%S')}")

        prev_open, prev_close = self.run_on_main(self.ibw.prev_rth_open_close, sym, nowt)
        next_open, next_close = self.run_on_main(self.ibw.next_rth_open_close, sym, nowt)

        today = nowt.date()
        if next_open.date() == today and nowt < next_open:
            use_date = prev_open.date()
            messagebox.showinfo("未開市", f"{sym}: 未開市，已自動讀取『上一交易日 {use_date.isoformat()}』首支K線。")
            self._set_status(f"{sym}: 正在以歷史數據回補上一交易日（{use_date}）首支 K 線…")
            self.log(f"[KLINE] 未開市 → 讀上一交易日 {use_date}")
            threading.Thread(target=self._fetch_candle_hist_for_date, args=(sym, tf, use_date, "IB 歷史"), daemon=True).start()
            return

        if prev_open.date() == today and prev_open <= nowt < prev_close:
            start_et, end_et = first_candle_window(today, tf)
            if nowt < end_et:
                self._set_status(f"{sym}: 收集首支 K 線（串流）中… 剩餘 {int((end_et - nowt).total_seconds())}s")
                self.log("[KLINE] 窗內 → 先串流聚合，如未得會自動回補歷史")
                threading.Thread(target=self._fetch_candle_during_window, args=(sym, tf, today), daemon=True).start()
                return
            else:
                self._set_status(f"{sym}: 正在以歷史數據回補首支 K 線（今日）…")
                self.log("[KLINE] 走歷史回補（今日）")
                threading.Thread(target=self._fetch_candle_hist_for_date, args=(sym, tf, today, "IB 歷史"), daemon=True).start()
                return

        use_date = prev_open.date()
        self._set_status(f"{sym}: 已收市/非交易日 → 正在回補『上一交易日 {use_date}』首支 K 線…")
        self.log(f"[KLINE] 收市/非交易日 → 讀上一交易日 {use_date}")
        threading.Thread(target=self._fetch_candle_hist_for_date, args=(sym, tf, use_date, "IB 歷史"), daemon=True).start()

    # == 兩個 worker ==
    def _fetch_candle_hist_for_date(self, sym, tf, date_, source_text):
        try:
            fh, fl, msg = self.run_on_main(self.ibw.hist_first_candle, sym, tf, date_)
            if fh is None:
                self._set_status(f"{sym}: 取得首支K線失敗（{msg}）")
                self.log(f"[HIST] 失敗：{msg}")
                return
            rng = max(fh - fl, 0.0001)
            # FIXED: Only take the first value (netliq) from the 5 returned values
            acct_vals = self.run_on_main(self.ibw.account_values)
            netliq = acct_vals[0] if acct_vals else 0

            qty = math.floor(0.005 * (netliq or 0) / rng) if netliq else 0
            self._update_latest_info(date_, fh, fl, rng, qty, source_text)
            self._set_status(f"{sym}: 首支K線 高={fh:.4f} 低={fl:.4f}（來源：{source_text}）")
            self.log(f"[HIST] 完成 {sym} {tf}m {date_.isoformat()} H={fh:.4f} L={fl:.4f}")
        except Exception as e:
            self.log(f"[ERR] _fetch_candle_hist_for_date: {e}")
            self._set_status(f"{sym}: 歷史回補錯誤：{e}")

    def _fetch_candle_during_window(self, sym, tf, trade_date):
        try:
            start_et, end_et = first_candle_window(trade_date, tf)
            bars = self.run_on_main(self.ibw.start_rt_bars, sym)
            hi = None; lo = None
            while now_et() < end_et:
                time.sleep(0.25)
                for b in list(bars):
                    t_et = b.time.replace(tzinfo=timezone.utc).astimezone(US_EASTERN)
                    if start_et <= t_et < end_et:
                        hi = b.high if hi is None else max(hi, b.high)
                        lo = b.low  if lo is None else min(lo,  b.low)
                secs_left = max(0, int((end_et - now_et()).total_seconds()))
                hi_txt = f"{hi:.4f}" if hi is not None else "—"
                lo_txt = f"{lo:.4f}" if lo is not None else "—"
                self._set_status(f"{sym}: 收集中… 剩餘 {secs_left}s｜暫時首支 H/L = {hi_txt}/{lo_txt}")
            self.run_on_main(self.ibw.cancel_rt_bars, bars)

            if hi is None or lo is None:
                self.log("[RT] 串流未齊 → 改用歷史回補（今日）")
                fh, fl, _ = self.run_on_main(self.ibw.hist_first_candle, sym, tf, trade_date)
                if fh is None:
                    self._set_status(f"{sym}: 取得首支K線失敗（歷史回補亦無）")
                    return
                hi, lo = fh, fl
                src_text = "IB 歷史"
            else:
                src_text = "IB 串流"

            rng = max(hi - lo, 0.0001)
            # FIXED: Only take the first value (netliq)
            acct_vals = self.run_on_main(self.ibw.account_values)
            netliq = acct_vals[0] if acct_vals else 0

            qty = math.floor(0.005 * (netliq or 0) / rng) if netliq else 0
            self._update_latest_info(trade_date, hi, lo, rng, qty, src_text)
            self._set_status(f"{sym}: 首支K線完成 高={hi:.4f} 低={lo:.4f}（來源：{src_text}）")
            self.log(f"[KLINE] 完成 {sym} {tf}m 來源={src_text} H={hi:.4f} L={lo:.4f}")
        except Exception as e:
            self.log(f"[ERR] _fetch_candle_during_window: {e}")
            self._set_status(f"{sym}: 串流聚合錯誤：{e}")

    # ---- 賣出條件單 ----
    def place_sell_break_selected_tf(self):
        """依據 UI 當前選取的時間框（1/5/30m）放置賣出條件單"""
        try:
            tf = int(self.tf_choice.get())
        except Exception:
            tf = 1
        self.place_sell_break_button(tf)

    def place_sell_break_button(self, tf: int):
        sym = self.ent_symbol.get().strip().upper()
        if not sym:
            messagebox.showwarning("提示","請輸入股票代號"); return
        if not self.ibw.ib.isConnected():
            messagebox.showwarning("提示","請先連線 IB。"); return

        self.log(f"[BTN] 放置 賣出條件單（Sell Stop）嚴格跌破首支低點：{sym} tf={tf}m")
        threading.Thread(target=self._place_sell_break_worker, args=(sym, tf), daemon=True).start()

    def _place_sell_break_worker(self, sym: str, tf: int):
        try:
            nowt = now_et()
            # 與「獲取K線」一致：非交易時段/週末 → 用上一個交易日；交易中 → 用今天
            prev_open, prev_close = self.run_on_main(self.ibw.prev_rth_open_close, sym, nowt)
            next_open, next_close = self.run_on_main(self.ibw.next_rth_open_close, sym, nowt)
            if nowt < next_open:
                date_use = prev_open.date()
                self.log(f"[SELL] 非交易日/未開市 → 使用上一交易日 {date_use} 的首支K線低點")
            elif prev_open <= nowt < prev_close:
                date_use = nowt.date()
                self.log(f"[SELL] 交易中 → 使用今日 {date_use} 的首支K線低點")
            else:
                date_use = prev_open.date()
                self.log(f"[SELL] 已收市 → 使用上一交易日 {date_use} 的首支K線低點")

            qty_pos = self.run_on_main(self.ibw.get_position_qty, sym)
            if qty_pos <= 0:
                msg = f"{sym}: 目前持倉數量=0，未落賣出條件單。"
                self._set_status(msg)
                messagebox.showinfo("提示", msg)
                self.log(f"[SELL] 無持倉：{sym}")
                return

            fh, fl, src, msg_take = self.run_on_main(self.ibw.first_candle_any, sym, tf, date_use)
            if fl is None:
                self._set_status(f"{sym}: 無法取得首支低點（{msg_take}）")
                self.log(f"[SELL] 取低點失敗：{msg_take}")
                return

            # === MODIFIED: Use place_stop_order for Sell Stop logic ===
            # Strict = True means (Low - minTick)
            ok, msg2 = self.run_on_main(self.ibw.place_stop_order, sym, float(fl), int(qty_pos), 'SELL', True)
            self._set_status(msg2)
            self.log(f"[SELL] 結果：{msg2}｜來源={src} 低點={fl:.4f} 持倉={qty_pos}")
        except Exception as e:
            self._set_status(f"{sym}: 放置賣出條件單錯誤：{e}")
            self.log(f"[ERR] _place_sell_break_worker: {e}")

    # ---- 監察（09:29 預熱；09:31:01 / 09:36:01 下單）----
    
    def _renumber_watch_rows(self):
        """Renumber the first '#' column of watch_list from 1..N and auto-fit columns."""
        try:
            children = self.watch_list.get_children('')
            for i, iid in enumerate(children, start=1):
                vals = list(self.watch_list.item(iid, "values"))
                if not vals:
                    continue
                # ensure we have 7 columns with idx at [0]
                if len(vals) == 6:
                    vals = [i] + vals
                else:
                    vals[0] = i
                self.watch_list.item(iid, values=tuple(vals))
            # autosize columns to minimal width
            try:
                m = len(children)
                idx_min = max(28, (len(str(m)) or 1) * 8 + 10)
                self._autosize_tree_columns(self.watch_list,
                    min_widths={'idx': idx_min, 'symbol': 70, 'tf': 40, 'source': 70, 'status': 120, 'trigger': 70, 'qty': 60},
                    max_widths={'symbol': 120, 'status': 220},
                    pad=12)
            except Exception:
                pass
        except Exception:
            pass

    def start_watch_auto(self):
        sym = self.ent_symbol.get().strip().upper()
        if not sym:
            messagebox.showwarning("提示","請輸入股票代號"); return
        tf = int(self.tf_choice.get())
        src = "IB 自動"

        if not self.ibw.ib.isConnected():
            messagebox.showwarning("提示","請先連線 IB。"); return

        if sym in self.watch_items and self.watch_items[sym].active:
            messagebox.showinfo("提示", f"{sym} 已在監察中。"); return

        nowt = now_et()
        open_dt, close_dt = self.run_on_main(self.ibw.next_rth_open_close, sym, nowt)
        if nowt >= close_dt:
            open_dt, close_dt = self.run_on_main(self.ibw.next_rth_open_close, sym, nowt + timedelta(minutes=1))

        w = WatchItem(sym, tf, src, open_dt_et=open_dt, close_dt_et=close_dt)
        w.status = f"已加入，目標 RTH 開市 {open_dt.strftime('%Y-%m-%d %H:%M ET')}，09:29 開始串流"
        self.watch_items[sym] = w
        self._refresh_watch_row(sym)

        th = threading.Thread(target=self._watch_loop_stream, args=(w,), daemon=True)
        self.watch_threads[sym] = th
        th.start()
        messagebox.showinfo("監察開始", f"{sym}（{tf}m，來源：{src}）已加入監察。")

    def _watch_loop_stream(self, w: WatchItem):
        while w.active:
            try:
                nowt = now_et()

                if nowt >= (w.close_dt_et or nowt):
                    new_open, new_close = self.run_on_main(self.ibw.next_rth_open_close, w.symbol, nowt + timedelta(seconds=1))
                    w.open_dt_et, w.close_dt_et = new_open, new_close
                    w.status = f"已收市，改為等待下一次 RTH {new_open.strftime('%Y-%m-%d %H:%M ET')}"
                    self.run_on_main(self._refresh_watch_row, w.symbol)
                    self.log(f"[WATCH] {w.symbol} 轉到下一 RTH：{new_open.strftime('%Y-%m-%d %H:%M ET')}")
                    time.sleep(POLL_SECONDS)
                    continue

                open_dt = w.open_dt_et
                close_dt= w.close_dt_et
                first_start = open_dt
                first_end   = open_dt + timedelta(minutes=w.tf_min)
                pre_time    = open_dt - timedelta(minutes=1)

                if nowt < pre_time:
                    left = int((pre_time - nowt).total_seconds())
                    w.status = f"未到 09:29 ET，還有 {left//60}m{left%60}s（下一 RTH：{open_dt.strftime('%Y-%m-%d')}）"
                    self.run_on_main(self._refresh_watch_row, w.symbol)
                    time.sleep(POLL_SECONDS)
                    continue

                if pre_time <= nowt < first_start:
                    w.status = "預備中（09:29~09:30），等待 09:30 開始聚合首支K線…"
                    self.run_on_main(self._refresh_watch_row, w.symbol)
                    time.sleep(POLL_SECONDS)
                    continue

                # 09:30–首支時間窗：等待窗口結束後用歷史數據
                if first_start <= nowt < first_end:
                    self.log(f"[WATCH] 等待首支K線窗口結束 {w.symbol} {w.tf_min}m @ {open_dt.date()}")
                    w.status = f"收集中...（{w.tf_min}分鐘K線窗口）"
                    self.run_on_main(self._refresh_watch_row, w.symbol)

                    # 等到時間窗口結束
                    while now_et() < first_end and w.active:
                        time.sleep(0.2)

                    # 用歷史數據獲取（同「獲取K線」按鈕）
                    self.log(f"[WATCH] 窗口結束 → 歷史數據獲取 {w.symbol}")
                    fh, fl, _ = self.run_on_main(self.ibw.hist_first_candle, w.symbol, w.tf_min, open_dt.date())
                else:
                    self.log(f"[WATCH] 首支時間窗已過 → 歷史回補 {w.symbol}")
                    fh, fl, _ = self.run_on_main(self.ibw.hist_first_candle, w.symbol, w.tf_min, open_dt.date())

                if fh is None or fl is None:
                    w.status = "未能取得首支K線，稍後重試"
                    self.run_on_main(self._refresh_watch_row, w.symbol)
                    time.sleep(POLL_SECONDS)
                    continue

                # 等到 :01 秒才下單
                if now_et() < (first_end + timedelta(seconds=1)):
                    while now_et() < (first_end + timedelta(seconds=1)) and w.active:
                        time.sleep(0.2)

                # FIXED: Only take the first value (netliq)
                acct_vals = self.run_on_main(self.ibw.account_values)
                netliq = acct_vals[0] if acct_vals else 0

                rng = max(fh - fl, 0.0001)
                w.qty = math.floor(0.005 * (netliq or 0) / rng) if netliq else 0
                self._update_latest_info(open_dt.date(), fh, fl, rng, w.qty, "IB 自動")

                w.first_high, w.first_low = fh, fl
                w.status = f"已取得首支K線：高={fh:.4f} 低={fl:.4f} → 準備下 BUY STOP 單（嚴格『>』首高）"
                self.run_on_main(self._refresh_watch_row, w.symbol)

                if not self.ibw.ib.isConnected():
                    w.status = "IB 未連線，不能落單"; self.run_on_main(self._refresh_watch_row, w.symbol); return
                if w.qty <= 0:
                    w.status = "建議股數=0，未落單"; self.run_on_main(self._refresh_watch_row, w.symbol); return

                # === MODIFIED: Use place_stop_order for Buy Breakout logic ===
                # Strict = True means (High + minTick)
                ok, msg2 = self.run_on_main(self.ibw.place_stop_order, w.symbol, w.first_high, w.qty, 'BUY', True)
                w.status = "條件單已提交" if ok else "條件單提交失敗"
                self._set_status(msg2)
                self.run_on_main(self._refresh_watch_row, w.symbol)
                self.log(f"[WATCH] 下單結果：{msg2}")
                # 成功提交後，從監察清單移除
                if ok:
                    try:
                        self.run_on_main(self._remove_watch, w.symbol)
                        self._set_status(f"{w.symbol} 已提交 STOP 單 → 已從監察清單移除。")
                    except Exception as _e:
                        self.log(f"[WARN] 無法移除監察項目：{_e}")
                return
            except Exception as e:
                w.status = f"錯誤：{e}"
                self.run_on_main(self._refresh_watch_row, w.symbol)
                self.log(f"[ERR] _watch_loop_stream: {e}")
                time.sleep(POLL_SECONDS)

    def _update_latest_info(self, used_date, fh, fl, rng, qty, source):
        hk_date = datetime.combine(used_date, dtime(0,0), tzinfo=US_EASTERN).astimezone(HK_TZ).date()
        self.lbl_date.config(text=f"交易日：{used_date.isoformat()}（ET）／ {hk_date.isoformat()}（HKT）")
        self.lbl_hilo.config(text=f"首支 K 線 High/Low：{fh:.4f} / {fl:.4f}")
        self.lbl_range.config(text=f"Range：{rng:.4f}")
        self.lbl_qty.config(text=f"建議股數（0.5% 資金）：{qty}")
        try:
            # FIXED: Handle 5 return values here
            avals = self.ibw.account_values() if self.ibw.ib.isConnected() else None
            netliq_val = avals[0] if avals else None

            est_cost = float(fh) * float(qty or 0)
            pct_use = (est_cost / float(netliq_val)) * 100.0 if (netliq_val and netliq_val > 0 and qty) else 0.0
            self.lbl_cap_pct.config(text=f"預估資金佔用（%NetLiq）：{pct_use:.2f}%（以首高×建議股數≈{est_cost:,.2f} USD）")
        except Exception:
            self.lbl_cap_pct.config(text="預估資金佔用（%NetLiq）：—")
    
    def _refresh_watch_row(self, symbol):
        try:
            w = self.watch_items.get(symbol)
            if not w:
                return

            # 刪除已有同 Symbol 的舊行（注意：第 0 欄是 #，第 1 欄才是 Symbol）
            try:
                for iid in self.watch_list.get_children():
                    vals = self.watch_list.item(iid, "values")
                    if vals and ((len(vals) > 1 and vals[1] == symbol) or (len(vals) >= 1 and vals[0] == symbol)):
                        self.watch_list.delete(iid)
            except Exception:
                pass

            trig = f"{w.first_high:.4f}" if getattr(w, "first_high", None) else "—"
            self.watch_list.insert("", "end",
                                   values=(0, w.symbol, f"{w.tf_min}m", w.source, w.status, trig, w.qty or "—"))
            self._renumber_watch_rows()
        except Exception as e:
            try:
                self._append_log(f"[ERR] _refresh_watch_row: {e}")
            except Exception:
                try:
                    self._status(f"_refresh_watch_row 失敗：{e}")
                except Exception:
                    pass

    def _remove_watch(self, symbol: str):
        """從監察清單中完全移除指定 Symbol（UI + 內部狀態）。"""
        try:
            # 刪 UI（# 在第 0 欄、Symbol 在第 1 欄；亦容錯單欄情形）
            for iid in self.watch_list.get_children():
                try:
                    vals = self.watch_list.item(iid, "values")
                    if vals and ((len(vals) > 1 and vals[1] == symbol) or (len(vals) >= 1 and vals[0] == symbol)):
                        self.watch_list.delete(iid)
                except Exception:
                    pass
            # 刪內部狀態
            if symbol in self.watch_items:
                try:
                    self.watch_items[symbol].active = False
                except Exception:
                    pass
                try:
                    del self.watch_items[symbol]
                except Exception:
                    pass
            if symbol in self.watch_threads:
                try:
                    del self.watch_threads[symbol]
                except Exception:
                    pass
            # 重編號
            try:
                self._renumber_watch_rows()
            except Exception:
                pass
        except Exception as e:
            self.log(f"[ERR] _remove_watch: {e}")


    def cancel_selected(self):



        sel = self.watch_list.selection()


        if not sel: return


        vals = self.watch_list.item(sel[0], "values")


        if not vals: return


        # 一般情況 Symbol 在第 2 欄（index=1）；容錯取第 1 欄


        sym = vals[1] if len(vals) > 1 else vals[0]


        try:


            self.run_on_main(self._remove_watch, sym)


            self._set_status(f"{sym} 監察已取消並移除。")


            self.log(f"[BTN] 取消並移除監察 {sym}")


        except Exception as e:


            self._set_status(f"{sym} 取消移除失敗：{e}")


            self.log(f"[ERR] 取消移除失敗 {sym}: {e}")


    # ---- 第一頁：取消選中訂單 ----
    def cancel_selected_open_order(self):
        sel = self.tree_oo.selection()
        if not sel:
            messagebox.showinfo("提示", "請先在『未執行訂單』表格選中一張訂單。")
            return
        vals = self.tree_oo.item(sel[0], "values")
        if not vals: return
        try:
            order_id = int(vals[1])
        except Exception:
            messagebox.showwarning("提示", "讀取 OrderId 失敗。")
            return

        self.log(f"[BTN] 取消訂單 OrderId={order_id}")
        threading.Thread(target=self._cancel_order_worker, args=(order_id,), daemon=True).start()

    def _cancel_order_worker(self, order_id: int):
        ok, msg = self.run_on_main(self.ibw.cancel_order_by_id, order_id)
        self._set_status(msg)
        self.log(f"[OO] {msg}")

    # ---- 週期刷新 ----
    def refresh_account_ui_loop(self):
        try:
            if self.ibw.ib.isConnected(): self.ib_connected.set("已連線")
            # --- MODIFIED: Unpack 5 values ---
            netliq, cash, excess, mktval, buypwr = self.ibw.account_values() if self.ibw.ib.isConnected() else (None,None,None,None,None)

            if netliq is not None: self.lbl_netliq.config(text=f"NetLiq：{human_usd(netliq)}")
            if cash   is not None: self.lbl_cash.config(text=f"現金餘額：{human_usd(cash)}")
            # --- MODIFIED: Update new labels ---
            if excess is not None: self.lbl_excess.config(text=f"Excess Liq：{human_usd(excess)}")
            if mktval is not None: self.lbl_mktval.config(text=f"MKT Val：{human_usd(mktval)}")
            if buypwr is not None: self.lbl_buypwr.config(text=f"Buying Power：{human_usd(buypwr)}")

            d, u, r = self.ibw.pnl_values()
            
            # --- MODIFIED: Calculate Daily P&L % ---
            # Formula: (Daily PnL / (NetLiq - Daily PnL)) * 100
            # Note: NetLiq is current equity. Start Equity approx = NetLiq - DailyPnL
            d_pct_txt = ""
            if d is not None and netliq is not None and netliq != 0:
                try:
                    prev_eq = netliq - float(d)
                    if abs(prev_eq) > 0.01:
                        pct = (float(d) / prev_eq) * 100.0
                        sign = "+" if pct > 0 else ""
                        d_pct_txt = f" ({sign}{pct:.2f}%)"
                except: pass

            self.lbl_daily.config(text=f"今日 P&L（Daily）：{human_usd(d)}{d_pct_txt}")
            self.lbl_unrl.config(text=f"未實現 P&L（Unrealized）：{human_usd(u)}")
            self.lbl_real.config(text=f"已實現 P&L（Realized 今日）：{human_usd(r)}")

            for row in self.tree.get_children(): self.tree.delete(row)
            rows = self.ibw.portfolio_rows() if self.ibw.ib.isConnected() else []
            nl_val = float(netliq or 0.0)

            for i, rrow in enumerate(rows, start=1):
                sym = rrow.get("symbol","—")
                position = rrow.get("position", 0)
                mkt_px = rrow.get("mkt_px", None)
                avg_cost = rrow.get("avg_cost", None)
                unrl = rrow.get("unrealized", 0.0)
                unrl_pct_val = rrow.get("unrealized_pct", None)

                mkt_px_txt  = f"{float(mkt_px):.2f}" if isinstance(mkt_px,(int,float)) and mkt_px is not None else "—"
                avg_cost_txt= f"{float(avg_cost):.2f}" if isinstance(avg_cost,(int,float)) and avg_cost is not None else "—"
                unrl_txt    = human_usd(unrl)
                unrl_pct_txt= f"{float(unrl_pct_val):.2f}%" if isinstance(unrl_pct_val,(int,float)) else "—"

                try:
                    mv = rrow.get('market_value')
                    if not isinstance(mv, (int, float)) or mv is None:
                        px = (float(mkt_px) if mkt_px is not None else (float(avg_cost) if avg_cost is not None else 0.0))
                        mv = abs(float(position) or 0.0) * px
                    pct_netliq_txt = f"{(abs(mv)/(nl_val if nl_val else 1))*100:.2f}%" if nl_val else "—"
                except:
                    pct_netliq_txt = "—"

                self.tree.insert("", "end", values=(i, sym, position, mkt_px_txt, avg_cost_txt, unrl_txt, unrl_pct_txt, pct_netliq_txt))
            # auto-fit positions columns (minimal width that fits content)
            try:
                n = len(rows)
                idx_min = max(28, (len(str(n)) or 1) * 8 + 10)
                self._autosize_tree_columns(self.tree,
                    min_widths={'idx': idx_min, 'symbol': 60, 'position': 60, 'mkt_px': 60, 'avg_cost': 60, 'unrl': 80, 'unrl_pct': 70, 'pct_netliq': 80},
                    max_widths={'symbol': 140, 'status': 140, 'unrl': 140, 'pct_netliq': 120},
                    pad=12)
            except Exception:
                pass
            try:
                _pos_apply_sort(self.tree)
            except Exception:
                pass


        except Exception as e:
            self._set_status(f"刷新帳戶資料錯誤：{e}")
            self.log(f"[ERR] refresh_account_ui_loop: {e}")
        finally:
            self.safe_after(1000, self.refresh_account_ui_loop)

    def _on_oo_select(self, event=None):
        """只保留最後一次選中的單一 OrderId（固定單選）。"""
        try:
            sel = self.tree_oo.selection()
            if not sel:
                self._oo_selected_id = None
                return
            vals = self.tree_oo.item(sel[0], "values")
            self._oo_selected_id = str(vals[1]) if vals and len(vals) > 1 else None
        except Exception:
            pass

    def refresh_open_orders_loop(self):
        try:
            if self.ibw.ib.isConnected():
                trades = self.ibw.open_orders_all()
                self.open_order_map.clear()
                try:
                    self.lf_oo.configure(text=f"未執行訂單（Open Orders｜全帳戶）（{len(trades)}）")
                except Exception:
                    pass
                last_selected_id = getattr(self, "_oo_selected_id", None)
                for row in self.tree_oo.get_children(): self.tree_oo.delete(row)
                item_for_restore = None
                for i, t in enumerate(trades, start=1):
                    o = t.order; c = t.contract; st = t.orderStatus
                    status_txt = st.status if isinstance(st, OrderStatus) else getattr(st, "status", "—")
                    
                    # === Modified logic for columns ===
                    # Trigger column (Conditions)
                    trig = ""
                    if getattr(o, "conditions", None):
                        try: trig = f"{o.conditions[0].price:.4f}"
                        except: trig = "Cond"
                    
                    # Stop Price column (Aux Price)
                    stop_val = ""
                    if o.auxPrice and o.auxPrice > 0 and o.auxPrice < 1.79e308:
                         stop_val = f"{o.auxPrice:.2f}"

                    oid = o.orderId or "—"
                    item_id = self.tree_oo.insert("", "end", values=(
                        i,
                        oid,
                        getattr(c, "symbol", "—"),
                        o.action or "—",
                        o.totalQuantity or "—",
                        o.orderType or "—",
                        status_txt or "—",
                        o.tif or "—",
                        trig,      # Trigger
                        stop_val   # Stop Price
                    ))
                    if isinstance(oid, int):
                        self.open_order_map[str(oid)] = t
                    if last_selected_id is not None and str(oid) == str(last_selected_id):
                        item_for_restore = item_id
            # auto-fit open orders columns (minimal width that fits content)
            try:
                m = len(trades)
                idx_min = max(28, (len(str(m)) or 1) * 8 + 10)
                self._autosize_tree_columns(self.tree_oo,
                    min_widths={'idx': idx_min, 'id': 60, 'symbol': 60, 'action': 50, 'qty': 60, 'type': 60, 'status': 80, 'tif': 50, 'aux': 70, 'stop_px': 70},
                    max_widths={'id': 100, 'symbol': 140, 'status': 160, 'aux': 120, 'stop_px': 120},
                    pad=12)
            except Exception:
                pass

                        # 僅恢復單一選取
            try:
                self.tree_oo.selection_remove(self.tree_oo.selection())
                if item_for_restore:
                    self.tree_oo.selection_set(item_for_restore)
                    self.tree_oo.focus(item_for_restore)
            except Exception:
                pass

        except Exception as e:
            self._set_status(f"刷新 Open Orders 錯誤：{e}")
            self.log(f"[ERR] refresh_open_orders_loop: {e}")
        finally:
            self.safe_after(1000, self.refresh_open_orders_loop)


# ====== 新增：收市前兩分鐘監察（SMA10 不含今日）＋ 半倉市價賣出 ======
        def _get_last_price(self, sym: str):
            try:
                from ib_insync import Stock
                if not self.ibw.ib.isConnected():
                    return None
                sym = sym.upper()
                c = Stock(sym, 'SMART', 'USD')
                self.ibw.ib.qualifyContracts(c)
                # 先 LIVE
                try:
                    self.ibw.ib.reqMarketDataType(1)
                    t = self.ibw.ib.reqMktData(c, "", False, False)
                    self.ibw.ib.sleep(1.0)
                    for k in ("last", "close", "bid", "ask"):
                        v = getattr(t, k, None)
                        if v is not None:
                            try:
                                px = float(v)
                                if px == px:
                                    try: self.ibw.ib.cancelMktData(t)
                                    except: pass
                                    return px
                            except: pass
                    try: self.ibw.ib.cancelMktData(t)
                    except: pass
                except Exception:
                    pass
                # 後備 DELAYED
                try:
                    self.ibw.ib.reqMarketDataType(3)
                    t2 = self.ibw.ib.reqMktData(c, "", False, False)
                    self.ibw.ib.sleep(1.0)
                    for k in ("last", "close", "bid", "ask"):
                        v = getattr(t2, k, None)
                        if v is not None:
                            try:
                                px = float(v)
                                if px == px:
                                    try: self.ibw.ib.cancelMktData(t2)
                                    except: pass
                                    return px
                            except: pass
                    try: self.ibw.ib.cancelMktData(t2)
                    except: pass
                except Exception:
                    pass
                return None
            except Exception as e:
                self.log(f"[ERR] _get_last_price: {e}")
                return None

        def _sma10_excl_today(self, sym: str):
            try:
                from ib_insync import Stock
                if not self.ibw.ib.isConnected():
                    return None, "IB 未連線"
                sym = sym.upper()
                nowt = now_et()
                if hasattr(self.ibw, "prev_rth_open_close"):
                    _, prev_close = self.ibw.prev_rth_open_close(sym, nowt)
                else:
                    _, prev_close = self.ibw.next_rth_open_close(sym, nowt - timedelta(days=1))
                c = Stock(sym, 'SMART', 'USD')
                self.ibw.ib.qualifyContracts(c)
                bars = self.ibw.ib.reqHistoricalData(
                    c, endDateTime=prev_close, durationStr="20 D",
                    barSizeSetting="1 day", whatToShow="TRADES",
                    useRTH=True, formatDate=2, keepUpToDate=False
                )
                closes = [float(b.close) for b in bars if getattr(b, "close", None) is not None]
                if len(closes) < 10:
                    return None, "歷史日線不足 10 根"
                sma = sum(closes[-10:]) / 10.0
                d0 = str(bars[-10].date)[:10] if len(bars) >= 10 else ""
                d1 = str(bars[-1].date)[:10] if len(bars) >= 1 else ""
                return sma, f"{d0} ~ {d1}"
            except Exception as e:
                self.log(f"[ERR] _sma10_excl_today: {e}")
                return None, str(e)

        def start_close_monitor_sma10(self):
            sym = (self.ent_symbol.get() or "").strip().upper()
            if not sym:
                messagebox.showwarning("提示","請輸入股票代號"); return
            if not self.ibw.ib.isConnected():
                messagebox.showwarning("提示","請先連線 IB。"); return
            self.log(f"[BTN] 開始監察收市前2分鐘（SMA10排今日）：{sym}")
            threading.Thread(target=self._close_monitor_worker, args=(sym,), daemon=True).start()

        def _close_monitor_worker(self, sym: str):
            try:
                sym = sym.upper()
                sma10, desc = self._sma10_excl_today(sym)
                if sma10 is None:
                    self._set_status(f"{sym}: 無法計算 SMA10（{desc}）"); return
                self.log(f"[CLOSEMON] {sym} SMA10={sma10:.4f} 期間 {desc}")
                self._set_status(f"{sym}: SMA10(排今日)={sma10:.4f}（{desc}）")
                nowt = now_et()
                _, close_dt = self.ibw.next_rth_open_close(sym, nowt)
                t_trigger = close_dt - timedelta(minutes=2)
                while now_et() < t_trigger and not getattr(self, "_closing", False):
                    left = int((t_trigger - now_et()).total_seconds())
                    if left < 0: break
                    m, s = divmod(left, 60)
                    self._set_status(f"{sym}: 等候收市前兩分鐘… 倒數 {m}分{s}秒")
                    time.sleep(1)
                if getattr(self, "_closing", False):
                    return
                self._set_status(f"{sym}: 進入最後兩分鐘，讀取現價…")
                px = self._get_last_price(sym)
                if px is None:
                    self._set_status(f"{sym}: 讀不到現價，未能落單。"); self.log("[CLOSEMON] 無現價"); return
                self.log(f"[CLOSEMON] {sym} 現價={px:.4f} vs SMA10={sma10:.4f}")
                if px < sma10:
                    qty = int(self.ibw.get_position_qty(sym))
                    if qty <= 0:
                        self._set_status(f"{sym}: 無持倉，跳過賣出"); return
                    from ib_insync import Stock, MarketOrder
                    c = Stock(sym,'SMART','USD'); self.ibw.ib.qualifyContracts(c)
                    self.ibw.ib.placeOrder(c, MarketOrder('SELL', qty))
                    self._set_status(f"{sym}: 現價<{sma10:.4f} → 已市價賣出 {qty} 股")
                    self.log(f"[CLOSEMON] 已送出市價賣出 {qty}")
                else:
                    self._set_status(f"{sym}: 未達條件（{px:.4f} ≥ {sma10:.4f}），不賣出")
            except Exception as e:
                self._set_status(f"{sym}: 收市監察錯誤：{e}")
                self.log(f"[ERR] _close_monitor_worker: {e}")

        def sell_half_now(self):
            sym = (self.ent_symbol.get() or "").strip().upper()
            if not sym:
                messagebox.showwarning("提示","請輸入股票代號"); return
            if not self.ibw.ib.isConnected():
                messagebox.showwarning("提示","請先連線 IB。"); return
            qty = int(self.ibw.get_position_qty(sym))
            if qty <= 0:
                messagebox.showinfo("提示", f"{sym}: 目前無持倉。"); return
            sell_qty = max(1, qty // 2)
            self.log(f"[BTN] 半倉賣出 {sym}: {sell_qty}/{qty}")
            try:
                from ib_insync import Stock, MarketOrder
                c = Stock(sym,'SMART','USD'); self.ibw.ib.qualifyContracts(c)
                self.ibw.ib.placeOrder(c, MarketOrder('SELL', sell_qty))
                self._set_status(f"{sym}: 已市價賣出一半（{sell_qty}/{qty}）")
            except Exception as e:
                self._set_status(f"{sym}: 半倉賣出失敗：{e}")
                self.log(f"[ERR] sell_half_now: {e}")

        # ====== 安全 after（關窗時不再觸發） ======

# ====== 安全 after（關窗時不再觸發） ======
# [main removed to allow integration]


# ===== Integration augmentations from ts_integrated_final_v2.py (modified for single-file) =====

# -*- coding: utf-8 -*-
"""
ts_integrated_final_v2.py
在 ts_integrated_final 基礎上更新：
- 「收市前2分鐘（SMA10不含今日）」由『只判一次』→ 改為『15:58:00 ~ 16:00:00 之間每 10 秒連續監察』
- 任何一次偵測到 現價 < SMA10 即刻以市價賣出全部持倉並結束；至 16:00 仍未觸發 → 標示未觸發並移除。
- 監察中（可取消）狀態欄會即時顯示倒數／監察窗口內的現價對比。

需要與 ts_closemon_clean_fix8.py 放同資料夾：
    python ts_integrated_final_v2.py
"""
import threading, time
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import timedelta
from ib_insync import IB, Stock, MarketOrder, util

US_EASTERN = US_EASTERN
now_et = now_et
WatchItem = WatchItem


def refresh_positions_ui(self):
    """Rebuild positions table from cache every call."""
    try:
        rows = self.ibw.portfolio_rows() if hasattr(self.ibw, "portfolio_rows") else []
        # FIXED: Unpack 5 values safely (take first for netliq)
        avals = self.ibw.account_values()
        netliq = avals[0] if avals else 0
        
        self.tree.delete(*self.tree.get_children(""))
        for i, r in enumerate(rows, 1):
            sym = r.get("symbol","—")
            qty = int(r.get("position",0) or 0)
            mpx = r.get("mkt_px","—")
            avg = r.get("avg_cost","—")
            unv = float(r.get("unrealized",0.0) or 0.0)
            upr = float(r.get("unrealized_pct",0.0) or 0.0)
            mv  = float(r.get("market_value",0.0) or 0.0)
            pct_nl = (mv / netliq * 100.0) if netliq else 0.0
            self.tree.insert("", "end", values=(i, sym, qty, mpx, avg, f"{unv:,.2f}", f"{upr:.2f}%", f"{pct_nl:.2f}%"))
        try:
            _pos_apply_sort(self.tree)
        except Exception:
            pass

    except Exception as e:
        try: self._append_log(f"[ERR] refresh_positions_ui: {e}")
        except Exception: pass
        try:
            _pos_apply_sort(self.tree)
        except Exception:
            pass


    def refresh_portfolio_loop(self):
        try:
            self.refresh_positions_ui()
        except Exception as e:
            try: self._append_log(f"[ERR] refresh_portfolio_loop: {e}")
            except Exception: pass
        if not getattr(self, "_closing", False):
            self.safe_after(1000, self.refresh_portfolio_loop)

class AppExtended(App):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 安裝持倉下拉 Combobox（可輸入/可選）
        try: self._install_symbol_dropdown()
        except Exception as e: self.log(f"[WARN] 安裝持倉下拉 Combobox 失敗：{e}")
        # 修補：account_values 一律在主線執行，避免背景線讀帳戶摘要報錯
        try:
            import types
            def _safe_account_values(ibw_self):
                def _summ():
                    try:
                        summary = {v.tag: v.value for v in ibw_self.ib.accountSummary()}
                        netliq = float(summary.get("NetLiquidation", "0") or 0)
                        cash   = float(summary.get("TotalCashValue", "0") or 0)
                        # --- MODIFIED: Return extra fields here too to match base class ---
                        excess = float(summary.get("ExcessLiquidity", "0") or 0)
                        mktval = float(summary.get("GrossPositionValue", "0") or 0)
                        buypower = float(summary.get("BuyingPower", "0") or 0)
                        return netliq, cash, excess, mktval, buypower
                    except Exception as e:
                        self._status(f"讀取帳戶摘要失敗：{e}")
                        return None, None, None, None, None
                try:
                    return self.run_on_main(_summ)
                except Exception as e:
                    self._status(f"讀取帳戶摘要失敗：{e}")
                    return None, None, None, None, None
            self.ibw.account_values = types.MethodType(_safe_account_values, self.ibw)
        except Exception as e:
            self.log(f"[WARN] 安裝帳戶摘要安全封裝失敗：{e}")

    # -------------------- 安裝持倉下拉 --------------------
    def _install_symbol_dropdown(self):
        ent = getattr(self, "ent_symbol", None)
        if ent is None or not isinstance(ent, (tk.Entry, ttk.Entry)):
            return
        info = ent.grid_info()
        parent = ent.master; row = int(info.get("row", 0)); col = int(info.get("column", 0))
        self.cbo_symbol = ttk.Combobox(parent, width=12, state="normal")
        self.cbo_symbol.grid(row=row, column=col+1, padx=(4,0), pady=info.get("pady",0), sticky=info.get("sticky","w"))
        def _on_pick(event=None):
            sym = (self.cbo_symbol.get() or "").strip().upper()
            try: ent.delete(0,"end"); ent.insert(0, sym)
            except Exception: pass
        self.cbo_symbol.bind("<<ComboboxSelected>>", _on_pick)
        def _refresh_combo():
            try:
                syms = self._get_position_symbols()
                manual = (ent.get() or "").strip().upper()
                vals = sorted(set([*syms, manual]) - {""})
                self.cbo_symbol.configure(values=vals)
            except Exception as e:
                self.log(f"[WARN] 刷新持倉下拉失敗：{e}")
        self.cbo_symbol.configure(postcommand=_refresh_combo)
        self.cbo_symbol.bind("<Return>", _on_pick)

    def _get_position_symbols(self):
        ib: IB = self.ibw.ib
        if not ib.isConnected(): return []
        try: poss = self.run_on_main(ib.positions)
        except Exception: poss = []
        out = []
        for p in poss:
            try:
                sym = getattr(p.contract, "symbol", "") or ""
                if sym: out.append(sym)
            except Exception: pass
        return sorted(set(out))

    # -------------------- 主線橋接（for worker threads） --------------------
    def _call_on_main(self, func, *args, **kwargs):
        if threading.current_thread() is threading.main_thread():
            return func(*args, **kwargs)
        result = {'v': None, 'e': None}
        ev = threading.Event()
        def _run():
            try: result['v'] = func(*args, **kwargs)
            except Exception as e: result['e'] = e
            finally: ev.set()
        try: self.after(0, _run)
        except Exception as e: result['e'] = e; ev.set()
        ev.wait()
        if result['e'] is not None: raise result['e']
        return result['v']

    # -------------------- 監察列表更新 --------------------
    def _watch_update_status(self, symbol: str, status: str):
        try:
            for iid in self.watch_list.get_children():
                vals = list(self.watch_list.item(iid, "values"))
                if not vals: continue
                sym = vals[1] if len(vals)>1 else vals[0]
                if sym == symbol:
                    if len(vals) >= 5:
                        vals[4] = status
                        self.watch_list.item(iid, values=tuple(vals))
                        self.watch_list.see(iid)
                        return True
        except Exception:
            pass
        return False

    def _watch_update_trigger(self, symbol: str, trigger_text: str):
        try:
            for iid in self.watch_list.get_children():
                vals = list(self.watch_list.item(iid, "values"))
                if not vals: continue
                sym = vals[1] if len(vals)>1 else vals[0]
                if sym == symbol:
                    if len(vals) >= 6:
                        vals[5] = trigger_text
                        self.watch_list.item(iid, values=tuple(vals))
                        self.watch_list.see(iid)
                        return True
        except Exception:
            pass
        return False

    # -------------------- SMA10（不含今日） --------------------
    def _sma10_excl_today(self, sym: str) -> float:
        ib: IB = self.ibw.ib
        def _fetch_hist():
            c = Stock(sym, 'SMART', 'USD'); ib.qualifyContracts(c)
            prev_open, prev_close = self.ibw.prev_rth_open_close(sym, now_et())
            return ib.reqHistoricalData(
                c, endDateTime=prev_close, durationStr="30 D",
                barSizeSetting="1 day", whatToShow="TRADES",
                useRTH=True, formatDate=2, keepUpToDate=False)
        try: bars = self._call_on_main(_fetch_hist)
        except Exception as e:
            self.log(f"[ERR] 取日線失敗 {sym}: {e}"); return float('nan')
        closes = []
        for b in bars:
            try: closes.append(float(b.close))
            except Exception: pass
        if len(closes) < 10: return float('nan')
        return float(sum(closes[-10:]) / 10.0)

    # -------------------- 現價 --------------------
    def _last_price_now(self, sym: str, timeout: float = 1.2) -> float:
        ib: IB = self.ibw.ib
        def _req(md_type):
            c = Stock(sym, 'SMART', 'USD'); ib.qualifyContracts(c)
            ib.reqMarketDataType(md_type)
            t = ib.reqMktData(c, "", False, False)
            return c, t
        def _cancel(md):
            c,t = md
            try: ib.cancelMktData(t)
            except Exception: pass
        md = self._call_on_main(lambda: _req(1))
        time.sleep(timeout)
        t = md[1]; last = None
        for cand in (getattr(t,"last",None), getattr(t,"close",None), getattr(t,"bid",None), getattr(t,"ask",None)):
            if cand is not None and not util.isNan(cand):
                last = float(cand); break
        self._call_on_main(lambda: _cancel(md))
        if last is not None: return last
        md2 = self._call_on_main(lambda: _req(3))
        time.sleep(timeout)
        t2 = md2[1]; last2 = None
        for cand in (getattr(t2,"last",None), getattr(t2,"close",None), getattr(t2,"bid",None), getattr(t2,"ask",None)):
            if cand is not None and not util.isNan(cand):
                last2 = float(cand); break
        self._call_on_main(lambda: _cancel(md2))
        return float(last2) if last2 is not None else float('nan')

    # -------------------- 開始監察（收市前2分鐘｜SMA10不含今日） --------------------
    def start_close_monitor_sma10(self):
        sym = (self.ent_symbol.get().strip().upper() or "")
        if not sym:
            messagebox.showwarning("提示", "請先輸入股票代號（或用下拉挑選持倉）。"); return
        if not self.ibw.ib.isConnected():
            messagebox.showwarning("提示", "請先連線 IB。"); return
        if sym in self.watch_items and getattr(self.watch_items[sym], "active", False):
            messagebox.showinfo("已在監察", f"{sym} 已在監察清單中。"); return

        self.log(f"[BTN] 開始監察 {sym}（收市前2分鐘｜SMA10不含今日｜10秒輪詢）")

        w = WatchItem(sym, 0, "收市前2m(連續)", open_dt_et=None, close_dt_et=None)
        w.status = "等待收市前兩分鐘…"
        w.active = True
        self.watch_items[sym] = w
        self._refresh_watch_row(sym)

        sma10 = self._sma10_excl_today(sym)
        if sma10 == sma10: self._watch_update_trigger(sym, f"{sma10:.4f}")
        else:               self._watch_update_trigger(sym, "—")

        th = threading.Thread(target=self._close_mon_worker, args=(sym,), daemon=True)
        self.watch_threads[sym] = th
        th.start()
        messagebox.showinfo("監察開始", f"{sym}（收市前2分鐘｜SMA10不含今日｜10秒輪詢）已加入監察。")

    # -------------------- 背景線：實際監察流程（15:58~16:00 每 10s 連續觀察） --------------------
    def _close_mon_worker(self, sym: str):
        try:
            def is_active():
                w = self.watch_items.get(sym); return (w is not None) and getattr(w, "active", False)

            def _calc_target():
                nowt = now_et()
                prev_open, prev_close = self.ibw.prev_rth_open_close(sym, nowt)
                next_open, next_close = self.ibw.next_rth_open_close(sym, nowt)
                if prev_open <= nowt <= prev_close:   return prev_close
                elif nowt < next_open:                 return next_close
                else:                                   return self.ibw.next_rth_open_close(sym, nowt + timedelta(minutes=1))[1]

            target_close = self._call_on_main(_calc_target)
            self.log(f"[CLOSEMON] 目標收市時間（ET）：{target_close.strftime('%Y-%m-%d %H:%M:%S')}")

            two_min_before = target_close - timedelta(minutes=2)

            # 等到 15:58:00（或該日 RTH 收市前 2 分鐘）
            while now_et() < two_min_before:
                if not is_active():
                    self._watch_update_status(sym, "已取消"); return
                left = int((two_min_before - now_et()).total_seconds())
                self._watch_update_status(sym, f"倒數 {left//60}分{left%60}秒")
                time.sleep(1)

            # 進入 15:58~16:00 監察窗口，先計一次 SMA10（固定用）
            sma10 = self._sma10_excl_today(sym)
            if sma10 == sma10: self._watch_update_trigger(sym, f"{sma10:.4f}")
            else:
                self._watch_update_status(sym, "SMA10不可用"); self._call_on_main(self._remove_watch, sym); return

            # 每 10 秒取一次現價直至 16:00:00
            POLL_SEC = 10
            while now_et() < target_close:
                if not is_active():
                    self._watch_update_status(sym, "已取消"); self._call_on_main(self._remove_watch, sym); return

                last = self._last_price_now(sym, 1.2)
                if last == last:
                    self._watch_update_status(sym, f"觀察中 現價={last:.4f} vs SMA10={sma10:.4f}")
                    if last < sma10:
                        qty = self.ibw.get_position_qty(sym)
                        if qty <= 0:
                            self._watch_update_status(sym, "持倉=0"); self._call_on_main(self._remove_watch, sym); return
                        def _place():
                            ib = self.ibw.ib
                            c = Stock(sym, 'SMART', 'USD'); ib.qualifyContracts(c)
                            o = MarketOrder('SELL', int(qty))
                            return ib.placeOrder(c, o)
                        self._watch_update_status(sym, f"觸發！賣出 {qty}")
                        self._call_on_main(_place)
                        self.log(f"[CLOSEMON] 已送出市價單 SELL {sym} x {qty}")
                        self._watch_update_status(sym, f"已賣出 {qty}")
                        self._call_on_main(self._remove_watch, sym)
                        return
                else:
                    self._watch_update_status(sym, "現價不可用/權限不足")

                # 等待下個輪詢點（同時計算離 16:00 的倒數）
                end = time.time() + POLL_SEC
                while time.time() < end and now_et() < target_close:
                    if not is_active():
                        self._watch_update_status(sym, "已取消"); self._call_on_main(self._remove_watch, sym); return
                    left = int((target_close - now_et()).total_seconds())
                    self._watch_update_status(sym, f"觀察中 剩餘 {left//60}分{left%60}秒（現價每{POLL_SEC}s刷新）")
                    time.sleep(1)

            # 到 16:00 仍未觸發
            self._watch_update_status(sym, "未觸發（現價≥SMA10）")
            self._call_on_main(self._remove_watch, sym)
        except Exception as e:
            self._watch_update_status(sym, f"錯誤：{e}")
            try: self._call_on_main(self._remove_watch, sym)
            except Exception: pass


if __name__ == "__main__":
    app = AppExtended()
    app.mainloop()

def flush_ticks_loop(self):
    try:
        if self.ibw and self.ibw.ib.isConnected():
            try:
                self.ibw.ib.pendingTickers()
            except Exception:
                pass
    finally:
        try:
            self.safe_after(300, self.flush_ticks_loop)
        except Exception:
            pass
