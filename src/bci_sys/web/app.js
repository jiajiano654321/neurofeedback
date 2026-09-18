/* IT4 单页：live 走 WS（snapshot+帧、断档自动重连），replay 走
 * GET /api/replay 一次取全量帧，本地按 offsets_ms 以 1x 调度。
 * 两种传输共用 applyEvent——浏览器所见严格镜像引擎，前端不发明相位。 */
"use strict";

const $ = (id) => document.getElementById(id);

const REASON_CN = {
  user_abort: "用户中止（POST /api/abort 或 Ctrl+C）",
  incomplete_data: "可用整段不足 4200（数据不完整）",
  wall_cap: "墙钟超过 25 分钟硬上限",
  freeze_failed: "S3 θ 冻结失败（成熟有效段不足或阈值非有限）",
  engine_error: "引擎内部错误（详见终端 traceback）",
};

function freshState() {
  return {
    mode: "live", started: false, terminal: false,
    phase: null, params: null,
    subj: null, dataVhdr: null, runDir: null,
    calK: -1, blockIdx: null, segInBlock: 0,
    blockValid: 0, blockHit: 0,
    totals: { valid: 0, hit: 0 },
    blockRates: [], blockInvalid: [],
    fr: null, smoothed: null, verdict: null,
    status: null, reason: null, k: null, t1: null,
    runInfo: null,
  };
}
let S = freshState();
let restEndsAt = 0, restTimer = null;

// ---------- WS（live） ----------
let ws = null, lastSeq = 0, wsShouldRun = true, reconnectMs = 300, reconnectTimer = null;

function connectWs() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  $("ws-dot").className = "dot live";
  ws.onmessage = (ev) => {
    let env;
    try { env = JSON.parse(ev.data); } catch (e) { scheduleReconnect(); return; }
    if (!env || !env.msg) { scheduleReconnect(); return; }
    if (env.msg.type === "snapshot") {
      applySnapshot(env.msg);
      lastSeq = env.seq;
      render();
      return;
    }
    // seq 断档（含丢帧）：不插值、不猜，重连凭 snapshot 追平
    if (env.seq !== lastSeq + 1) { scheduleReconnect(); return; }
    lastSeq = env.seq;
    applyEvent(env.msg);
    render();
    if (env.msg.type === "final" || env.msg.type === "aborted") {
      wsShouldRun = false; S.terminal = true;
    }
  };
  ws.onclose = () => {
    $("ws-dot").className = "off dot";
    scheduleReconnect();  // 重连唯一入口；终态后 wsShouldRun=false 自动不排
  };
  ws.onerror = () => { try { ws.close(); } catch (e) {} };
}
function scheduleReconnect() {
  if (!wsShouldRun) return;
  try { ws.close(); } catch (e) {}  // 已 CLOSED 时 close() 是空操作，不会重复触发
  // 去重：消息路径关流与随之而来的 onclose 都会进本函数，只保留一个挂起重连
  clearTimeout(reconnectTimer);
  reconnectTimer = setTimeout(connectWs, reconnectMs);
}

// ---------- 状态应用（live 帧与 replay 帧共用） ----------
function applySnapshot(m) {
  if (m.block_idx != null && m.block_idx !== S.blockIdx) {
    // 重连追平时已跨块：本块计数从 0 重新累计（snapshot 不携带本块分母，
    // 历史分母不可重建；同块内重连则保留断线前计数，gap 期间帧不可追回）
    S.blockValid = 0; S.blockHit = 0; S.verdict = null;
    document.body.classList.remove("invalid");
  }
  S.phase = m.phase;
  S.blockIdx = m.block_idx;
  S.segInBlock = m.seg_in_phase || 0;
  S.k = m.k; S.t1 = m.t1;
  S.params = m.params;
  S.fr = (m.b_fp1 == null) ? null : {
    b_fp1: m.b_fp1, b_fp2: m.b_fp2, theta_thr_uV2: m.theta_thr,
  };
  S.totals = { valid: m.n_valid || 0, hit: m.n_hit || 0 };
  S.blockRates = m.block_rates || [];
  S.smoothed = m.last_smoothed;
  S.verdict = m.last_verdict;
  S.status = m.status; S.reason = m.reason;
  if (m.k != null) S.started = true;
}

function applyEvent(f) {
  S.started = true;
  switch (f.type) {
    case "calib_tick":
    case "calib_seg":
      S.phase = "calibration"; S.calK = f.k; S.t1 = f.t1; break;
    case "freeze":
      S.fr = f; S.phase = "freeze_card";
      S.k = null; S.t1 = f.t_sec; break;
    case "phase":
      if (f.phase === "block_start") {
        S.phase = "block"; S.blockIdx = f.block_idx;
        S.segInBlock = 0; S.blockValid = 0; S.blockHit = 0;
        S.verdict = null; document.body.classList.remove("invalid");
        S.t1 = f.t1;
      } else if (f.phase === "rest_start") {
        S.phase = "rest";
        restEndsAt = performance.now() + S.params.rest_sec * 1000;
        startRestTicker();
      } else if (f.phase === "rest_end") {
        /* 相位以随后的 block_start 为准，这里不动 */
      }
      break;
    case "seg":
      S.phase = "block"; S.blockIdx = f.block_idx;
      S.segInBlock = f.seg_in_block + 1; S.k = f.k; S.t1 = f.t1;
      S.verdict = { valid: f.valid, reasons: f.reasons,
                    fz_clean_p2p_uv: f.fz_clean_p2p_uv,
                    eog_p2p_uv: f.eog_p2p_uv };
      if (f.valid) {
        S.blockValid += 1; S.totals.valid += 1;
        S.smoothed = f.smoothed_uV2;
        if (f.hit) { S.blockHit += 1; S.totals.hit += 1; }
        // 无效类统一由 renderBlock 按 verdict 切换，此处不动 body 类
      }
      break;
    case "block_summary":
      S.blockRates[f.block_idx] = f.hit_rate;
      S.blockInvalid[f.block_idx] = f.n_invalid;
      break;
    case "final":
      S.phase = "finished"; S.status = "ok"; S.terminal = true;
      S.blockRates = f.block_rates;
      if (S.mode === "live") fetchRunInfo();
      break;
    case "aborted":
      S.phase = "aborted"; S.status = "aborted";
      S.reason = f.reason; S.k = f.k; S.t1 = f.t1;
      S.terminal = true;
      if (S.mode === "live") fetchRunInfo();
      break;
  }
}

async function fetchRunInfo() {
  try {
    S.runInfo = await (await fetch("/api/run")).json();
    render();  // 终态后重拉 run.json：墙钟/数据时长/产物路径要立即反映到 finished 屏
  } catch (e) {}
}

// ---------- 渲染 ----------
function show(id) {
  for (const s of document.querySelectorAll(".screen"))
    s.classList.toggle("hidden", s.id !== id);
}
function mmss(x) {
  x = Math.max(0, Math.round(x));
  return `${String(Math.floor(x / 60)).padStart(2, "0")}:` +
         `${String(x % 60).padStart(2, "0")}`;
}
function pct(x) { return (x * 100).toFixed(1) + "%"; }

function render() {
  renderConfirm();
  if (!S.started && !(S.phase === "finished" || S.phase === "aborted")) {
    show("screen-confirm"); return;
  }
  switch (S.phase) {
    case "calibration": renderCalibration(); break;
    case "freeze_card": renderFreeze(); break;
    case "block": renderBlock(); break;
    case "rest": break; // 倒计时由 ticker 自己画，相位切换已 show
    case "finished": renderFinished(); break;
    case "aborted": renderAborted(); break;
  }
  if (S.phase === "rest") show("screen-rest");
}

function renderConfirm() {
  const kv = $("confirm-kv");
  if (!S.params && S.mode === "replay" && RP.meta) S.params = RP.meta.params;
  if (!S.params) return;
  const p = S.params;
  kv.innerHTML =
    `被试：<b>sub-${S.subj ?? "001"}</b>　数据：<b>${S.dataVhdr ?? "—"}</b><br>` +
    `靶点：<b>θ ${p.theta_band_hz[0]}–${p.theta_band_hz[1]} Hz @ Fz，` +
    `direction = ${targetDir()}</b><br>` +
    `S3 校准 <b>${p.qc_calibration_sec}s</b>　S4 <b>${p.n_blocks}×${p.block_sec}s</b>` +
    `　休息 <b>${p.rest_sec}s ×${p.n_blocks - 1}</b>（数据钟暂停）<br>` +
    `θ 平滑 <b>W=${p.theta_smooth_segs} 段（仅有效段）</b>　` +
    `阈值 <b>S3 P${p.theta_quantile * 100}</b>` +
    `（目标达标率 ${p.theta_target_rate}，仅冻结用）<br>` +
    `QC：L1 残差 p2p &gt; ${p.l1_p2p_uv}µV　` +
    `L3 FP1/FP2 任一 p2p &gt; ${p.l3_eog_p2p_uv}µV<br>` +
    `墙钟硬上限 ${p.session_wall_cap_sec}s`;
  $("btn-start").textContent =
    S.mode === "replay" ? "开始回放（1x）" : "确认参数并开始 S3 校准";
}
function targetDir() {
  return S.runInfo?.target?.direction ?? (RP.meta?.target?.direction) ?? "up";
}

let dotsBuilt = false;
function renderCalibration() {
  show("screen-calibration");
  if (!dotsBuilt) {
    const box = $("cal-dots");
    for (let i = 0; i < 600; i++) {
      const d = document.createElement("i");
      box.appendChild(d);
    }
    dotsBuilt = true;
  }
  const kids = $("cal-dots").children;
  for (let i = 0; i < 600; i++)
    kids[i].classList.toggle("on", i <= S.calK);
  const sec = Math.min((S.calK + 1) * (S.params?.segment_sec ?? 0.2),
                       S.params?.qc_calibration_sec ?? 120);
  $("cal-time").textContent = mmss(sec);
}

function renderFreeze() {
  show("screen-freeze");
  const f = S.fr;
  $("freeze-kv").innerHTML =
    `b_FP1 = <b>${f.b_fp1.toFixed(4)}</b>　b_FP2 = <b>${f.b_fp2.toFixed(4)}</b><br>` +
    `θ 阈值 = <b>${f.theta_thr_uV2.toFixed(2)} µV²</b>` +
    `（S3 有效段 P${(f.quantile ?? S.params.theta_quantile) * 100}，` +
    `n=${f.n_thr_segs}，实测校准达标率 ${(f.calib_hit_rate * 100).toFixed(1)}%）<br>` +
    `direction = <b>${f.direction ?? targetDir()}</b>`;
}

function renderBlock() {
  show("screen-block");
  const b = S.blockIdx ?? 0;
  $("block-dots").innerHTML =
    `<b>${"●".repeat(b + 1)}</b>${"○".repeat(Math.max(0, 5 - b))}`;
  $("block-label").textContent = `BLOCK ${b + 1} / ${S.params.n_blocks}`;
  const left = S.params.block_sec - S.segInBlock * S.params.segment_sec;
  $("block-timeleft").textContent = `剩余 ${mmss(left)}`;

  const invalid = !!(S.verdict && !S.verdict.valid);
  document.body.classList.toggle("invalid", invalid);
  const chip = $("qc-chip");
  if (invalid) {
    chip.className = "chip bad";
    chip.textContent = `✗ 无效段 · 反馈暂停（${(S.verdict.reasons || []).join("/") || "?"}）`;
  } else {
    chip.className = "chip ok";
    chip.textContent = "● 有效 · 反馈运行";
  }

  const rate = S.blockValid ? S.blockHit / S.blockValid : 0;
  $("hit-fill").style.width = pct(rate);
  $("hit-bar").classList.toggle("paused", invalid);
  $("hit-label").textContent =
    `本块达标 ${S.blockHit} / ${S.blockValid}（白线＝80% 目标刻线；仅有效段计分）`;

  if (S.smoothed != null && S.fr) {
    const w = Math.max(0, Math.min(100,
      S.smoothed / (2 * S.fr.theta_thr_uV2) * 100));
    $("theta-fill").style.width = w + "%";
  }
  $("theta-label").textContent =
    "θ 平滑功率 " + (S.smoothed == null ? "—（窗未成熟）"
      : S.smoothed.toFixed(1) + " µV²") + "　黄线＝S3 冻结阈值";

  $("invalid-banner").textContent = invalid
    ? `本段无效（${(S.verdict.reasons || []).join("、") || "未知原因"}）` +
      " · 反馈暂停 · 不计入达标率"
    : "";
}

function startRestTicker() {
  clearInterval(restTimer);
  show("screen-rest");
  const prev = S.blockRates.length ? S.blockRates[S.blockRates.length - 1] : null;
  $("rest-prev").textContent = prev == null ? ""
    : `上一块达标率 ${pct(prev)}（仅记录，阈值不变）`;
  restTimer = setInterval(() => {
    const left = (restEndsAt - performance.now()) / 1000;
    $("rest-countdown").textContent = mmss(left);
    if (left <= 0) clearInterval(restTimer);
  }, 200);
  $("rest-countdown").textContent = mmss(S.params.rest_sec);
}

function renderFinished() {
  show("screen-finished");
  const n = S.params.n_blocks;
  $("rates-head").innerHTML = "<th>block</th>" +
    Array.from({ length: n }, (_, i) => `<th>${i + 1}</th>`).join("");
  $("rates-rate").innerHTML = "<th>达标率</th>" +
    Array.from({ length: n }, (_, i) =>
      `<td>${S.blockRates[i] == null ? "—" : pct(S.blockRates[i])}</td>`).join("");
  $("rates-invalid").innerHTML = "<th>无效段</th>" +
    Array.from({ length: n }, (_, i) =>
      `<td>${S.blockInvalid[i] ?? "—"}</td>`).join("");
  let wall, data;
  if (S.runInfo && S.runInfo.wall_elapsed_s != null) {
    wall = S.runInfo.wall_elapsed_s; data = S.runInfo.data_elapsed_s;
    S.runDir = S.runInfo.run_dir;
  } else if (S.mode === "replay") {
    wall = RP.offs[RP.offs.length - 1] / 1000;
    data = S.params.qc_calibration_sec + S.params.n_blocks * S.params.block_sec;
  } else { wall = "—"; data = S.t1 ?? "—"; }
  $("finished-kv").innerHTML =
    `会话完成 · sub-${S.subj ?? "001"} · 数据 <b>${data}s</b>` +
    ` / 墙钟 <b>${typeof wall === "number" ? wall.toFixed(1) : wall}s</b>`;
  const rd = $("finished-rundir");
  if (S.runDir) { rd.style.display = ""; rd.textContent = `产物目录：${S.runDir}`; }
  else { rd.style.display = "none"; }
}

function renderAborted() {
  show("screen-aborted");
  $("aborted-kv").innerHTML =
    `原因：<b>${REASON_CN[S.reason] ?? S.reason}</b><br>` +
    `已完成到：block ${S.blockIdx == null ? "—" : S.blockIdx + 1}` +
    `（k=${S.k ?? "—"}，t1=${S.t1 ?? "—"}s）<br>` +
    `run.json status=aborted；本次数据已留痕，标记为中止不删除。`;
}

// ---------- replay 播放器 ----------
const RP = { meta: null, frames: [], offs: [], boundaries: [],
             idx: -1, playing: false, timer: null,
             anchorWall: 0, anchorOff: 0 };

async function bootReplay(file) {
  document.body.classList.add("replay");
  document.querySelectorAll(".abortbtn").forEach((b) => { b.style.display = "none"; });
  $("replay-bar").classList.remove("hidden");
  wsShouldRun = false;
  $("ws-dot").className = "dot";
  S.mode = "replay";
  const pl = await (await fetch(`/api/replay?file=${encodeURIComponent(file)}`)).json();
  RP.meta = pl.meta; RP.frames = pl.frames; RP.offs = pl.offsets_ms;
  RP.boundaries = pl.boundaries;
  S.params = pl.meta.params;
  const seek = $("replay-seek");
  seek.max = String(RP.offs[RP.offs.length - 1] || 0);
  renderReplayTime();
  render();
}

function replayStep() {
  if (!RP.playing) return;
  if (RP.idx + 1 >= RP.frames.length) {
    RP.playing = false; $("btn-replay-play").textContent = "▶ 播放";
    return;
  }
  const next = RP.idx + 1;
  const wait = RP.offs[next] - RP.anchorOff - (performance.now() - RP.anchorWall);
  RP.timer = setTimeout(() => {
    RP.idx = next;
    applyEvent(RP.frames[RP.idx]);
    render();
    $("replay-seek").value = String(RP.offs[RP.idx]);
    renderReplayTime();
    replayStep();
  }, Math.max(0, wait));
}
function replayPlay() {
  if (RP.playing) { // 暂停
    RP.playing = false; clearTimeout(RP.timer);
    $("btn-replay-play").textContent = "▶ 继续";
    return;
  }
  RP.playing = true;
  if (RP.idx + 1 >= RP.frames.length) { RP.idx = -1; resetReplayState(); }
  if (RP.idx < 0) {  // 首帧 200ms 后才到，先定格校准屏，避免所有 section 同闪
    S.phase = "calibration"; S.calK = -1; render();
  }
  RP.anchorWall = performance.now();
  RP.anchorOff = RP.idx >= 0 ? RP.offs[RP.idx] : 0;
  $("btn-replay-play").textContent = "⏸ 暂停";
  replayStep();
}
function resetReplayState() {
  const subj = S.subj, vhdr = S.dataVhdr, mode = S.mode;
  S = freshState();
  S.mode = mode; S.subj = subj; S.dataVhdr = vhdr; S.params = RP.meta.params;
  document.body.classList.remove("invalid");
}
function replaySeekTo(wallMs) {
  // 吸附到该时刻或之前最近相位边界，从头 reduce 重建 snapshot（与后端口径一致）
  RP.playing = false; clearTimeout(RP.timer);
  $("btn-replay-play").textContent = "▶ 继续";
  let bi = -1;
  for (const bIdx of RP.boundaries) {
    if (RP.offs[bIdx] <= wallMs) bi = bIdx; else break;
  }
  resetReplayState();
  for (let i = 0; i <= bi; i++) applyEvent(RP.frames[i]);
  RP.idx = bi;
  RP.anchorOff = bi >= 0 ? RP.offs[bi] : 0;
  render();
  renderReplayTime();
}
function renderReplayTime() {
  const cur = RP.idx >= 0 ? RP.offs[RP.idx] / 1000 : 0;
  const end = (RP.offs[RP.offs.length - 1] || 0) / 1000;
  $("replay-time").textContent = `${mmss(cur)} / ${mmss(end)}`;
}

// ---------- 启动 ----------
async function boot() {
  const info = await (await fetch("/api/run")).json();
  S.subj = info.subj; S.dataVhdr = info.data_vhdr; S.runInfo = info;
  for (const id of ["btn-abort-cal", "btn-abort-freeze", "btn-abort-block",
                    "btn-abort-rest"])
    $(id).addEventListener("click", doAbort);
  $("btn-start").addEventListener("click", onStart);
  $("btn-replay-play").addEventListener("click", replayPlay);
  $("replay-seek").addEventListener("input", (e) =>
    replaySeekTo(Number(e.target.value)));
  if (info.mode === "replay") {
    await bootReplay(info.replay_file);
  } else {
    connectWs();
  }
  render();
}

function onStart() {
  if (S.mode === "replay") { replayPlay(); return; }
  $("btn-start").disabled = true;
  fetch("/api/start", { method: "POST" })
    .then((r) => { if (!r.ok) throw new Error(String(r.status)); })
    .then(() => { S.started = true; render(); })
    .catch((e) => { $("btn-start").disabled = false;
                    alert("启动失败：" + e.message); });
}
async function doAbort() {
  const r = await fetch("/api/abort", { method: "POST" });
  if (!r.ok) alert("中止请求失败：" + r.status);  // 202 Accepted 属成功，不报警
}

boot();
