/* Run 3 live dashboard. Vanilla JS plus uPlot. No build step. */
(function () {
  'use strict';

  var POLL_MS = 15000;
  var $ = function (s, r) { return (r || document).querySelector(s); };

  function h(tag, attrs) {
    var e = document.createElement(tag);
    if (attrs) for (var k in attrs) {
      if (k === 'class') e.className = attrs[k];
      else if (k === 'text') e.textContent = attrs[k];
      else if (k === 'style') e.setAttribute('style', attrs[k]);
      else if (k.slice(0, 2) === 'on') e.addEventListener(k.slice(2), attrs[k]);
      else e.setAttribute(k, attrs[k]);
    }
    var kids = [];
    for (var i = 2; i < arguments.length; i++) kids = kids.concat(arguments[i]);
    for (var j = 0; j < kids.length; j++) {
      var c = kids[j];
      if (c == null || c === false) continue;
      e.appendChild(typeof c === 'string' || typeof c === 'number' ? document.createTextNode(String(c)) : c);
    }
    return e;
  }
  function isNum(v) { return typeof v === 'number' && isFinite(v); }
  function num(v) { return isNum(v) ? v : null; }
  function clear(e) { while (e.firstChild) e.removeChild(e.firstChild); }
  function tget(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }
  function tset(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* storage may be blocked */ } }

  /* ---------- formatting ---------- */
  function fmtNum(v) {
    if (!isNum(v)) return '–';
    var a = Math.abs(v);
    if (a === 0) return '0';
    if (a < 0.001 || a >= 1e5) return v.toExponential(1).replace('e+', 'e');
    if (a < 1) return v.toFixed(3);
    if (a < 100) return v.toFixed(2);
    return v.toFixed(0);
  }
  function fmtPct(v) { return isNum(v) ? (v * 100).toFixed(v < 0.1 && v > 0 ? 1 : 0) + '%' : '–'; }
  function fmtUsd(v) { return isNum(v) ? '$' + v.toFixed(2) : '–'; }
  function fmtDur(s) {
    if (!isNum(s) || s < 0) return '–';
    s = Math.round(s);
    var hh = Math.floor(s / 3600), mm = Math.floor((s % 3600) / 60);
    if (hh > 0) return hh + ' h ' + (mm < 10 ? '0' : '') + mm + ' min';
    if (mm > 0) return mm + ' min';
    return s + ' s';
  }
  function fmtAgo(s) {
    if (s < 0) s = 0;
    if (s < 90) return Math.round(s) + ' s ago';
    if (s < 5400) return Math.round(s / 60) + ' min ago';
    return (s / 3600).toFixed(1) + ' h ago';
  }
  function parseT(t) { var d = Date.parse(t); return isNaN(d) ? null : d; }
  function fmtClock(ms) {
    var d = new Date(ms);
    var p = function (n) { return (n < 10 ? '0' : '') + n; };
    var today = new Date();
    var day = d.toDateString() === today.toDateString() ? '' : (d.getMonth() + 1) + '/' + d.getDate() + ' ';
    return day + p(d.getHours()) + ':' + p(d.getMinutes());
  }

  /* ---------- theme ---------- */
  var themeMode = tget('run3-theme') || 'auto';
  function applyTheme() {
    var root = document.documentElement;
    if (themeMode === 'auto') root.removeAttribute('data-theme'); else root.setAttribute('data-theme', themeMode);
    $('#theme').textContent = themeMode;
  }
  function isDark() {
    var t = document.documentElement.getAttribute('data-theme');
    if (t) return t === 'dark';
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  }
  $('#theme').addEventListener('click', function () {
    themeMode = themeMode === 'auto' ? 'light' : themeMode === 'light' ? 'dark' : 'auto';
    tset('run3-theme', themeMode);
    applyTheme();
    rebuildCharts();
  });
  if (window.matchMedia) {
    var mq = window.matchMedia('(prefers-color-scheme: dark)');
    var onMq = function () { rebuildCharts(); };
    if (mq.addEventListener) mq.addEventListener('change', onMq); else if (mq.addListener) mq.addListener(onMq);
  }
  applyTheme();

  var cssCache = null;
  function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  function palette() {
    return {
      ink: cssVar('--alm-ink'), ink2: cssVar('--alm-ink2'), hair: cssVar('--alm-hairline'),
      outline: cssVar('--alm-outline'),
      sage: cssVar('--alm-sage'), oxblood: cssVar('--alm-oxblood'), amber: cssVar('--alm-amber'),
      plum: cssVar('--alm-plum'), paper: cssVar('--alm-paper')
    };
  }
  function alpha(hex, a) {
    var m = /^#([0-9a-f]{6})$/i.exec(hex);
    if (!m) return hex;
    var n = parseInt(m[1], 16);
    return 'rgba(' + (n >> 16) + ',' + ((n >> 8) & 255) + ',' + (n & 255) + ',' + a + ')';
  }

  /* ---------- data store ---------- */
  var S = {
    state: null, evals: null, notes: null, samples: null,
    metrics: { rows: [], offset: 0 }, system: { rows: [], offset: 0 }, events: { rows: [], offset: 0 },
    skew: 0, lastOk: 0, errors: {}
  };

  function noteDate(resp) {
    var d = resp.headers.get('Date');
    if (d) { var ms = Date.parse(d); if (!isNaN(ms)) S.skew = ms - Date.now(); }
  }
  function fetchJSON(path) {
    return fetch(path, { cache: 'no-store' }).then(function (r) {
      noteDate(r);
      if (r.status === 404) return null;
      if (!r.ok) throw new Error(path + ' ' + r.status);
      return r.json();
    });
  }

  /* Incremental JSON-lines reader using HTTP Range requests. */
  function fetchLines(path, store) {
    var headers = {};
    if (store.offset > 0) headers.Range = 'bytes=' + store.offset + '-';
    return fetch(path, { cache: 'no-store', headers: headers }).then(function (r) {
      noteDate(r);
      if (r.status === 404) { store.rows = []; store.offset = 0; return false; }
      if (r.status === 204) return false;
      if (r.status === 416) {
        var cr = r.headers.get('Content-Range') || '';
        var m = /\*\/(\d+)/.exec(cr);
        if (m && parseInt(m[1], 10) === store.offset) return false;
        store.rows = []; store.offset = 0;
        return fetchLines(path, store);
      }
      if (!r.ok) throw new Error(path + ' ' + r.status);
      var partial = r.status === 206;
      return r.arrayBuffer().then(function (buf) {
        var bytes = new Uint8Array(buf);
        var end = bytes.lastIndexOf(10);
        if (end < 0) return false;
        var text = new TextDecoder().decode(bytes.subarray(0, end + 1));
        var rows = partial ? store.rows.slice() : [];
        var changed = false;
        text.split('\n').forEach(function (line) {
          line = line.trim();
          if (!line) return;
          try { rows.push(JSON.parse(line)); changed = true; } catch (e) { /* skip a bad line */ }
        });
        store.rows = rows;
        store.offset = (partial ? store.offset : 0) + end + 1;
        return changed;
      });
    });
  }

  /* ---------- header, cost, progress ---------- */
  var STATE_TONE = {
    running: 'sage', done: 'sage', failed: 'oxblood', idle: '',
    creating_pod: 'ochre', bootstrapping: 'ochre', restarting: 'ochre', recreating_pod: 'ochre',
    stopping: 'ochre', uploading: 'ochre', deleting: 'ochre'
  };
  function nowMs() { return Date.now() + S.skew; }

  function renderHeader() {
    var st = S.state;
    var badge = $('#badge');
    if (!st) { $('#badge-text').textContent = 'no data'; $('#updated').textContent = 'waiting for the monitor'; return; }
    var rs = st.run_state || 'unknown';
    badge.className = 'badge ' + rs + ' tone-' + (STATE_TONE[rs] || '');
    $('#badge-text').textContent = rs.replace(/_/g, ' ');
    $('#demo-banner').hidden = !st.demo;
    var phase = st.current_phase;
    var g = st.grpo || {};
    var bits = [];
    if (phase) bits.push('phase: ' + phase);
    if (isNum(g.step) && isNum(g.max_steps)) bits.push('step ' + g.step + '/' + g.max_steps);
    $('#phase-now').textContent = bits.join(' · ');
    var eb = $('#error-banner');
    if (st.last_error) { eb.hidden = false; eb.textContent = 'Last error: ' + String(st.last_error).slice(0, 400); } else eb.hidden = true;
    tickUpdated();
  }
  function tickUpdated() {
    var st = S.state;
    if (!st) return;
    var t = parseT(st.updated);
    var el = $('#updated'), sb = $('#stale-banner');
    if (t == null) { el.textContent = ''; return; }
    var age = (nowMs() - t) / 1000;
    el.textContent = 'updated ' + fmtAgo(age);
    var active = ['running', 'creating_pod', 'bootstrapping', 'restarting', 'recreating_pod', 'stopping', 'uploading', 'deleting'].indexOf(st.run_state) >= 0;
    if (active && !st.demo && age > 300) {
      sb.hidden = false;
      sb.textContent = 'The monitor has not reported for ' + fmtDur(age) + '. The numbers below may be out of date.';
    } else sb.hidden = true;
  }

  function stat(label, value, unit, sub, cls) {
    return h('div', { class: 'stat' + (cls ? ' ' + cls : '') },
      h('span', { class: 'alm-label', text: label }),
      h('div', { class: 'num' }, value, unit ? h('span', { class: 'unit', text: unit }) : null),
      h('div', { class: 'sub-line', text: sub || ' ' }));
  }

  function renderCost() {
    var st = S.state || {};
    var box = $('#cost-stats'); clear(box);
    var budget = isNum(st.budget_usd) ? st.budget_usd : 50;
    var spend = isNum(st.spend_usd) ? st.spend_usd : 0;
    var rate = st.pod && isNum(st.pod.cost_per_hr) ? st.pod.cost_per_hr : null;
    var sys = S.system.rows;
    if (rate == null && sys.length > 20) {
      var a = sys[sys.length - 1], b = sys[sys.length - 21];
      var dt = (parseT(a.t) - parseT(b.t)) / 3600000;
      if (dt > 0 && isNum(a.spend_usd) && isNum(b.spend_usd)) rate = (a.spend_usd - b.spend_usd) / dt;
    }
    var eta = st.eta ? parseT(st.eta) : null;
    var remainH = eta != null ? Math.max(0, (eta - nowMs()) / 3600000) : null;
    var finished = st.run_state === 'done';
    var projected = finished ? spend : (remainH != null && rate != null ? spend + rate * remainH : null);
    var over = projected != null && projected > budget;
    box.appendChild(stat('Spent so far', fmtUsd(spend), 'of ' + fmtUsd(budget), Math.round(spend / budget * 100) + '% of the cap'));
    box.appendChild(stat('Rate', rate != null ? fmtUsd(rate) : '–', '/ hour', st.pod && st.pod.gpu ? st.pod.gpu : ''));
    box.appendChild(stat('Time left', finished ? 'Done' : (remainH != null ? fmtDur(remainH * 3600) : '–'),
      '', eta != null && !finished ? 'about ' + fmtClock(eta) : (st.started ? 'started ' + fmtClock(parseT(st.started)) : '')));
    box.appendChild(stat('Projected total', projected != null ? fmtUsd(projected) : '–', '',
      projected != null ? (over ? 'over the cap by ' + fmtUsd(projected - budget) : fmtUsd(budget - projected) + ' to spare') : 'needs an estimated finish time', over ? 'bad' : ''));
    var m = $('#budget-meter'); clear(m);
    m.appendChild(h('div', { class: 'fill', style: 'width:' + Math.min(100, spend / budget * 100) + '%' }));
    if (projected != null && !finished) {
      var pp = Math.min(100, projected / budget * 100);
      m.appendChild(h('div', { class: 'proj' + (pp > 80 ? ' flip' : ''), style: 'left:' + pp + '%' }));
    }
    var bits = [];
    if (st.pod && isNum(st.pod.generation)) bits.push('pod generation ' + st.pod.generation);
    if (isNum(st.restarts)) bits.push(st.restarts + ' restart' + (st.restarts === 1 ? '' : 's'));
    if (st.pod && isNum(st.pod.uptime_s)) bits.push('pod up ' + fmtDur(st.pod.uptime_s));
    $('#cost-note').textContent = bits.join(' · ');
  }

  function renderProgress() {
    var st = S.state || {};
    var phases = Array.isArray(st.phases) ? st.phases : [];
    var bar = $('#phasebar'); clear(bar);
    var list = $('#phase-list'); clear(list);
    phases.forEach(function (p) {
      var status = p.status || 'pending';
      bar.appendChild(h('div', { class: 'seg ' + status, title: p.name + ': ' + status }));
      var t0 = parseT(p.started), t1 = parseT(p.finished);
      var dur = t0 != null ? ((t1 != null ? t1 : nowMs()) - t0) / 1000 : null;
      var right = status === 'pending' ? 'waiting' : (dur != null ? fmtDur(dur) : '') + (status === 'running' ? ' so far' : '');
      list.appendChild(h('li', { class: 'st-' + status },
        h('span', { class: 'dot' }),
        h('span', null, p.name.replace(/_/g, ' '), ' ', h('span', { class: 'alm-data', text: status === 'running' ? '— running' : (status === 'failed' ? '— failed' : '') })),
        h('span', { class: 'alm-data', text: right })));
    });
    if (!phases.length) list.appendChild(h('li', null, h('span'), h('span', { class: 'empty', text: 'No phases reported yet.' }), h('span')));
    // GRPO
    var g = st.grpo || {};
    var tr = $('#grpo-track'); clear(tr);
    var wrap = $('#grpo');
    if (!isNum(g.step) || !isNum(g.max_steps) || g.max_steps <= 0) { wrap.hidden = true; return; }
    wrap.hidden = false;
    var pct = Math.min(100, g.step / g.max_steps * 100);
    tr.appendChild(h('div', { class: 'fill', style: 'width:' + pct + '%' }));
    if (isNum(g.last_checkpoint) && g.last_checkpoint > 0) tr.appendChild(h('div', { class: 'tick', style: 'left:' + (g.last_checkpoint / g.max_steps * 100) + '%' }, h('span', { text: 'saved ' + g.last_checkpoint })));
    if (isNum(g.synced_checkpoint) && g.synced_checkpoint > 0 && g.synced_checkpoint !== g.last_checkpoint) tr.appendChild(h('div', { class: 'tick synced', style: 'left:' + (g.synced_checkpoint / g.max_steps * 100) + '%' }, h('span', { text: 'on HF ' + g.synced_checkpoint })));
    else if (isNum(g.synced_checkpoint) && g.synced_checkpoint > 0) tr.lastChild.firstChild.textContent = 'saved + synced ' + g.synced_checkpoint;
    $('#grpo-text').textContent = 'step ' + g.step + ' of ' + g.max_steps + ' (' + pct.toFixed(0) + '%)';
    var rows = S.metrics.rows.slice(-10), per = [];
    rows.forEach(function (r) { var t = (num(r['time/rollout_s']) || 0) + (num(r['time/train_s']) || 0) + (num(r['time/sync_s']) || 0); if (t > 0) per.push(t); });
    var note = '';
    if (per.length) {
      var avg = per.reduce(function (a, b) { return a + b; }, 0) / per.length;
      note = 'About ' + fmtDur(avg) + ' per step recently';
      if (st.run_state === 'running' && g.step < g.max_steps) note += '; about ' + fmtDur(avg * (g.max_steps - g.step)) + ' of training left.';
      else note += '.';
    }
    $('#grpo-note').textContent = note;
  }

  /* ---------- metric charts ---------- */
  var charts = [];
  var smoothing = parseFloat(tget('run3-smooth'));
  if (!isFinite(smoothing)) smoothing = 0.6;
  var smoothInput = $('#smooth');
  smoothInput.value = smoothing;
  $('#smooth-val').textContent = smoothing.toFixed(2);

  function ema(arr, s) {
    var out = new Array(arr.length), prev = null;
    for (var i = 0; i < arr.length; i++) {
      var v = arr[i];
      if (v == null) { out[i] = null; continue; }
      prev = prev == null ? v : s * prev + (1 - s) * v;
      out[i] = prev;
    }
    return out;
  }
  function frac(a, b) { return isNum(a) && isNum(b) && b > 0 ? a / b : null; }

  function pctAxis(u, vals) { return vals.map(function (v) { return Math.round(v * 100) + '%'; }); }
  function numAxis(u, vals) { return vals.map(fmtNum); }
  function stepAxis(u, vals) { return vals.map(function (v) { return Number.isInteger(v) ? String(v) : ''; }); }

  var METRIC_SPECS = [
    // group, id, title, desc, series, options
    ['g-signal', 'success', 'Success rate', 'Share of training attempts that finished the task.',
      [{ key: 'success/overall', label: 'success', tone: 'sage', ema: true }], { frac: true }],
    ['g-signal', 'reward', 'Reward', 'Average reward per attempt, including bonuses and penalties.',
      [{ key: 'reward/total', label: 'total', tone: 'ink', ema: true }, { key: 'reward/success', label: 'from success', tone: 'sage', ema: true, dash: [4, 3] }], { min0: true }],
    ['g-signal', 'eval', 'Mid-training exam', 'Success on a fixed exam run every few steps.',
      [{ key: 'eval/success', label: 'exam success', tone: 'plum', pts: true }], { frac: true }],
    ['g-signal', 'fail', 'Why attempts fail', 'Share of attempts ending each way (smoothed).',
      [{ key: 'failure/ran_out_of_turns', label: 'ran out of turns', tone: 'amber', ema: true },
       { key: 'failure/wrong_action', label: 'wrong action', tone: 'oxblood', ema: true, dash: [5, 3] },
       { key: 'failure/invalid_action', label: 'invalid action', tone: 'plum', ema: true, dash: [1, 3] },
       { key: 'failure/format_error', label: 'format error', tone: 'ink', ema: true, dash: [8, 3, 2, 3] }], { frac: true }],
    ['g-signal', 'turns', 'Turns per attempt', 'Average number of actions before the attempt ended.',
      [{ key: 'turns/mean', label: 'turns', tone: 'ink', ema: true }], { min0: true }],
    ['g-signal', 'epc', 'Attempts over time', 'Cumulative attempts played versus attempts used for learning.',
      [{ key: 'episodes/count', label: 'played', tone: 'ink', cum: true },
       { key: 'episodes/trained', label: 'trained on', tone: 'sage', cum: true }], { min0: true, noTipRaw: true }],

    ['g-groups', 'mixed', 'Mixed and wasted groups', 'Mixed groups (some wins, some losses) are the only ones that teach.',
      [{ get: function (r) { return frac(r['dyn/mixed_groups'], r['dyn/groups_rolled']); }, label: 'mixed (of rolled)', tone: 'sage', ema: true },
       { key: 'groups/zero_variance_frac', label: 'zero-variance', tone: 'oxblood', ema: true, dash: [5, 3] }], { frac: true }],
    ['g-groups', 'rolled', 'Groups rolled and used', 'Groups played to fill one step versus groups that went into the update.',
      [{ key: 'dyn/groups_rolled', label: 'rolled', tone: 'ink', ema: true }, { key: 'groups/used', label: 'used', tone: 'sage', ema: true }], { min0: true }],
    ['g-groups', 'rounds', 'Dynamic sampling rounds', 'Rounds needed to find enough mixed groups.',
      [{ key: 'dyn/rounds', label: 'rounds', tone: 'amber', ema: true }], { min0: true }],
    ['g-groups', 'gstd', 'Reward spread inside groups', 'Mean standard deviation of reward within a group.',
      [{ key: 'groups/mean_std', label: 'mean std', tone: 'ink', ema: true }], { min0: true }],

    ['g-optim', 'loss', 'Loss', 'The training loss at each step.',
      [{ key: 'loss', label: 'loss', tone: 'ink', ema: true }], {}],
    ['g-optim', 'kl', 'KL from the starting model', 'How far the policy has drifted.',
      [{ key: 'train/approx_kl', label: 'approx KL', tone: 'plum', ema: true }], { min0: true }],
    ['g-optim', 'gn', 'Gradient norm', 'Size of the update before clipping.',
      [{ key: 'grad_norm', label: 'grad norm', tone: 'ink', ema: true }], { min0: true }],
    ['g-optim', 'clip', 'Clip fraction', 'Share of updates cut by the safety limit.',
      [{ key: 'train/clip_frac', label: 'clip fraction', tone: 'amber', ema: true }], { min0: true }],
    ['g-optim', 'ent', 'Entropy', 'How undecided the model is between answers.',
      [{ key: 'train/entropy_proxy', label: 'entropy', tone: 'ink', ema: true }], { min0: true }],
    ['g-optim', 'lr', 'Learning rate', 'How big each update step is allowed to be.',
      [{ key: 'lr', label: 'learning rate', tone: 'ink' }], { min0: true }],
    ['g-optim', 'ratio', 'Policy ratio', 'Largest and mean ratio of new to old probabilities.',
      [{ key: 'train/ratio_max', label: 'max', tone: 'oxblood', ema: true }, { key: 'train/ratio_mean', label: 'mean', tone: 'ink', ema: true }], {}],
    ['g-optim', 'mism', 'Sampler mismatch', 'Gap between the sampling engine and the trainer.',
      [{ key: 'sampler/mean_abs_logp_diff', label: 'mean |log p diff|', tone: 'ink', ema: true }], { min0: true }],

    ['g-speed', 'time', 'Step time, split', 'Seconds per step spent playing attempts, learning, and syncing weights.',
      [{ key: 'time/rollout_s', label: 'rollout', tone: 'amber' }, { key: 'time/train_s', label: 'train', tone: 'ink' }, { key: 'time/sync_s', label: 'sync', tone: 'plum' }], { stack: true, unit: 's' }],
    ['g-speed', 'eps', 'Attempts per second', 'Rollout throughput.',
      [{ key: 'time/episodes_per_s', label: 'attempts/s', tone: 'ink', ema: true }], { min0: true }],
    ['g-speed', 'gpumem', 'Trainer GPU memory', 'Peak allocated and total used, in GB.',
      [{ key: 'gpu/device_used_gb', label: 'device used', tone: 'ink' }, { key: 'gpu/peak_allocated_gb', label: 'peak allocated', tone: 'amber', dash: [5, 3] }], { min0: true }]
  ];

  var CLAIMED_PREFIX = ['success/', 'task_rate/', 'task_weight/', 'format_error/'];
  var CLAIMED_KEYS = { step: 1, 'success/overall': 1 };
  METRIC_SPECS.forEach(function (s) { s[4].forEach(function (x) { if (x.key) CLAIMED_KEYS[x.key] = 1; }); });
  ['episodes/crashed', 'dyn/mixed_groups', 'groups/count', 'groups/mean_reward', 'failure/success', 'reward/format_penalty', 'reward/efficiency_bonus', 'micro_batch_episodes', 'sampler/length_mismatch_skips', 'reward/success'].forEach(function (k) { CLAIMED_KEYS[k] = CLAIMED_KEYS[k] || 0; });

  function toneColor(P, tone) { return P[tone] || P.ink; }

  function Chart(spec, parent) {
    this.spec = spec;
    this.parent = parent;
    this.root = h('div', { class: 'chart' });
    this.titleEl = h('span', { class: 'chart-title', text: spec.title });
    this.lastEl = h('span', { class: 'chart-last' });
    this.descEl = h('div', { class: 'chart-desc', text: spec.desc || '' });
    this.plot = h('div', { class: 'plot' });
    this.legend = h('div', { class: 'legend' });
    this.root.appendChild(h('div', { class: 'chart-head' }, this.titleEl, this.lastEl));
    this.root.appendChild(this.descEl);
    this.root.appendChild(this.plot);
    this.root.appendChild(this.legend);
    this.root.hidden = true;
    parent.appendChild(this.root);
    this.u = null; this.sig = '';
    var self = this;
    if (window.ResizeObserver) {
      this.ro = new ResizeObserver(function () {
        if (self.u) { var w = self.plot.clientWidth; if (w > 20 && Math.abs(w - self.u.width) > 1) self.u.setSize({ width: w, height: self.height() }); }
      });
      this.ro.observe(this.plot);
    }
  }
  Chart.prototype.height = function () { return this.spec.height || 170; };
  Chart.prototype.destroy = function () { if (this.u) { this.u.destroy(); this.u = null; } this.sig = ''; };

  /* series: [{label, tone, dash, ema, pts, cum, vals(array)}], xs array */
  Chart.prototype.update = function (xs, series, opts) {
    opts = opts || {};
    var spec = this.spec;
    var live = series.filter(function (s) { return s.vals.some(function (v) { return v != null; }); });
    if (!live.length || !xs.length || typeof uPlot === 'undefined') { this.root.hidden = true; this.destroy(); return; }
    this.root.hidden = false;
    var P = palette();
    var s = smoothing;
    var uSeries = [], data = [xs], tipRows = [], legendRows = [];
    var stack = !!opts.stack;
    var running = stack ? xs.map(function () { return 0; }) : null;
    var cumLive = live.map(function (x) { return x; });
    if (stack) {
      // totals first so later (lower) layers paint on top
      var cum = [], acc = xs.map(function () { return 0; });
      live.forEach(function (x) { acc = acc.map(function (a, i) { return a + (x.vals[i] || 0); }); cum.push(acc.slice()); });
      for (var k = live.length - 1; k >= 0; k--) {
        var col = toneColor(P, live[k].tone);
        uSeries.push({ label: live[k].label, stroke: col, width: 1.5, fill: alpha(col, 0.35), points: { show: false }, spanGaps: true });
        data.push(cum[k]);
      }
      live.forEach(function (x) { tipRows.push({ label: x.label, color: toneColor(P, x.tone), vals: x.vals, strong: true, dash: x.dash }); });
      live.forEach(function (x) { legendRows.push({ label: x.label, color: toneColor(P, x.tone), kind: 'block' }); });
    } else {
      live.forEach(function (x) {
        var col = toneColor(P, x.tone);
        var vals = x.vals;
        if (x.cum) { var c = 0; vals = vals.map(function (v) { if (v != null) c += v; return v == null ? null : c; }); }
        var useEma = x.ema && s > 0;
        if (useEma) {
          uSeries.push({ label: x.label + ' (raw)', stroke: alpha(col, 0.32), width: 1, dash: x.dash, points: { show: false }, spanGaps: true });
          data.push(vals);
          var sm = ema(vals, s);
          uSeries.push({ label: x.label, stroke: col, width: 2, dash: x.dash, points: { show: false }, spanGaps: true });
          data.push(sm);
          tipRows.push({ label: x.label, color: col, vals: sm, strong: true, dash: x.dash });
          tipRows.push({ label: 'raw', color: alpha(col, 0.5), vals: vals, strong: false, dash: x.dash });
        } else {
          var pts = !!x.pts || vals.filter(function (v) { return v != null; }).length < xs.length * 0.6;
          var dotsOnly = !!x.dotsOnly;
          uSeries.push({
            label: x.label, stroke: col, width: x.cum ? 1.75 : 2, dash: x.dash, spanGaps: true,
            paths: dotsOnly ? function () { return null; } : undefined,
            points: { show: pts || dotsOnly, size: dotsOnly ? 5 : 6, fill: col, stroke: P.paper, width: 1 }
          });
          data.push(vals);
          tipRows.push({ label: x.label, color: col, vals: vals, strong: true, dash: x.dash });
        }
        legendRows.push({ label: x.label, color: col, dash: x.dash, dots: x.dotsOnly });
      });
    }
    // y range
    var all = [];
    for (var i = 1; i < data.length; i++) for (var j = 0; j < data[i].length; j++) if (data[i][j] != null) all.push(data[i][j]);
    var ymin = Math.min.apply(null, all), ymax = Math.max.apply(null, all);
    var frac01 = !!opts.frac;
    var yRange = function () {
      if (frac01) return [0, ymax > 0.6 || opts.fullScale ? 1 : Math.max(0.05, Math.min(1, ymax * 1.15))];
      var lo = opts.min0 ? 0 : ymin, hi = ymax;
      if (hi === lo) { hi = lo + 1; }
      var pad = (hi - lo) * 0.08;
      return [opts.min0 ? 0 : lo - pad, hi + pad];
    };
    var sig = [spec.id, uSeries.length, uSeries.map(function (q) { return q.label; }).join('|'), isDark() ? 'd' : 'l', opts.time ? 't' : 's'].join('#');
    var self = this;
    if (this.u && this.sig === sig) {
      this.rawX = xs; this.tipRows = tipRows;
      this.u.setData(data, true);
    } else {
      this.destroy();
      this.rawX = xs; this.tipRows = tipRows;
      var tip = h('div', { class: 'tip', hidden: '' });
      var axisFont = '10.5px "IBM Plex Mono", monospace';
      var axBase = { stroke: P.ink2, font: axisFont, grid: { stroke: alpha(P.hair, 1), width: 1 }, ticks: { show: false }, border: { show: false } };
      var o = {
        width: Math.max(200, this.plot.clientWidth || 300), height: this.height(),
        padding: [8, 10, 0, 0], legend: { show: false },
        cursor: { points: { size: 7, stroke: P.ink, fill: P.paper, width: 1.5 }, drag: { x: false, y: false }, x: true, y: false },
        scales: { x: { time: !!opts.time }, y: { range: yRange } },
        axes: [
          Object.assign({}, axBase, { size: 26, gap: 4, space: 70, values: opts.time ? undefined : stepAxis }),
          Object.assign({}, axBase, { size: frac01 ? 40 : 52, gap: 4, space: 34, values: frac01 ? pctAxis : numAxis })
        ],
        series: [{}].concat(uSeries),
        hooks: {
          setCursor: [function (u) {
            var idx = u.cursor.idx;
            if (idx == null) { tip.hidden = true; return; }
            clear(tip);
            var xv = self.rawX[idx];
            tip.appendChild(h('div', { class: 'th', text: opts.time ? new Date(xv * 1000).toLocaleString() : (opts.xlabel || 'step') + ' ' + xv }));
            self.tipRows.forEach(function (r) {
              var v = r.vals[idx];
              if (v == null) return;
              var txt = frac01 ? fmtPct(v) : (spec.unit ? fmtNum(v) + ' ' + spec.unit : fmtNum(v));
              tip.appendChild(h('div', { class: 'tr', style: '--c:' + r.color }, h('i'), h('b', { text: txt }), h('span', { text: r.label })));
            });
            tip.hidden = false;
            var ow = u.over.clientWidth, tw = tip.offsetWidth;
            var left = u.cursor.left + 14;
            if (left + tw > ow) left = u.cursor.left - tw - 14;
            tip.style.left = Math.max(0, left) + 'px';
            tip.style.top = Math.max(0, Math.min(u.cursor.top - 10, u.over.clientHeight - tip.offsetHeight)) + 'px';
          }]
        }
      };
      clear(this.plot);
      this.u = new uPlot(o, data, this.plot);
      this.u.over.appendChild(tip);
      this.sig = sig;
    }
    // legend + last value
    clear(this.legend);
    if (legendRows.length > 1) {
      legendRows.forEach(function (r) {
        var i = h('i', { style: '--c:' + r.color, class: r.kind || (r.dots ? 'dots' : (r.dash ? 'dashed' : '')) });
        self.legend.appendChild(h('span', null, i, r.label));
      });
    }
    var first = live[0], lastv = null;
    if (!stack) { for (var q = first.vals.length - 1; q >= 0; q--) if (first.vals[q] != null) { lastv = first.vals[q]; break; } }
    else { for (var q2 = xs.length - 1; q2 >= 0; q2--) { var t = 0, ok = false; live.forEach(function (x) { if (x.vals[q2] != null) { t += x.vals[q2]; ok = true; } }); if (ok) { lastv = t; break; } } }
    if (first.cum) { lastv = null; first.vals.forEach(function (v) { if (v != null) lastv = (lastv || 0) + v; }); }
    this.lastEl.textContent = lastv == null ? '' : (frac01 ? fmtPct(lastv) : fmtNum(lastv) + (spec.unit ? ' ' + spec.unit : ''));
  };

  function mountSpecs() {
    METRIC_SPECS.forEach(function (sp) {
      var parent = document.getElementById(sp[0]);
      var spec = { id: sp[1], title: sp[2], desc: sp[3], defs: sp[4], opts: sp[5] || {}, unit: (sp[5] || {}).unit };
      var c = new Chart(spec, parent);
      c.kind = 'metric';
      charts.push(c);
    });
  }

  var extraCharts = {}, taskCharts = {}, sysCharts = [];

  function dedupeRows(rows) {
    var byStep = {};
    rows.forEach(function (r) { if (isNum(r.step)) byStep[r.step] = r; });
    return Object.keys(byStep).map(Number).sort(function (a, b) { return a - b; }).map(function (k) { return byStep[k]; });
  }

  function renderMetricCharts() {
    var rows = dedupeRows(S.metrics.rows);
    var xs = rows.map(function (r) { return r.step; });
    charts.forEach(function (c) {
      if (c.kind !== 'metric') return;
      var series = c.spec.defs.map(function (d) {
        var vals = rows.map(function (r) { return d.get ? num(d.get(r)) : num(r[d.key]); });
        return { label: d.label, tone: d.tone, dash: d.dash, ema: d.ema, pts: d.pts, cum: d.cum, dotsOnly: d.dotsOnly, vals: vals };
      });
      try { c.update(xs, series, c.spec.opts); } catch (e) { console.error('chart', c.spec.id, e); c.root.hidden = true; }
    });
    // other numeric keys
    var keys = {};
    rows.forEach(function (r) { Object.keys(r).forEach(function (k) { if (isNum(r[k])) keys[k] = 1; }); });
    var extra = Object.keys(keys).filter(function (k) {
      if (k in CLAIMED_KEYS) return false;
      return !CLAIMED_PREFIX.some(function (p) { return k.indexOf(p) === 0; });
    }).sort().slice(0, 30);
    $('#more-wrap').hidden = !extra.length;
    var grid = $('#g-more');
    extra.forEach(function (k) {
      if (!extraCharts[k]) { extraCharts[k] = new Chart({ id: 'x-' + k, title: k, desc: '', height: 130 }, grid); extraCharts[k].kind = 'extra'; }
      extraCharts[k].update(xs, [{ label: k, tone: 'ink', ema: true, vals: rows.map(function (r) { return num(r[k]); }) }], {});
    });
    // per task
    var tasks = {};
    rows.forEach(function (r) { Object.keys(r).forEach(function (k) { if (k.indexOf('success/') === 0 && k !== 'success/overall') tasks[k.slice(8)] = 1; else if (k.indexOf('task_rate/') === 0) tasks[k.slice(10)] = 1; }); });
    var names = Object.keys(tasks).sort();
    var tgrid = $('#g-tasks');
    if (!names.length && !tgrid.children.length) tgrid.appendChild(h('div', { class: 'empty', text: 'No per-task numbers yet.' }));
    if (names.length) { var em = tgrid.querySelector('.empty'); if (em) em.remove(); }
    names.forEach(function (t) {
      if (!taskCharts[t]) { taskCharts[t] = new Chart({ id: 't-' + t, title: t, desc: '', height: 110 }, tgrid); taskCharts[t].kind = 'task'; }
      var c = taskCharts[t];
      c.update(xs, [
        { label: 'rate', tone: 'sage', vals: rows.map(function (r) { return num(r['task_rate/' + t]); }) },
        { label: 'rollouts', tone: 'ink', dotsOnly: true, vals: rows.map(function (r) { return num(r['success/' + t]); }) }
      ], { frac: true });
    });
  }

  function renderSystemCharts() {
    var rows = S.system.rows;
    var grid = $('#g-system');
    if (!sysCharts.length) {
      [['sys-util', 'GPU utilisation', 'Percent of time the GPU was busy.', 'gpu_util', 'sage', { max100: true }, ''],
       ['sys-mem', 'GPU memory', 'Gigabytes in use.', 'gpu_mem_mb', 'ink', {}, 'GB'],
       ['sys-pw', 'GPU power', 'Watts drawn.', 'gpu_power_w', 'amber', {}, 'W'],
       ['sys-cost', 'Spend', 'Dollars spent since the run began.', 'spend_usd', 'oxblood', {}, '']].forEach(function (d) {
        var c = new Chart({ id: d[0], title: d[1], desc: d[2], unit: d[6] }, grid);
        c.kind = 'sys'; c.key = d[3]; c.tone = d[4]; c.o = d[5]; sysCharts.push(c);
      });
    }
    var xs = rows.map(function (r) { return (parseT(r.t) || 0) / 1000; });
    sysCharts.forEach(function (c) {
      var vals = rows.map(function (r) { var v = num(r[c.key]); if (v != null && c.key === 'gpu_mem_mb') v = v / 1024; return v; });
      if (c.key === 'spend_usd') c.spec.unit = '';
      try { c.update(xs, [{ label: c.spec.title, tone: c.tone, vals: vals }], { time: true, min0: true }); } catch (e) { console.error(e); c.root.hidden = true; }
    });
    if (!rows.length && !grid.querySelector('.empty')) grid.appendChild(h('div', { class: 'empty', text: 'No GPU readings yet.' }));
    else if (rows.length) { var em = grid.querySelector('.empty'); if (em) em.remove(); }
  }

  function allCharts() {
    var a = charts.concat(sysCharts);
    Object.keys(extraCharts).forEach(function (k) { a.push(extraCharts[k]); });
    Object.keys(taskCharts).forEach(function (k) { a.push(taskCharts[k]); });
    return a;
  }
  function rebuildCharts() {
    allCharts().forEach(function (c) { c.destroy(); });
    renderMetricCharts(); renderSystemCharts();
  }

  smoothInput.addEventListener('input', function () {
    smoothing = parseFloat(smoothInput.value);
    $('#smooth-val').textContent = smoothing.toFixed(2);
    tset('run3-smooth', String(smoothing));
    renderMetricCharts();
  });

  /* ---------- evals ---------- */
  function wilson(p, n) {
    if (!n) return [0, 0];
    var z = 1.96, d = 1 + z * z / n, c = (p + z * z / (2 * n)) / d;
    var hw = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d;
    return [Math.max(0, c - hw), Math.min(1, c + hw)];
  }
  var evalSel = {};
  function pickEval(names, kind) {
    var isBase = function (n) { return /base|before|pre[-_]|start|zero/i.test(n); };
    var inKind = names.filter(function (n) { return kind === 'held' ? /held/i.test(n) : !/held/i.test(n); });
    var pool = inKind.length ? inKind : [];
    var bases = pool.filter(isBase), afters = pool.filter(function (n) { return !isBase(n); });
    return { before: bases.length ? bases[bases.length - 1] : (pool[0] || ''), after: afters.length ? afters[afters.length - 1] : '', pool: pool };
  }
  function evalRow(name, b, a) {
    var row = h('div', { class: 'erow' + (name === 'mean' ? ' mean' : '') });
    row.appendChild(h('div', { class: 'name', text: name === 'mean' ? 'overall' : name }));
    var tr = h('div', { class: 'etrack' });
    var add = function (cls, o) {
      if (!o) return;
      tr.appendChild(h('div', { class: 'ebar ' + cls, style: 'width:' + (o.p * 100) + '%' + (cls === 'a' ? ';--c:' + (b && o.p < b.p ? 'var(--alm-oxblood)' : (b && o.p === b.p ? 'var(--alm-ink2)' : 'var(--alm-sage)')) : '') }));
      tr.appendChild(h('div', { class: 'ewh ' + cls, style: 'left:' + (o.lo * 100) + '%;width:' + Math.max(0, (o.hi - o.lo) * 100) + '%' }));
    };
    add('b', b); add('a', a);
    row.appendChild(tr);
    var txt = h('div', { class: 'vals' });
    if (b) txt.appendChild(document.createTextNode(fmtPct(b.p) + ' → '));
    if (a) {
      txt.appendChild(h('b', { text: fmtPct(a.p) }));
      if (b) {
        var dlt = Math.round((a.p - b.p) * 100), clear_ = a.lo > b.hi || a.hi < b.lo;
        txt.appendChild(document.createTextNode('  ' + (dlt > 0 ? '+' : '') + dlt + ' pts' + (clear_ ? ' •' : '')));
      }
    } else txt.appendChild(document.createTextNode('–'));
    row.title = name + (b ? ': before ' + fmtPct(b.p) + ' [' + fmtPct(b.lo) + ', ' + fmtPct(b.hi) + '] (n=' + b.n + ')' : '') + (a ? '; after ' + fmtPct(a.p) + ' [' + fmtPct(a.lo) + ', ' + fmtPct(a.hi) + '] (n=' + a.n + ')' : '');
    row.appendChild(txt);
    return row;
  }
  function pointOf(ev, task) {
    var t = ev && ev.tasks && ev.tasks[task];
    if (task === 'mean') {
      if (!ev || !isNum(ev.mean)) return null;
      var ci = Array.isArray(ev.ci95) && ev.ci95.length === 2 ? ev.ci95 : null;
      var n = 0; Object.keys(ev.tasks || {}).forEach(function (k) { n += ev.tasks[k].n || 0; });
      if (!ci) ci = wilson(ev.mean, n);
      return { p: ev.mean, lo: ci[0], hi: ci[1], n: n };
    }
    if (!t || !isNum(t.success)) return null;
    var w = wilson(t.success, t.n || 0);
    return { p: t.success, lo: w[0], hi: w[1], n: t.n || 0 };
  }
  function renderEvals() {
    var ev = S.evals || {};
    var names = Object.keys(ev);
    var host = $('#eval-panels'); clear(host);
    if (!names.length) { host.appendChild(h('div', { class: 'empty', text: 'No exams have finished yet.' })); clear($('#eval-table')); return; }
    [['held', 'Held-out tasks'], ['pool', 'Pool tasks']].forEach(function (kd) {
      var kind = kd[0], pick = pickEval(names, kind);
      var sel = evalSel[kind] || {};
      if (!sel.before || names.indexOf(sel.before) < 0) sel.before = pick.before;
      if (!sel.after || names.indexOf(sel.after) < 0) sel.after = pick.after;
      evalSel[kind] = sel;
      var panel = h('div', { class: 'evalp' });
      var mk = function (which) {
        var s = h('select', { 'aria-label': which + ' exam' });
        s.appendChild(h('option', { value: '', text: '(none)' }));
        (pick.pool.length ? pick.pool : names).forEach(function (n) { var o = h('option', { value: n, text: n }); if (n === sel[which]) o.selected = true; s.appendChild(o); });
        if (sel[which] && (pick.pool.length ? pick.pool : names).indexOf(sel[which]) < 0) { var o2 = h('option', { value: sel[which], text: sel[which] }); o2.selected = true; s.appendChild(o2); }
        s.addEventListener('change', function () { sel[which] = s.value; renderEvals(); });
        return s;
      };
      panel.appendChild(h('div', { class: 'evalp-head' }, h('h3', { text: kd[1] }),
        h('div', { class: 'evalp-pick' }, 'before', mk('before'), 'after', mk('after'))));
      if (!pick.pool.length) { panel.appendChild(h('div', { class: 'empty', text: 'No ' + kd[1].toLowerCase() + ' exam has finished yet.' })); host.appendChild(panel); return; }
      var B = ev[sel.before], A = ev[sel.after];
      panel.appendChild(h('div', { class: 'eleg' }, h('span', null, h('i', { class: 'b' }), 'before: ' + (sel.before || '–')), h('span', null, h('i', { class: 'a' }), 'after: ' + (sel.after || '–')), h('span', { text: '• = intervals do not overlap' })));
      var tasks = {};
      [B, A].forEach(function (e) { if (e && e.tasks) Object.keys(e.tasks).forEach(function (k) { tasks[k] = 1; }); });
      var list = Object.keys(tasks);
      list.sort(function (x, y) {
        var dx = (pointOf(A, x) || { p: 0 }).p - (pointOf(B, x) || { p: 0 }).p, dy = (pointOf(A, y) || { p: 0 }).p - (pointOf(B, y) || { p: 0 }).p;
        return dy - dx || (x < y ? -1 : 1);
      });
      panel.appendChild(evalRow('mean', pointOf(B, 'mean'), pointOf(A, 'mean')));
      list.forEach(function (t) { panel.appendChild(evalRow(t, pointOf(B, t), pointOf(A, t))); });
      host.appendChild(panel);
    });
    var tb = $('#eval-table'); clear(tb);
    tb.appendChild(h('thead', null, h('tr', null, ['Exam', 'Track', 'Mode', 'Tasks', 'Mean success', '95% interval'].map(function (t) { return h('th', { text: t }); }))));
    var body = h('tbody');
    names.forEach(function (n) {
      var e = ev[n], nt = e.tasks ? Object.keys(e.tasks).length : 0;
      var ci = Array.isArray(e.ci95) ? fmtPct(e.ci95[0]) + ' to ' + fmtPct(e.ci95[1]) : '–';
      body.appendChild(h('tr', null, h('td', { text: n }), h('td', { text: e.track || '–' }), h('td', { text: e.mode || '–' }), h('td', { text: String(nt) }), h('td', { text: fmtPct(e.mean) }), h('td', { text: ci })));
    });
    tb.appendChild(body);
  }

  /* ---------- samples ---------- */
  var epCache = {}, openDir = null;
  function parseClick(turn) {
    var srcs = [turn.action || '', turn.response || ''];
    for (var i = 0; i < srcs.length; i++) {
      var s = srcs[i];
      var m = /coordinate["']?\s*[=:]\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]/.exec(s) ||
              /click\(\s*\[?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]?\s*\)/.exec(s) ||
              /"x"\s*:\s*(-?\d+(?:\.\d+)?)\s*,\s*"y"\s*:\s*(-?\d+(?:\.\d+)?)/.exec(s);
      if (m) return [parseFloat(m[1]), parseFloat(m[2])];
    }
    return null;
  }
  function shot(dir, turn) {
    var wrap = h('div', { class: 'shot' });
    var img = h('img', { alt: 'Screenshot the model saw', loading: 'lazy', src: 'data/samples/' + encodeURIComponent(dir) + '/' + encodeURIComponent(turn.image || '') });
    img.addEventListener('error', function () { img.style.display = 'none'; wrap.appendChild(h('span', { class: 'ph' })); });
    wrap.appendChild(img);
    var xy = parseClick(turn);
    if (xy) wrap.appendChild(h('div', { class: 'xh', style: 'left:' + Math.max(0, Math.min(100, xy[0] / 10)) + '%;top:' + Math.max(0, Math.min(100, xy[1] / 10)) + '%' }));
    return wrap;
  }
  function loadEpisode(dir) {
    if (epCache[dir]) return Promise.resolve(epCache[dir]);
    return fetchJSON('data/samples/' + encodeURIComponent(dir) + '/episode.json').then(function (e) { if (e) epCache[dir] = e; return e; });
  }
  function openSample(item) {
    openDir = item.dir;
    var box = $('#sample-detail'); clear(box); box.hidden = false;
    box.appendChild(h('div', { class: 'empty', text: 'Loading…' }));
    loadEpisode(item.dir).then(function (ep) {
      clear(box);
      if (!ep) { box.appendChild(h('div', { class: 'empty', text: 'This sample is no longer available.' })); return; }
      box.appendChild(h('div', { class: 'detail-head' },
        h('h3', { text: ep.task + ' · step ' + ep.step }),
        h('span', { class: 'alm-data', text: 'seed ' + ep.seed + ' · reward ' + fmtNum(ep.reward) + ' · ' + (ep.success ? 'success' : 'not solved') }),
        h('button', { class: 'alm-btn alm-btn--outline alm-btn--dense', type: 'button', text: 'Close', onclick: function () { box.hidden = true; openDir = null; renderGallery(); } })));
      var turns = h('div', { class: 'turns' });
      (ep.turns || []).forEach(function (t, i) {
        var d = h('div', { class: 'turn' }, h('div', { class: 'alm-label', text: 'turn ' + (i + 1) }), shot(item.dir, t));
        if (t.action) d.appendChild(h('div', { class: 'cap' }, h('b', { text: t.action })));
        if (t.response) d.appendChild(h('pre', { text: t.response }));
        if (t.error) d.appendChild(h('div', { class: 'err', text: t.error }));
        turns.appendChild(d);
      });
      box.appendChild(turns);
      box.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }).catch(function () { clear(box); box.appendChild(h('div', { class: 'empty', text: 'Could not load this sample.' })); });
    renderGallery();
  }
  function renderGallery() {
    var g = $('#gallery');
    var idx = Array.isArray(S.samples) ? S.samples : [];
    var sig = idx.slice(0, 12).map(function (i) { return i.dir; }).join(',') + '|' + openDir;
    if (g.dataset.sig === sig) return;
    g.dataset.sig = sig; clear(g);
    if (!idx.length) { g.appendChild(h('div', { class: 'empty', text: 'No sample rollouts yet.' })); return; }
    idx.slice(0, 12).forEach(function (item) {
      var btn = h('button', { class: 'sample', type: 'button', 'aria-pressed': item.dir === openDir ? 'true' : 'false', onclick: function () { openSample(item); } });
      var holder = h('div', { class: 'shot' }, h('span', { class: 'ph' }));
      btn.appendChild(holder);
      btn.appendChild(h('div', { class: 'cap' }, h('b', { text: item.task }), h('br'),
        'step ' + item.step + ' · ', h('span', { class: item.success ? 'ok' : 'no', text: item.success ? 'solved' : 'missed' }), ' · r ' + fmtNum(item.reward)));
      g.appendChild(btn);
      loadEpisode(item.dir).then(function (ep) {
        if (!ep || !ep.turns || !ep.turns.length) return;
        var s = shot(item.dir, ep.turns[0]);
        btn.replaceChild(s, holder);
      }).catch(function () { /* leave placeholder */ });
    });
  }

  /* ---------- events, notes, links ---------- */
  var KIND_COLOR = {
    phase_done: 'var(--alm-sage)', upload_verified: 'var(--alm-sage)', pod_created: 'var(--alm-ink)', phase_started: 'var(--alm-ink2)',
    restart: 'var(--alm-amber)', pod_recreated: 'var(--alm-amber)', budget_soft_stop: 'var(--alm-amber)',
    pod_lost: 'var(--alm-oxblood)', error: 'var(--alm-oxblood)', pod_deleted: 'var(--alm-plum)', note: 'var(--alm-plum)'
  };
  var showAllEvents = false;
  function renderEvents() {
    var ol = $('#events'); clear(ol);
    var rows = S.events.rows.slice().sort(function (a, b) { return (parseT(b.t) || 0) - (parseT(a.t) || 0); });
    if (!rows.length) { ol.appendChild(h('li', null, h('span', { class: 'empty', text: 'Nothing has happened yet.' }))); return; }
    (showAllEvents ? rows : rows.slice(0, 40)).forEach(function (e) {
      var t = parseT(e.t);
      ol.appendChild(h('li', { style: '--c:' + (KIND_COLOR[e.kind] || 'var(--alm-ink2)') },
        h('div', { class: 'when', text: (t ? fmtClock(t) : '') + ' · ' + String(e.kind || '').replace(/_/g, ' ') }),
        h('div', { class: 'msg', text: String(e.msg || '') })));
    });
    if (!showAllEvents && rows.length > 40) ol.appendChild(h('li', null, h('button', { class: 'alm-btn alm-btn--outline alm-btn--dense', type: 'button', text: 'Show all ' + rows.length, onclick: function () { showAllEvents = true; renderEvents(); } })));
  }

  function esc(s) { return s.replace(/[&<>"']/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]; }); }
  function inline(s) {
    s = esc(s);
    s = s.replace(/`([^`]+)`/g, '<code>$1</code>');
    s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    s = s.replace(/(^|[^*])\*([^*\s][^*]*)\*/g, '$1<em>$2</em>');
    s = s.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
    return s;
  }
  function md(text) {
    var out = [];
    String(text || '').split(/\n{2,}/).forEach(function (block) {
      var lines = block.split('\n').filter(function (l) { return l.trim() !== ''; });
      if (!lines.length) return;
      if (lines.every(function (l) { return /^\s*[-*]\s+/.test(l); })) {
        out.push('<ul>' + lines.map(function (l) { return '<li>' + inline(l.replace(/^\s*[-*]\s+/, '')) + '</li>'; }).join('') + '</ul>');
      } else if (/^#{1,4}\s/.test(lines[0])) {
        out.push('<h4>' + inline(lines[0].replace(/^#+\s*/, '')) + '</h4>');
        if (lines.length > 1) out.push('<p>' + lines.slice(1).map(inline).join('<br>') + '</p>');
      } else out.push('<p>' + lines.map(inline).join('<br>') + '</p>');
    });
    return out.join('');
  }
  function renderNotes() {
    var box = $('#notes'); clear(box);
    var notes = Array.isArray(S.notes) ? S.notes.slice().reverse() : [];
    if (!notes.length) { box.appendChild(h('div', { class: 'empty', text: 'No notes yet.' })); return; }
    notes.slice(0, 20).forEach(function (n) {
      var t = parseT(n.t);
      var d = h('div', { class: 'note' }, h('div', { class: 'when', text: t ? fmtClock(t) : '' }));
      var body = h('div'); body.innerHTML = md(n.text); // text is escaped first, see esc()
      d.appendChild(body); box.appendChild(d);
    });
  }
  function renderLinks() {
    var box = $('#links'); clear(box);
    var st = S.state || {};
    var add = function (label, url) { if (typeof url === 'string' && /^https?:\/\//.test(url)) box.appendChild(h('a', { href: url, target: '_blank', rel: 'noopener noreferrer', text: label + ' ↗' })); };
    add('Weights & Biases run', st.wandb_url); add('Hugging Face adapter', st.hf_url);
  }

  /* ---------- polling ---------- */
  var polling = false;
  function poll() {
    if (polling) return;
    polling = true;
    var tasks = {
      state: fetchJSON('data/state.json').then(function (v) { if (v) S.state = v; }),
      evals: fetchJSON('data/evals.json').then(function (v) { S.evals = v; }),
      notes: fetchJSON('data/notes.json').then(function (v) { S.notes = v; }),
      samples: fetchJSON('data/samples/index.json').then(function (v) { S.samples = v; }),
      metrics: fetchLines('data/metrics.jsonl', S.metrics),
      system: fetchLines('data/system.jsonl', S.system),
      events: fetchLines('data/events.jsonl', S.events)
    };
    var names = Object.keys(tasks);
    Promise.all(names.map(function (k) { return tasks[k].then(function () { delete S.errors[k]; }, function (e) { S.errors[k] = e && e.message || 'failed'; }); })).then(function () {
      polling = false;
      if (names.length > Object.keys(S.errors).length) S.lastOk = Date.now();
      renderAll();
    });
  }
  function safe(fn) { try { fn(); } catch (e) { console.error(e); } }
  function renderAll() {
    safe(renderHeader); safe(renderCost); safe(renderProgress);
    safe(renderMetricCharts); safe(renderSystemCharts);
    safe(renderEvals); safe(renderGallery); safe(renderEvents); safe(renderNotes); safe(renderLinks);
    if (Object.keys(S.errors).length === 7) { var eb = $('#error-banner'); eb.hidden = false; eb.textContent = 'Cannot reach the data feed. Retrying every 15 seconds.'; }
  }

  /* ---------- live log ---------- */
  var MAX_LINES = 2000;
  var logBox = $('#logbox'), logFilter = $('#log-filter'), autoBtn = $('#log-auto');
  var autoScroll = true, programmatic = false, logLines = [];
  var ANSI = /\x1b\[[0-9;?]*[A-Za-z]/g;
  function lineClass(t) { return /traceback|error|exception|failed|fatal|oom|out of memory/i.test(t) ? 'ln err' : (/warn/i.test(t) ? 'ln warn' : 'ln'); }
  function matches(t) { var f = logFilter.value.trim().toLowerCase(); return !f || t.toLowerCase().indexOf(f) >= 0; }
  function paint(el, text) {
    var f = logFilter.value.trim();
    clear(el);
    if (!f) { el.textContent = text; return; }
    var i = text.toLowerCase().indexOf(f.toLowerCase());
    if (i < 0) { el.textContent = text; return; }
    el.appendChild(document.createTextNode(text.slice(0, i)));
    el.appendChild(h('mark', { text: text.slice(i, i + f.length) }));
    el.appendChild(document.createTextNode(text.slice(i + f.length)));
  }
  function addLog(lines, reset) {
    if (reset) { logLines = []; clear(logBox); }
    var frag = document.createDocumentFragment();
    lines.forEach(function (raw) {
      var t = String(raw).replace(ANSI, '');
      var el = h('div', { class: lineClass(t) });
      paint(el, t);
      if (!matches(t)) el.hidden = true;
      logLines.push({ t: t, el: el });
      frag.appendChild(el);
    });
    logBox.appendChild(frag);
    while (logLines.length > MAX_LINES) { var o = logLines.shift(); o.el.remove(); }
    updateCount();
    if (autoScroll) scrollEnd();
  }
  function scrollEnd() { programmatic = true; logBox.scrollTop = logBox.scrollHeight; setTimeout(function () { programmatic = false; }, 60); }
  function updateCount() {
    var shown = 0; logLines.forEach(function (l) { if (!l.el.hidden) shown++; });
    $('#log-count').textContent = logFilter.value.trim() ? shown + ' of ' + logLines.length + ' lines match' : logLines.length + ' lines';
  }
  logFilter.addEventListener('input', function () {
    logLines.forEach(function (l) { l.el.hidden = !matches(l.t); paint(l.el, l.t); });
    updateCount(); if (autoScroll) scrollEnd();
  });
  function setAuto(on) { autoScroll = on; autoBtn.textContent = 'Auto-scroll: ' + (on ? 'on' : 'off'); autoBtn.setAttribute('aria-pressed', on ? 'true' : 'false'); if (on) scrollEnd(); }
  autoBtn.addEventListener('click', function () { setAuto(!autoScroll); });
  logBox.addEventListener('scroll', function () {
    if (programmatic) return;
    var atEnd = logBox.scrollHeight - logBox.scrollTop - logBox.clientHeight < 24;
    if (!atEnd && autoScroll) setAuto(false);
    else if (atEnd && !autoScroll) setAuto(true);
  });
  var es = null;
  function connectLog() {
    if (!window.EventSource) { $('#log-status').textContent = 'live log needs a newer browser'; return; }
    es = new EventSource('api/log/stream');
    es.addEventListener('open', function () { $('#log-status').textContent = 'streaming'; });
    es.addEventListener('init', function (e) { addLog(JSON.parse(e.data).lines || [], true); $('#log-status').textContent = 'streaming'; });
    es.addEventListener('reset', function (e) { addLog(JSON.parse(e.data).lines || [], true); });
    es.addEventListener('append', function (e) { addLog(JSON.parse(e.data).lines || [], false); });
    es.addEventListener('error', function () { $('#log-status').textContent = 'reconnecting…'; });
  }

  /* ---------- go ---------- */
  mountSpecs();
  poll();
  setInterval(poll, POLL_MS);
  setInterval(function () { safe(tickUpdated); }, 1000);
  document.addEventListener('visibilitychange', function () { if (!document.hidden) poll(); });
  connectLog();
})();
