/* Spacecraft Health Monitoring System - front-end behaviour */
(function () {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const root = document.documentElement;

  /* ---------------- dropdown menus ---------------- */
  function closeMenus(except) {
    $$("[data-menu].open").forEach((m) => {
      if (m === except) return;
      m.classList.remove("open");
      const b = $("[data-menu-button]", m);
      if (b) b.setAttribute("aria-expanded", "false");
    });
  }
  $$("[data-menu]").forEach((menu) => {
    const btn = $("[data-menu-button]", menu);
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const open = !menu.classList.contains("open");
      closeMenus(menu);
      menu.classList.toggle("open", open);
      btn.setAttribute("aria-expanded", String(open));
      if (open) {
        const first = $(".menu-panel button, .menu-panel a, .menu-panel input", menu);
        if (first) setTimeout(() => first.focus(), 30);
      }
    });
    menu.addEventListener("click", (e) => e.stopPropagation());
  });
  document.addEventListener("click", () => closeMenus());
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    const open = $("[data-menu].open");
    if (open) {
      closeMenus();
      $("[data-menu-button]", open).focus();
    }
  });

  /* mobile nav */
  const navToggle = $("[data-nav-toggle]");
  if (navToggle) {
    navToggle.addEventListener("click", () => {
      const links = $("#nav-links");
      const open = links.classList.toggle("open");
      navToggle.setAttribute("aria-expanded", String(open));
    });
  }

  /* ---------------- accessibility preferences ---------------- */
  const A11Y_KEY = "shms-a11y";
  function loadPrefs() {
    try { return JSON.parse(localStorage.getItem(A11Y_KEY) || "{}"); } catch (e) { return {}; }
  }
  function savePrefs(p) {
    try { localStorage.setItem(A11Y_KEY, JSON.stringify(p)); } catch (e) { /* ignore */ }
  }
  function applyPrefs(p) {
    if (p.text) root.dataset.text = p.text; else delete root.dataset.text;
    if (p.contrast) root.dataset.contrast = "high"; else delete root.dataset.contrast;
    if (p.motion) root.dataset.motion = "reduce"; else delete root.dataset.motion;
    if (p.readable) root.dataset.readable = "on"; else delete root.dataset.readable;
    $$("[data-text-size]").forEach((b) => b.setAttribute("aria-pressed", String((p.text || "") === b.dataset.textSize)));
    $$("[data-a11y]").forEach((c) => { c.checked = !!p[c.dataset.a11y]; });
  }
  let prefs = loadPrefs();
  applyPrefs(prefs);
  $$("[data-text-size]").forEach((b) => b.addEventListener("click", () => {
    prefs.text = b.dataset.textSize; savePrefs(prefs); applyPrefs(prefs);
  }));
  $$("[data-a11y]").forEach((c) => c.addEventListener("change", () => {
    prefs[c.dataset.a11y] = c.checked; savePrefs(prefs); applyPrefs(prefs);
  }));
  const reset = $("[data-a11y-reset]");
  if (reset) reset.addEventListener("click", () => { prefs = {}; savePrefs(prefs); applyPrefs(prefs); });

  /* ---------------- small helpers ---------------- */
  $$("[data-dismiss]").forEach((b) => b.addEventListener("click", () => b.closest(".flash").remove()));
  $$("[data-print]").forEach((b) => b.addEventListener("click", () => window.print()));

  $$("[data-reveal]").forEach((b) => b.addEventListener("click", () => {
    const input = document.getElementById(b.dataset.reveal);
    const show = input.type === "password";
    input.type = show ? "text" : "password";
    b.setAttribute("aria-pressed", String(show));
    $(".sr-only", b).textContent = show ? "Hide password" : "Show password";
    b.title = show ? "Hide password" : "Show password";
  }));

  /* read aloud (speech synthesis) */
  let speaking = null;
  $$("[data-speak]").forEach((b) => b.addEventListener("click", () => {
    if (!("speechSynthesis" in window)) { b.disabled = true; return; }
    const label = $("span", b);
    if (speaking) {
      window.speechSynthesis.cancel();
      speaking = null;
      label.textContent = "Read report aloud";
      return;
    }
    const target = $(b.dataset.speak);
    const verdict = $("#verdict-title");
    const summary = $(".summary-card p");
    const text = [verdict && verdict.textContent, target && target.innerText, summary && "In short. " + summary.textContent]
      .filter(Boolean).join(". ");
    speaking = new SpeechSynthesisUtterance(text);
    speaking.rate = 0.98;
    speaking.onend = () => { speaking = null; label.textContent = "Read report aloud"; };
    window.speechSynthesis.speak(speaking);
    label.textContent = "Stop reading";
  }));

  /* ---------------- analyze page ---------------- */
  const form = $("#analyze-form");
  if (form) {
    const input = $("#file-input");
    const zone = $("#dropzone");
    const chip = $("#file-chip");
    const sample = $("#sample");
    const btn = $("#analyze-btn");
    const hint = $("#analyze-hint");

    const fmtSize = (n) => n > 1048576 ? (n / 1048576).toFixed(1) + " MB" : Math.max(1, Math.round(n / 1024)) + " KB";
    function refresh() {
      const file = input.files && input.files[0];
      if (file) {
        $("#file-name").textContent = file.name;
        $("#file-size").textContent = fmtSize(file.size) + " · ready to analyse";
        $("#file-ext").textContent = (file.name.split(".").pop() || "").toUpperCase().slice(0, 4);
        chip.classList.add("show");
        if (sample) sample.value = "";
      } else {
        chip.classList.remove("show");
      }
      const ready = !!file || (sample && sample.value);
      btn.disabled = !ready;
      hint.textContent = file ? "Your file is ready. Press Analyze."
        : (sample && sample.value) ? "NASA sample " + sample.value + " selected. Press Analyze."
        : "Choose a file or a sample to begin.";
    }
    input.addEventListener("change", refresh);
    if (sample) sample.addEventListener("change", () => {
      if (sample.value) input.value = "";
      refresh();
    });
    $("#file-clear").addEventListener("click", () => { input.value = ""; refresh(); input.focus(); });

    ["dragenter", "dragover"].forEach((ev) => zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add("drag"); }));
    ["dragleave", "drop"].forEach((ev) => zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.remove("drag"); }));
    zone.addEventListener("drop", (e) => {
      if (e.dataTransfer.files.length) {
        input.files = e.dataTransfer.files;
        refresh();
      }
    });

    form.addEventListener("submit", (e) => {
      if (btn.disabled) { e.preventDefault(); return; }
      const overlay = $("#overlay");
      overlay.hidden = false;
      btn.disabled = true;
      const steps = $$("#loader-steps li");
      const delays = [0, 700, 1500, 3200, 5200];
      steps.forEach((li, i) => setTimeout(() => {
        steps.forEach((s, j) => {
          s.classList.toggle("done", j < i);
          s.classList.toggle("active", j === i);
        });
      }, delays[i]));
    });
    /* if the user comes back with the browser's back button */
    window.addEventListener("pageshow", () => { $("#overlay").hidden = true; refresh(); });
    refresh();
  }

  /* ---------------- history page ---------------- */
  const search = $("#history-search");
  const filter = $("#history-filter");
  if (search && filter) {
    const items = $$("#history-list .h-item");
    const update = () => {
      const q = search.value.trim().toLowerCase();
      const st = filter.value;
      let shown = 0;
      items.forEach((it) => {
        const ok = (!q || it.dataset.text.includes(q)) && (!st || it.dataset.status === st);
        it.hidden = !ok;
        if (ok) shown++;
      });
      $("#no-match").hidden = shown > 0;
    };
    search.addEventListener("input", update);
    filter.addEventListener("change", update);
  }

  /* confirm-before-delete modal */
  const modal = $("#confirm-modal");
  if (modal) {
    let pendingForm = null;
    let opener = null;
    $$("[data-confirm]").forEach((b) => b.addEventListener("click", () => {
      pendingForm = document.getElementById(b.dataset.confirm);
      opener = b;
      $("#confirm-title").textContent = b.dataset.confirmTitle || "Are you sure?";
      $("#confirm-text").textContent = b.dataset.confirmText || "";
      modal.showModal();
      $("[data-modal-cancel]", modal).focus();
    }));
    $("[data-modal-cancel]", modal).addEventListener("click", () => modal.close());
    $("[data-modal-ok]", modal).addEventListener("click", () => { if (pendingForm) pendingForm.submit(); });
    modal.addEventListener("close", () => { if (opener) opener.focus(); });
    modal.addEventListener("click", (e) => { if (e.target === modal) modal.close(); });
  }

  /* ---------------- Gemini report (loaded after the page opens) ---------------- */
  const reportCard = $("#report-card");
  if (reportCard) {
    const loading = $("#report-loading");
    const notice = $("#report-notice");
    const text = $("#report-text");
    const source = $("#report-source span");
    const retry = $("#report-retry");

    const showFailure = (msg) => {
      loading.hidden = true;
      text.hidden = false;
      $("#report-notice-title").textContent = msg || "Gemini didn't respond this time.";
      notice.hidden = false;
      source.textContent = "Built-in report";
      retry.disabled = false;
    };

    async function loadReport() {
      loading.hidden = false;
      notice.hidden = true;
      text.hidden = true;
      retry.disabled = true;
      source.textContent = "Gemini is writing...";
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 75000);   // never spin forever
      try {
        const res = await fetch(reportCard.dataset.reportUrl, { method: "POST", signal: ctrl.signal });
        const data = await res.json();
        if (!data.ok) return showFailure(data.error);
        text.innerHTML = data.report_html;
        $("#summary-text").textContent = data.summary;
        loading.hidden = true;
        text.hidden = false;
        source.textContent = "Written by Gemini from the analysis above";
        reportCard.dataset.reportState = "gemini";
      } catch (err) {
        showFailure(err.name === "AbortError" ? "Gemini took too long to answer." : "Could not reach the server.");
      } finally {
        clearTimeout(timer);
      }
    }

    retry.addEventListener("click", loadReport);
    if (reportCard.dataset.reportState === "pending") loadReport();
  }

  /* ---------------- AERIS assistant (Gemini chat) ---------------- */
  const panel = $("#chat-panel");
  const fab = $("[data-chat-toggle]");
  const body = $("#chat-body");
  const chatForm = $("#chat-form");
  const chatInput = $("#chat-input");
  const loggedIn = panel && panel.dataset.loggedIn === "yes";
  const STORE = "shms-chat-" + (panel ? panel.dataset.user : "");
  const analysisId = document.body.dataset.analysisId || null;
  let history = [];
  try { history = JSON.parse(sessionStorage.getItem(STORE) || "[]"); } catch (e) { history = []; }

  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }
  function inline(s) {
    return s.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*])\*(?!\s)(.+?)\*(?!\*)/g, "$1<em>$2</em>")
      .replace(/`(.+?)`/g, "<code>$1</code>");
  }
  function md(text) {
    const lines = escapeHtml(text).split(/\r?\n/);
    let out = "", list = null, para = [];
    const flushPara = () => { if (para.length) { out += "<p>" + inline(para.join(" ")) + "</p>"; para = []; } };
    const closeList = () => { if (list) { out += "</" + list + ">"; list = null; } };
    lines.forEach((raw) => {
      const l = raw.trim();
      const b = l.match(/^[-*+]\s+(.*)/), n = l.match(/^\d+[.)]\s+(.*)/), h = l.match(/^#{1,4}\s+(.*)/);
      if (!l) { flushPara(); closeList(); }
      else if (h) { flushPara(); closeList(); out += "<p><strong>" + inline(h[1]) + "</strong></p>"; }
      else if (b || n) {
        flushPara();
        const tag = b ? "ul" : "ol";
        if (list !== tag) { closeList(); out += "<" + tag + ">"; list = tag; }
        out += "<li>" + inline((b || n)[1]) + "</li>";
      } else { closeList(); para.push(l); }
    });
    flushPara(); closeList();
    return out;
  }
  function addMsg(role, text) {
    const div = document.createElement("div");
    div.className = "msg " + (role === "user" ? "msg-user" : role === "error" ? "msg-error" : "msg-bot");
    if (role === "assistant") div.innerHTML = md(text); else div.textContent = text;
    body.appendChild(div);
    body.scrollTop = body.scrollHeight;
    return div;
  }
  function welcome() {
    body.innerHTML = "";
    addMsg("assistant", loggedIn
      ? "Hi! I'm the **AERIS Assistant**. Ask me about your analyses, how SHMS works, telemetry, or anything about space."
      : "Hi! I'm the **AERIS Assistant**. Please log in so I can help with your analyses.");
    if (!loggedIn) return;
    const chips = document.createElement("div");
    chips.className = "chips";
    const ideas = analysisId
      ? ["Explain this analysis simply", "Is the spacecraft in danger?", "What do precision and recall mean here?"]
      : ["Summarise my recent analyses", "What is telemetry?", "How does AERIS detect anomalies?"];
    ideas.forEach((q) => {
      const c = document.createElement("button");
      c.type = "button"; c.className = "chip"; c.textContent = q;
      c.addEventListener("click", () => ask(q));
      chips.appendChild(c);
    });
    body.appendChild(chips);
  }
  function render() {
    welcome();
    history.forEach((m) => addMsg(m.role, m.text));
  }
  function openChat(open) {
    panel.classList.toggle("open", open);
    fab.setAttribute("aria-expanded", String(open));
    if (open && chatInput) setTimeout(() => chatInput.focus(), 120);
  }

  let busy = false;
  async function ask(text) {
    text = (text || "").trim();
    if (!text || busy || !loggedIn) return;
    openChat(true);
    busy = true;
    const chips = $(".chips", body);
    if (chips) chips.remove();
    addMsg("user", text);
    const typing = document.createElement("div");
    typing.className = "msg msg-bot typing";
    typing.setAttribute("aria-label", "The assistant is typing");
    typing.innerHTML = "<i></i><i></i><i></i>";
    body.appendChild(typing);
    body.scrollTop = body.scrollHeight;
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, history: history.slice(-12), analysis_id: analysisId }),
      });
      const data = await res.json().catch(() => ({ error: "Unexpected answer from the server." }));
      typing.remove();
      if (data.reply) {
        history.push({ role: "user", text }, { role: "assistant", text: data.reply });
        try { sessionStorage.setItem(STORE, JSON.stringify(history.slice(-30))); } catch (e) { /* ignore */ }
        addMsg("assistant", data.reply);
      } else {
        addMsg("error", data.error || "Something went wrong.");
      }
    } catch (err) {
      typing.remove();
      addMsg("error", "Could not reach the server. Is the app still running?");
    } finally {
      busy = false;
    }
  }

  if (panel && fab) {
    render();
    fab.addEventListener("click", () => openChat(!panel.classList.contains("open")));
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && panel.classList.contains("open")) { openChat(false); fab.focus(); }
    });
    $("[data-chat-clear]").addEventListener("click", () => {
      history = [];
      try { sessionStorage.removeItem(STORE); } catch (e) { /* ignore */ }
      render();
    });
    $$("[data-chat-ask]").forEach((b) => b.addEventListener("click", () => ask(b.dataset.chatAsk)));
    if (chatForm) {
      chatForm.addEventListener("submit", (e) => {
        e.preventDefault();
        const t = chatInput.value;
        chatInput.value = "";
        chatInput.style.height = "";
        ask(t);
      });
      chatInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); chatForm.requestSubmit(); }
      });
      chatInput.addEventListener("input", () => {
        chatInput.style.height = "auto";
        chatInput.style.height = Math.min(chatInput.scrollHeight, 140) + "px";
      });
    }
  }
})();
