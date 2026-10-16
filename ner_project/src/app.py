"""
Flask Web Service - Article Named Entity Recognition
POST /extract-entities  ->  returns JSON with detected entities
GET  /                  ->  interactive browser panel (HTML)
GET  /health            ->  health check
"""

import json
import logging
import os
import sys

from flask import Flask, jsonify, request, render_template_string

# Allow imports from src/
sys.path.insert(0, os.path.dirname(__file__))
from ner_pipeline import extract_entities, supported_labels, MAX_TEXT_LENGTH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# HTML Panel (single-page, no external framework)
# ---------------------------------------------------------------------------
PANEL_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Article NER Service – Maybank News</title>
  <meta name="description" content="Named Entity Recognition service for online news articles. Enter a title and body to extract people, organizations, locations, dates and more."/>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet"/>
  <style>
    :root {
      --bg: #0a0e1a;
      --surface: #111827;
      --surface2: #1a2236;
      --border: #1e2d45;
      --accent: #f59e0b;
      --accent2: #3b82f6;
      --accent3: #10b981;
      --text: #e2e8f0;
      --muted: #64748b;
      --danger: #ef4444;
      --radius: 12px;
    }
    *{box-sizing:border-box;margin:0;padding:0}
    body{
      font-family:'Inter',sans-serif;
      background:var(--bg);
      color:var(--text);
      min-height:100vh;
      background-image:
        radial-gradient(ellipse at 20% 20%, rgba(245,158,11,0.08) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(59,130,246,0.08) 0%, transparent 50%);
    }
    header{
      padding:1.5rem 2rem;
      border-bottom:1px solid var(--border);
      background:rgba(17,24,39,0.8);
      backdrop-filter:blur(12px);
      position:sticky;top:0;z-index:100;
      display:flex;align-items:center;gap:1rem;
    }
    .logo{
      width:40px;height:40px;border-radius:10px;
      background:linear-gradient(135deg,var(--accent),#d97706);
      display:flex;align-items:center;justify-content:center;
      font-weight:800;font-size:1.1rem;color:#000;
    }
    header h1{font-size:1.25rem;font-weight:700;letter-spacing:-0.02em}
    header p{font-size:0.8rem;color:var(--muted);margin-top:1px}
    .badge{
      margin-left:auto;padding:0.3rem 0.8rem;border-radius:20px;
      background:rgba(245,158,11,0.15);border:1px solid rgba(245,158,11,0.3);
      color:var(--accent);font-size:0.75rem;font-weight:600;
    }
    main{max-width:1100px;margin:0 auto;padding:2rem;}
    .grid{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;}
    @media(max-width:768px){.grid{grid-template-columns:1fr}}
    .card{
      background:var(--surface);
      border:1px solid var(--border);
      border-radius:var(--radius);
      padding:1.5rem;
    }
    .card-title{
      font-size:0.85rem;font-weight:600;
      color:var(--muted);text-transform:uppercase;letter-spacing:0.05em;
      margin-bottom:1rem;
    }
    label{display:block;font-size:0.875rem;font-weight:500;margin-bottom:0.4rem;color:var(--text)}
    input[type=text],textarea{
      width:100%;background:var(--surface2);
      border:1px solid var(--border);
      border-radius:8px;
      padding:0.75rem 1rem;
      color:var(--text);
      font-family:'Inter',sans-serif;font-size:0.9rem;
      outline:none;transition:border-color 0.2s,box-shadow 0.2s;resize:vertical;
    }
    input[type=text]:focus,textarea:focus{
      border-color:var(--accent);
      box-shadow:0 0 0 3px rgba(245,158,11,0.15);
    }
    .form-group{margin-bottom:1.2rem}
    .char-count{font-size:0.75rem;color:var(--muted);text-align:right;margin-top:0.3rem}
    .char-count.warn{color:var(--danger)}
    .btn{
      width:100%;padding:0.85rem 1.5rem;
      background:linear-gradient(135deg,var(--accent),#d97706);
      border:none;border-radius:8px;
      color:#000;font-weight:700;font-size:0.95rem;
      cursor:pointer;transition:all 0.2s;
      display:flex;align-items:center;justify-content:center;gap:0.5rem;
    }
    .btn:hover{transform:translateY(-1px);box-shadow:0 8px 25px rgba(245,158,11,0.35)}
    .btn:active{transform:translateY(0)}
    .btn:disabled{opacity:0.5;cursor:not-allowed;transform:none}
    .spinner{
      width:18px;height:18px;border:2px solid rgba(0,0,0,0.3);
      border-top-color:#000;border-radius:50%;
      animation:spin 0.7s linear infinite;display:none;
    }
    @keyframes spin{to{transform:rotate(360deg)}}
    .result-card{
      background:var(--surface);border:1px solid var(--border);
      border-radius:var(--radius);padding:1.5rem;margin-top:1.5rem;
      display:none;animation:fadeUp 0.3s ease;
    }
    @keyframes fadeUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
    .result-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:1.2rem}
    .result-header h2{font-size:1rem;font-weight:600}
    .meta-chips{display:flex;gap:0.5rem;flex-wrap:wrap}
    .chip{
      padding:0.25rem 0.7rem;border-radius:20px;
      font-size:0.75rem;font-weight:600;
      background:rgba(59,130,246,0.15);border:1px solid rgba(59,130,246,0.3);color:var(--accent2);
    }
    table{width:100%;border-collapse:collapse;font-size:0.875rem}
    thead th{
      text-align:left;padding:0.6rem 0.8rem;
      border-bottom:1px solid var(--border);
      color:var(--muted);font-size:0.78rem;font-weight:600;text-transform:uppercase;
    }
    tbody tr{transition:background 0.15s}
    tbody tr:hover{background:rgba(255,255,255,0.03)}
    tbody td{padding:0.65rem 0.8rem;border-bottom:1px solid rgba(30,45,69,0.5)}
    .label-badge{
      padding:0.2rem 0.6rem;border-radius:6px;
      font-size:0.75rem;font-weight:700;letter-spacing:0.03em;
    }
    .label-PERSON{background:rgba(167,139,250,0.2);color:#a78bfa;border:1px solid rgba(167,139,250,0.3)}
    .label-ORG{background:rgba(59,130,246,0.2);color:#60a5fa;border:1px solid rgba(59,130,246,0.3)}
    .label-GPE{background:rgba(16,185,129,0.2);color:#34d399;border:1px solid rgba(16,185,129,0.3)}
    .label-LOC{background:rgba(16,185,129,0.2);color:#34d399;border:1px solid rgba(16,185,129,0.3)}
    .label-DATE{background:rgba(245,158,11,0.2);color:#fbbf24;border:1px solid rgba(245,158,11,0.3)}
    .label-EVENT{background:rgba(239,68,68,0.2);color:#f87171;border:1px solid rgba(239,68,68,0.3)}
    .label-MONEY{background:rgba(52,211,153,0.2);color:#34d399;border:1px solid rgba(52,211,153,0.3)}
    .label-NORP{background:rgba(251,191,36,0.2);color:#fbbf24;border:1px solid rgba(251,191,36,0.3)}
    .label-default{background:rgba(100,116,139,0.2);color:#94a3b8;border:1px solid rgba(100,116,139,0.3)}
    .highlighted-text{
      line-height:1.9;font-size:0.9rem;color:var(--text);
      background:var(--surface2);border-radius:8px;padding:1rem;
      max-height:280px;overflow-y:auto;
    }
    mark.ent{border-radius:4px;padding:1px 4px;cursor:default;transition:opacity 0.2s}
    mark.ent:hover{opacity:0.8}
    .empty-state{
      text-align:center;padding:2.5rem;color:var(--muted);
    }
    .empty-state svg{margin:0 auto 1rem;display:block;opacity:0.4}
    .error-box{
      background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);
      border-radius:8px;padding:1rem;color:#f87171;font-size:0.875rem;
      margin-top:1rem;display:none;
    }
    .stats-row{display:flex;gap:1rem;margin-bottom:1.2rem}
    .stat{
      flex:1;background:var(--surface2);border-radius:8px;padding:0.8rem 1rem;
      text-align:center;border:1px solid var(--border);
    }
    .stat-value{font-size:1.4rem;font-weight:700;color:var(--accent)}
    .stat-label{font-size:0.72rem;color:var(--muted);margin-top:2px}
    .tabs{display:flex;gap:0.5rem;margin-bottom:1rem}
    .tab{
      padding:0.4rem 1rem;border-radius:8px;font-size:0.825rem;
      font-weight:600;cursor:pointer;transition:all 0.2s;
      border:1px solid var(--border);color:var(--muted);background:transparent;
    }
    .tab.active{background:rgba(245,158,11,0.15);border-color:rgba(245,158,11,0.4);color:var(--accent)}
  </style>
</head>
<body>
<header>
  <div class="logo">NER</div>
  <div>
    <h1>Article NER Service</h1>
    <p>Named Entity Recognition · Maybank News Dataset</p>
  </div>
  <span class="badge" id="modelBadge">Loading model…</span>
</header>

<main>
  <div class="grid">
    <!-- Input Panel -->
    <div>
      <div class="card">
        <div class="card-title">📰 Article Input</div>
        <div style="display:flex;gap:0.5rem;margin-bottom:1rem;flex-wrap:wrap;align-items:center;">
          <span style="font-size:0.75rem;color:var(--muted);font-weight:600;">Presets:</span>
          <button type="button" class="tab" onclick="loadSample(1)" style="font-size:0.75rem;padding:0.25rem 0.6rem;">📊 Maybank Q3</button>
          <button type="button" class="tab" onclick="loadSample(2)" style="font-size:0.75rem;padding:0.25rem 0.6rem;">🤝 CIMB ESG</button>
          <button type="button" class="tab" onclick="loadSample(3)" style="font-size:0.75rem;padding:0.25rem 0.6rem;">🌱 Maybank CSR</button>
          <button type="button" class="tab" onclick="clearInput()" style="font-size:0.75rem;padding:0.25rem 0.6rem;color:var(--danger);border-color:rgba(239,68,68,0.3);">✕ Clear</button>
        </div>
        <div class="form-group">
          <label for="titleInput">Article Title</label>
          <input type="text" id="titleInput" placeholder="Enter article title…" maxlength="500"/>
          <div class="char-count" id="titleCount">0 / 500</div>
        </div>
        <div class="form-group">
          <label for="bodyInput">Article Body</label>
          <textarea id="bodyInput" rows="10" placeholder="Paste the full article body here…"></textarea>
          <div class="char-count" id="bodyCount">0 / 50000</div>
        </div>
        <button class="btn" id="submitBtn" onclick="analyzeArticle()">
          <span class="spinner" id="spinner"></span>
          <span id="btnText">⚡ Extract Entities</span>
        </button>
        <div class="error-box" id="errorBox"></div>
      </div>
    </div>

    <!-- Results Panel -->
    <div>
      <div class="card" id="placeholderCard">
        <div class="empty-state">
          <svg width="64" height="64" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"/>
          </svg>
          <p>Enter article title and body, then click <strong>Extract Entities</strong></p>
        </div>
      </div>

      <div class="result-card" id="resultCard">
        <div class="result-header">
          <h2>🔍 Extracted Entities</h2>
          <div class="meta-chips">
            <span class="chip" id="modelChip"></span>
            <span class="chip" id="lengthChip"></span>
          </div>
        </div>
        <div class="stats-row" id="statsRow"></div>
        <div class="tabs">
          <button class="tab active" id="tabTable" onclick="showTab('table')">Table View</button>
          <button class="tab" id="tabHighlight" onclick="showTab('highlight')">Highlighted Text</button>
        </div>
        <div id="tableView">
          <table id="entitiesTable">
            <thead><tr><th>#</th><th>Entity Text</th><th>Type</th><th>Start</th><th>End</th></tr></thead>
            <tbody id="entitiesBody"></tbody>
          </table>
        </div>
        <div id="highlightView" style="display:none">
          <div class="highlighted-text" id="highlightedText"></div>
        </div>
      </div>
    </div>
  </div>
</main>

<script>
const LABEL_COLORS = {
  PERSON:  { bg:'rgba(167,139,250,0.25)', border:'#a78bfa' },
  ORG:     { bg:'rgba(59,130,246,0.25)',  border:'#60a5fa' },
  GPE:     { bg:'rgba(16,185,129,0.25)',  border:'#34d399' },
  LOC:     { bg:'rgba(16,185,129,0.25)',  border:'#34d399' },
  DATE:    { bg:'rgba(245,158,11,0.25)',  border:'#fbbf24' },
  EVENT:   { bg:'rgba(239,68,68,0.25)',   border:'#f87171' },
  MONEY:   { bg:'rgba(52,211,153,0.25)',  border:'#34d399' },
  NORP:    { bg:'rgba(251,191,36,0.25)',  border:'#fbbf24' },
  DEFAULT: { bg:'rgba(100,116,139,0.25)', border:'#94a3b8' },
};

let lastData = null;
let currentTab = 'table';

const SAMPLES = {
  1: {
    title: "Maybank posts record net profit of RM2.5b in Q3 2026",
    body: "Kuala Lumpur: Maybank Group, headquartered in Malaysia, reported a record net profit of RM2.5 billion for the third quarter of 2026. Group President and CEO Datuk Khairussaleh Ramli stated that strong loan growth across ASEAN markets including Singapore and Indonesia drove the positive outcome. Total assets rose to RM1.2 trillion as of 30 September 2026."
  },
  2: {
    title: "CIMB Group expands green financing initiative across Southeast Asia",
    body: "CIMB Group Malaysia announced an additional allocation of RM5 billion for sustainable financing in Kuala Lumpur. Group leadership affirmed commitment to regional ESG targets across ASEAN operations by 2050, partnering with Bank Negara Malaysia."
  },
  3: {
    title: "Maybank Foundation empowers 200 youths in rural Sabah community project",
    body: "Maybank Foundation launched its flagship digital upskilling program in Kota Kinabalu, Sabah on 15 September 2026. The initiative aims to provide vocational tech training to underrepresented youth across East Malaysia."
  }
};

function loadSample(idx){
  const s = SAMPLES[idx];
  if(!s) return;
  document.getElementById('titleInput').value = s.title;
  document.getElementById('bodyInput').value = s.body;
  document.getElementById('titleCount').textContent = s.title.length + ' / 500';
  document.getElementById('bodyCount').textContent = s.body.length + ' / 50000';
  clearError();
}

function clearInput(){
  document.getElementById('titleInput').value = '';
  document.getElementById('bodyInput').value = '';
  document.getElementById('titleCount').textContent = '0 / 500';
  document.getElementById('bodyCount').textContent = '0 / 50000';
  clearError();
}

// Character counters
document.getElementById('titleInput').addEventListener('input', function(){
  const n = this.value.length;
  const el = document.getElementById('titleCount');
  el.textContent = n + ' / 500';
  el.classList.toggle('warn', n > 450);
});
document.getElementById('bodyInput').addEventListener('input', function(){
  const n = this.value.length;
  const el = document.getElementById('bodyCount');
  el.textContent = n + ' / 50000';
  el.classList.toggle('warn', n > 45000);
});

// Fetch model info on load
fetch('/health').then(r=>r.json()).then(d=>{
  document.getElementById('modelBadge').textContent = d.model || 'en_core_web_sm';
}).catch(()=>{
  document.getElementById('modelBadge').textContent = 'en_core_web_sm';
});

function setLoading(on){
  document.getElementById('spinner').style.display = on ? 'block' : 'none';
  document.getElementById('btnText').textContent = on ? 'Analyzing…' : '⚡ Extract Entities';
  document.getElementById('submitBtn').disabled = on;
}

function showError(msg){
  const el = document.getElementById('errorBox');
  el.textContent = '⚠ ' + msg;
  el.style.display = 'block';
}
function clearError(){
  document.getElementById('errorBox').style.display = 'none';
}

async function analyzeArticle(){
  const title = document.getElementById('titleInput').value.trim();
  const body = document.getElementById('bodyInput').value.trim();
  clearError();
  if(!title){ showError('Title is required.'); return; }
  if(!body){ showError('Body is required.'); return; }

  setLoading(true);
  try {
    const res = await fetch('/extract-entities', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({title, body})
    });
    const data = await res.json();
    if(!res.ok){
      showError(data.error || `Server error ${res.status}`);
      return;
    }
    lastData = data;
    renderResults(data, title + '\n\n' + body);
  } catch(e){
    showError('Network error: ' + e.message);
  } finally {
    setLoading(false);
  }
}

function getLabelClass(label){
  const known = ['PERSON','ORG','GPE','LOC','DATE','EVENT','MONEY','NORP'];
  return known.includes(label) ? 'label-'+label : 'label-default';
}

function renderResults(data, combinedText){
  document.getElementById('placeholderCard').style.display = 'none';
  const card = document.getElementById('resultCard');
  card.style.display = 'block';

  document.getElementById('modelChip').textContent = '🤖 ' + data.meta.model;
  document.getElementById('lengthChip').textContent = '📄 ' + data.meta.text_length + ' chars';

  // Stats
  const byType = {};
  data.entities.forEach(e => { byType[e.label] = (byType[e.label]||0)+1; });
  const topTypes = Object.entries(byType).sort((a,b)=>b[1]-a[1]).slice(0,4);
  const statsHtml = topTypes.map(([l,c])=>`
    <div class="stat"><div class="stat-value">${c}</div><div class="stat-label">${l}</div></div>
  `).join('') + `<div class="stat"><div class="stat-value">${data.meta.entity_count}</div><div class="stat-label">Total</div></div>`;
  document.getElementById('statsRow').innerHTML = statsHtml;

  // Table
  const tbody = document.getElementById('entitiesBody');
  if(data.entities.length === 0){
    tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--muted);padding:2rem">No entities detected.</td></tr>';
  } else {
    tbody.innerHTML = data.entities.map((e,i)=>`
      <tr>
        <td style="color:var(--muted)">${i+1}</td>
        <td><strong>${escHtml(e.text)}</strong></td>
        <td><span class="label-badge ${getLabelClass(e.label)}">${e.label}</span></td>
        <td style="color:var(--muted)">${e.start}</td>
        <td style="color:var(--muted)">${e.end}</td>
      </tr>
    `).join('');
  }

  // Highlighted text
  renderHighlight(data.entities, combinedText);
  showTab(currentTab);
}

function renderHighlight(entities, text){
  // Sort by start offset, handle overlaps greedily
  const sorted = [...entities].sort((a,b)=>a.start-b.start);
  let html = '';
  let pos = 0;
  for(const ent of sorted){
    if(ent.start < pos) continue; // skip overlapping
    html += escHtml(text.slice(pos, ent.start));
    const c = LABEL_COLORS[ent.label] || LABEL_COLORS.DEFAULT;
    html += `<mark class="ent" title="${ent.label}" style="background:${c.bg};border-bottom:2px solid ${c.border};border-radius:3px;padding:1px 3px">${escHtml(ent.text)}<sup style="font-size:0.65rem;color:${c.border};margin-left:2px">${ent.label}</sup></mark>`;
    pos = ent.end;
  }
  html += escHtml(text.slice(pos));
  document.getElementById('highlightedText').innerHTML = html.replace(/\n/g,'<br/>');
}

function escHtml(s){ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;') }

function showTab(tab){
  currentTab = tab;
  document.getElementById('tableView').style.display = tab==='table' ? 'block' : 'none';
  document.getElementById('highlightView').style.display = tab==='highlight' ? 'block' : 'none';
  document.getElementById('tabTable').classList.toggle('active', tab==='table');
  document.getElementById('tabHighlight').classList.toggle('active', tab==='highlight');
}

// Allow Enter key in title field to submit
document.getElementById('titleInput').addEventListener('keypress', e=>{
  if(e.key==='Enter') analyzeArticle();
});
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template_string(PANEL_HTML)


@app.route("/health")
def health():
    try:
        labels = supported_labels()
        model_info = os.environ.get("NER_MODEL", "en_core_web_sm")
        trained_path = os.path.join(os.path.dirname(__file__), "..", "data", "trained_model")
        if os.path.isdir(trained_path):
            model_info = "trained-model"
        return jsonify({"status": "ok", "model": model_info, "supported_labels": labels})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/extract-entities", methods=["POST"])
def extract_entities_route():
    # 1. Must be JSON
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Malformed JSON body"}), 400

    # 2. Required fields must be strings
    title = data.get("title")
    body = data.get("body")

    if not isinstance(title, str):
        return jsonify({"error": "'title' must be a non-null string"}), 400
    if not isinstance(body, str):
        return jsonify({"error": "'body' must be a non-null string"}), 400

    # 3. Run NER (validation also happens inside)
    try:
        result = extract_entities(title, body)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("NER processing error")
        return jsonify({"error": "Internal NER error", "detail": str(exc)}), 500

    return jsonify(result), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
