"""
Flask Web Service - Article Named Entity Recognition
POST /extract-entities  ->  returns JSON with detected entities
GET  /                  ->  interactive browser panel (HTML)
GET  /evaluation        ->  model card and held-out evaluation summary (HTML)
GET  /health            ->  health check
"""

import json
import logging
import os
import sys

from flask import Flask, jsonify, render_template, request, render_template_string

# Allow imports from src/
sys.path.insert(0, os.path.dirname(__file__))
from ner_pipeline import extract_entities, supported_labels, MAX_TEXT_LENGTH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))

# ---------------------------------------------------------------------------
# HTML Panel - Monochrome Minimalist Black & White Edition
# ---------------------------------------------------------------------------
PANEL_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Kelompok 7 &bull; Named Entity Recognition</title>
  <meta name="description" content="Monochrome NER service for news article entity extraction."/>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet"/>
  <style>
    :root {
      --bg: #09090b;
      --surface: #121215;
      --surface-elevated: #18181b;
      --surface-hover: #222226;
      --border: #27272a;
      --border-light: #3f3f46;
      --border-bright: #71717a;
      --text: #f4f4f5;
      --text-muted: #a1a1aa;
      --text-dim: #71717a;
      --white: #ffffff;
      --black: #000000;
      --radius: 10px;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      -webkit-font-smoothing: antialiased;
      line-height: 1.5;
    }

    /* Subtle grid background pattern */
    body::before {
      content: "";
      position: fixed;
      inset: 0;
      background-image: 
        linear-gradient(to right, rgba(255, 255, 255, 0.02) 1px, transparent 1px),
        linear-gradient(to bottom, rgba(255, 255, 255, 0.02) 1px, transparent 1px);
      background-size: 32px 32px;
      pointer-events: none;
      z-index: 0;
    }

    header {
      position: sticky;
      top: 0;
      z-index: 100;
      background: rgba(9, 9, 11, 0.85);
      backdrop-filter: blur(14px);
      border-bottom: 1px solid var(--border);
      padding: 1.1rem 2rem;
    }

    .header-inner {
      max-width: 1280px;
      margin: 0 auto;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1.5rem;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 0.9rem;
    }

    .brand-logo {
      width: 36px;
      height: 36px;
      background: var(--white);
      color: var(--black);
      border-radius: 8px;
      font-weight: 800;
      font-size: 0.95rem;
      letter-spacing: -0.04em;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .brand-text h1 {
      font-size: 1.05rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      color: var(--white);
    }

    .brand-text p {
      font-size: 0.75rem;
      color: var(--text-muted);
      letter-spacing: 0.02em;
    }

    .header-actions {
      display: flex;
      align-items: center;
      justify-content: flex-end;
      gap: 0.65rem;
      flex-wrap: wrap;
    }

    .nav-link {
      display: inline-flex;
      align-items: center;
      padding: 0.35rem 0.8rem;
      border: 1px solid var(--border-light);
      border-radius: 9999px;
      background: var(--surface);
      color: var(--text-muted);
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.7rem;
      text-decoration: none;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      transition: color 0.15s ease, border-color 0.15s ease, background 0.15s ease;
    }

    .nav-link:hover {
      color: var(--white);
      border-color: var(--border-bright);
      background: var(--surface-elevated);
    }

    .badge-status {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.35rem 0.85rem;
      border-radius: 9999px;
      background: var(--surface-elevated);
      border: 1px solid var(--border-light);
      font-size: 0.72rem;
      font-family: 'JetBrains Mono', monospace;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }

    .badge-dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: var(--white);
      box-shadow: 0 0 8px rgba(255, 255, 255, 0.6);
    }

    main {
      position: relative;
      z-index: 1;
      max-width: 1280px;
      margin: 0 auto;
      padding: 2rem 1.5rem;
    }

    .grid {
      display: grid;
      grid-template-columns: 1.05fr 1fr;
      gap: 1.5rem;
      align-items: start;
    }

    @media (max-width: 900px) {
      .grid {
        grid-template-columns: 1fr;
      }
    }

    @media (max-width: 700px) {
      .header-inner {
        flex-wrap: wrap;
      }

      .header-actions {
        width: 100%;
        justify-content: flex-start;
      }
    }

    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 1.5rem;
      transition: border-color 0.2s ease;
    }

    .card:hover {
      border-color: var(--border-light);
    }

    .card-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 1.25rem;
      padding-bottom: 0.85rem;
      border-bottom: 1px solid var(--border);
    }

    .card-title {
      font-size: 0.8rem;
      font-weight: 700;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .presets-container {
      margin-bottom: 1.25rem;
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 0.45rem;
    }

    .presets-label {
      font-size: 0.72rem;
      color: var(--text-dim);
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-right: 0.2rem;
    }

    .preset-btn {
      background: var(--surface-elevated);
      border: 1px solid var(--border);
      color: var(--text-muted);
      padding: 0.35rem 0.75rem;
      border-radius: 6px;
      font-size: 0.75rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.15s ease;
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
    }

    .preset-btn:hover {
      background: var(--white);
      color: var(--black);
      border-color: var(--white);
      transform: translateY(-1px);
    }

    .preset-btn.clear-btn {
      color: var(--text-dim);
      border-style: dashed;
    }

    .preset-btn.clear-btn:hover {
      background: #27272a;
      color: var(--white);
      border-color: var(--border-bright);
    }

    .form-group {
      margin-bottom: 1.2rem;
    }

    label {
      display: block;
      font-size: 0.8rem;
      font-weight: 600;
      color: var(--text);
      margin-bottom: 0.45rem;
      letter-spacing: -0.01em;
    }

    input[type="text"],
    textarea {
      width: 100%;
      background: var(--surface-elevated);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.8rem 1rem;
      color: var(--white);
      font-family: 'Inter', sans-serif;
      font-size: 0.88rem;
      outline: none;
      transition: all 0.2s ease;
    }

    input[type="text"]:focus,
    textarea:focus {
      border-color: var(--white);
      box-shadow: 0 0 0 1px var(--white);
      background: #1c1c20;
    }

    textarea {
      resize: vertical;
      line-height: 1.6;
      min-height: 200px;
    }

    .char-count {
      font-size: 0.7rem;
      font-family: 'JetBrains Mono', monospace;
      color: var(--text-dim);
      text-align: right;
      margin-top: 0.35rem;
    }

    .char-count.warn {
      color: var(--white);
      font-weight: 700;
    }

    .btn-submit {
      width: 100%;
      padding: 0.9rem 1.5rem;
      background: var(--white);
      color: var(--black);
      border: none;
      border-radius: 8px;
      font-weight: 700;
      font-size: 0.9rem;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 0.6rem;
      letter-spacing: -0.01em;
      margin-top: 0.5rem;
    }

    .btn-submit:hover {
      background: #e4e4e7;
      transform: translateY(-1px);
      box-shadow: 0 6px 20px rgba(255, 255, 255, 0.12);
    }

    .btn-submit:active {
      transform: translateY(0);
    }

    .btn-submit:disabled {
      background: #27272a;
      color: #71717a;
      cursor: not-allowed;
      transform: none;
      box-shadow: none;
    }

    .spinner {
      width: 16px;
      height: 16px;
      border: 2px solid rgba(0, 0, 0, 0.2);
      border-top-color: #000;
      border-radius: 50%;
      animation: spin 0.6s linear infinite;
      display: none;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    /* Result Panel */
    .empty-card {
      background: var(--surface);
      border: 1px dashed var(--border-light);
      border-radius: var(--radius);
      padding: 3.5rem 2rem;
      text-align: center;
      color: var(--text-muted);
    }

    .empty-card svg {
      margin: 0 auto 1.25rem;
      stroke: var(--text-dim);
      opacity: 0.6;
    }

    .empty-card h3 {
      font-size: 0.95rem;
      font-weight: 600;
      color: var(--text);
      margin-bottom: 0.35rem;
    }

    .empty-card p {
      font-size: 0.8rem;
      color: var(--text-dim);
      max-width: 320px;
      margin: 0 auto;
    }

    .result-container {
      display: none;
      animation: fadeIn 0.3s ease;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .result-meta-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 0.75rem;
      margin-bottom: 1.25rem;
      padding-bottom: 0.85rem;
      border-bottom: 1px solid var(--border);
    }

    .result-title {
      font-size: 0.95rem;
      font-weight: 700;
      color: var(--white);
      letter-spacing: -0.01em;
    }

    .chips-group {
      display: flex;
      gap: 0.5rem;
      flex-wrap: wrap;
    }

    .chip {
      padding: 0.25rem 0.65rem;
      border-radius: 9999px;
      font-size: 0.72rem;
      font-family: 'JetBrains Mono', monospace;
      font-weight: 500;
      background: var(--surface-elevated);
      border: 1px solid var(--border-light);
      color: var(--text-muted);
    }

    /* Stats Grid */
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(90px, 1fr));
      gap: 0.75rem;
      margin-bottom: 1.25rem;
    }

    .stat-card {
      background: var(--surface-elevated);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.85rem 0.65rem;
      text-align: center;
      transition: all 0.15s ease;
    }

    .stat-card:hover {
      border-color: var(--border-bright);
    }

    .stat-val {
      font-size: 1.35rem;
      font-weight: 800;
      color: var(--white);
      font-family: 'JetBrains Mono', monospace;
      letter-spacing: -0.03em;
    }

    .stat-lbl {
      font-size: 0.68rem;
      font-weight: 600;
      color: var(--text-dim);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-top: 0.2rem;
    }

    /* Tab Switcher */
    .view-tabs {
      display: flex;
      background: var(--surface-elevated);
      border: 1px solid var(--border);
      padding: 3px;
      border-radius: 8px;
      margin-bottom: 1.25rem;
    }

    .view-tab {
      flex: 1;
      padding: 0.45rem 1rem;
      border-radius: 6px;
      font-size: 0.78rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s ease;
      background: transparent;
      border: none;
      color: var(--text-muted);
      text-align: center;
    }

    .view-tab.active {
      background: var(--white);
      color: var(--black);
      font-weight: 700;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
    }

    /* Table View */
    .table-wrap {
      overflow-x: auto;
      border: 1px solid var(--border);
      border-radius: 8px;
      max-height: 380px;
      overflow-y: auto;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.82rem;
      text-align: left;
    }

    thead th {
      position: sticky;
      top: 0;
      background: #151518;
      color: var(--text-muted);
      font-size: 0.7rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      padding: 0.65rem 0.85rem;
      border-bottom: 1px solid var(--border);
      z-index: 2;
    }

    tbody tr {
      border-bottom: 1px solid var(--border);
      transition: background 0.15s ease;
    }

    tbody tr:hover {
      background: rgba(255, 255, 255, 0.03);
    }

    tbody td {
      padding: 0.65rem 0.85rem;
      color: var(--text);
    }

    /* Monochrome Entity Badges */
    .entity-tag {
      display: inline-flex;
      align-items: center;
      padding: 0.15rem 0.55rem;
      border-radius: 4px;
      font-size: 0.7rem;
      font-weight: 700;
      font-family: 'JetBrains Mono', monospace;
      letter-spacing: 0.04em;
    }

    .tag-PERSON {
      background: var(--white);
      color: var(--black);
    }

    .tag-ORG {
      background: #27272a;
      color: var(--white);
      border: 1px solid var(--border-bright);
    }

    .tag-GPE, .tag-LOC {
      background: #18181b;
      color: #d4d4d8;
      border: 1px solid var(--border-light);
    }

    .tag-DATE {
      background: transparent;
      color: #e4e4e7;
      border: 1px dashed var(--border-bright);
    }

    .tag-MONEY {
      background: #222226;
      color: var(--white);
      border-left: 3px solid var(--white);
    }

    .tag-EVENT {
      background: #18181b;
      color: var(--white);
      border-bottom: 2px solid var(--white);
    }

    .tag-default {
      background: #18181b;
      color: var(--text-muted);
      border: 1px solid var(--border);
    }

    /* Highlight View */
    .highlight-box {
      background: var(--surface-elevated);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1.25rem;
      line-height: 2;
      font-size: 0.9rem;
      color: var(--text);
      max-height: 380px;
      overflow-y: auto;
    }

    mark.ner-ent {
      background: rgba(255, 255, 255, 0.12);
      border-bottom: 2px solid var(--white);
      color: var(--white);
      padding: 1px 4px;
      border-radius: 3px;
      cursor: default;
      transition: background 0.15s ease;
    }

    mark.ner-ent:hover {
      background: rgba(255, 255, 255, 0.22);
    }

    mark.ner-ent sup {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.65rem;
      font-weight: 700;
      margin-left: 4px;
      color: #a1a1aa;
    }

    /* Error Box */
    .error-card {
      margin-top: 1rem;
      padding: 0.85rem 1rem;
      border-radius: 8px;
      background: #1c1414;
      border: 1px solid #7f1d1d;
      color: #fca5a5;
      font-size: 0.82rem;
      display: none;
    }

    footer {
      max-width: 1280px;
      margin: 3rem auto 1rem;
      padding: 1.5rem;
      border-top: 1px solid var(--border);
      text-align: center;
      font-size: 0.75rem;
      color: var(--text-dim);
    }
  </style>
</head>
<body>

<header>
  <div class="header-inner">
    <div class="brand">
      <div class="brand-logo">7</div>
      <div class="brand-text">
        <h1>Kelompok 7 &bull; Named Entity Recognition</h1>
        <p>NLP Financial News Entity Extraction</p>
      </div>
    </div>
    <div class="header-actions">
      <a class="nav-link" href="/evaluation">Model &amp; Evaluation</a>
      <div class="badge-status" id="modelBadgeContainer">
        <span class="badge-dot"></span>
        <span id="modelBadge">LOADING...</span>
      </div>
    </div>
  </div>
</header>

<main>
  <div class="grid">
    <!-- Left: Article Input -->
    <div>
      <div class="card">
        <div class="card-header">
          <span class="card-title">Article Input</span>
        </div>

        <!-- Presets -->
        <div class="presets-container">
          <span class="presets-label">Presets:</span>
          <button type="button" class="preset-btn" onclick="loadSample(1)">Maybank Q3</button>
          <button type="button" class="preset-btn" onclick="loadSample(2)">CIMB ESG</button>
          <button type="button" class="preset-btn" onclick="loadSample(3)">Maybank CSR</button>
          <button type="button" class="preset-btn clear-btn" onclick="clearInput()">Clear</button>
        </div>

        <!-- Title -->
        <div class="form-group">
          <label for="titleInput">Article Title</label>
          <input type="text" id="titleInput" placeholder="Enter headline or article title..." maxlength="500"/>
          <div class="char-count" id="titleCount">0 / 500</div>
        </div>

        <!-- Body -->
        <div class="form-group">
          <label for="bodyInput">Article Body</label>
          <textarea id="bodyInput" rows="10" placeholder="Paste news article body or full text here..."></textarea>
          <div class="char-count" id="bodyCount">0 / 50000</div>
        </div>

        <!-- Submit Button -->
        <button class="btn-submit" id="submitBtn" onclick="analyzeArticle()">
          <span class="spinner" id="spinner"></span>
          <span id="btnText">Extract Entities</span>
        </button>

        <!-- Error Container -->
        <div class="error-card" id="errorBox"></div>
      </div>
    </div>

    <!-- Right: Entity Output -->
    <div>
      <!-- Empty Placeholder -->
      <div class="empty-card" id="placeholderCard">
        <svg width="48" height="48" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
        <h3>No Document Analyzed Yet</h3>
        <p>Select a preset above or type your own news article, then click <strong>Extract Entities</strong>.</p>
      </div>

      <!-- Result Card -->
      <div class="card result-container" id="resultCard">
        <div class="result-meta-row">
          <span class="result-title">Extracted Entities</span>
          <div class="chips-group">
            <span class="chip" id="modelChip">MODEL</span>
            <span class="chip" id="lengthChip">0 CHARS</span>
          </div>
        </div>

        <!-- Stats Grid -->
        <div class="stats-grid" id="statsRow"></div>

        <!-- View Switcher -->
        <div class="view-tabs">
          <button type="button" class="view-tab active" id="tabTable" onclick="showTab('table')">Table View</button>
          <button type="button" class="view-tab" id="tabHighlight" onclick="showTab('highlight')">Highlighted Text</button>
        </div>

        <!-- Table View -->
        <div id="tableView">
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th style="width: 40px;">#</th>
                  <th>Entity Text</th>
                  <th>Type</th>
                  <th style="width: 60px;">Start</th>
                  <th style="width: 60px;">End</th>
                </tr>
              </thead>
              <tbody id="entitiesBody"></tbody>
            </table>
          </div>
        </div>

        <!-- Highlight View -->
        <div id="highlightView" style="display:none">
          <div class="highlight-box" id="highlightedText"></div>
        </div>
      </div>
    </div>
  </div>
</main>

<footer>
  <p>Kelompok 7 Data Science &bull; Named Entity Recognition Project</p>
</footer>

<script>
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

let currentTab = 'table';

function loadSample(idx) {
  const s = SAMPLES[idx];
  if(!s) return;
  document.getElementById('titleInput').value = s.title;
  document.getElementById('bodyInput').value = s.body;
  document.getElementById('titleCount').textContent = s.title.length + ' / 500';
  document.getElementById('bodyCount').textContent = s.body.length + ' / 50000';
  clearError();
}

function clearInput() {
  document.getElementById('titleInput').value = '';
  document.getElementById('bodyInput').value = '';
  document.getElementById('titleCount').textContent = '0 / 500';
  document.getElementById('bodyCount').textContent = '0 / 50000';
  clearError();
}

// Counter listeners
document.getElementById('titleInput').addEventListener('input', function() {
  const n = this.value.length;
  const el = document.getElementById('titleCount');
  el.textContent = n + ' / 500';
  el.classList.toggle('warn', n > 450);
});

document.getElementById('bodyInput').addEventListener('input', function() {
  const n = this.value.length;
  const el = document.getElementById('bodyCount');
  el.textContent = n + ' / 50000';
  el.classList.toggle('warn', n > 45000);
});

// Check health / model name
fetch('/health')
  .then(r => r.json())
  .then(d => {
    document.getElementById('modelBadge').textContent = (d.model || 'en_core_web_sm').toUpperCase();
  })
  .catch(() => {
    document.getElementById('modelBadge').textContent = 'EN_CORE_WEB_SM';
  });

function setLoading(on) {
  document.getElementById('spinner').style.display = on ? 'inline-block' : 'none';
  document.getElementById('btnText').textContent = on ? 'Analyzing Article...' : 'Extract Entities';
  document.getElementById('submitBtn').disabled = on;
}

function showError(msg) {
  const el = document.getElementById('errorBox');
  el.textContent = msg;
  el.style.display = 'block';
}

function clearError() {
  document.getElementById('errorBox').style.display = 'none';
}

async function analyzeArticle() {
  const title = document.getElementById('titleInput').value.trim();
  const body = document.getElementById('bodyInput').value.trim();
  clearError();

  if(!title) { showError('Please provide an article title.'); return; }
  if(!body) { showError('Please provide the article body.'); return; }

  setLoading(true);
  try {
    const res = await fetch('/extract-entities', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title, body })
    });
    const data = await res.json();
    if(!res.ok) {
      showError(data.error || ('Server error ' + res.status));
      return;
    }
    renderResults(data, title + '\n\n' + body);
  } catch(e) {
    showError('Network error: ' + e.message);
  } finally {
    setLoading(false);
  }
}

function getLabelClass(label) {
  const known = ['PERSON', 'ORG', 'GPE', 'LOC', 'DATE', 'EVENT', 'MONEY'];
  return known.includes(label) ? ('tag-' + label) : 'tag-default';
}

function renderResults(data, combinedText) {
  document.getElementById('placeholderCard').style.display = 'none';
  const card = document.getElementById('resultCard');
  card.style.display = 'block';

  document.getElementById('modelChip').textContent = 'MODEL: ' + (data.meta.model || 'en_core_web_sm').toUpperCase();
  document.getElementById('lengthChip').textContent = data.meta.text_length + ' CHARS';

  // Compute stats
  const byType = {};
  data.entities.forEach(e => { byType[e.label] = (byType[e.label] || 0) + 1; });
  const topTypes = Object.entries(byType).sort((a,b) => b[1] - a[1]).slice(0, 4);

  let statsHtml = `<div class="stat-card"><div class="stat-val">${data.meta.entity_count}</div><div class="stat-lbl">TOTAL</div></div>`;
  statsHtml += topTypes.map(([lbl, cnt]) => `
    <div class="stat-card"><div class="stat-val">${cnt}</div><div class="stat-lbl">${lbl}</div></div>
  `).join('');

  document.getElementById('statsRow').innerHTML = statsHtml;

  // Table
  const tbody = document.getElementById('entitiesBody');
  if(data.entities.length === 0) {
    tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--text-dim);padding:2.5rem">No named entities found in this text.</td></tr>';
  } else {
    tbody.innerHTML = data.entities.map((e, i) => `
      <tr>
        <td style="color:var(--text-dim);font-family:\'JetBrains Mono\',monospace;">${i + 1}</td>
        <td><strong style="color:var(--white)">${escHtml(e.text)}</strong></td>
        <td><span class="entity-tag ${getLabelClass(e.label)}">${e.label}</span></td>
        <td style="color:var(--text-dim);font-family:\'JetBrains Mono\',monospace;">${e.start}</td>
        <td style="color:var(--text-dim);font-family:\'JetBrains Mono\',monospace;">${e.end}</td>
      </tr>
    `).join('');
  }

  // Highlight View
  renderHighlight(data.entities, combinedText);
  showTab(currentTab);
}

function renderHighlight(entities, text) {
  const sorted = [...entities].sort((a, b) => a.start - b.start);
  let html = '';
  let pos = 0;
  for(const ent of sorted) {
    if(ent.start < pos) continue;
    html += escHtml(text.slice(pos, ent.start));
    html += `<mark class="ner-ent" title="${ent.label}">${escHtml(ent.text)}<sup>${ent.label}</sup></mark>`;
    pos = ent.end;
  }
  html += escHtml(text.slice(pos));
  document.getElementById('highlightedText').innerHTML = html.replace(/\n/g, '<br/>');
}

function escHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function showTab(tab) {
  currentTab = tab;
  document.getElementById('tableView').style.display = tab === 'table' ? 'block' : 'none';
  document.getElementById('highlightView').style.display = tab === 'highlight' ? 'block' : 'none';
  document.getElementById('tabTable').classList.toggle('active', tab === 'table');
  document.getElementById('tabHighlight').classList.toggle('active', tab === 'highlight');
}

document.getElementById('titleInput').addEventListener('keypress', e => {
  if(e.key === 'Enter') analyzeArticle();
});
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Model / evaluation summary data
# ---------------------------------------------------------------------------

def _load_json_file(filename):
    """Load a project JSON artifact without breaking the web service."""
    path = os.path.join(DATA_DIR, filename)
    try:
        with open(path, "r", encoding="utf-8-sig") as file_handle:
            return json.load(file_handle)
    except FileNotFoundError:
        logger.warning("Summary artifact not found: %s", path)
    except (OSError, ValueError) as exc:
        logger.warning("Could not read summary artifact %s: %s", path, exc)
    return {}


def _fmt_number(value, digits=4):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_int(value):
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "N/A"


def _resolved_model_name():
    trained_path = os.path.join(DATA_DIR, "trained_model")
    if os.path.isdir(trained_path):
        return "trained-model"
    return os.environ.get("NER_MODEL", "en_core_web_sm")


def _evaluation_context():
    training = _load_json_file("training_report.json")
    report = _load_json_file("evaluation_report.json")
    examples = _load_json_file("evaluation_examples.json")
    if not isinstance(examples, list):
        examples = []

    model_settings = training.get("model_settings") or {}
    baseline = (
        (report.get("baseline") or {}).get("overall")
        or (training.get("baseline_metrics") or {}).get("overall")
        or {}
    )
    trained = (
        (report.get("trained") or {}).get("overall")
        or (training.get("test_metrics") or {}).get("overall")
        or {}
    )
    validation = (training.get("validation_metrics") or {}).get("overall") or {}
    per_label = (
        (report.get("trained") or {}).get("per_label")
        or (training.get("test_metrics") or {}).get("per_label")
        or {}
    )

    per_label_rows = []
    for label, metrics in per_label.items():
        f1_value = metrics.get("f1")
        try:
            weak = float(f1_value) < 0.5
        except (TypeError, ValueError):
            weak = False
        per_label_rows.append(
            {
                "label": label,
                "precision": _fmt_number(metrics.get("precision"), 3),
                "recall": _fmt_number(metrics.get("recall"), 3),
                "f1": _fmt_number(f1_value, 3),
                "support": _fmt_int(metrics.get("support", "N/A")),
                "weak": weak,
            }
        )

    tuning_rows = []
    configs = (training.get("validation_tuning") or {}).get("configs") or []
    for config in configs:
        best = (config.get("best_validation") or {}).get("overall") or {}
        tuning_rows.append(
            {
                "name": config.get("name", "N/A"),
                "epoch": config.get("best_epoch", "N/A"),
                "precision": _fmt_number(best.get("precision")),
                "recall": _fmt_number(best.get("recall")),
                "f1": _fmt_number(best.get("f1")),
                "seconds": _fmt_number(config.get("training_seconds"), 1),
                "selected": config.get("name") == model_settings.get("selected_config"),
            }
        )

    data_profile = training.get("data_profile") or {}
    language_distribution = data_profile.get("language_distribution") or {}
    near_duplicate = data_profile.get("near_duplicate_audit") or {}
    data_split = training.get("data_split") or {}
    split_rows = []
    for key, label in (("train", "Train"), ("validation", "Validation"), ("test", "Test")):
        split = data_split.get(key) or {}
        split_rows.append(
            {
                "label": label,
                "documents": _fmt_int(split.get("documents", "N/A")),
                "clusters": _fmt_int(split.get("near_duplicate_clusters", "N/A")),
                "spans": _fmt_int(split.get("spans", "N/A")),
            }
        )

    statuses = sorted({str(example.get("review_status", "unknown")) for example in examples})
    reviewers = sorted(
        {str(example.get("reviewer")) for example in examples if example.get("reviewer")}
    )
    reviewed_dates = sorted(
        {str(example.get("reviewed_at")) for example in examples if example.get("reviewed_at")}
    )
    all_human_reviewed = bool(examples) and statuses == ["human-reviewed"]
    if all_human_reviewed:
        review_label = "Human-reviewed"
    elif examples:
        review_label = "Pending human review"
    else:
        review_label = "No review metadata"

    gold_spans = sum(len(example.get("gold_entities") or []) for example in examples)
    entity_rich = sum(1 for example in examples if example.get("gold_entities"))

    try:
        delta_f1 = f"{float(trained.get('f1')) - float(baseline.get('f1')):+.4f}"
    except (TypeError, ValueError):
        delta_f1 = "N/A"

    return {
        "has_reports": bool(training or report or examples),
        "model_name": _resolved_model_name(),
        "max_text_length": MAX_TEXT_LENGTH,
        "selected_config": model_settings.get("selected_config", "N/A"),
        "selected_epoch": model_settings.get("selected_epoch", "N/A"),
        "base_model": model_settings.get("base_model", "N/A"),
        "base_model_version": model_settings.get("base_model_version", "N/A"),
        "spacy_version": model_settings.get("spacy_version", "N/A"),
        "seed": model_settings.get("seed", "N/A"),
        "training_seconds": _fmt_number(training.get("training_seconds"), 1),
        "training_seconds_total": _fmt_number(training.get("training_seconds_total"), 1),
        "baseline_precision": _fmt_number(baseline.get("precision")),
        "baseline_recall": _fmt_number(baseline.get("recall")),
        "baseline_f1": _fmt_number(baseline.get("f1")),
        "test_precision": _fmt_number(trained.get("precision")),
        "test_recall": _fmt_number(trained.get("recall")),
        "test_f1": _fmt_number(trained.get("f1")),
        "delta_f1": delta_f1,
        "validation_precision": _fmt_number(validation.get("precision")),
        "validation_recall": _fmt_number(validation.get("recall")),
        "validation_f1": _fmt_number(validation.get("f1")),
        "per_label_rows": per_label_rows,
        "tuning_rows": tuning_rows,
        "review_label": review_label,
        "all_human_reviewed": all_human_reviewed,
        "reviewer": ", ".join(reviewers) if reviewers else "N/A",
        "reviewed_at": ", ".join(reviewed_dates) if reviewed_dates else "N/A",
        "examples_count": len(examples),
        "gold_spans": gold_spans,
        "entity_rich": entity_rich,
        "negative_control": len(examples) - entity_rich,
        "raw_rows": _fmt_int(data_profile.get("raw_rows", "N/A")),
        "unique_rows": _fmt_int(data_profile.get("unique_rows", "N/A")),
        "duplicates": _fmt_int(data_profile.get("duplicates_removed", "N/A")),
        "english_rows": _fmt_int(language_distribution.get("en", "N/A")),
        "near_duplicate_clusters": _fmt_int(near_duplicate.get("clusters", "N/A")),
        "near_duplicate_multi": _fmt_int(near_duplicate.get("multi_document_clusters", "N/A")),
        "split_rows": split_rows,
        "limitations": [
            "Training labels are weak supervision from en_core_web_sm, not human gold.",
            "EVENT and MONEY remain the weakest classes in the held-out evaluation.",
            "The evaluation set is small (12 excerpts / 122 spans), so per-label numbers are noisy.",
            "The service is English-only; Malay and Chinese text is not supported well.",
            "Exact-span matching is strict; determiners and punctuation can cause FP/FN.",
        ],
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template_string(PANEL_HTML)


@app.route("/evaluation")
def evaluation_page():
    try:
        context = _evaluation_context()
    except Exception:
        logger.exception("Could not build evaluation summary")
        context = {
            "has_reports": False,
            "model_name": _resolved_model_name(),
            "max_text_length": MAX_TEXT_LENGTH,
            "review_label": "Summary unavailable",
            "all_human_reviewed": False,
            "reviewer": "N/A",
            "reviewed_at": "N/A",
            "per_label_rows": [],
            "tuning_rows": [],
            "split_rows": [],
            "limitations": ["The evaluation reports could not be loaded."],
        }
    return render_template("evaluation.html", **context)


@app.route("/health")
def health():
    try:
        labels = supported_labels()
        model_info = _resolved_model_name()
        return jsonify(
            {
                "status": "ok",
                "model": model_info,
                "supported_labels": labels,
                "max_text_length": MAX_TEXT_LENGTH,
            }
        )
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
    if not isinstance(data, dict):
        return jsonify({"error": "JSON body must be an object"}), 400

    # 2. Required fields must be strings
    title = data.get("title")
    body = data.get("body")

    if not isinstance(title, str):
        return jsonify({"error": "'title' must be a non-null string"}), 400
    if not isinstance(body, str):
        return jsonify({"error": "'body' must be a non-null string"}), 400

    # 3. Run NER
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
