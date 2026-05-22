from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from datetime import date
import pandas as pd
import statsapi
import joblib

app = FastAPI()
model = joblib.load("best_pitch_model.pkl")

ENCODE_DICT = {0:"CH",1:"CU",2:"FC",3:"FF",4:"FS",5:"KC",6:"SI",7:"SL",8:"ST"}
CODE_TO_NAME = {"ST":"Sweeper","CH":"Changeup","FF":"Four Seam Fastball","SI":"Sinker","SL":"Slider","FC":"Cutter","CU":"Curveball","KC":"Knuckleball","FS":"Split-finger"}
FALLBACK_DATA = {"pitcher":"605397","batter":"606466","on_1b":0,"on_2b":0,"on_3b":0,"if_fielding_alignment":"standard","of_fielding_alignment":"standard","prev_pitch_type":"FF","inning":1,"balls":0,"strikes":0,"outs_when_up":0,"score_diff":0}

EMBEDDED_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Padres Pitch Predictor</title>
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;900&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
:root{--bg:#060d08;--surf:#0c1a0f;--surf2:#122016;--bdr:#1e3524;--acc:#c8f04a;--acc2:#4af07a;--gold:#ffc425;--txt:#e8f5ea;--muted:#5a7a5e;--red:#ff5252;}
*{box-sizing:border-box;margin:0;padding:0;}
html,body{height:100%;overflow:hidden;}
body{background:var(--bg);color:var(--txt);font-family:'IBM Plex Mono',monospace;display:flex;align-items:center;justify-content:center;}
body::before{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(200,240,74,0.03) 1px,transparent 1px),linear-gradient(90deg,rgba(200,240,74,0.03) 1px,transparent 1px);background-size:40px 40px;pointer-events:none;}
.wrap{width:100%;max-width:900px;padding:0.75rem 1.25rem;position:relative;z-index:1;}
.hdr{display:flex;align-items:baseline;justify-content:space-between;margin-bottom:0.5rem;}
.hdr-title{font-family:'Barlow Condensed',sans-serif;font-size:2rem;font-weight:900;text-transform:uppercase;letter-spacing:-0.5px;line-height:1;}
.hdr-title span{color:var(--acc);}
.hdr-right{display:flex;align-items:center;gap:0.5rem;}
.live-badge{display:inline-flex;align-items:center;gap:5px;background:rgba(255,82,82,0.12);border:1px solid rgba(255,82,82,0.3);border-radius:2px;padding:2px 8px;font-size:10px;letter-spacing:2px;color:var(--red);text-transform:uppercase;}
.live-dot{width:5px;height:5px;border-radius:50%;background:var(--red);animation:blink 1.2s infinite;}
/* Auto toggle */
.auto-toggle{display:flex;align-items:center;gap:5px;font-size:10px;letter-spacing:1px;color:var(--muted);text-transform:uppercase;cursor:pointer;user-select:none;}
.auto-toggle input{display:none;}
.toggle-track{width:28px;height:14px;background:var(--bdr);border-radius:7px;position:relative;transition:background 0.2s;}
.toggle-track.on{background:var(--acc);}
.toggle-thumb{position:absolute;top:2px;left:2px;width:10px;height:10px;background:var(--muted);border-radius:50%;transition:all 0.2s;}
.toggle-track.on .toggle-thumb{left:16px;background:var(--bg);}
/* Waiting pulse on result card */
.result.waiting{border-color:var(--muted);animation:none;}
.result.waiting .result-hdr{background:var(--surf2);}
.result.waiting .result-hdr-label{color:var(--muted);}
.result.waiting .result-code{color:var(--muted);animation:none;}
.result.waiting .result-blink{background:var(--muted);}
.countdown{font-size:10px;color:var(--muted);letter-spacing:1px;text-align:right;margin-bottom:0.3rem;min-height:14px;}
.main{display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;margin-bottom:0.5rem;}
.card{background:var(--surf);border:1px solid var(--bdr);border-radius:4px;padding:0.5rem 0.75rem;}
.card-label{font-size:9px;letter-spacing:2px;color:var(--muted);text-transform:uppercase;margin-bottom:0.25rem;}
.matchup{display:flex;align-items:center;justify-content:space-between;gap:0.5rem;}
.player-name{font-family:'Barlow Condensed',sans-serif;font-size:1.2rem;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:140px;}
.pitcher-name{color:var(--acc);}
.batter-name{color:var(--gold);text-align:right;}
.vs{font-family:'Barlow Condensed',sans-serif;font-size:0.85rem;font-weight:900;color:var(--muted);flex-shrink:0;}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--bdr);border-radius:4px;overflow:hidden;border:1px solid var(--bdr);margin-bottom:0.5rem;}
.stat-cell{background:var(--surf);padding:0.4rem 0.6rem;}
.stat-label{font-size:9px;letter-spacing:1.5px;color:var(--muted);text-transform:uppercase;margin-bottom:2px;}
.stat-val{font-family:'Barlow Condensed',sans-serif;font-size:1.3rem;font-weight:700;line-height:1;}
.stat-val.acc{color:var(--acc);}
.stat-val.gold{color:var(--gold);}
.count-bases{display:grid;grid-template-columns:1fr 1fr 1fr auto;gap:0.5rem;margin-bottom:0.5rem;}
.count-box{background:var(--surf);border:1px solid var(--bdr);border-radius:4px;padding:0.4rem 0.6rem;}
.count-label{font-size:9px;letter-spacing:1.5px;color:var(--muted);text-transform:uppercase;margin-bottom:0.3rem;}
.dots{display:flex;gap:4px;align-items:center;}
.dot{width:11px;height:11px;border-radius:50%;border:1.5px solid var(--bdr);background:transparent;transition:all 0.25s;}
.dot.ball.on{background:var(--acc2);border-color:var(--acc2);box-shadow:0 0 6px var(--acc2);}
.dot.strike.on{background:var(--red);border-color:var(--red);box-shadow:0 0 6px var(--red);}
.dot.out.on{background:var(--gold);border-color:var(--gold);box-shadow:0 0 6px var(--gold);}
.bases-box{background:var(--surf);border:1px solid var(--bdr);border-radius:4px;padding:0.4rem 0.6rem;display:flex;align-items:center;gap:0.5rem;}
.btn{width:100%;padding:0.7rem;background:var(--acc);color:var(--bg);border:none;border-radius:4px;font-family:'Barlow Condensed',sans-serif;font-size:1.2rem;font-weight:900;letter-spacing:3px;text-transform:uppercase;cursor:pointer;transition:all 0.15s;margin-bottom:0.35rem;}
.btn:hover{transform:translateY(-1px);box-shadow:0 6px 20px rgba(200,240,74,0.25);}
.btn:active{transform:translateY(0);}
.btn:disabled{opacity:0.4;cursor:not-allowed;transform:none;box-shadow:none;}
.result{display:none;border:1px solid var(--acc);border-radius:4px;overflow:hidden;margin-bottom:0.5rem;}
.result.show{display:block;animation:pop 0.3s ease both;}
.result-hdr{background:var(--acc);padding:0.3rem 0.75rem;display:flex;align-items:center;justify-content:space-between;}
.result-hdr-label{font-size:9px;letter-spacing:3px;color:var(--bg);text-transform:uppercase;font-weight:700;}
.result-blink{width:6px;height:6px;border-radius:50%;background:var(--bg);animation:blink 1s infinite;}
.result-body{padding:0.6rem 1rem;display:flex;align-items:center;justify-content:center;gap:1rem;}
.result-code{font-family:'Barlow Condensed',sans-serif;font-size:3.5rem;font-weight:900;color:var(--acc);line-height:1;animation:scaleIn 0.3s ease both;}
.result-name{font-family:'Barlow Condensed',sans-serif;font-size:1.1rem;font-weight:700;color:var(--txt);letter-spacing:1px;text-transform:uppercase;}
.result-sub{font-size:10px;color:var(--muted);letter-spacing:1px;margin-top:2px;}
.legend{display:grid;grid-template-columns:repeat(9,1fr);gap:1px;background:var(--bdr);border:1px solid var(--bdr);border-radius:4px;overflow:hidden;}
.leg{background:var(--surf);padding:0.3rem 0.4rem;text-align:center;transition:background 0.15s;}
.leg:hover{background:var(--surf2);}
.leg.active{background:rgba(200,240,74,0.1);}
.leg-code{font-family:'Barlow Condensed',sans-serif;font-size:0.95rem;font-weight:700;color:var(--acc);line-height:1;}
.leg-name{font-size:8px;color:var(--muted);letter-spacing:0.3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.raw-toggle{background:none;border:1px solid var(--bdr);border-radius:4px;color:var(--muted);font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:1px;padding:0.3rem 0.6rem;cursor:pointer;transition:all 0.15s;text-transform:uppercase;width:100%;margin-bottom:0.35rem;}
.raw-toggle:hover{border-color:var(--muted);color:var(--txt);}
.raw-panel{display:none;background:var(--surf);border:1px solid var(--bdr);border-radius:4px;padding:0.5rem 0.75rem;margin-bottom:0.5rem;}
.raw-panel.show{display:grid;grid-template-columns:1fr 1fr;gap:0 1rem;}
.raw-row{display:flex;justify-content:space-between;padding:0.2rem 0;border-bottom:1px solid var(--bdr);font-size:10px;}
.raw-row:last-child{border-bottom:none;}
.raw-key{color:var(--muted);}
.raw-val{color:var(--acc);}
.err{display:none;background:rgba(255,82,82,0.1);border:1px solid var(--red);border-radius:4px;padding:0.4rem 0.75rem;font-size:11px;color:var(--red);margin-bottom:0.4rem;}
.err.show{display:block;}
.spinner{display:inline-block;width:13px;height:13px;border:2px solid var(--bg);border-top-color:transparent;border-radius:50%;animation:spin 0.6s linear infinite;vertical-align:middle;margin-right:6px;}
.scanline{position:fixed;top:0;left:0;right:0;height:2px;background:linear-gradient(transparent,rgba(200,240,74,0.05),transparent);animation:scanline 8s linear infinite;pointer-events:none;z-index:999;}
@keyframes blink{0%,100%{opacity:1;}50%{opacity:0.15;}}
@keyframes spin{to{transform:rotate(360deg);}}
@keyframes pop{from{opacity:0;transform:scale(0.97);}to{opacity:1;transform:scale(1);}}
@keyframes scaleIn{from{opacity:0;transform:scale(0.6);}to{opacity:1;transform:scale(1);}}
@keyframes scanline{0%{transform:translateY(-100%);}100%{transform:translateY(100vh);}}
</style>
</head>
<body>
<div class="scanline"></div>
<div class="wrap">
  <div class="hdr">
    <div class="hdr-title">Pitch <span>Predictor</span> <span style="font-size:0.9rem;color:var(--gold);font-weight:600;">· SD Padres</span></div>
    <div class="hdr-right">
      <label class="auto-toggle" title="Auto-predict on every new pitch">
        <input type="checkbox" id="autoCheck" onchange="toggleAuto()"/>
        <div class="toggle-track" id="toggleTrack"><div class="toggle-thumb"></div></div>
        Auto
      </label>
      <div class="live-badge" id="liveBadge"><div class="live-dot"></div><span id="liveText">Ready</span></div>
    </div>
  </div>

  <div class="main">
    <div class="card">
      <div class="card-label">Matchup</div>
      <div class="matchup">
        <div>
          <div style="font-size:9px;letter-spacing:1.5px;color:var(--muted);text-transform:uppercase;">Pitcher</div>
          <div class="player-name pitcher-name" id="pitcherName">—</div>
        </div>
        <div class="vs">VS</div>
        <div style="text-align:right">
          <div style="font-size:9px;letter-spacing:1.5px;color:var(--muted);text-transform:uppercase;">Batter</div>
          <div class="player-name batter-name" id="batterName">—</div>
        </div>
      </div>
    </div>
    <div class="card" style="display:flex;align-items:center;justify-content:center;">
      <svg width="90" height="90" viewBox="0 0 100 100" fill="none">
        <path d="M12 82 Q50 10 88 82" stroke="#1e3524" stroke-width="1.5" fill="none"/>
        <circle cx="50" cy="62" r="26" stroke="#1e3524" stroke-width="1" fill="none" stroke-dasharray="3 3"/>
        <line x1="50" y1="84" x2="76" y2="58" stroke="#2e4a32" stroke-width="1.5"/>
        <line x1="50" y1="84" x2="24" y2="58" stroke="#2e4a32" stroke-width="1.5"/>
        <line x1="24" y1="58" x2="50" y2="32" stroke="#2e4a32" stroke-width="1.5"/>
        <line x1="76" y1="58" x2="50" y2="32" stroke="#2e4a32" stroke-width="1.5"/>
        <rect id="dBase1" x="68" y="50" width="11" height="11" rx="1" fill="#1e3524" stroke="#2e4a32" stroke-width="1.5" transform="rotate(45 73.5 55.5)"/>
        <rect id="dBase2" x="44" y="24" width="11" height="11" rx="1" fill="#1e3524" stroke="#2e4a32" stroke-width="1.5" transform="rotate(45 49.5 29.5)"/>
        <rect id="dBase3" x="20" y="50" width="11" height="11" rx="1" fill="#1e3524" stroke="#2e4a32" stroke-width="1.5" transform="rotate(45 25.5 55.5)"/>
        <polygon points="50,76 44,82 50,88 56,82" fill="#1e3524" stroke="#c8f04a" stroke-width="1.5"/>
        <circle cx="50" cy="58" r="3.5" fill="#c8f04a" opacity="0.85"/>
      </svg>
    </div>
  </div>

  <div class="stats">
    <div class="stat-cell"><div class="stat-label">Inning</div><div class="stat-val acc" id="ctxInning">–</div></div>
    <div class="stat-cell"><div class="stat-label">Score Diff</div><div class="stat-val" id="ctxScore">–</div></div>
    <div class="stat-cell"><div class="stat-label">Last Pitch</div><div class="stat-val gold" id="ctxPrev">–</div></div>
    <div class="stat-cell"><div class="stat-label">IF Align</div><div class="stat-val" id="ctxIF" style="font-size:0.85rem;">–</div></div>
  </div>

  <div class="count-bases">
    <div class="count-box"><div class="count-label">Balls</div><div class="dots" id="balls"><div class="dot ball"></div><div class="dot ball"></div><div class="dot ball"></div><div class="dot ball"></div></div></div>
    <div class="count-box"><div class="count-label">Strikes</div><div class="dots" id="strikes"><div class="dot strike"></div><div class="dot strike"></div><div class="dot strike"></div></div></div>
    <div class="count-box"><div class="count-label">Outs</div><div class="dots" id="outs"><div class="dot out"></div><div class="dot out"></div><div class="dot out"></div></div></div>
    <div class="bases-box">
      <div style="font-size:9px;letter-spacing:1.5px;color:var(--muted);text-transform:uppercase;">Bases</div>
      <svg width="38" height="38" viewBox="0 0 48 48">
        <rect id="sBase1" x="28" y="18" width="13" height="13" rx="1" fill="#1e3524" stroke="#2e4a32" stroke-width="1.5" transform="rotate(45 34.5 24.5)"/>
        <rect id="sBase2" x="17.5" y="8" width="13" height="13" rx="1" fill="#1e3524" stroke="#2e4a32" stroke-width="1.5" transform="rotate(45 24 14.5)"/>
        <rect id="sBase3" x="7" y="18" width="13" height="13" rx="1" fill="#1e3524" stroke="#2e4a32" stroke-width="1.5" transform="rotate(45 13.5 24.5)"/>
      </svg>
    </div>
  </div>

  <div class="err" id="errBanner"></div>
  <div class="countdown" id="countdown"></div>
  <button class="btn" id="predictBtn" onclick="runPrediction()">Predict Next Pitch</button>

  <div class="result" id="resultCard">
    <div class="result-hdr"><span class="result-hdr-label" id="resultLabel">Prediction</span><div class="result-blink"></div></div>
    <div class="result-body">
      <div class="result-code" id="resultCode">FF</div>
      <div><div class="result-name" id="resultName">Four Seam Fastball</div><div class="result-sub" id="resultSub">Model output · Real-time</div></div>
    </div>
  </div>

  <button class="raw-toggle" id="rawToggle" onclick="toggleRaw()" style="display:none">▼ Show Input Features</button>
  <div class="raw-panel" id="rawPanel"></div>
  <div class="legend" id="legend"></div>
</div>

<script>
const C2N={ST:'Sweeper',CH:'Changeup',FF:'Four Seam Fastball',SI:'Sinker',SL:'Slider',FC:'Cutter',CU:'Curveball',KC:'Knuckleball',FS:'Split-finger'};
const legEl=document.getElementById('legend');
Object.entries(C2N).forEach(([c,n])=>{const d=document.createElement('div');d.className='leg';d.id='leg-'+c;d.innerHTML=`<div class="leg-code">${c}</div><div class="leg-name">${n}</div>`;legEl.appendChild(d);});

// --- State ---
let autoMode=false;
let pollTimer=null;
let countdownTimer=null;
let lastFingerprint=null;  // tracks last seen play+pitch count
let pollInterval=4000;     // ms between polls
let countdownSec=0;

function setDots(id,n){document.querySelectorAll(`#${id} .dot`).forEach((d,i)=>{d.classList.remove('on');if(i<n)d.classList.add('on');});}
function setBase(id,on){const e=document.getElementById(id);if(!e)return;e.setAttribute('fill',on?'#c8f04a':'#1e3524');e.setAttribute('stroke',on?'#c8f04a':'#2e4a32');e.style.filter=on?'drop-shadow(0 0 5px rgba(200,240,74,0.8))'  :'';}

function updateUI(d,pn,bn){
  document.getElementById('pitcherName').textContent=pn||d.pitcher;
  document.getElementById('batterName').textContent=bn||d.batter;
  document.getElementById('ctxInning').textContent=d.inning??'–';
  const sd=d.score_diff;
  document.getElementById('ctxScore').textContent=sd>0?`+${sd}`:sd<0?`${sd}`:'Tied';
  document.getElementById('ctxPrev').textContent=d.prev_pitch_type||'–';
  document.getElementById('ctxIF').textContent=(d.if_fielding_alignment||'standard')[0].toUpperCase()+(d.if_fielding_alignment||'').slice(1);
  setDots('balls',d.balls??0);setDots('strikes',d.strikes??0);setDots('outs',d.outs_when_up??0);
  setBase('sBase1',d.on_1b);setBase('sBase2',d.on_2b);setBase('sBase3',d.on_3b);
  setBase('dBase1',d.on_1b);setBase('dBase2',d.on_2b);setBase('dBase3',d.on_3b);
}

function populateRaw(d){document.getElementById('rawPanel').innerHTML=Object.entries(d).map(([k,v])=>`<div class="raw-row"><span class="raw-key">${k}</span><span class="raw-val">${v}</span></div>`).join('');}
function toggleRaw(){const p=document.getElementById('rawPanel'),b=document.getElementById('rawToggle');p.classList.toggle('show');b.textContent=p.classList.contains('show')?'▲ Hide Input Features':'▼ Show Input Features';}
function showErr(msg){const e=document.getElementById('errBanner');e.textContent='⚠ '+msg;e.classList.add('show');setTimeout(()=>e.classList.remove('show'),6000);}
function setLive(txt,active=true){document.getElementById('liveBadge').innerHTML=`<div class="live-dot" style="${active?'':'background:var(--muted)'}"></div><span>${txt}</span>`;}

function showResult(pitchCode,pitchName,waiting=false){
  const card=document.getElementById('resultCard');
  card.classList.remove('show','waiting');
  void card.offsetWidth; // force reflow for animation replay
  card.classList.add('show');
  if(waiting)card.classList.add('waiting');
  document.getElementById('resultCode').textContent=pitchCode;
  document.getElementById('resultName').textContent=pitchName;
  document.getElementById('resultLabel').textContent=waiting?'Waiting for pitch…':'Prediction';
  document.getElementById('resultSub').textContent=waiting?'Pitch thrown — updating…':'Model output · Real-time';
  document.querySelectorAll('.leg').forEach(l=>l.classList.remove('active'));
  if(!waiting){const l=document.getElementById('leg-'+pitchCode);if(l)l.classList.add('active');}
}

// --- Auto mode ---
function toggleAuto(){
  autoMode=document.getElementById('autoCheck').checked;
  document.getElementById('toggleTrack').classList.toggle('on',autoMode);
  if(autoMode){
    setLive('Auto · On');
    startPolling();
  } else {
    stopPolling();
    setLive('Ready');
    document.getElementById('countdown').textContent='';
  }
}

function startPolling(){
  stopPolling();
  poll(); // immediate first poll
}

function stopPolling(){
  if(pollTimer)clearTimeout(pollTimer);
  if(countdownTimer)clearInterval(countdownTimer);
  pollTimer=null;countdownTimer=null;
}

function startCountdown(sec){
  if(countdownTimer)clearInterval(countdownTimer);
  countdownSec=sec;
  document.getElementById('countdown').textContent=`Next poll in ${countdownSec}s`;
  countdownTimer=setInterval(()=>{
    countdownSec--;
    if(countdownSec<=0){clearInterval(countdownTimer);document.getElementById('countdown').textContent='Polling…';}
    else document.getElementById('countdown').textContent=`Next poll in ${countdownSec}s`;
  },1000);
}

async function poll(){
  if(!autoMode)return;
  try{
    // Lightweight fingerprint check first
    const res=await fetch('/fingerprint');
    if(!res.ok)throw new Error('poll failed');
    const {fingerprint,no_game}=await res.json();

    if(no_game){
      setLive('No game today',false);
      document.getElementById('countdown').textContent='No Padres game in progress';
      pollTimer=setTimeout(poll,30000); // check less often when no game
      return;
    }

    if(fingerprint!==lastFingerprint){
      // New pitch detected — run full prediction
      lastFingerprint=fingerprint;
      await runPrediction(true); // true = called from auto
    } else {
      setLive('Auto · Watching');
    }
  }catch(e){
    setLive('Auto · Error',false);
  }
  if(autoMode){
    startCountdown(Math.round(pollInterval/1000));
    pollTimer=setTimeout(poll,pollInterval);
  }
}

// --- Manual / auto prediction ---
async function runPrediction(fromAuto=false){
  const btn=document.getElementById('predictBtn');
  if(!fromAuto){btn.disabled=true;btn.innerHTML='<span class="spinner"></span>Analyzing…';}
  if(!fromAuto){document.getElementById('rawToggle').style.display='none';document.getElementById('rawPanel').classList.remove('show');}

  try{
    const res=await fetch('/predict',{method:'POST'});
    if(!res.ok)throw new Error(`Server error: ${res.status}`);
    const j=await res.json();
    updateUI(j.inputs,j.pitcher_name,j.batter_name);
    populateRaw(j.inputs);
    showResult(j.pitch_code,j.pitch_name,false);
    document.getElementById('rawToggle').style.display='block';
    if(!fromAuto)setLive('Live data');
    else setLive('Auto · Live');
  }catch(err){
    if(!fromAuto){
      showErr(err.message||'Failed to connect.');
      setLive('Demo mode',false);
      runDemo();
    }
  }finally{
    if(!fromAuto){btn.disabled=false;btn.textContent='Predict Next Pitch';}
  }
}

function runDemo(){
  const keys=Object.keys(C2N);
  const rc=keys[Math.floor(Math.random()*keys.length)];
  const d={pitcher:'605397',batter:'606466',on_1b:Math.random()>.6?1:0,on_2b:Math.random()>.75?1:0,on_3b:Math.random()>.9?1:0,if_fielding_alignment:'standard',of_fielding_alignment:'standard',prev_pitch_type:keys[Math.floor(Math.random()*keys.length)],inning:Math.ceil(Math.random()*9),balls:Math.floor(Math.random()*4),strikes:Math.floor(Math.random()*3),outs_when_up:Math.floor(Math.random()*3),score_diff:Math.floor(Math.random()*7)-3};
  updateUI(d,'Dylan Cease','Yordan Alvarez');populateRaw(d);
  showResult(rc,C2N[rc],false);
  document.getElementById('rawToggle').style.display='block';
}
</script>
</body>
</html>
"""

def get_game_id():
    schedule = statsapi.schedule(start_date=date.today(), end_date=date.today(), team="135", sportId=1)
    return schedule[0]["game_id"] if schedule else None

def get_live_feed(game_id):
    live_feed = statsapi.get("game_playByPlay", {"gamePk": game_id})
    all_plays = live_feed.get("allPlays", [])
    return all_plays

def build_context(all_plays):
    if not all_plays:
        return None
    current_play = all_plays[-1]
    play_events = current_play.get("playEvents", [])
    prev_pitch_type = (
        play_events[-1].get("details", {}).get("type", {}).get("code", "firstPitch")
        if play_events else "firstPitch"
    )
    home_score = current_play["result"].get("homeScore", 0) or 0
    away_score = current_play["result"].get("awayScore", 0) or 0
    return {
        "pitcher": current_play["matchup"].get("pitcher", {}).get("id", "None"),
        "batter":  current_play["matchup"].get("batter",  {}).get("id", "None"),
        "on_1b": int("postOnFirst"  in current_play["matchup"]),
        "on_2b": int("postOnSecond" in current_play["matchup"]),
        "on_3b": int("postOnThird"  in current_play["matchup"]),
        "if_fielding_alignment": current_play["matchup"].get("ifFieldingAlignment", "standard"),
        "of_fielding_alignment": current_play["matchup"].get("ofFieldingAlignment", "standard"),
        "prev_pitch_type": prev_pitch_type,
        "inning":       current_play["about"].get("inning"),
        "balls":        current_play["count"].get("balls"),
        "strikes":      current_play["count"].get("strikes"),
        "outs_when_up": current_play["count"].get("outs"),
        "score_diff":   home_score - away_score,
    }

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return HTMLResponse(content=EMBEDDED_HTML)

@app.get("/fingerprint")
async def fingerprint():
    """Lightweight endpoint — returns a fingerprint of the current game state.
    The frontend polls this every few seconds; only calls /predict when it changes."""
    game_id = get_game_id()
    if not game_id:
        return JSONResponse({"fingerprint": None, "no_game": True})
    all_plays = get_live_feed(game_id)
    if not all_plays:
        return JSONResponse({"fingerprint": None, "no_game": False})
    current_play = all_plays[-1]
    play_idx = current_play.get("atBatIndex", 0)
    pitch_count = len(current_play.get("playEvents", []))
    fp = f"{play_idx}-{pitch_count}"
    return JSONResponse({"fingerprint": fp, "no_game": False})

@app.post("/predict")
async def predict():
    game_id = get_game_id()
    all_plays = get_live_feed(game_id) if game_id else []
    data = build_context(all_plays)
    used_fallback = data is None
    if used_fallback:
        data = FALLBACK_DATA.copy()
    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)
    pitch_code = ENCODE_DICT.get(prediction[0], "FF")
    pitch_name = CODE_TO_NAME.get(pitch_code, "Unknown")
    pitcher_name = batter_name = None
    try:
        pitcher_info = statsapi.get("person", {"personId": data["pitcher"]})
        pitcher_name = pitcher_info["people"][0]["fullName"]
        batter_info  = statsapi.get("person", {"personId": data["batter"]})
        batter_name  = batter_info["people"][0]["fullName"]
    except Exception:
        pass
    return JSONResponse({"pitch_code": pitch_code, "pitch_name": pitch_name, "pitcher_name": pitcher_name, "batter_name": batter_name, "inputs": data, "fallback": used_fallback})