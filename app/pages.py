"""Server-rendered HTML pages for NAK AI Detector (own branding, inline CSS/JS).

Premium responsive theme: light, blue/cyan brand system, mobile-first.
Signatures are part of the main.py contract - do not change them.
"""

from __future__ import annotations

import html as _h

BASE_CSS = """
:root {
  --bg: #eaf3fb;
  --card: #ffffff;
  --ink: #0f172a;
  --muted: #64748b;
  --faint: #94a3b8;
  --brand: #0284c7;
  --brand-dark: #0369a1;
  --brand-soft: #e0f2fe;
  --accent: #38bdf8;
  --line: #e2e8f0;
  --ok: #16a34a;
  --err: #dc2626;
  --warn: #b45309;
  --radius: 16px;
  --shadow: 0 10px 30px rgba(2, 132, 199, .10), 0 2px 8px rgba(15, 23, 42, .06);
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
               'Helvetica Neue', Arial, sans-serif;
  background: linear-gradient(180deg, #e3f1fc 0%, var(--bg) 320px);
  color: var(--ink);
  margin: 0;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  font-size: 16px;
  line-height: 1.55;
}
/* ---------- header ---------- */
.sitehead {
  background: rgba(255, 255, 255, .92);
  backdrop-filter: blur(8px);
  border-bottom: 1px solid var(--line);
  position: sticky;
  top: 0;
  z-index: 60;
}
.headin {
  max-width: 960px;
  margin: 0 auto;
  padding: 10px 16px;
  display: flex;
  align-items: center;
  gap: 12px;
}
.hlogo { height: 42px; width: auto; border-radius: 8px; background: #fff; }
.hbrand { line-height: 1.2; }
.hbrand b { display: block; font-size: 17px; letter-spacing: .2px; }
.hbrand span { font-size: 12px; color: var(--muted); }
.hnav { margin-left: auto; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.hnav a { color: var(--brand); font-size: 14px; font-weight: 600; text-decoration: none; }
.hnav a:hover { text-decoration: underline; }
/* ---------- layout ---------- */
.wrap {
  width: 100%;
  max-width: 760px;
  margin: 0 auto;
  padding: 20px 16px 40px;
  flex: 1;
}
.card {
  background: var(--card);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 26px;
  border: 1px solid rgba(226, 232, 240, .8);
}
.card + .card { margin-top: 18px; }
.card h2 { margin: 0 0 6px; font-size: 20px; letter-spacing: .1px; }
.card .lede { color: var(--muted); font-size: 14px; margin: 0 0 14px; }
.section { border-top: 1px solid var(--line); margin-top: 22px; padding-top: 18px; }
.section h2 { font-size: 18px; margin: 0 0 8px; }
.sitefoot {
  text-align: center;
  color: var(--faint);
  font-size: 12.5px;
  padding: 18px 16px 26px;
}
/* ---------- pills / misc ---------- */
.pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: var(--brand-soft);
  border: 1px solid #bae6fd;
  color: #0c4a6e;
  border-radius: 999px;
  padding: 7px 15px;
  font-size: 14px;
  font-weight: 600;
  white-space: nowrap;
}
.pill b { color: var(--brand-dark); }
.muted { color: var(--muted); font-size: 13.5px; }
.small { font-size: 13px; }
/* ---------- forms ---------- */
textarea, input[type=text], input[type=password], input[type=number] {
  width: 100%;
  border-radius: 12px;
  border: 1.5px solid #cbd5e1;
  background: #f8fafc;
  color: var(--ink);
  padding: 13px 14px;
  font-size: 16px;
  margin: 6px 0;
  font-family: inherit;
}
textarea:focus, input:focus {
  outline: none;
  border-color: var(--brand);
  box-shadow: 0 0 0 3px rgba(2, 132, 199, .15);
  background: #fff;
}
input[type=hidden] { display: none !important; }
textarea { min-height: 150px; resize: vertical; }
button, .btn {
  background: linear-gradient(180deg, #0ea5e9, var(--brand));
  color: #fff;
  border: none;
  border-radius: 12px;
  padding: 13px 24px;
  font-size: 16px;
  font-weight: 700;
  cursor: pointer;
  text-decoration: none;
  display: inline-block;
  min-height: 48px;
  box-shadow: 0 4px 14px rgba(2, 132, 199, .30);
}
button:hover, .btn:hover { background: var(--brand-dark); }
button:disabled { opacity: .55; cursor: default; box-shadow: none; }
button.danger {
  background: linear-gradient(180deg, #ef4444, #b91c1c);
  box-shadow: 0 4px 14px rgba(220, 38, 38, .25);
}
button.ghost {
  background: #fff;
  color: var(--brand-dark);
  border: 1.5px solid #bae6fd;
  box-shadow: none;
}
.row { display: flex; align-items: center; gap: 12px; margin-top: 12px; flex-wrap: wrap; }
/* custom checkboxes */
.check {
  display: flex;
  gap: 10px;
  align-items: flex-start;
  background: #f8fafc;
  border: 1.5px solid var(--line);
  border-radius: 12px;
  padding: 11px 13px;
  margin: 8px 0;
  cursor: pointer;
  font-size: 14.5px;
}
.check input { width: 20px; height: 20px; margin: 1px 0 0; accent-color: var(--brand); flex: none; }
.check em { display: block; font-style: normal; color: var(--muted); font-size: 12.5px; }
.opt-note {
  font-size: 13px;
  color: var(--muted);
  background: #f0f9ff;
  border: 1px solid #bae6fd;
  border-radius: 10px;
  padding: 9px 13px;
  margin: 10px 0 0;
}
.opt-note b { color: var(--brand-dark); }
/* API key creation form */
.keyform { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-top: 4px; }
.keyform input[type=text] { flex: 2 1 220px; margin: 0; }
.keyform input[type=number] { flex: 1 1 120px; margin: 0; min-width: 0; }
.keyform .btn-sm { flex: none; }
/* file drop zone */
.filehidden { position: absolute; width: 1px; height: 1px; opacity: 0; overflow: hidden; }
.drop {
  display: flex;
  gap: 14px;
  align-items: center;
  border: 2px dashed #7dd3fc;
  background: #f0f9ff;
  border-radius: 14px;
  padding: 20px;
  cursor: pointer;
  margin: 8px 0;
}
.drop:hover { border-color: var(--brand); background: #e0f2fe; }
.drop-ic { font-size: 34px; }
.drop b { font-size: 16px; }
.filename { font-size: 13.5px; color: var(--brand-dark); font-weight: 600; margin: 4px 0 0;
            word-break: break-all; }
/* ---------- result widgets ---------- */
#count { color: var(--faint); font-size: 13px; }
#result { margin-top: 18px; display: none; }
.verdict { font-size: 21px; font-weight: 800; margin-bottom: 10px; }
.likely_ai { color: #dc2626; }
.likely_human { color: var(--ok); }
.uncertain { color: #ca8a04; }
.barwrap { background: #e2e8f0; border-radius: 10px; height: 22px; overflow: hidden; }
#bar { height: 100%; width: 0%;
       background: linear-gradient(90deg, #22c55e, #facc15, #ef4444);
       transition: width .4s; }
#meta { color: var(--muted); font-size: 13px; margin-top: 8px; }
.err {
  color: #991b1b;
  background: #fef2f2;
  border: 1px solid #fecaca;
  border-radius: 10px;
  padding: 10px 14px;
  margin-top: 12px;
  display: none;
  font-size: 14px;
}
.ok {
  color: #065f46;
  background: #ecfdf5;
  border: 1px solid #a7f3d0;
  border-radius: 10px;
  padding: 10px 14px;
  margin-top: 12px;
  font-size: 14px;
}
.ok a, #scanok a { color: var(--brand-dark); font-weight: 700; }
.apikey {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  background: #0f172a;
  color: #7dd3fc;
  padding: 12px 14px;
  border-radius: 10px;
  word-break: break-all;
  font-size: 13px;
}
/* ---------- feature strip ---------- */
.feats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 16px; }
.feat {
  background: #f8fafc;
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 14px 12px;
  text-align: center;
  font-size: 13px;
  color: var(--muted);
}
.feat b { display: block; color: var(--ink); font-size: 14px; margin-bottom: 2px; }
.feat .fi { font-size: 24px; display: block; margin-bottom: 6px; }
/* ---------- admin table ---------- */
.twrap { overflow-x: auto; margin: 0 -4px; padding: 0 4px; }
table.admin { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 14.5px; }
table.admin th, table.admin td {
  border-bottom: 1px solid var(--line);
  padding: 10px 8px;
  text-align: left;
  vertical-align: middle;
}
table.admin th { color: var(--muted); font-weight: 700; font-size: 12.5px;
                 text-transform: uppercase; letter-spacing: .4px; }
table.admin td .mini { display: inline-flex; gap: 6px; flex-wrap: wrap; }
table.admin input[type=number] { width: 76px; margin: 0; padding: 8px 10px; font-size: 14px; }
.btn-sm { padding: 9px 14px; font-size: 13.5px; min-height: 40px; border-radius: 10px; }
.status { font-size: 12.5px; font-weight: 700; padding: 4px 12px; border-radius: 999px; }
.status.on { background: #dcfce7; color: #166534; }
.status.off { background: #fee2e2; color: #991b1b; }
/* ---------- mobile ---------- */
@media (max-width: 640px) {
  body { font-size: 15px; }
  .wrap { padding: 14px 12px 32px; }
  .card { padding: 20px 16px; border-radius: 14px; }
  .hlogo { height: 36px; }
  .hbrand b { font-size: 15px; }
  .hbrand span { display: none; }
  .hhide { display: none; }
  button, .btn { width: 100%; text-align: center; }
  .row .btn, .row button { flex: 1 1 100%; }
  .feats { grid-template-columns: 1fr; }
  .drop { padding: 16px; }
  /* admin table -> stacked cards */
  table.admin thead { display: none; }
  table.admin, table.admin tbody, table.admin tr, table.admin td {
    display: block; width: 100%;
  }
  table.admin tr {
    border: 1.5px solid var(--line);
    border-radius: 12px;
    margin-bottom: 12px;
    padding: 6px 14px;
    background: #fbfdff;
  }
  table.admin td { border: none; padding: 9px 0; }
  table.admin td.acts { border-top: 1px dashed var(--line); }
  table.admin td::before {
    content: attr(data-label);
    display: block;
    font-size: 11px;
    font-weight: 700;
    color: var(--faint);
    text-transform: uppercase;
    letter-spacing: .5px;
    margin-bottom: 3px;
  }
  table.admin td.acts::before { margin-bottom: 8px; }
  table.admin td.acts .mini { display: flex; flex-direction: column; align-items: stretch; }
  table.admin td.acts form { display: flex !important; gap: 8px; margin: 0; }
  table.admin td.acts form + form { margin-top: 8px; }
  table.admin td.acts input[type=number] { flex: 1 1 auto; width: auto; min-width: 0; }
  table.admin td.acts .btn-sm { flex: 1 1 auto; }
  .repo-del { flex-direction: column; align-items: stretch; }
  .repo-del .btn-sm { width: 100%; }
  .keyform { flex-direction: column; align-items: stretch; }
  .keyform input[type=text], .keyform input[type=number] { flex: 1 1 auto; width: 100%; }
  .keyform .btn-sm { width: 100%; }
}
@media (min-width: 641px) {
  .card { padding: 32px; }
}
"""


def _page(title: str, body: str, right: str = "") -> str:
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_h.escape(title)} - NAK AI Detector</title>
<style>{BASE_CSS}</style></head>
<body>
<header class="sitehead"><div class="headin">
<img src="/assets/logo.webp" class="hlogo" alt="NAK AI Detector">
<div class="hbrand"><b>NAK AI Detector</b><span>Own model &middot; Honest results</span></div>
<div class="hnav">{right}</div>
</div></header>
<main class="wrap">{body}</main>
<footer class="sitefoot">NAK AI Detector &mdash; own model, honest results.<br>
<span style="font-size:11.5px">Reports auto-delete 30 minutes after scanning.</span></footer>
</body></html>"""


def signup_page(firebase_config: dict | None = None, error: str = "") -> str:
    """Futuristic signup with phone OTP + Google. Matches 2100 theme."""
    err = f'<div class="err" style="margin-bottom:16px">{_h.escape(error)}</div>' if error else ""
    fb = firebase_config or {}
    fb_json = _h.escape(__import__("json").dumps(fb))
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sign Up — NAK AI Detector</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#030014;color:#e2e8f0;min-height:100vh;display:flex;align-items:center;justify-content:center;padding:24px;
background-image:radial-gradient(ellipse 60% 40% at 20% 10%,rgba(0,229,255,.08) 0%,transparent 60%),radial-gradient(ellipse 50% 40% at 85% 80%,rgba(139,92,246,.1) 0%,transparent 60%)}}
.card{{background:rgba(255,255,255,.03);border:1px solid rgba(0,229,255,.2);border-radius:24px;padding:44px;max-width:460px;width:100%;backdrop-filter:blur(12px);box-shadow:0 0 60px rgba(0,229,255,.1)}}
.logo{{font-size:20px;font-weight:800;background:linear-gradient(90deg,#00e5ff,#8b5cf6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;text-align:center;margin-bottom:8px}}
h2{{text-align:center;color:#fff;margin-bottom:6px;font-size:26px}}
.sub{{text-align:center;color:#64748b;font-size:14px;margin-bottom:28px}}
.free{{text-align:center;background:rgba(0,229,255,.08);border:1px solid rgba(0,229,255,.25);border-radius:12px;padding:12px;margin-bottom:24px;color:#00e5ff;font-weight:700;font-size:14px}}
label{{display:block;color:#94a3b8;font-size:13px;margin-bottom:6px;font-weight:600}}
input,select{{width:100%;padding:14px 16px;background:rgba(255,255,255,.04);border:1px solid rgba(0,229,255,.2);border-radius:10px;font-size:15px;color:#fff;margin-bottom:14px;outline:none}}
input:focus,select:focus{{border-color:#00e5ff;box-shadow:0 0 16px rgba(0,229,255,.2)}}
input::placeholder{{color:#475569}}
select option{{background:#0a0620}}
.phone-row{{display:flex;gap:10px}}
.phone-row select{{width:110px;flex-shrink:0}}
.btn{{width:100%;padding:15px;border:none;border-radius:12px;font-weight:700;font-size:15px;cursor:pointer;transition:.25s;margin-bottom:12px}}
.btn-pri{{background:linear-gradient(90deg,#00c8e5,#8b5cf6);color:#fff;box-shadow:0 0 24px rgba(0,229,255,.3)}}
.btn-pri:hover{{box-shadow:0 0 36px rgba(0,229,255,.5)}}
.btn-google{{background:#fff;color:#333;display:flex;align-items:center;justify-content:center;gap:10px}}
.btn-google:hover{{box-shadow:0 4px 20px rgba(255,255,255,.2)}}
.divider{{text-align:center;color:#475569;font-size:13px;margin:18px 0;position:relative}}
.divider:before,.divider:after{{content:"";position:absolute;top:50%;width:40%;height:1px;background:rgba(0,229,255,.15)}}
.divider:before{{left:0}}.divider:after{{right:0}}
.err{{background:rgba(255,43,100,.1);border:1px solid rgba(255,43,100,.3);color:#ff6b8a;padding:12px;border-radius:8px;font-size:14px}}
.ok{{background:rgba(0,229,255,.1);border:1px solid rgba(0,229,255,.3);color:#00e5ff;padding:12px;border-radius:8px;font-size:14px}}
.hidden{{display:none}}
.step{{margin-bottom:8px}}
.back{{text-align:center;margin-top:16px}}
.back a{{color:#64748b;font-size:13px;text-decoration:none}}
.back a:hover{{color:#00e5ff}}
#recaptcha{{margin-bottom:14px}}
</style></head><body>
<div class="card">
<div class="logo">◈ NAK AI DETECTOR</div>
<h2>Create Account</h2>
<p class="sub">Join the future of content verification</p>
<div class="free">🎁 Get 1 FREE credit on signup!</div>
{err}
<div id="msg"></div>
<!-- STEP 1: Phone (no OTP - instant) -->
<div id="step1" class="step">
<label>Phone Number</label>
<div class="phone-row">
<select id="cc">
<option value="+92" selected>🇵🇰 +92</option>
<option value="+91">🇮🇳 +91</option>
<option value="+1">🇺🇸 +1</option>
<option value="+44">🇬🇧 +44</option>
<option value="+971">🇦🇪 +971</option>
<option value="+966">🇸🇦 +966</option>
<option value="+61">🇦🇺 +61</option>
<option value="+49">🇩🇪 +49</option>
<option value="+33">🇫🇷 +33</option>
<option value="+81">🇯🇵 +81</option>
</select>
<input type="tel" id="phone" placeholder="300 1234567" style="margin-bottom:0">
</div>
<label style="margin-top:14px">Choose a Password</label>
<input type="password" id="pw" placeholder="Min 6 characters">
<button class="btn btn-pri" onclick="phoneSignup()" style="margin-top:6px">Create Account →</button>
</div>
<div class="divider">OR</div>
<button class="btn btn-google" onclick="googleSignup()">
<svg width="18" height="18" viewBox="0 0 24 24"><path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/><path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/><path fill="#FBBC05" d="M5.84 14.1c-.22-.66-.35-1.36-.35-2.1s.13-1.44.35-2.1V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l3.66-2.84z"/><path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z"/></svg>
Continue with Google
</button>
<div class="back"><a href="/">← Back to home</a> &nbsp;·&nbsp; <a href="#" onclick="openLogin(event)">Already have an account? Log in</a></div>
</div>
<script src="https://www.gstatic.com/firebasejs/10.7.0/firebase-app-compat.js"></script>
<script src="https://www.gstatic.com/firebasejs/10.7.0/firebase-auth-compat.js"></script>
<script>
const FB = {fb_json};
let app = null, confirmResult = null;
if (FB.apiKey) {{
  app = firebase.initializeApp(FB);
  window.recaptchaVerifier = new firebase.auth.RecaptchaVerifier('recaptcha', {{size: 'normal'}});
}}
function showMsg(t, ok) {{
  const m = document.getElementById('msg');
  m.innerHTML = '<div class="'+(ok?'ok':'err')+'">'+t+'</div>';
}}
async function phoneSignup() {{
  const cc = document.getElementById('cc').value;
  const ph = document.getElementById('phone').value.replace(/[^0-9]/g,'');
  const pw = document.getElementById('pw').value;
  if (ph.length < 7) {{ showMsg('Please enter a valid phone number.'); return; }}
  if (pw.length < 6) {{ showMsg('Password must be at least 6 characters.'); return; }}
  const username = cc + ph;
  showMsg('Creating your account...', true);
  const r = await fetch('/api/signup-simple', {{
    method: 'POST', headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{username: username, password: pw, method: 'phone'}})
  }});
  const d = await r.json();
  if (d.ok) {{
    showMsg('🎉 Account created! 1 free credit added. Redirecting...', true);
    setTimeout(()=>{{ window.location.href = '/'; }}, 1500);
  }} else {{ showMsg(d.error || 'Signup failed.'); }}
}}
async function googleSignup() {{
  if (!app) {{ showMsg('Google signup not configured yet.'); return; }}
  showMsg('Opening Google...', true);
  try {{
    const provider = new firebase.auth.GoogleAuthProvider();
    const cred = await firebase.auth().signInWithPopup(provider);
    const token = await cred.user.getIdToken();
    await finishSignup(token, 'google');
  }} catch(e) {{ showMsg('Google signup failed: ' + e.message); }}
}}
async function finishSignup(token, method) {{
  showMsg('Creating your account...', true);
  const r = await fetch('/api/signup', {{
    method: 'POST', headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{id_token: token, method: method}})
  }});
  const d = await r.json();
  if (d.ok) {{
    showMsg('🎉 Account created! 1 free credit added. Redirecting...', true);
    setTimeout(()=>{{ window.location.href = '/'; }}, 1500);
  }} else {{ showMsg(d.error || 'Signup failed.'); }}
}}
function openLogin(e){{e.preventDefault();window.location.href='/?login=1'}}
</script></body></html>"""


def landing_login(error: str = "") -> str:
    err = f'<div class="err" style="display:block">{_h.escape(error)}</div>' if error else ""
    return _page("Login", f"""
<div class="card" style="max-width:480px;margin:32px auto;padding:36px 32px;text-align:center">
<div style="font-size:48px;margin-bottom:12px">🛡️</div>
<h2 style="margin-bottom:6px">NAK AI Detector</h2>
<p class="lede" style="margin-bottom:24px">Secure client portal — AI detection & plagiarism checking</p>
<form method="post" action="/login" style="text-align:left">
<label style="font-size:13px;color:#888;margin-bottom:4px;display:block">Username</label>
<input type="text" name="username" placeholder="Enter your username" required autocomplete="username" style="margin-bottom:14px">
<label style="font-size:13px;color:#888;margin-bottom:4px;display:block">Password</label>
<input type="password" name="password" placeholder="Enter your password" required autocomplete="current-password" style="margin-bottom:20px">
<div class="row" style="justify-content:center"><button type="submit" style="width:100%;padding:14px;font-size:16px">🔐 Log in securely</button></div>
</form>{err}
<p class="muted" style="margin-top:20px;font-size:13px">No account? Contact your administrator.<br>Your documents auto-delete 30 minutes after scanning.</p>
</div>""")


def dashboard(user: dict, error: str = "") -> str:
    err = f'<div class="err" style="display:block">{_h.escape(error)}</div>' if error else ""
    credits = int(user.get("credits", 0))
    username = _h.escape(user["username"])
    right = (f'<span class="pill">Balance: <b>{credits}</b> credits</span>'
             f'<span class="muted hhide">Hi, {username}</span>'
             f'<a href="/logout">Log out</a>')
    head = f"""
<div class="card">
<h2>Scan a document</h2>
<p class="lede">AI detection <b>+</b> plagiarism check in one scan &mdash; 1 credit per document.</p>
<label class="drop" for="file"><span class="drop-ic">&#128196;</span>
<span><b>Choose a document</b><br>
<span class="muted">.txt / .pdf / .docx &mdash; up to 100MB</span></span></label>
<input type="file" id="file" class="filehidden" accept=".txt,.pdf,.doc,.docx">
<div class="filename" id="filename"></div>
<label class="check"><input type="checkbox" id="opt_repo" name="store_repo">
<span>Save to document repository<em>Default off &mdash; with this off, re-scanning the same file never self-matches.</em></span></label>
<label class="check"><input type="checkbox" id="opt_bib" name="exclude_bibliography">
<span>Exclude bibliography<em>Plagiarism only &mdash; references are not counted or highlighted.</em></span></label>
<label class="check"><input type="checkbox" id="opt_quo" name="exclude_quotes">
<span>Exclude quoted text<em>Plagiarism only &mdash; quoted passages are not counted or highlighted.</em></span></label>
<p class="opt-note">Note: AI detection needs at least <b>300 words</b> of prose for a
reliable score &mdash; shorter texts show a low-confidence notice.</p>
<div class="row"><button id="scanbtn" onclick="scan()">Upload &amp; scan</button></div>
<div class="err" id="scanerr"></div><div class="ok" id="scanok" style="display:none"></div>
</div>
<div class="card">
<h2>Recent scans</h2>
<p class="lede">History auto-deletes 30 minutes after each scan.</p>
<div id="histwrap"><p class="muted">Loading…</p></div>
</div>
<div class="card">
<h2>Check text</h2>
<p class="lede">Quick AI-likelihood estimate &mdash; 1 credit per check.</p>
<textarea id="text" placeholder="Paste at least 50 characters of text..."></textarea>
<div class="row"><button id="btn" onclick="check()">Check text</button><span id="count">0 chars</span></div>
<div class="err" id="err"></div>
<div id="result"><div class="verdict" id="verdict"></div>
<div class="barwrap"><div id="bar"></div></div><div id="meta"></div></div>
</div>
<div class="card">
<h2>API access</h2>
<p class="lede">Integrate NAK AI Detector with your own tools via the API
(<b>POST /api/detect</b> and <b>POST /api/scan</b> with your key as the
<b>X-API-Key</b> header).</p>
<p class="muted">Your personal API key was shared with you when your account was
created. Contact the admin if you need it again &mdash; keys are never shown here.</p>
</div>
<div class="feats">
<div class="feat"><span class="fi">&#129302;</span><b>Own AI model</b>Sentence-level detection</div>
<div class="feat"><span class="fi">&#128270;</span><b>Plagiarism check</b>45,000+ reference docs</div>
<div class="feat"><span class="fi">&#128273;</span><b>Private</b>Reports auto-delete in 30 min</div>
</div>
{err}
"""
    # NOTE: plain (non-f) string so the JS braces don't need escaping.
    script = """
<script>
const ta = document.getElementById('text');
ta.addEventListener('input', () => { document.getElementById('count').textContent = ta.value.length + ' chars'; });
const fi = document.getElementById('file');
fi.addEventListener('change', () => {
  document.getElementById('filename').textContent =
    fi.files.length ? 'Selected: ' + fi.files[0].name : '';
});
function esc(s){ return String(s==null?'':s).replace(/[&<>"']/g, function(c){ return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]; }); }
async function loadHistory(){
  const w = document.getElementById('histwrap');
  if (!w) return;
  try {
    const r = await fetch('/api/history');
    const d = await r.json();
    if (!r.ok || !d.scans || !d.scans.length) {
      w.innerHTML = '<p class="muted">No scans yet &mdash; upload a document to get started.</p>';
      return;
    }
    let h = '<div class="twrap"><table class="admin hist"><thead><tr><th>S.No</th><th>File name</th><th>AI %</th><th>Plagiarism %</th><th>AI PDF</th><th>Plag PDF</th></tr></thead><tbody>';
    d.scans.forEach(function(s, i){
      h += '<tr><td data-label="S.No">' + (i+1) + '</td>'
        + '<td data-label="File name"><b>' + esc(s.filename) + '</b><br><span class="muted small">' + esc(s.created_at || '') + '</span></td>'
        + '<td data-label="AI %">' + Number(s.ai_pct).toFixed(1) + '%</td>'
        + '<td data-label="Plagiarism %">' + Number(s.sim_pct).toFixed(1) + '%</td>'
        + '<td data-label="AI PDF"><a class="btn btn-sm ghost" href="' + esc(s.ai_pdf_url) + '">&#8681; AI</a></td>'
        + '<td data-label="Plag PDF"><a class="btn btn-sm ghost" href="' + esc(s.plag_pdf_url) + '">&#8681; Plag</a></td></tr>';
    });
    h += '</tbody></table></div>';
    w.innerHTML = h;
  } catch (e) {
    w.innerHTML = '<p class="muted">Could not load history.</p>';
  }
}
loadHistory();
function labels(v){
  if (v === 'likely_ai') return ['Likely AI-generated', 'likely_ai'];
  if (v === 'likely_human') return ['Likely human-written', 'likely_human'];
  return ['Uncertain', 'uncertain'];
}
async function check(){
  const err = document.getElementById('err');
  const result = document.getElementById('result');
  const btn = document.getElementById('btn');
  err.style.display = 'none'; result.style.display = 'none';
  const text = ta.value.trim();
  if (text.length < 50) { err.textContent = 'Please enter at least 50 characters.'; err.style.display = 'block'; return; }
  btn.disabled = true; btn.textContent = 'Checking...';
  try {
    const r = await fetch('/api/detect', { method: 'POST',
      headers: {'Content-Type': 'application/json'}, body: JSON.stringify({text}) });
    const d = await r.json();
    if (!r.ok) { err.textContent = d.error || 'Request failed.'; err.style.display = 'block'; return; }
    const lbl = labels(d.verdict);
    const v = document.getElementById('verdict');
    v.textContent = lbl[0]; v.className = 'verdict ' + lbl[1];
    document.getElementById('bar').style.width = (d.ai_probability * 100).toFixed(1) + '%';
    document.getElementById('meta').textContent =
      'AI probability: ' + (d.ai_probability * 100).toFixed(1) + '%' +
      '  |  confidence: ' + d.confidence + '  |  credits left: ' + d.credits_remaining;
    result.style.display = 'block';
  } catch (e) { err.textContent = 'Network error. Please try again.'; err.style.display = 'block'; }
  btn.disabled = false; btn.textContent = 'Check text';
}
async function scan(){
  const f = fi.files[0];
  const err = document.getElementById('scanerr');
  const ok = document.getElementById('scanok');
  const btn = document.getElementById('scanbtn');
  err.style.display = 'none'; ok.style.display = 'none';
  if (!f) { err.textContent = 'Choose a file first.'; err.style.display = 'block'; return; }
  btn.disabled = true; btn.textContent = 'Scanning...';
  try {
    const fd = new FormData(); fd.append('file', f);
    // scan options: only sent when enabled (backend treats presence as on)
    if (document.getElementById('opt_repo').checked) fd.append('store_repo', 'on');
    if (document.getElementById('opt_bib').checked) fd.append('exclude_bibliography', 'on');
    if (document.getElementById('opt_quo').checked) fd.append('exclude_quotes', 'on');
    const r = await fetch('/api/scan', { method: 'POST', body: fd });
    const d = await r.json();
    if (!r.ok) { err.textContent = d.error || 'Scan failed.'; err.style.display = 'block'; return; }
    var simTxt = (typeof d.similarity_percentage === 'number')
      ? ' &middot; <b>' + d.similarity_percentage.toFixed(1) + '% similar</b>' : '';
    ok.innerHTML = 'Scan complete: <b>' + d.ai_text_percentage.toFixed(1) + '% AI text</b>' + simTxt +
      ' (' + d.word_count.toLocaleString() + ' words). <a href="' + d.report_url +
      '">Open full report</a>';
    ok.style.display = 'block';
    loadHistory();
  } catch (e) { err.textContent = 'Network error. Please try again.'; err.style.display = 'block'; }
  btn.disabled = false; btn.textContent = 'Upload & scan';
}
</script>"""
    return _page("Dashboard", head + script, right)


def admin_login(error: str = "", misconfigured: bool = False) -> str:
    if misconfigured:
        body = ('<div class="card" style="max-width:460px;margin:24px auto">'
                '<div class="err" style="display:block">Admin access is not configured '
                '(ADMIN_PASSWORD env var missing).</div></div>')
    else:
        err = f'<div class="err" style="display:block">{_h.escape(error)}</div>' if error else ""
        body = f"""
<div class="card" style="max-width:460px;margin:24px auto">
<h2>Admin login</h2>
<p class="lede">Restricted area.</p>
<form method="post" action="/admin/login">
<input type="password" name="password" placeholder="Admin password" required autocomplete="current-password">
<div class="row"><button type="submit">Log in</button></div>
</form>{err}</div>"""
    return _page("Admin", body)


def _api_keys_section(keys: list[dict] | None) -> str:
    """API Access card for the admin panel.

    Contract (backend): each key dict has
      id (str, used for revoke), label (str), owner (str),
      created (str, display-ready), last_used (str|None, display-ready),
      daily_quota (int), credits (int).
    Routes: POST /admin/api/keys/create  (fields: label, daily_quota, credits, unlimited=1)
            POST /admin/api/keys/revoke  (field: key_id)
            POST /admin/api/keys/add_credits (fields: key_id, credits)
    Credits: -1 = unlimited, 0 = empty/exhausted, >0 = finite balance.
    """
    keys = keys or []
    rows = []
    for k in keys:
        kid = _h.escape(str(k.get("id", "")))
        label = _h.escape(str(k.get("label", "")))
        owner = _h.escape(str(k.get("owner", "")))
        created = _h.escape(str(k.get("created", "—")))
        last_used = _h.escape(str(k.get("last_used") or "never"))
        quota = int(k.get("daily_quota", 0))
        credits = int(k.get("credits", 0))
        is_revoked = "[revoked]" in str(k.get("label", ""))
        credits_txt = ("∞ Unlimited" if credits < 0 else
                       "0 (empty)" if credits == 0 else str(credits))
        # Revoked keys get Delete button, active keys get Revoke button
        if is_revoked:
            action_btn = (
                f"<form method='post' action='/admin/api/keys/delete' "
                f"style='display:inline' onsubmit=\"return confirm('Permanently delete this key?')\">"
                f"<input type='hidden' name='key_id' value='{kid}'>"
                f"<button type='submit' class='btn-sm danger'>Delete</button></form>"
            )
        else:
            action_btn = (
                f"<form method='post' action='/admin/api/keys/revoke' "
                f"style='display:inline'>"
                f"<input type='hidden' name='key_id' value='{kid}'>"
                f"<button type='submit' class='btn-sm danger'>Revoke</button></form>"
            )
        rows.append(
            f"<tr><td data-label=\"Label\"><b>{label}</b></td>"
            f"<td data-label=\"Owner\">{owner}</td>"
            f"<td data-label=\"Created\">{created}</td>"
            f"<td data-label=\"Last used\">{last_used}</td>"
            f"<td data-label=\"Daily quota\">{quota}</td>"
            f"<td data-label=\"Credits\"><b>{credits_txt}</b></td>"
            f"<td data-label=\"Action\" style=\"white-space:nowrap\">"
            f"<form method='post' action='/admin/api/keys/add_credits' style='display:inline'>"
            f"<input type='hidden' name='key_id' value='{kid}'>"
            f"<input type='number' name='credits' placeholder='+credits' min='1' style='width:80px'>"
            f"<button type='submit' class='btn-sm'>Add</button></form> "
            f"{action_btn}"
            f"</td></tr>"
        )
    table = ("<div class='twrap'><table class='admin'><thead><tr><th>Label</th><th>Owner</th>"
             "<th>Created</th><th>Last used</th><th>Daily quota</th><th>Credits</th><th>Action</th></tr></thead><tbody>"
             + "".join(rows) + "</tbody></table></div>") if rows else \
        '<p class="muted">No API keys yet.</p>'
    return f"""
<div class="card">
<h2>API Access</h2>
<p class="lede">Service-level API keys for integrations. Keys are shown <b>once</b> at
creation &mdash; they never appear on client pages. Credits = total scans (Rs. 120 each).</p>
{table}
<div class="section"><h2 style="font-size:16px">Create API key</h2>
<form method="post" action="/admin/api/keys/create">
<div class="keyform">
<input type="text" name="label" placeholder="Label (e.g. Website integration)" required>
<input type="number" name="daily_quota" placeholder="Daily quota" value="100" min="0">
<input type="number" name="credits" placeholder="Credits (scans)" value="10" min="0">
<label style="font-size:13px;white-space:nowrap"><input type="checkbox" name="unlimited" value="1" checked> Unlimited</label>
<button type="submit" class="btn-sm">Create key</button>
</div></form></div></div>"""


def admin_api_keys(keys: list[dict], msg: str = "") -> str:
    """Standalone API-keys page (for GET /admin/api/keys if the backend wants
    a dedicated route instead of the inline admin-panel section)."""
    msg_html = f'<div class="ok">{_h.escape(msg)}</div>' if msg else ""
    right = '<a href="/admin">Back to panel</a>'
    return _page("API Access", f"{msg_html}{_api_keys_section(keys)}", right)


def admin_key_created(label: str, api_key: str) -> str:
    """Once-only display of a newly created service API key (admin only)."""
    right = '<a href="/admin">Back to panel</a>'
    return _page("API key created", f"""
<div class="card" style="max-width:560px;margin:24px auto">
<div class="ok" style="display:block">API key <b>{_h.escape(label)}</b> created.</div>
<p class="lede" style="margin-top:14px">Copy now &mdash; shown only once, never again:</p>
<div class="apikey">{_h.escape(api_key)}</div>
<div class="row"><a class="btn ghost" href="/admin">Back to panel</a></div>
</div>""", right)


def admin_panel(users: list[dict], msg: str = "",
                repo_docs: list[dict] | None = None,
                api_keys: list[dict] | None = None,
                ai_min_words: int = 300) -> str:
    rows = []
    for u in users:
        if u.get("is_admin"):
            continue
        uname = _h.escape(u["username"])
        suspended = bool(u.get("suspended"))
        status = ('<span class="status off">Suspended</span>' if suspended
                  else '<span class="status on">Active</span>')
        rows.append(
            f"<tr><td data-label=\"User\"><b>{uname}</b></td>"
            f"<td data-label=\"Credits\">{int(u['credits'])}</td>"
            f"<td data-label=\"Status\">{status}</td>"
            f"<td data-label=\"Actions\" class=\"acts\"><span class=\"mini\">"
            f"<form method='post' action='/admin/credits' style='display:inline-flex;gap:6px'>"
            f"<input type='hidden' name='username' value='{uname}'>"
            f"<input type='number' name='delta' value='5' aria-label='Credit change'>"
            f"<button type='submit' class='btn-sm ghost'>+/- credits</button></form>"
            f"<form method='post' action='/admin/suspend' style='display:inline'>"
            f"<input type='hidden' name='username' value='{uname}'>"
            f"<button type='submit' class='btn-sm {'ghost' if suspended else 'danger'}'>"
            f"{'Unsuspend' if suspended else 'Suspend'}</button></form>"
            f"<form method='post' action='/admin/delete_user' style='display:inline' "
            f"onsubmit=\"return confirm('Permanently delete this account and all its data? "
            f"This cannot be undone.')\">"
            f"<input type='hidden' name='username' value='{uname}'>"
            f"<input type='hidden' name='confirm' value='yes'>"
            f"<button type='submit' class='btn-sm danger'>Delete</button></form>"
            f"</span></td></tr>"
        )
    table = ("<div class='twrap'><table class='admin'><thead><tr><th>User</th><th>Credits</th>"
             "<th>Status</th><th>Actions</th></tr></thead><tbody>"
             + "".join(rows) + "</tbody></table></div>") if rows else \
        '<p class="muted">No client accounts yet.</p>'
    msg_html = f'<div class="ok">{_h.escape(msg)}</div>' if msg else ""
    right = '<a href="/admin/logout">Log out</a>'
    # ---- document repository (doc-vs-doc fingerprints) ----
    repo_docs = repo_docs or []
    rrows = []
    for d in repo_docs[:200]:
        ref = _h.escape(d["doc_ref"])
        fn = _h.escape(d.get("filename") or "—")
        words = f'{int(d.get("words", 0)):,}'
        rrows.append(
            f"<tr><td data-label=\"File\"><b>{fn}</b></td>"
            f"<td data-label=\"Words\">{words}</td>"
            f"<td data-label=\"Action\"><form method='post' action='/admin/repo/delete' "
            f"style='display:inline'>"
            f"<input type='hidden' name='doc_ref' value='{ref}'>"
            f"<button type='submit' class='btn-sm danger'>Remove</button></form>"
            f"</td></tr>"
        )
    rtable = ("<div class='twrap'><table class='admin'><thead><tr><th>File</th><th>Words</th>"
              "<th>Action</th></tr></thead><tbody>"
              + "".join(rrows) + "</tbody></table></div>") if rrows else \
        '<p class="muted">Repository is empty — no document fingerprints stored.</p>'
    repo_html = f"""
<div class="card">
<h2>Document repository</h2>
<p class="lede">Fingerprints of past scans used for document-vs-document comparison.
Only irreversible fingerprints are kept — documents auto-delete after 30 minutes.
New scans are <b>not</b> stored unless the client ticks "Save to document repository".</p>
{rtable}
<form method="post" action="/admin/repo/delete" class="repo-del" style="margin-top:12px;display:flex;gap:8px">
<input type="text" name="filename" placeholder="Delete by filename (or part of it)">
<button type="submit" class="btn-sm danger">Delete matches</button>
</form></div>"""
    return _page("Admin panel", f"""
<div class="card">
<h2>Scan a document</h2>
<p class="lede">Admin scans are free &mdash; <span class="pill">Credits: <b>Unlimited</b></span></p>
<label class="drop" for="afile"><span class="drop-ic">&#128196;</span>
<span><b>Choose a document</b><br>
<span class="muted">.txt / .pdf / .docx &mdash; up to 100MB</span></span></label>
<input type="file" id="afile" class="filehidden" accept=".txt,.pdf,.doc,.docx">
<div class="filename" id="afilename"></div>
<label class="check"><input type="checkbox" id="aopt_repo" name="store_repo">
<span>Save to document repository<em>Default off &mdash; with this off, re-scanning the same file never self-matches.</em></span></label>
<label class="check"><input type="checkbox" id="aopt_bib" name="exclude_bibliography">
<span>Exclude bibliography<em>Plagiarism only &mdash; references are not counted or highlighted.</em></span></label>
<label class="check"><input type="checkbox" id="aopt_quo" name="exclude_quotes">
<span>Exclude quoted text<em>Plagiarism only &mdash; quoted passages are not counted or highlighted.</em></span></label>
<p class="opt-note">Note: AI detection needs at least <b>300 words</b> of prose for a
reliable score &mdash; shorter texts show a low-confidence notice.</p>
<div class="row"><button id="ascanbtn" onclick="ascan()">Upload &amp; scan</button></div>
<div class="err" id="ascanerr"></div><div class="ok" id="ascanok" style="display:none"></div>
</div>
<div class="card">
<h2>Client accounts</h2>
<p class="lede">Create accounts, top up credits, suspend or permanently delete clients.</p>
{msg_html}{table}</div>
{_api_keys_section(api_keys)}
<div class="card">
<h2>Detection settings</h2>
<p class="lede">AI scoring needs at least this many words; shorter documents get
no AI score (similarity still runs). Per-scan <b>ai_min_words</b> API param overrides it.</p>
<form method="post" action="/admin/settings">
<div class="keyform">
<input type="number" name="ai_min_words" value="{ai_min_words}" min="0" max="5000"
 aria-label="AI minimum words">
<button type="submit" class="btn-sm">Save</button>
</div></form></div>
<div class="card">
<h2>Create client account</h2>
<form method="post" action="/admin/create">
<input type="text" name="username" placeholder="Username" required autocomplete="off">
<input type="password" name="password" placeholder="Password" required autocomplete="new-password">
<input type="number" name="credits" placeholder="Starting credits" value="0" min="0">
<div class="row"><button type="submit">Create account</button></div>
</form>
<p class="muted" style="margin-top:10px">The client's API key is shown once after creation &mdash; copy it for them.</p>
</div>{repo_html}""" + _ASCAN_SCRIPT, right)


# Admin scan JS: plain (non-f) string so the JS braces don't need escaping.
_ASCAN_SCRIPT = """
<script>
const afi = document.getElementById('afile');
afi.addEventListener('change', () => {
  document.getElementById('afilename').textContent =
    afi.files.length ? 'Selected: ' + afi.files[0].name : '';
});
async function ascan(){
  const f = afi.files[0];
  const err = document.getElementById('ascanerr');
  const ok = document.getElementById('ascanok');
  const btn = document.getElementById('ascanbtn');
  err.style.display = 'none'; ok.style.display = 'none';
  if (!f) { err.textContent = 'Choose a file first.'; err.style.display = 'block'; return; }
  btn.disabled = true; btn.textContent = 'Scanning...';
  try {
    const fd = new FormData(); fd.append('file', f);
    if (document.getElementById('aopt_repo').checked) fd.append('store_repo', 'on');
    if (document.getElementById('aopt_bib').checked) fd.append('exclude_bibliography', 'on');
    if (document.getElementById('aopt_quo').checked) fd.append('exclude_quotes', 'on');
    const r = await fetch('/api/scan', { method: 'POST', body: fd });
    const d = await r.json();
    if (!r.ok) { err.textContent = d.error || 'Scan failed.'; err.style.display = 'block'; return; }
    var simTxt = (typeof d.similarity_percentage === 'number')
      ? ' &middot; <b>' + d.similarity_percentage.toFixed(1) + '% similar</b>' : '';
    ok.innerHTML = 'Scan complete: <b>' + d.ai_text_percentage.toFixed(1) + '% AI text</b>' + simTxt +
      ' (' + d.word_count.toLocaleString() + ' words). <a href="' + d.report_url +
      '">Open full report</a>';
    ok.style.display = 'block';
  } catch (e) { err.textContent = 'Network error. Please try again.'; err.style.display = 'block'; }
  btn.disabled = false; btn.textContent = 'Upload & scan';
}
</script>"""


def admin_created(username: str) -> str:
    right = '<a href="/admin">Back to panel</a>'
    return _page("Account created", f"""
<div class="card" style="max-width:560px;margin:24px auto">
<div class="ok" style="display:block">Account <b>{_h.escape(username)}</b> created.</div>
<p class="lede" style="margin-top:14px">The client logs in with their username
and password. API keys are never displayed &mdash; manage service-level keys
under <b>API Access</b>.</p>
<div class="row"><a class="btn ghost" href="/admin">Back to panel</a></div>
</div>""", right)
