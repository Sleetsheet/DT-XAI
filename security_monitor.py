"""Streamlit-native login protection, SQLite audit/triage, opt-in host USB.
The login score uses transparent rules, NOT the network-traffic models.
"""
from contextlib import contextmanager
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import secrets
import sqlite3
import threading
import time

import streamlit as st
from usb_inventory import inventory

ROOT = Path(__file__).resolve().parent
STATE = Path(os.getenv('STATE_DIR', str(ROOT / 'state')))
LOCK = threading.RLock()

@contextmanager
def db():
    STATE.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(STATE / 'workspace.sqlite3', timeout=15)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init():
    with db() as c:
        c.executescript('''
        CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, ts REAL, ip TEXT, device TEXT,
            event TEXT, risk REAL, level TEXT, parts TEXT);
        CREATE INDEX IF NOT EXISTS event_time ON events(ts);
        CREATE INDEX IF NOT EXISTS event_ip ON events(ip,ts);
        CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE IF NOT EXISTS triage(key TEXT PRIMARY KEY, status TEXT, updated REAL);
        ''')


def setting(key, default=None):
    with db() as c:
        row = c.execute('SELECT value FROM settings WHERE key=?', (key,)).fetchone()
    return json.loads(row['value']) if row else default


def save_setting(key, value):
    with db() as c:
        c.execute('INSERT INTO settings VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                  (key, json.dumps(value)))


def password():
    value = os.getenv('APP_PASSWORD', '')
    if not value:
        try:
            value = str(st.secrets.get('app_password', ''))
        except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
            pass
    return value


@st.cache_resource(show_spinner=False)
def password_reference(secret):
    salt = secrets.token_bytes(32)
    hashed = hashlib.pbkdf2_hmac('sha256', secret.encode(), salt, 250_000)
    return salt, hashed


def client():
    ip, agent = 'unknown-source', ''
    try:
        # st.context supplies server-observed metadata. Do not trust arbitrary XFF.
        observed = st.context.ip_address
        ip = observed if isinstance(observed, str) and observed else 'unknown-source'
        header = st.context.headers.get('User-Agent', '')
        agent = header if isinstance(header, str) else ''
    except (AttributeError, RuntimeError):
        pass
    label = 'Unknown browser'
    for token, name in [('iphone','iPhone'),('ipad','iPad'),('android','Android'),
                        ('windows','Windows PC'),('macintosh','Mac'),('linux','Linux')]:
        if token in agent.lower():
            label = name
            break
    return str(ip)[:100], label


def risk_level(score):
    return 'Low' if score < 40 else 'Medium' if score <= 70 else 'High'


def login_risk(f10, f60, recognised, usb_new):
    parts = {'Failed attempts · 10 min': 50 * min(f10, 5) / 5,
             'Attempt speed · 60 seconds': 20 * min(f60, 3) / 3,
             'Unrecognised Streamlit session': 0 if recognised else 20,
             'New USB on Streamlit host': 10 if usb_new else 0}
    score = min(100., sum(parts.values()))
    return score, risk_level(score), parts


@st.cache_data(ttl=5, show_spinner=False)
def current_usb():
    try:
        devices = inventory()
        return {'devices': devices, 'error': '', 'seen': time.time(), 'supported': True}
    except Exception as exc:
        return {'devices': [], 'error': str(exc), 'seen': time.time(), 'supported': False}


def usb_status():
    if not setting('usb_enabled', False):
        return {'enabled': False, 'supported': False, 'devices': [], 'new': [], 'error': ''}
    result = current_usb()
    baseline = setting('usb_baseline', None)
    if result['supported'] and baseline is None:
        baseline = result['devices']
        save_setting('usb_baseline', baseline)
    return {**result, 'enabled': True,
            'new': sorted(set(result['devices']) - set(baseline or [])) if result['supported'] else []}


def attempt(secret_input):
    now = time.time()
    ip, device = client()
    with LOCK:
        with db() as c:
            failed = c.execute("SELECT ts FROM events WHERE ip=? AND event='failed' AND ts>?", (ip, now-600)).fetchall()
            global_count = c.execute("SELECT COUNT(*) FROM events WHERE event='failed' AND ts>?", (now-60,)).fetchone()[0]
        times = [r['ts'] for r in failed]
        wait = max(0, math.ceil(120-(now-max(times)))) if len(times) >= 5 else 0
        if global_count >= 50:
            wait = max(wait, 60)
        valid = False
        if not wait and 0 < len(secret_input) <= 1024:
            salt, reference = password_reference(password())
            supplied = hashlib.pbkdf2_hmac('sha256', secret_input.encode(), salt, 250_000)
            valid = hmac.compare_digest(reference, supplied)
        kind = 'blocked' if wait else 'success' if valid else 'failed'
        extra = int(kind == 'failed')
        usb = usb_status()
        score, level, parts = login_risk(len(times)+extra, sum(now-t<=60 for t in times)+extra,
                                       bool(st.session_state.get('recognised_session')), bool(usb['new']))
        with db() as c:
            latest = c.execute('SELECT ts FROM events WHERE ip=? ORDER BY id DESC LIMIT 1', (ip,)).fetchone()
            if kind != 'blocked' or not latest or now-latest['ts']>=5:
                c.execute('INSERT INTO events(ts,ip,device,event,risk,level,parts) VALUES(?,?,?,?,?,?,?)',
                          (now, ip, device, kind, score, level, json.dumps(parts)))
            c.execute('DELETE FROM events WHERE ts<?', (now-30*86400,))
        if valid:
            st.session_state['auth_at'] = now
            st.session_state['recognised_session'] = True
        return kind, wait


def authenticated():
    return time.time() - st.session_state.get('auth_at', 0) < 8*3600


def require_auth():
    if not authenticated():
        st.rerun(scope='app')


def logout():
    st.session_state.pop('auth_at', None)
    # Remove widget passwords and private assessment selections from this session.
    for key in list(st.session_state):
        if key.startswith(('password_', 'selected_', 'manual_', 'triage_')):
            del st.session_state[key]


def events():
    with db() as c:
        rows = c.execute('SELECT * FROM events ORDER BY id DESC LIMIT 500').fetchall()
        recent = c.execute('SELECT * FROM events WHERE ts>?', (time.time()-600,)).fetchall()
    result = [dict(r) for r in rows]
    for row in result:
        row['parts'] = json.loads(row['parts'])
    return {'events': result, 'failed': sum(r['event']=='failed' for r in recent),
            'flagged': sum(r['level']!='Low' for r in recent),
            'sources': len({r['ip'] for r in recent}),
            'peak': max((r['risk'] for r in recent), default=0.)}


def triage_states(prefix):
    with db() as c:
        return {r['key'].removeprefix(prefix): r['status'] for r in c.execute(
            'SELECT key,status FROM triage WHERE substr(key,1,?)=?', (len(prefix),prefix))}


def save_triage(key, status):
    if status not in ('Open', 'Investigating', 'Resolved'):
        raise ValueError('Invalid triage status')
    with db() as c:
        c.execute('INSERT INTO triage VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET status=excluded.status,updated=excluded.updated',
                  (key, status, time.time()))
