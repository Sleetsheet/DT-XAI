"""Verdant — standalone Streamlit edition.
Run: pip install -r requirements.txt && streamlit run app.py
Place real Edge-IIoTset data.csv beside this file; set APP_PASSWORD or st.secrets.
Core modules: collection, preprocessing, detection, risk, XAI; plus security/USB.
"""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import base64
import hashlib
import html
import os
from pathlib import Path
import threading
import time

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pipeline as pipe
import security_monitor as sec

ROOT = Path(__file__).resolve().parent
PALETTE = {'Low': '#35654E', 'Medium': '#B49C71', 'High': '#A16650'}
PAPER, EMERALD, MUTED = '#F8F5ED', '#174D3F', '#74766C'
load_data = st.cache_data(show_spinner='Reading and sampling the real CSV…', max_entries=2)(pipe.load_data)
clean_data = st.cache_data(show_spinner='Preparing numeric traffic features…', max_entries=2)(pipe.clean_data)
train_models = st.cache_resource(show_spinner='Learning from known traffic. This configuration is cached.', max_entries=2)(pipe.train_models)


def esc(value):
    return html.escape(str(value))


@st.cache_data(show_spinner=False)
def font_css():
    result = ''
    for name, path in [('VerdantSerif','serif.woff2'),('VerdantSans','sans.woff2')]:
        file = ROOT / 'assets' / path
        if file.is_file():
            data = base64.b64encode(file.read_bytes()).decode()
            result += f"@font-face{{font-family:{name};src:url(data:font/woff2;base64,{data}) format('woff2');font-display:swap;}}"
    return result


def inject_style():
    st.markdown('<style>'+font_css()+'''
    .stApp {background:#F3F0E8;color:#29352F;}
    html,body,[data-testid="stAppViewContainer"],button,input,textarea {font-family:VerdantSans,Arial,sans-serif;}
    h1,h2,h3,h4,[data-testid="stHeading"] h1,[data-testid="stHeading"] h2,[data-testid="stHeading"] h3 {font-family:VerdantSerif,Georgia,serif!important;font-weight:400!important;letter-spacing:-.025em;}
    .block-container {max-width:1440px;padding-top:3.8rem;padding-bottom:1rem;}
    [data-testid="stSidebar"] {background:#EFEAE1;border-right:1px solid #DDD8CC;}
    [data-testid="stSidebar"] .block-container {padding-top:2rem;}
    [data-testid="stSidebar"] [role="radiogroup"] {gap:.65rem;}
    [data-testid="stSidebar"] [role="radiogroup"] label {padding:9px 12px;border-radius:8px;}
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {background:#DFE5D9;box-shadow:inset 2px 2px 5px #cbd2c4,inset -2px -2px 5px #f8fcf3;}
    [data-testid="stMetric"] {border-radius:13px;padding:20px;background:linear-gradient(135deg,#FBF8F1,#F0ECE3);border:1px solid #FFFFFF99;box-shadow:6px 6px 17px #d8d3c780,-5px -5px 14px #ffffffb3;}
    [data-testid="stMetricValue"],[data-testid="stMetricValue"] * {font-family:VerdantSerif,Georgia,serif!important;font-size:2.65rem!important;color:#214D3B;}
    [data-testid="stMetricLabel"] {font-size:.76rem;color:#7A7D6D;}
    [data-testid="stVerticalBlockBorderWrapper"]>div,[data-testid="stLayoutWrapper"]>div[data-testid="stVerticalBlockBorderWrapper"] {border-radius:14px!important;}
    [data-testid="stVerticalBlockBorderWrapper"] {border-radius:14px;background:#F8F5ED;box-shadow:6px 6px 18px #d8d3c766,-5px -5px 15px #ffffff99;border-color:#ffffff99!important;}
    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlockBorderWrapper"] {box-shadow:none;}
    [data-testid="stButton"] button,[data-testid="stFormSubmitButton"] button {border-radius:8px;}
    [data-testid="stButton"] button[kind="secondary"] {background:#F5F2E9;border-color:#DDD9CB;color:#406248;box-shadow:3px 3px 6px #ddd8cb,-3px -3px 6px #fff9;}
    [data-testid="stButton"] button[kind="primary"],[data-testid="stFormSubmitButton"] button[kind="primary"] {background:#174D3F;border-color:#174D3F;color:#F5F3E8;}
    [data-testid="stTextInput"] input {border-radius:7px;}
    .eyebrow {font-family:VerdantSans,Arial,sans-serif;font-size:10px;letter-spacing:.17em;text-transform:uppercase;color:#8A8E7E;margin-bottom:8px;}
    [class*="st-key-card_"] {background:#F8F5ED!important;border:1px solid #FFFFFFAA!important;border-radius:14px!important;box-shadow:6px 6px 18px #d8d3c766,-5px -5px 15px #ffffff99;}
    [data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li,[data-testid="stCaptionContainer"],label,button,input {font-family:VerdantSans,Arial,sans-serif!important;}
    .hero {padding:10px 0 23px;}
    .hero h1 {font-size:clamp(40px,4.8vw,68px);line-height:1.05;margin:12px 0 16px;padding:0;}
    .hero h1 em {color:#275C43;font-weight:400;}
    .hero p {font-size:13px;color:#818574;max-width:740px;line-height:1.8;}
    .brand {font-family:VerdantSerif,Georgia,serif;font-size:43px;color:#244D37;letter-spacing:-.05em;line-height:1;}
    .brand-note {font-size:8px;letter-spacing:.2em;color:#8B917D;margin:9px 0 30px;}
    .dark-card {background:radial-gradient(ellipse at 95% 80%,#38644666,transparent 65%),linear-gradient(130deg,#1A4E3E,#123C31);padding:30px;border-radius:14px;color:#ECEDDA;min-height:290px;box-shadow:7px 7px 20px #bec8b580;}
    .dark-card .eyebrow {color:#A6B99B;}
    .dark-card h2 {color:#EFF1DA;font-size:42px;line-height:1.05;margin:20px 0;}
    .dark-card h2 em {color:#B3C6A8;}
    .dark-card p {font-size:12px;color:#BAC9AE;line-height:1.9;}
    .dark-card .big {font:46px VerdantSerif,Georgia,serif;color:#E7ECD4;margin-top:24px;}
    .dark-card small {font-size:10px;color:#AABD9D;}
    .note {color:#8D927F;font-size:11px;line-height:1.8;border-top:1px solid #E2DECF;padding-top:14px;margin-top:16px;}
    .pill {display:inline-block;font-size:10px;border-radius:5px;padding:5px 9px;background:#E3E9DC;color:#52714D;margin:0 6px 8px 0;}
    .score {font-family:VerdantSerif,Georgia,serif;font-size:58px;line-height:1.1;color:#27563C;}
    .score small {font-size:20px;color:#939A84;}
    .workspace-footer {display:flex;justify-content:space-between;font-size:9px;letter-spacing:.1em;color:#9B9F8C;border-top:1px solid #DEDCCD;padding:20px 0;margin-top:30px;}
    .login-panel {padding:70px 40px;background:linear-gradient(130deg,#1B4F3E,#143C30);border-radius:18px;min-height:480px;color:#E2EBD0;}
    .login-panel h1 {font-size:58px;line-height:1.08;color:#EBEBD8;margin:80px 0 25px;}
    .login-panel p {color:#AABD9A;font-size:13px;line-height:1.9;}
    .login-panel .brand {color:#DCE7CE;}
    [data-testid="stCaptionContainer"] {color:#8B927D;font-size:11px;}
    [data-testid="stDataFrame"] {border-radius:8px;}
    @media(max-width:700px){.block-container{padding-left:1rem;padding-right:1rem;}.hero h1{font-size:43px}.login-panel{min-height:280px;padding:30px}.login-panel h1{font-size:40px;margin-top:35px}.dark-card h2{font-size:35px}.workspace-footer{font-size:7px}}
    </style>''', unsafe_allow_html=True)


def heading(eyebrow, title, subtitle=''):
    # Only title is trusted author markup; all dataset-derived strings are escaped.
    st.markdown(f'<div class="hero"><div class="eyebrow">{esc(eyebrow)}</div><h1>{title}</h1><p>{esc(subtitle)}</p></div>', unsafe_allow_html=True)


def section(eyebrow, title):
    st.markdown(f'<div class="eyebrow">{esc(eyebrow)}</div>', unsafe_allow_html=True)
    st.subheader(title)


def note(text):
    st.markdown(f'<div class="note">{esc(text)}</div>', unsafe_allow_html=True)


def login():
    if sec.authenticated():
        return True
    left, right = st.columns([1.1, 1], gap='large')
    with left:
        st.markdown('''<div class="login-panel"><div class="brand">verdant</div><div class="eyebrow">IIoT RISK OBSERVATORY</div>
        <h1>Understand the risk.<br/><em>See beyond the signal.</em></h1><p>An explainable view of industrial traffic, unseen attack families, and the devices at your edge.</p></div>''', unsafe_allow_html=True)
    with right:
        st.write(''); st.write('')
        heading('A PRIVATE ANALYTICAL WORKSPACE', 'Welcome inside.', 'Sign in to your risk observatory.')
        secret = sec.password()
        if len(secret) < 12 or secret.startswith(('REPLACE-', 'replace-')):
            st.warning('Set a unique password of at least 12 characters before opening this workspace.')
            st.code('APP_PASSWORD=your-own-strong-password\n# Or .streamlit/secrets.toml:\napp_password = "your-own-strong-password"', language='text')
            return False
        with st.form('sign_in', clear_on_submit=True):
            supplied = st.text_input('Workspace password', type='password', max_chars=1024, key='password_input')
            submitted = st.form_submit_button('Enter workspace', type='primary', width='stretch')
        if submitted:
            kind, wait = sec.attempt(supplied)
            if kind == 'success':
                st.rerun()
            elif kind == 'blocked':
                st.error(f'Too many attempts. Try again in {wait} seconds.')
            else:
                st.error('Incorrect password. This attempt was logged.')
        note('Sign-in attempts are logged and rule-scored. Repeated failures temporarily lock access. Session access expires after eight hours.')
    return False


# Modules 1–3: real data, training-only transformations and reproducible model cache.
@st.cache_data(show_spinner=False, max_entries=2)
def file_identity(path, signature):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@st.cache_data(show_spinner=False, max_entries=6)
def scores(key, pool, _bundle):
    frame = _bundle['zero_raw'] if pool == 'zero' else _bundle['test_raw']
    return pipe.assess_risk(_bundle, frame)


@st.cache_data(show_spinner=False, max_entries=3)
def metrics(key, _bundle):
    b = _bundle
    y, pred = b['y_test'], b['rf'].predict(b['test_scaled'])
    return {'Accuracy': accuracy_score(y, pred), 'Precision · attack': precision_score(y, pred, zero_division=0),
            'Recall · attack': recall_score(y, pred, zero_division=0), 'F1 · attack': f1_score(y, pred, zero_division=0)}, confusion_matrix(y, pred, labels=[0,1]), classification_report(y, pred, labels=[0,1], target_names=['Normal','Attack'], zero_division=0, digits=3)


@st.cache_data(show_spinner=False, max_entries=150)
def local_values(key, values, _bundle):
    frame = pd.DataFrame([values], columns=_bundle['features'])
    return pipe.assess_risk(_bundle, frame).iloc[0], pipe.explain_rows(_bundle, frame)[0]


# Module 5: asynchronous global SHAP; workers never call Streamlit functions.
@st.cache_resource(show_spinner=False)
def explanation_worker():
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix='verdant-shap'), OrderedDict(), threading.Lock()


def global_job(key, bundle):
    executor, jobs, lock = explanation_worker()
    with lock:
        if key not in jobs:
            def work():
                rows = bundle['test_raw'].sample(n=min(500, bundle['test_count']), random_state=42)
                values = pipe.explain_rows(bundle, rows)
                return len(rows), np.abs(values).mean(axis=0)
            jobs[key] = executor.submit(work)
            while len(jobs) > 2:
                _, old = jobs.popitem(last=False)
                old.cancel()
        jobs.move_to_end(key)
        return jobs[key]


def bar_plot(names, values, xlabel, signed=False):
    values = np.asarray(values)
    order = np.argsort(np.abs(values))
    fig = Figure(figsize=(7, max(2.3, len(names)*.29)), facecolor=PAPER)
    ax = fig.subplots(); ax.set_facecolor(PAPER)
    colors = np.where(values[order] >= 0, '#477757', '#7D96A3') if signed else '#527F5A'
    ax.barh(np.asarray(names)[order], values[order], height=.55, color=colors)
    if signed:
        ax.axvline(0, color='#B1B8A3', linewidth=.8)
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    ax.tick_params(axis='both', labelsize=9, colors=MUTED, length=0)
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, width='stretch')


@st.fragment(run_every=3)
def global_panel(key, _bundle):
    sec.require_auth()
    future = global_job(key, _bundle)
    if not future.done():
        st.caption('Global SHAP is computing in the background. Other views remain available.')
        st.progress(.12, text='Up to 500 known-test records · RF only')
        return
    try:
        count, values = future.result()
        bar_plot(_bundle['features'], values, 'Mean absolute class-1 SHAP contribution')
        st.caption(f'{count} sampled known-test records · seed 42. RF component only, not the combined risk.')
    except Exception as exc:
        st.error(f'Global explanation could not finish: {exc}')


def local_explanation(bundle, key, row):
    values_tuple = tuple(float(row[f]) for f in bundle['features'])
    with st.spinner('Tracing feature contributions…'):
        risk, values = local_values(key, values_tuple, bundle)
    st.markdown(f'<div class="score">{float(risk["Risk score"]):.1f}<small> / 100</small></div>', unsafe_allow_html=True)
    level = risk['Risk level']
    {'High':st.error, 'Medium':st.warning, 'Low':st.success}[level](f'{level} risk')
    c1, c2 = st.columns(2)
    c1.caption('RF attack probability'); c1.write(f"**{risk['RF attack probability']:.1%}**")
    c2.caption('Normalized anomaly'); c2.write(f"**{risk['Normalized anomaly']:.3f}**")
    positive = np.flatnonzero(values > 1e-10)
    phrases = []
    for i in sorted(positive, key=lambda i: values[i], reverse=True)[:2]:
        f = bundle['features'][i]; ref = bundle['reference'][f]; value = float(row[f])
        position = 'above the training 75th percentile' if value > ref['75%'] else 'below the training 25th percentile' if value < ref['25%'] else 'within the middle half of training values'
        phrases.append(f'{f} ({value:.4g}, {position})')
    st.write('The strongest features pushing the RF toward attack are '+ ' and '.join(phrases)+'.' if phrases else 'No feature has a positive contribution toward the RF attack output.')
    bar_plot(bundle['features'], values, 'Positive → attack · negative → Normal', signed=True)
    note('SHAP explains Random Forest only—not Isolation Forest or the combined score. These are model associations, not causes.')


# Module 4: risk queues, row selection and persistent investigation state.
def record_queue(ctx, pool='known', top_only=False):
    b, key = ctx['bundle'], ctx['key']
    raw = b['zero_raw'] if pool == 'zero' else b['test_raw']
    frame = raw[['Attack_type']].join(scores(key, pool, b)).sort_values('Risk score', ascending=False)
    prefix = f'{key}:{pool}:'
    states = sec.triage_states(prefix)
    frame['Triage'] = [states.get(str(i), 'Open') for i in frame.index]
    if top_only:
        frame = frame.head(10)
    else:
        l, r = st.columns([3,1])
        query = l.text_input('Search record or family', placeholder='Record ID or attack family', key=f'search_{key}_{pool}')
        level = r.selectbox('Risk level', ['All','High','Medium','Low'], key=f'filter_{key}_{pool}')
        if query:
            frame = frame.loc[frame.Attack_type.str.contains(query, case=False, regex=False) | frame.index.astype(str).str.contains(query, regex=False)]
        if level != 'All':
            frame = frame.loc[frame['Risk level'].eq(level)]
    if frame.empty:
        st.info('No records match these filters.'); return
    total = len(frame)
    if not top_only:
        page = st.number_input('Record page', min_value=1, max_value=max(1, int(np.ceil(total/25))), value=1, step=1, key=f'page_{key}_{pool}_{total}')
        frame = frame.iloc[(page-1)*25:page*25]
    display = frame.reset_index().rename(columns={'index':'record_id'})
    if 'record_id' not in display:
        display = display.rename(columns={display.columns[0]:'record_id'})
    st.caption(f'{total:,} records · select a row to inspect its RF explanation')
    selected = st.dataframe(display, hide_index=True, width='stretch', on_select='rerun', selection_mode='single-row',
                            key=f'queue_{key}_{pool}_{top_only}_{tuple(frame.index)}',
                            column_config={'Risk score': st.column_config.ProgressColumn('Risk / 100', min_value=0, max_value=100, format='%.1f'),
                                           'RF attack probability': st.column_config.NumberColumn('RF probability', format='%.3f'),
                                           'Normalized anomaly': st.column_config.NumberColumn('Anomaly', format='%.3f')})
    chosen = int(display.iloc[selected.selection.rows[0]]['record_id']) if selected.selection.rows else None
    # Native fallback also supports keyboard users and Streamlit testing tools.
    with st.expander('Choose a record by ID'):
        manual_id = st.selectbox('Record ID', frame.index.tolist(), index=None, placeholder='Choose a record', format_func=lambda v: f'#{v}', key=f'selected_{key}_{pool}_{top_only}')
    if chosen is None:
        chosen = manual_id
    if chosen is not None and chosen in raw.index:
        left, right = st.columns([2,1], gap='large')
        with left, st.container(border=True, key="card_01"):
            section('LOCAL EXPLAINABILITY', f'Record #{chosen}')
            local_explanation(b, key, raw.loc[chosen])
        with right, st.container(border=True, key="card_02"):
            section('INVESTIGATION TRAIL', 'Analyst triage')
            st.caption('A triage status is not a change to the model prediction or ground truth.')
            options = ['Open','Investigating','Resolved']
            previous = states.get(str(chosen), 'Open')
            updated = st.selectbox('Status', options, index=options.index(previous), key=f'triage_{key}_{pool}_{chosen}')
            if updated != previous:
                sec.save_triage(prefix+str(chosen), updated)
                st.rerun()
            st.dataframe(raw.loc[[chosen], b['features']].T.rename(columns={chosen:'Original value'}), width='stretch')


def overview(ctx=None):
    heading('THE INDUSTRIAL RISK LANDSCAPE', 'Clarity in every <em>signal.</em>', 'A considered view of known threats, hidden patterns, and the evidence behind them.')
    security = sec.events()
    b = ctx['bundle'] if ctx else None
    known = scores(ctx['key'], 'known', b) if ctx else None
    zero = scores(ctx['key'], 'zero', b) if ctx else None
    values = metrics(ctx['key'], b)[0] if ctx else None
    cards = st.columns(4)
    cards[0].metric('Attack recall', f"{values['Recall · attack']:.1%}" if values else '—', help='RF on the known-class test partition')
    cards[1].metric('High-risk records', int(known['Risk level'].eq('High').sum()) if ctx else '—')
    cards[2].metric('Unseen-family detection', f"{zero['Risk score'].ge(40).mean():.1%}" if ctx else '—')
    cards[3].metric('Login alerts · 10 min', security['flagged'])
    st.write('')
    left, right = st.columns([1.65,1], gap='large')
    with left, st.container(border=True, key="card_03"):
        section('KNOWN-CLASS TEST', 'The risk spectrum')
        if ctx:
            levels = known['Risk level'].value_counts().reindex(['Low','Medium','High'], fill_value=0)
            chart, legend = st.columns([1.3,1])
            with chart:
                fig = Figure(figsize=(4,2.6), facecolor=PAPER); ax=fig.subplots()
                ax.pie(levels.values, colors=list(PALETTE.values()), startangle=90, counterclock=False, wedgeprops={'width':.13,'edgecolor':PAPER,'linewidth':2})
                ax.text(0,.04,f'{len(known):,}',ha='center',va='center',fontsize=25,color=EMERALD,fontfamily='serif')
                ax.text(0,-.23,'TEST RECORDS',ha='center',fontsize=7,color=MUTED)
                fig.tight_layout(); st.pyplot(fig, width='stretch')
            with legend:
                for name in ['Low','Medium','High']:
                    st.write(f'**{name}** — {int(levels[name]):,}')
                    st.caption(f'{levels[name]/len(known):.1%} of known-test records')
        else:
            st.info('Your real dataset will populate this spectrum. No synthetic metrics are displayed.')
        st.caption('Low <40 · Medium 40–70 inclusive · High >70')
    with right:
        attack = esc(ctx['holdout'].replace('_',' ')) if ctx else 'A reserved attack family'
        rate = f"{zero['Risk score'].ge(40).mean():.1%}" if ctx else '—'
        st.markdown(f'<div class="dark-card"><div class="eyebrow">BEYOND THE KNOWN</div><h2>One family.<br/><em>Entirely unseen.</em></h2><p>{attack} is excluded from training, feature selection, scaling, and calibration.</p><div class="big">{rate}</div><small>Combined flag rate · risk ≥40</small></div>', unsafe_allow_html=True)
    st.write('')
    left, right = st.columns([1.65,1], gap='large')
    with left, st.container(border=True, key="card_04"):
        section('GLOBAL RF EXPLANATION', 'What shapes the decision')
        if ctx:
            global_panel(ctx['key'], b)
        else:
            st.caption('Real class-1 feature contributions appear automatically after training.')
    with right, st.container(border=True, key="card_05"):
        section('ACCESS & ENDPOINTS', 'A closer watch')
        st.write(f"**{security['failed']}** failed sign-ins in the last ten minutes")
        usb = sec.usb_status()
        if not usb['enabled']: st.caption('Host USB monitoring is off. Enable it explicitly in Security Monitor.')
        elif not usb['supported']: st.warning('Host USB inventory is unavailable—not an all-clear.')
        else: st.write(f"**{len(usb['new'])}** new USB devices relative to the host baseline")
        note('Login rules and USB context are separate from the Edge-IIoTset traffic models. A remote visitor’s USB devices cannot be read by this server.')
    if ctx:
        st.write('')
        with st.container(border=True, key="card_06"):
            section('HIGHEST-RISK RECORDS', 'Priority investigation')
            record_queue(ctx, top_only=True)
    note('Research assessment, not a production IDS. Random row splits can overstate generalisation. Risk is a fixed heuristic—not a calibrated probability of compromise.')


def threat_explorer(ctx):
    heading('INVESTIGATION WORKSPACE', 'Follow the evidence.', 'Known-class detection performance, persistent triage, and record-level RF explanations.')
    b = ctx['bundle']; values, matrix, report = metrics(ctx['key'], b)
    for col, (name, value) in zip(st.columns(4), values.items()): col.metric(name, f'{value:.3f}')
    st.write('')
    left, right = st.columns(2, gap='large')
    with left, st.container(border=True, key="card_07"):
        section('CONFUSION MATRIX', 'Detection, in context')
        st.dataframe(pd.DataFrame(matrix, index=['Actual Normal','Actual Attack'], columns=['Predicted Normal','Predicted Attack']), width='stretch')
        with st.expander('Classification report'): st.code(report, language='text')
        st.caption('100-tree Random Forest · seed 42. The selected family is absent from this test split.')
    with right, st.container(border=True, key="card_08"):
        section('COMBINED ASSESSMENT', 'Distribution of risk')
        fig = Figure(figsize=(6,2.6), facecolor=PAPER); ax=fig.subplots(); ax.set_facecolor(PAPER)
        ax.hist(scores(ctx['key'],'known',b)['Risk score'], bins=np.arange(0,101,10), color='#72916B', rwidth=.8)
        ax.set_xlabel('Risk score',fontsize=9,color=MUTED); ax.set_ylabel('Records',fontsize=9,color=MUTED)
        ax.tick_params(labelsize=8,colors=MUTED)
        for spine in ax.spines.values(): spine.set_visible(False)
        fig.tight_layout(); st.pyplot(fig, width='stretch')
    with st.container(border=True, key="card_09"):
        section('SELECT A RECORD TO EXPLAIN', 'Investigation queue')
        record_queue(ctx)


def zero_day(ctx):
    heading('HELD-OUT FAMILY EXPERIMENT', 'Beyond familiar threats.', 'A deliberate blind spot in training. An honest look at detection and false positives.')
    b, key = ctx['bundle'], ctx['key']; zero=scores(key,'zero',b); known=scores(key,'known',b)
    normal=known.loc[b['y_test'].eq(0)]
    combined=float(zero['Risk score'].ge(40).mean()); rf=float(zero['RF attack probability'].gt(.5).mean())
    st.markdown(f'<div class="pill">{esc(ctx["holdout"])} · excluded from all fitting</div>', unsafe_allow_html=True)
    c1,c2,c3=st.columns(3); c1.metric('Reserved attack records',len(zero));c2.metric('Combined · risk ≥40',f'{combined:.1%}');c3.metric('Random Forest alone',f'{rf:.1%}')
    st.write('')
    with st.container(border=True, key="card_10"):
        section('DETECTION VS FALSE POSITIVES', 'The trade-off, not just the headline')
        comparison=pd.DataFrame({'Decision rule':['Combined risk ≥40','RF probability >0.50','RF probability ≥0.40 · sensitivity check'],
            'Held-out attacks flagged (%)':[100*combined,100*rf,100*zero['RF attack probability'].ge(.4).mean()],
            'Normal false-positive rate (%)':[100*normal['Risk score'].ge(40).mean(),100*normal['RF attack probability'].gt(.5).mean(),100*normal['RF attack probability'].ge(.4).mean()]})
        st.dataframe(comparison,hide_index=True,width='stretch')
        note('Different thresholds, not a matched-FPR comparison. The reserved set contains only attacks: these are detection rates, not accuracy or precision. Fusion is not guaranteed to outperform RF.')
    with st.container(border=True, key="card_11"):
        section('RESERVED ATTACK RECORDS', 'The unseen investigation queue')
        record_queue(ctx,pool='zero')
    note('A held-out family is a zero-day simulation, not proof of operational protection. With RF probability 0, anomaly evidence alone can produce at most risk 40—not High.')


def fill_manual(ctx, mode):
    b,key=ctx['bundle'],ctx['key']
    if mode=='median':
        row=b['reference'].loc['50%']; description='Training medians · hypothetical combination'
    else:
        pool=b['test_raw'].loc[b['test_raw'].Attack_label.eq(0 if mode=='normal' else 1)]
        if pool.empty: return
        row=pool.sample(1,random_state=42).iloc[0];description=f'Known-test {mode} record #{row.name}'
    clipped=[]
    for f in b['features']:
        value=float(np.clip(row[f], b['reference'][f]['min'], b['reference'][f]['max']))
        if value!=row[f]:clipped.append(f)
        st.session_state[f'manual_{key}_{f}']=value
    st.session_state[f'manual_source_{key}']=description+(f' · {len(clipped)} values clipped to training bounds' if clipped else '')


@st.fragment
def manual_studio(ctx):
    sec.require_auth()
    b,key=ctx['bundle'],ctx['key']
    left,right=st.columns([1.15,1],gap='large')
    with left,st.container(border=True, key="card_12"):
        section('TRAINING-SELECTED FEATURES','Shape the signal')
        a,c,d=st.columns(3)
        a.button('Normal example',on_click=fill_manual,args=(ctx,'normal'),width='stretch')
        c.button('Attack example',on_click=fill_manual,args=(ctx,'attack'),width='stretch')
        d.button('Medians',on_click=fill_manual,args=(ctx,'median'),width='stretch')
        st.caption(st.session_state.get(f'manual_source_{key}','Training medians · hypothetical combination'))
        values={}; columns=st.columns(2)
        for i,f in enumerate(b['features']):
            ref=b['reference'][f];low,high=float(ref['min']),float(ref['max'])
            with columns[i%2]:
                widget_key=f'manual_{key}_{f}'
                default=None if widget_key in st.session_state else float(ref['50%'])
                values[f]=st.slider(f,min_value=low,max_value=high,value=default,step=(high-low)/1000,format='%.6g',key=widget_key)
        note('The assessment updates when a slider change is committed. Values stay within training bounds. Arbitrary combinations may not describe valid network traffic.')
    with right,st.container(border=True, key="card_13"):
        section('LIVE MODEL RESPONSE','A responsive assessment')
        local_explanation(b,key,pd.Series(values))


def data_ledger(ctx):
    heading('PROVENANCE & PREPARATION','A clear chain of evidence.','The real file, the rows retained, and the choices made before a model sees a feature.')
    b=ctx['bundle']
    for col,name,value in zip(st.columns(4),['CSV rows','Stratified sample','Clean records','Selected features'],[ctx['shape'][0],len(ctx['sample']),len(ctx['clean']),len(b['features'])]):
        col.metric(name,f'{value:,}')
    st.write('')
    left,right=st.columns(2,gap='large')
    with left,st.container(border=True, key="card_14"):
        section('MODULE 1 · DATA COLLECTION','Class composition')
        counts=pd.concat([ctx['counts'],ctx['sample'].Attack_type.value_counts().rename('Sample'),ctx['clean'].Attack_type.value_counts().rename('Clean')],axis=1).fillna(0).astype(int)
        st.dataframe(counts,width='stretch')
    with right,st.container(border=True, key="card_15"):
        section('MODULE 2 · PRE-PROCESSING','Preparation ledger')
        st.dataframe(pd.DataFrame({'Stage':list(ctx['cleaning']),'Count':list(ctx['cleaning'].values())}),hide_index=True,width='stretch')
        with st.expander('Removed columns'):
            st.write('Identifiers / metadata:',ctx['removed_ids']);st.write('Other nonnumeric fields:',ctx['removed_text']);st.write('Training constants:',b['constants'])
        st.caption(f'{ctx["missing_types"]:,} missing/blank attack types removed before sampling. Labels and row IDs are never predictors.')
    left,right=st.columns(2,gap='large')
    with left,st.container(border=True, key="card_16"):
        section('TRAINING-ONLY SELECTION','The selected features')
        bar_plot(b['importances'].index,b['importances'].values,'Preliminary RF importance—not SHAP')
    with right,st.container(border=True, key="card_17"):
        section('REPRODUCIBILITY','Experimental contract')
        st.write(f'**Train / known test:** {b["train_count"]:,} / {b["test_count"]:,}')
        st.write(f'**Normal-only IF training:** {b["normal_train_count"]:,}')
        st.write(f'**Held-out attack:** {ctx["holdout"]}')
        st.write('**Risk:** 100 × (0.6 RF probability + 0.4 normalized anomaly)')
        st.write('**Anomaly bounds:** known-training 1st and 99th percentiles, clipped to [0,1].')
        st.caption('File SHA-256: '+ctx['fingerprint'])
        note('Random splitting and deduplication do not guarantee independence across time, captures or devices.')
    with st.expander('Source preview · first ten sampled records'):
        st.dataframe(ctx['sample'].head(10),width='stretch')
        st.caption('Source fields are shown for provenance. Excluded identifiers and text are not used by the models.')


@st.fragment(run_every=5)
def live_security():
    sec.require_auth()
    data=sec.events(); usb=sec.usb_status()
    columns=st.columns(4)
    for col,label,value in zip(columns,['Failed sign-ins · 10 min','Network sources · 10 min','Peak login risk · 10 min','New host USB devices'],[data['failed'],data['sources'],f'{data["peak"]:.1f}',len(usb['new']) if usb['enabled'] and usb['supported'] else '—']): col.metric(label,value)
    st.write('')
    left,right=st.columns([1.3,1],gap='large')
    with left,st.container(border=True, key="card_18"):
        section('LIVE · REFRESHES EVERY FIVE SECONDS','Access journal')
        if data['events']:
            rows=pd.DataFrame([{'ID':e['id'],'Time':time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(e['ts'])),'Device':e['device'],'Source':e['ip'],'Event':e['event'],'Risk':round(e['risk'],1),'Level':e['level']} for e in data['events'][:100]])
            st.dataframe(rows,hide_index=True,width='stretch',height=350)
        else: st.info('No sign-in attempts have been recorded.')
        st.caption('Latest 100 events. Passwords are never written to the journal. Distinct source addresses are not unique people.')
    with right,st.container(border=True, key="card_19"):
        section('LOGIN RULE EXPLANATION','Every point, accounted for')
        if data['events']:
            ids=[e['id'] for e in data['events']]
            chosen=st.selectbox('Login event',ids,key='login_event',format_func=lambda i:f'Event #{i}')
            event=next(e for e in data['events'] if e['id']==chosen)
            st.markdown(f'<div class="score">{event["risk"]:.1f}<small> / 100</small></div>',unsafe_allow_html=True)
            {'High':st.error,'Medium':st.warning,'Low':st.success}[event['level']](event['level']+' login-context risk')
            bar_plot(list(event['parts']),list(event['parts'].values()),'Rule points—not SHAP contributions')
        note('Failures (50), speed (20), unrecognised session (20), new host USB (10). This score is independent of the traffic models. USB context is not proof that the login came from that device.')
    with st.container(border=True, key="card_20"):
        section('THE PHYSICAL EDGE','Connected, with consent.')
        if not usb['enabled']:
            st.info('Host USB monitoring is off. This endpoint is not being monitored.')
        elif not usb['supported']:
            st.warning('USB inventory unavailable: '+usb['error'])
        else:
            st.caption('Current inventory checked at '+time.strftime('%H:%M:%S',time.localtime(usb['seen']))+' on the Streamlit server host.')
            if usb['new']: st.warning(f'{len(usb["new"])} new device(s) relative to the host baseline.');st.write(usb['new'])
            else: st.success('No new devices relative to this host baseline. This is not a security certification.')
            with st.expander(f'Reported inventory ({len(usb["devices"])})'):st.write(usb['devices'] or ['No USB devices exposed by the operating system.'])
        note('When Streamlit is hosted in the cloud, this is the cloud server—not your laptop. A remote phone opening the app cannot expose its USB inventory. Run Streamlit on the computer you want to monitor.')


def security_page():
    heading('ACCESS & ENDPOINT AWARENESS','Know who. Notice what.','Transparent login rules and explicitly enabled host USB monitoring. No dataset required.')
    with st.expander('Host USB settings · explicit opt-in',expanded=not sec.setting('usb_enabled',False)):
        enabled=st.toggle('Monitor USB on the machine running Streamlit',value=sec.setting('usb_enabled',False),key='usb_enabled_toggle',help='This does not inspect the browser device. On cloud hosting it inspects the cloud server.')
        if enabled!=sec.setting('usb_enabled',False):
            sec.save_setting('usb_enabled',enabled);sec.current_usb.clear()
        st.caption('The first successful inventory establishes a baseline. Existing devices initially count as normal. Local Windows, macOS and Linux enumeration is supported when OS permissions allow it.')
        confirm=st.checkbox('I have reviewed this host inventory and want to trust it as the baseline.',key='usb_confirm')
        if st.button('Trust current host inventory',disabled=not(enabled and confirm)):
            current=sec.current_usb()
            if current['supported']:
                sec.save_setting('usb_baseline',current['devices']);st.success('Host baseline updated.')
            else:st.error('Inventory unavailable; baseline was not changed.')
    live_security()
    note('Five failed attempts from a source in ten minutes trigger a two-minute lockout from its last failure. Streamlit recognition lasts only within this session; it is not the Vercel edition’s signed browser cookie. Use HTTPS and appropriate access controls for shared hosting.')


def main():
    st.set_page_config(page_title='Verdant · IIoT Risk Observatory',page_icon='🌿',layout='wide')
    inject_style()
    try:sec.init()
    except (OSError, sec.sqlite3.Error) as exc:st.error(f'STATE_DIR must be writable: {exc}');st.stop()
    if not login():st.stop()
    with st.sidebar:
        st.markdown('<div class="brand">verdant</div><div class="brand-note">IIoT RISK OBSERVATORY</div>',unsafe_allow_html=True)
        st.caption('RESEARCH WORKSPACE')
        page=st.radio('Observatory',['Overview','Threat explorer','Zero-day lab','Scenario studio','Security monitor','Data & methodology'],label_visibility='collapsed',key='navigation')
        st.divider()
        recent=sec.events()
        if recent['flagged']:st.caption(f'{recent["flagged"]} Medium / High login events in the last 10 minutes')
        st.caption('Local Python workspace · persistent audit & triage')
        if st.button('Sign out',width='stretch'):
            sec.logout();st.rerun()
    if page=='Security monitor':
        security_page();return
    path=Path(os.getenv('DATA_PATH',str(ROOT/'data.csv')))
    if not path.is_file():
        st.info('Your workspace is ready for its data. Place the real Edge-IIoTset ML CSV beside app.py as data.csv, or set DATA_PATH. Security Monitor works independently.')
        if page=='Overview':overview()
        else:heading('REAL DATA REQUIRED','Evidence comes first.','This view will populate once your real CSV is available. No synthetic data is substituted.')
        return
    try:
        signature=(str(path.resolve()),path.stat().st_mtime_ns,path.stat().st_size)
        sample,shape,counts,missing=load_data(str(path),signature)
        clean,cleaning,removed_ids,removed_text=clean_data(sample)
        attacks=sorted(clean.loc[clean.Attack_type.ne('Normal'),'Attack_type'].unique().tolist())
        if len(attacks)<2 or 'Normal' not in clean.Attack_type.unique():raise ValueError('Normal and at least two attack families must remain after cleaning.')
        fingerprint=file_identity(str(path),signature)
        with st.sidebar:
            st.divider();st.caption('ACTIVE HOLDOUT')
            holdout=st.selectbox('Attack family excluded from training',attacks,index=attacks.index('SQL_injection') if 'SQL_injection' in attacks else 0,key='holdout')
            st.caption(f'{len(clean):,} clean records · seed 42\n\n75 / 25 known-class split · up to 12 features')
            st.caption('Data SHA-256: '+fingerprint[:12])
        b=train_models(clean,holdout)
        key=f'{fingerprint}:{holdout}'
        ctx={'bundle':b,'key':key,'holdout':holdout,'fingerprint':fingerprint,'sample':sample,'clean':clean,
             'shape':shape,'counts':counts,'missing_types':missing,'cleaning':cleaning,'removed_ids':removed_ids,'removed_text':removed_text}
        if len(b['features'])<12:st.warning(f'Only {len(b["features"])} variable numeric features remain; all are used.')
        if b['anomaly_high']-b['anomaly_low']<=1e-12:st.warning('Training anomaly scores have no spread. Their normalized contribution is set to zero.')
        if page=='Overview':overview(ctx)
        elif page=='Threat explorer':threat_explorer(ctx)
        elif page=='Zero-day lab':zero_day(ctx)
        elif page=='Scenario studio':
            heading('MANUAL FEATURE EXPLORATION','A space for what if.','Adjust original-unit features and see the same fitted pipeline respond—without a submit-and-next-page workflow.')
            manual_studio(ctx)
        elif page=='Data & methodology':data_ledger(ctx)
    except (ValueError,OSError,pd.errors.ParserError) as exc:
        st.error(f'This dataset or experiment needs attention: {exc}')
    st.markdown('<div class="workspace-footer"><span>VERDANT / IIoT RISK OBSERVATORY</span><span>Observe carefully. Interpret responsibly.</span></div>',unsafe_allow_html=True)


if __name__=='__main__':
    main()
