import os, warnings
warnings.filterwarnings('ignore')
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retention Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
/* ── Reset & base ── */
[data-testid="collapsedControl"]  { display: none; }
.block-container { padding: 1.2rem 2rem 2rem 2rem; max-width: 1300px; }

/* ── Top nav bar ── */
.topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 30px 0 16px 0; border-bottom: 1px solid rgba(128,128,128,0.18);
    margin-bottom: 18px;
}
.topbar-title { font-size: 20px; font-weight: 600; display: flex; align-items: center; gap: 10px; }
.topbar-sub   { font-size: 12px; color: #888; margin-top: 3px; }
.groq-pill {
    font-size: 11px; font-weight: 600; padding: 4px 12px; border-radius: 20px;
    background: rgba(29,158,117,0.15); color: #1D9E75;
    border: 1px solid rgba(29,158,117,0.35); letter-spacing: 0.03em;
}
.groq-pill-off {
    font-size: 11px; font-weight: 600; padding: 4px 12px; border-radius: 20px;
    background: rgba(186,117,23,0.12); color: #BA7517;
    border: 1px solid rgba(186,117,23,0.3); letter-spacing: 0.03em;
}

/* ── KPI cards ── */
.kpi-row  { display: flex; gap: 12px; margin-bottom: 22px; }
.kpi-card {
    flex: 1; background: var(--secondary-background-color);
    border-radius: 10px; padding: 14px 18px;
    border: 1px solid rgba(128,128,128,0.12);
}
.kpi-label { font-size: 11px; color: #888; text-transform: uppercase;
             letter-spacing: 0.05em; margin-bottom: 5px; }
.kpi-value { font-size: 26px; font-weight: 600; line-height: 1.1; }
.kpi-sub   { font-size: 11px; margin-top: 5px; }
.kpi-red   { color: #D85A30; }
.kpi-green { color: #1D9E75; }
.kpi-amber { color: #EF9F27; }
.kpi-purple{ color: #7F77DD; }

/* ── Filter row ── */
.filter-strip {
    display: flex; gap: 12px; align-items: flex-end;
    background: var(--secondary-background-color);
    border-radius: 10px; padding: 12px 16px;
    border: 1px solid rgba(128,128,128,0.12);
    margin-bottom: 20px;
}

/* ── Section header ── */
.sec-hdr {
    font-size: 13px; font-weight: 600; color: #888;
    text-transform: uppercase; letter-spacing: 0.06em;
    margin: 20px 0 8px 0;
}

/* ── AI panel ── */
.ai-panel {
    background: var(--secondary-background-color);
    border: 1px solid rgba(127,119,221,0.3);
    border-left: 3px solid #7F77DD;
    border-radius: 10px; padding: 16px 20px; margin: 12px 0;
}
.ai-panel-hdr {
    font-size: 11px; font-weight: 700; color: #7F77DD;
    text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 10px;
    display: flex; align-items: center; gap: 6px;
}
.ai-panel p  { font-size: 13.5px; line-height: 1.75; margin: 0 0 6px 0; }
.ai-panel li { font-size: 13.5px; line-height: 1.8; }
.ai-panel ul { margin: 4px 0; padding-left: 18px; }

/* ── Segment badge colors ── */
.seg-premium { color: #1D9E75; font-weight: 600; }
.seg-churn   { color: #D85A30; font-weight: 600; }
.seg-occ     { color: #EF9F27; font-weight: 600; }
.seg-deal    { color: #7F77DD; font-weight: 600; }

/* ── Tab styling ── */
[data-testid="stTabs"] button { font-size: 13px; font-weight: 500; }

/* ── Table clean ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Streamlit metric override ── */
div[data-testid="metric-container"] { padding: 0 !important; background: none !important; border: none !important; }

/* ── Filter expander ── */
[data-testid="stExpander"] {
    border: 1px solid rgba(128,128,128,0.15) !important;
    border-radius: 10px !important;
    margin-bottom: 16px !important;
}
[data-testid="stExpander"] summary {
    font-size: 13px !important;
    font-weight: 500 !important;
    color: var(--text-color) !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
COLORS = {
    'Premium Loyalists'      : '#1D9E75',
    'Churn Risk'             : '#D85A30',
    'Occasional Shoppers'    : '#EF9F27',
    'High-Value Deal Seekers': '#7F77DD',
}
SEG_ORDER   = ['Premium Loyalists','Occasional Shoppers','Churn Risk','High-Value Deal Seekers']
RISK_COLORS = {'High':'#D85A30','Medium':'#EF9F27','Low':'#1D9E75'}

GROQ_KEY = os.getenv("GROQ_API_KEY","")

SYSTEM = """You are a senior retail data scientist. Give bullet-pointed, specific, 
actionable insights based on the numbers provided. No filler phrases. Max 180 words."""

def ask_groq(prompt, max_tokens=420):
    if not GROQ_KEY:
        return "_Add_ `GROQ_API_KEY` _to your_ `.env` _file to enable AI insights._"
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type":"application/json"},
            json={"model":"llama-3.3-70b-versatile",
                  "messages":[{"role":"system","content":SYSTEM},{"role":"user","content":prompt}],
                  "max_tokens":max_tokens,"temperature":0.35},
            timeout=20)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"_Groq error: {e}_"

def ctx(dff):
    s = dff.groupby('segment_name').agg(
        n=('household_key','count'), spend=('monetary','median'),
        freq=('frequency','median'), churn=('churned','mean'),
        prob=('churn_proba','mean'), high_risk=('churn_risk_tier',lambda x:(x=='High').sum()),
        recency=('recency','mean'), discount=('pct_spend_on_discount','mean'),
    ).round(3)
    rev = dff[dff['churn_risk_tier']=='High']['monetary'].sum()
    return (f"{len(dff):,} households | churn {dff['churned'].mean()*100:.1f}% | "
            f"${rev:,.0f} revenue at risk\n\n{s.to_string()}")

# ─── Load ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load(): return pd.read_csv('data/final_scored.csv')
df = load()

# ─── Top nav bar ──────────────────────────────────────────────────────────────
nc1, nc2 = st.columns([5, 1])
with nc1:
    st.markdown("""
    <div class="topbar">
      <div>
        <div class="topbar-title">Customer Retention Dashboard</div>
        <div class="topbar-sub">Dunnhumby Complete Journey · 2,499 households · 2-year window · Random Forest AUC 0.879</div>
      </div>
    </div>""", unsafe_allow_html=True)
with nc2:
    badge = 'groq-pill' if GROQ_KEY else 'groq-pill-off'
    icon  = '● Groq AI active' if GROQ_KEY else '○ Groq AI — add key'
    st.markdown(f"<br><span class='{badge}'>{icon}</span>", unsafe_allow_html=True)

# ─── Defaults ─────────────────────────────────────────────────────────────────
lo, hi = int(df['monetary'].min()), int(df['monetary'].max())

# Initialise session state so filters persist across reruns
if 'seg_f'    not in st.session_state: st.session_state['seg_f']    = SEG_ORDER
if 'risk_f'   not in st.session_state: st.session_state['risk_f']   = ['High','Medium','Low']
if 'spend_lo' not in st.session_state: st.session_state['spend_lo'] = lo
if 'spend_hi' not in st.session_state: st.session_state['spend_hi'] = hi

# ─── Collapsible filter panel ─────────────────────────────────────────────────
is_filtered = (
    st.session_state['seg_f']    != SEG_ORDER or
    st.session_state['risk_f']   != ['High','Medium','Low'] or
    st.session_state['spend_lo'] != lo or
    st.session_state['spend_hi'] != hi
)
filter_label = " Filters — active ✦" if is_filtered else " Filters"

with st.expander(filter_label, expanded=False):
    fa, fb, fc = st.columns([2, 1.5, 2.5])
    with fa:
        seg_f = st.multiselect(
            "Segment", SEG_ORDER,
            default=st.session_state['seg_f'],
            help="Select which customer segments to include",
        )
        st.session_state['seg_f'] = seg_f or SEG_ORDER
    with fb:
        risk_f = st.multiselect(
            "Risk tier", ['High','Medium','Low'],
            default=st.session_state['risk_f'],
            help="High = churn prob > 50%, Medium = 20–50%, Low = < 20%",
        )
        st.session_state['risk_f'] = risk_f or ['High','Medium','Low']
    with fc:
        spend_f = st.slider(
            "Household total spend ($)",
            lo, hi,
            (st.session_state['spend_lo'], st.session_state['spend_hi']),
            step=50,
            help="Filter by how much a household spent over the 2-year window",
        )
        st.session_state['spend_lo'], st.session_state['spend_hi'] = spend_f

    if st.button("↺ Reset all filters"):
        for k, v in [('seg_f', SEG_ORDER), ('risk_f', ['High','Medium','Low']),
                     ('spend_lo', lo), ('spend_hi', hi)]:
            st.session_state[k] = v
        st.rerun()

# ─── Apply filters ────────────────────────────────────────────────────────────
dff = df[
    df['segment_name'].isin(st.session_state['seg_f']) &
    df['churn_risk_tier'].isin(st.session_state['risk_f']) &
    df['monetary'].between(st.session_state['spend_lo'], st.session_state['spend_hi'])
].copy()
dff['priority_score'] = dff['churn_proba'] * np.log1p(dff['monetary'])

rev_at_risk  = dff[dff['churn_risk_tier']=='High']['monetary'].sum()
churn_delta  = (dff['churned'].mean() - df['churned'].mean()) * 100

# ─── KPI row (HTML cards for full control) ───────────────────────────────────
delta_col  = "kpi-red" if churn_delta >= 0 else "kpi-green"
delta_arrow= "▲" if churn_delta >= 0 else "▼"

st.markdown(f"""
<div class="kpi-row">
  <div class="kpi-card">
    <div class="kpi-label">Households</div>
    <div class="kpi-value">{len(dff):,}</div>
    <div class="kpi-sub" style="color:#888">of 2,499 total</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Churn rate</div>
    <div class="kpi-value kpi-red">{dff['churned'].mean()*100:.1f}%</div>
    <div class="kpi-sub {delta_col}">{delta_arrow} {abs(churn_delta):.1f}pp vs overall avg</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">High-risk households</div>
    <div class="kpi-value kpi-amber">{(dff['churn_risk_tier']=='High').sum():,}</div>
    <div class="kpi-sub" style="color:#888">churn probability &gt; 50%</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Revenue at risk</div>
    <div class="kpi-value kpi-red">${rev_at_risk:,.0f}</div>
    <div class="kpi-sub" style="color:#888">from high-risk HHs</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Avg churn score</div>
    <div class="kpi-value kpi-amber">{dff['churn_proba'].mean()*100:.1f}%</div>
    <div class="kpi-sub" style="color:#888">model probability</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Avg spend / household</div>
    <div class="kpi-value kpi-green">${dff['monetary'].mean():,.0f}</div>
    <div class="kpi-sub" style="color:#888">2-year total</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    " Segments",
    "Churn Risk",
    "Action List",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SEGMENTS
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="sec-hdr">Who are your customers?</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        sa = dff.groupby('segment_name').agg(
            households  =('household_key','count'),
            median_spend=('monetary','median'),
            trips       =('frequency','median'),
            churn_pct   =('churned','mean'),
        ).reset_index()
        sa['churn_pct'] = (sa['churn_pct']*100).round(1)
        sa = sa.set_index('segment_name').reindex(SEG_ORDER).reset_index()

        fig = px.bar(sa, x='segment_name', y='households',
                     color='segment_name', color_discrete_map=COLORS,
                     text='households', title='Households per segment',
                     custom_data=['median_spend','churn_pct','trips'])
        fig.update_traces(
            textposition='outside', textfont_size=12,
            hovertemplate='<b>%{x}</b><br>Households: <b>%{y}</b><br>'
                          'Median spend: <b>$%{customdata[0]:,.0f}</b><br>'
                          'Churn rate: <b>%{customdata[1]}%</b><br>'
                          'Median trips: <b>%{customdata[2]}</b><extra></extra>')
        fig.update_layout(showlegend=False, height=320,
                          margin=dict(t=40,b=10,l=0,r=0),
                          xaxis_title='', yaxis_title='Number of households',
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          title_font_size=13)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        metric_opts = {
            'monetary'             : 'Total 2-year spend ($)',
            'frequency'            : 'Shopping trips',
            'avg_basket'           : 'Avg basket size ($)',
            'recency'              : 'Days since last purchase',
            'pct_spend_on_discount': '% spend on discount',
        }
        mc = st.selectbox("Compare metric across segments",
                           list(metric_opts.keys()),
                           format_func=lambda x: metric_opts[x])

        # Median per segment + overall average reference line
        med = (dff.groupby('segment_name')[mc]
               .median().reindex(SEG_ORDER).reset_index())
        med.columns = ['segment_name', 'value']
        overall_avg = dff[mc].median()

        # Format labels nicely
        if mc in ['monetary', 'avg_basket']:
            med['label'] = med['value'].apply(lambda x: f'${x:,.0f}')
            avg_label    = f'Overall median: ${overall_avg:,.0f}'
        elif mc == 'pct_spend_on_discount':
            med['label'] = med['value'].apply(lambda x: f'{x*100:.1f}%')
            avg_label    = f'Overall median: {overall_avg*100:.1f}%'
        else:
            med['label'] = med['value'].apply(lambda x: f'{x:,.0f}')
            avg_label    = f'Overall median: {overall_avg:,.0f}'

        fig2 = px.bar(
            med, x='value', y='segment_name', orientation='h',
            color='segment_name', color_discrete_map=COLORS,
            text='label',
            title=f'Median {metric_opts[mc]} per segment',
        )
        fig2.update_traces(textposition='outside', textfont_size=11)
        fig2.add_vline(x=overall_avg, line_dash='dash', line_color='gray',
                       line_width=1.2, opacity=0.6,
                       annotation_text=avg_label,
                       annotation_position='top right',
                       annotation_font=dict(size=9, color='gray'))
        fig2.update_layout(
            showlegend=False, height=320,
            margin=dict(t=40, b=10, l=0, r=80),
            xaxis_title=metric_opts[mc], yaxis_title='',
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            title_font_size=13,
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="sec-hdr">Segment summary</div>', unsafe_allow_html=True)
    tbl = dff.groupby('segment_name').agg(
        Households    =('household_key','count'),
        Median_Spend  =('monetary','median'),
        Median_Trips  =('frequency','median'),
        Churn_Rate_Pct=('churned','mean'),
        Avg_Recency   =('recency','mean'),
        High_Risk_HHs =('churn_risk_tier', lambda x:(x=='High').sum()),
    ).reindex(SEG_ORDER).reset_index()
    tbl.columns = ['Segment','Households','Median Spend ($)','Median Trips',
                   'Churn Rate','Avg Recency (days)','High-Risk HHs']
    tbl['Churn Rate']       = tbl['Churn Rate'].apply(lambda x: f'{x*100:.1f}%')
    tbl['Median Spend ($)'] = tbl['Median Spend ($)'].apply(lambda x: f'${x:,.0f}')
    tbl['Avg Recency (days)'] = tbl['Avg Recency (days)'].round(1)
    st.dataframe(tbl, use_container_width=True, hide_index=True, height=210)

    st.markdown('<div class="ai-panel"><div class="ai-panel-hdr">🤖 AI Segment Analysis</div>', unsafe_allow_html=True)
    st.caption("Identifies the biggest opportunity, most urgent concern, and 3 specific actions based on the current segment view.")
    if st.button("Run Segment Analysis", use_container_width=True, key="btn_ai1"):
        with st.spinner("Asking Groq (Llama 3)..."):
            st.session_state['res_ai1'] = ask_groq(
                f"{ctx(dff)}\n\n"
                "1. What is the most urgent business concern in this data?\n"
                "2. Which segment has the biggest growth opportunity?\n"
                "3. Give exactly 3 specific, data-backed retention recommendations."
            )
    if 'res_ai1' in st.session_state:
        st.markdown(st.session_state['res_ai1'])
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CHURN RISK
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sec-hdr">Where is churn risk concentrated?</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        # Simple grouped bar: model churn score vs actual churn rate per segment
        bub = dff.groupby('segment_name').agg(
            model_score  =('churn_proba','mean'),
            actual_churn =('churned','mean'),
            households   =('household_key','count'),
        ).reindex(SEG_ORDER).reset_index()
        bub['model_score_pct']  = (bub['model_score']  * 100).round(1)
        bub['actual_churn_pct'] = (bub['actual_churn'] * 100).round(1)

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            name='Avg churn score (model)',
            x=bub['segment_name'], y=bub['model_score_pct'],
            marker_color='#7F77DD', text=bub['model_score_pct'].apply(lambda x: f'{x:.1f}%'),
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Avg model score: <b>%{y:.1f}%</b><extra></extra>',
        ))
        fig3.add_trace(go.Bar(
            name='Actual churn rate',
            x=bub['segment_name'], y=bub['actual_churn_pct'],
            marker_color='#D85A30', text=bub['actual_churn_pct'].apply(lambda x: f'{x:.1f}%'),
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Actual churn: <b>%{y:.1f}%</b><extra></extra>',
        ))
        fig3.update_layout(
            barmode='group', height=340,
            title='Predicted churn score vs actual churn rate by segment',
            xaxis_title='', yaxis_title='Percentage (%)',
            yaxis=dict(ticksuffix='%'),
            legend=dict(orientation='h', y=-0.2),
            margin=dict(t=40, b=10, l=0, r=0),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            title_font_size=13,
        )
        st.plotly_chart(fig3, use_container_width=True)

    with c2:
        # % of households in each risk tier per segment — immediately readable by anyone
        tier_counts = (dff.groupby(['segment_name','churn_risk_tier'])
                       .size().reset_index(name='count'))
        tier_totals = tier_counts.groupby('segment_name')['count'].transform('sum')
        tier_counts['pct'] = (tier_counts['count'] / tier_totals * 100).round(1)
        tier_counts = tier_counts[tier_counts['segment_name'].isin(SEG_ORDER)]

        fig4 = go.Figure()
        for tier, color in [('High','#D85A30'),('Medium','#EF9F27'),('Low','#1D9E75')]:
            td = tier_counts[tier_counts['churn_risk_tier']==tier].set_index('segment_name').reindex(SEG_ORDER).reset_index()
            fig4.add_trace(go.Bar(
                name=tier,
                x=td['segment_name'],
                y=td['pct'].fillna(0),
                marker_color=color,
                text=td['pct'].fillna(0).apply(lambda x: f'{x:.0f}%' if x >= 6 else ''),
                textposition='inside',
                textfont=dict(color='white', size=11),
                hovertemplate='<b>%{x}</b><br>' + tier + ' risk: <b>%{y:.1f}%</b> of segment<extra></extra>',
            ))
        fig4.update_layout(
            barmode='stack', height=340,
            title='What % of each segment is at risk?',
            xaxis_title='', yaxis_title='% of households in segment',
            yaxis=dict(ticksuffix='%', range=[0, 105]),
            legend=dict(title='Risk tier', orientation='h', y=-0.2),
            margin=dict(t=40, b=10, l=0, r=0),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            title_font_size=13,
        )
        st.plotly_chart(fig4, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        rp = pd.crosstab(dff['segment_name'], dff['churn_risk_tier'], normalize='index')*100
        rp = rp.reindex(SEG_ORDER, fill_value=0)
        fig5 = go.Figure()
        for tier, color in [('High','#D85A30'),('Medium','#EF9F27'),('Low','#1D9E75')]:
            if tier in rp.columns:
                v = rp[tier].values
                fig5.add_trace(go.Bar(
                    name=tier, x=rp.index, y=v, marker_color=color,
                    text=[f'{x:.0f}%' if x>6 else '' for x in v],
                    textposition='inside', textfont=dict(color='white'),
                    hovertemplate='<b>%{x}</b><br>'+tier+': %{y:.1f}%<extra></extra>'))
        fig5.update_layout(barmode='stack', height=300,
                           title='Risk tier breakdown per segment',
                           showlegend=True,
                           legend=dict(title='Risk tier', orientation='h', y=-0.25),
                           margin=dict(t=40,b=10,l=0,r=0),
                           xaxis_title='', yaxis_title='% of segment',
                           plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                           title_font_size=13)
        st.plotly_chart(fig5, use_container_width=True)

    with c4:
        rev = (dff[dff['churn_risk_tier']=='High']
               .groupby('segment_name')['monetary']
               .agg(revenue_at_risk='sum', hh_count='count')
               .reset_index().sort_values('revenue_at_risk'))
        if len(rev):
            fig6 = px.bar(rev, x='revenue_at_risk', y='segment_name', orientation='h',
                          color='segment_name', color_discrete_map=COLORS,
                          text=rev['revenue_at_risk'].apply(lambda x: f'${x:,.0f}'),
                          custom_data=['hh_count'],
                          title='Revenue at risk — high-risk households only')
            fig6.update_traces(textposition='outside',
                hovertemplate='<b>%{y}</b><br>Revenue at risk: <b>$%{x:,.0f}</b><br>HHs: <b>%{customdata[0]}</b><extra></extra>')
            fig6.update_layout(height=300, showlegend=False,
                               xaxis_title='Total spend ($)', yaxis_title='',
                               margin=dict(t=40,b=10,l=0,r=10),
                               plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                               title_font_size=13)
            st.plotly_chart(fig6, use_container_width=True)
        else:
            st.info("No high-risk households in current filter.")

    st.markdown('<div class="ai-panel"><div class="ai-panel-hdr"> AI Churn Risk Deep-Dive</div>', unsafe_allow_html=True)
    st.caption("Explains behavioural patterns driving churn and recommends the most cost-effective action.")
    if st.button("Run Churn Analysis", use_container_width=True, key="btn_ai2"):
        with st.spinner("Asking Groq (Llama 3)..."):
            hr = dff[dff['churn_risk_tier']=='High']
            if len(hr):
                hrs = hr.groupby('segment_name').agg(
                    n=('household_key','count'),
                    recency=('recency','mean'), spend=('monetary','mean'),
                    discount=('pct_spend_on_discount','mean'),
                    coupon=('coupon_redemption_rate','mean'),
                    ret=('return_rate','mean'),
                ).round(3)
                st.session_state['res_ai2'] = ask_groq(
                    f"High-risk households ({len(hr)}, ${hr['monetary'].sum():,.0f} at risk):\n\n{hrs.to_string()}\n\n"
                    "What behavioural patterns explain churn in each segment? "
                    "What is the single most cost-effective intervention for the largest at-risk group? "
                    "Are there any counter-intuitive findings?",
                    450)
            else:
                st.session_state['res_ai2'] = "No high-risk households in current filter."
    if 'res_ai2' in st.session_state:
        st.markdown(st.session_state['res_ai2'])
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ACTION LIST & AI Q&A
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec-hdr">Who should you contact first?</div>', unsafe_allow_html=True)

    st.info(
        "**Priority score** = churn probability × log(total spend).  "
        "This ranks households by **both** how likely they are to churn **and** how valuable they are — "
        "so you don't waste budget on low-value churners.",
        icon="ℹ️"
    )

    ac1, ac2, ac3 = st.columns([2, 1.5, 1])
    with ac1:
        tier_f = st.multiselect("Show risk tier", ['High','Medium','Low'],
                                 default=['High','Medium'], key='act_tier')
    with ac2:
        seg_f2 = st.multiselect("Show segment", SEG_ORDER, default=SEG_ORDER, key='act_seg')
    with ac3:
        n_show = st.select_slider("Rows to show", [10,25,50,100,200], value=25)

    act = (dff[
        dff['churn_risk_tier'].isin(tier_f or ['High','Medium']) &
        dff['segment_name'].isin(seg_f2 or SEG_ORDER)
    ][['household_key','segment_name','monetary','frequency','recency',
       'churn_proba','churn_risk_tier','priority_score']]
    .sort_values('priority_score', ascending=False)
    .reset_index(drop=True))
    act.index += 1

    act.columns = ['HH Key','Segment','Total Spend ($)','Trips',
                   'Days Since Purchase','Churn Probability','Risk Tier','Priority Score']
    act['Total Spend ($)']   = act['Total Spend ($)'].round(0).astype(int)
    act['Churn Probability'] = act['Churn Probability'].apply(lambda x: f'{x:.1%}')
    act['Priority Score']    = act['Priority Score'].round(3)

    st.dataframe(act.head(n_show), use_container_width=True, height=430)

    st.caption(f"Showing {min(n_show, len(act))} of {len(act):,} households matching filter.")

    dl1, _ = st.columns([1,4])
    with dl1:
        st.download_button("⬇ Download full list (.csv)",
                           data=act.to_csv().encode('utf-8'),
                           file_name='retention_action_list.csv',
                           mime='text/csv', use_container_width=True)

    st.markdown('<div class="ai-panel"><div class="ai-panel-hdr">🤖 Free-text Q&A</div>', unsafe_allow_html=True)
    st.caption("Ask anything about the data — strategy, campaigns, anomalies, trade-offs.")

    examples = [
        "Which segment should I prioritise for a limited budget win-back campaign?",
        "Should I use discounts or loyalty points to retain Churn Risk customers?",
        "What does a 15.7% churn rate in the Churn Risk segment mean for annual revenue?",
        "How should I sequence my retention actions over the next 30 days?",
    ]
    chosen = st.selectbox("Or pick an example question:", [""] + examples,
                           label_visibility='collapsed', key='eg')
    q = st.text_input("Your question:", value=chosen if chosen else "",
                       placeholder="e.g. How should I design a win-back campaign?",
                       label_visibility='collapsed')
    if st.button("Ask Groq →", key="btn_ai3", use_container_width=False) and q:
        with st.spinner("Thinking..."):
            st.session_state['res_ai3'] = ask_groq(f"{ctx(dff)}\n\nQuestion: {q}")
    if 'res_ai3' in st.session_state:
        st.markdown(st.session_state['res_ai3'])
    st.markdown('</div>', unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:#666;font-size:11px'>"
    "Dunnhumby Complete Journey · Random Forest ROC-AUC 0.879 · "
    "Churn = no purchase in final 90 days · AI powered by Groq Llama 3 · "
    "Priority = churn_proba × log(spend)"
    "</div>", unsafe_allow_html=True)