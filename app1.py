import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# KONFIQURASIYA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Data Analitik Portfolio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# QLOBAL STİL
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

:root {
    --navy:    #0A1628;
    --navy2:   #112240;
    --steel:   #1E3A5F;
    --accent:  #2E86AB;
    --accent2: #4BA3C3;
    --silver:  #8B9BB4;
    --light:   #C8D6E8;
    --white:   #F0F4F8;
}

/* Genel arka plan */
.stApp { background-color: #0D1B2A; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A1628 0%, #112240 100%);
    border-right: 1px solid #1E3A5F;
}
section[data-testid="stSidebar"] * { color: #C8D6E8 !important; }
section[data-testid="stSidebar"] .stRadio label { font-size: 0.95rem; padding: 4px 0; }

/* Başlıqlar */
h1, h2, h3, h4 { color: #F0F4F8 !important; font-weight: 600; letter-spacing: -0.3px; }
p, li, span, label, div { color: #C8D6E8; }

/* Metrik kartlar */
[data-testid="metric-container"] {
    background: #112240;
    border: 1px solid #1E3A5F;
    border-radius: 8px;
    padding: 16px 20px;
    transition: border-color 0.2s;
}
[data-testid="metric-container"]:hover { border-color: #2E86AB; }
[data-testid="stMetricValue"] { color: #4BA3C3 !important; font-family: 'IBM Plex Mono', monospace; font-size: 1.8rem !important; }
[data-testid="stMetricLabel"] { color: #8B9BB4 !important; font-size: 0.8rem !important; text-transform: uppercase; letter-spacing: 0.8px; }

/* Dataframe */
.stDataFrame { border-radius: 8px; overflow: hidden; border: 1px solid #1E3A5F; }
.stDataFrame thead tr th { background: #112240 !important; color: #4BA3C3 !important; font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; }

/* Düymələr */
.stButton > button {
    background: #2E86AB; color: #fff; border: none;
    border-radius: 6px; font-weight: 500; font-size: 0.9rem;
    padding: 10px 24px; transition: background 0.2s, transform 0.1s;
}
.stButton > button:hover { background: #4BA3C3; transform: translateY(-1px); }

/* Separator */
.section-title {
    border-left: 3px solid #2E86AB;
    padding-left: 12px;
    margin: 24px 0 16px 0;
    color: #F0F4F8;
    font-size: 1.1rem;
    font-weight: 600;
    letter-spacing: 0.2px;
}

/* KPI badge */
.kpi-badge {
    background: #112240;
    border: 1px solid #1E3A5F;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
}
.kpi-val { font-family: 'IBM Plex Mono', monospace; font-size: 2rem; color: #4BA3C3; font-weight: 600; }
.kpi-lbl { font-size: 0.75rem; color: #8B9BB4; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }

/* Alert / info bokslar */
.stAlert { border-radius: 8px; }

/* Tabs */
button[data-baseweb="tab"] { color: #8B9BB4 !important; font-size: 0.9rem; }
button[data-baseweb="tab"][aria-selected="true"] { color: #4BA3C3 !important; border-bottom: 2px solid #2E86AB; }

/* Expander */
details { border: 1px solid #1E3A5F !important; border-radius: 8px !important; background: #112240; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PLOTLY TEMASI
# ─────────────────────────────────────────────
PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="#112240",
        plot_bgcolor="#0D1B2A",
        font=dict(family="IBM Plex Sans", color="#C8D6E8", size=12),
        title_font=dict(family="IBM Plex Sans", color="#F0F4F8", size=15),
        xaxis=dict(gridcolor="#1E3A5F", linecolor="#1E3A5F", tickcolor="#8B9BB4", zerolinecolor="#1E3A5F"),
        yaxis=dict(gridcolor="#1E3A5F", linecolor="#1E3A5F", tickcolor="#8B9BB4", zerolinecolor="#1E3A5F"),
        colorway=["#2E86AB", "#4BA3C3", "#A8DADC", "#457B9D", "#1D3557", "#8ECAE6", "#219EBC", "#023047"],
        legend=dict(bgcolor="#112240", bordercolor="#1E3A5F", borderwidth=1, font=dict(color="#C8D6E8")),
        margin=dict(l=40, r=20, t=50, b=40),
    )
)

def apply_theme(fig):
    fig.update_layout(**PLOTLY_TEMPLATE["layout"])
    return fig

# ─────────────────────────────────────────────
# VERİ GENERASİYASI
# ─────────────────────────────────────────────
@st.cache_data
def generate_data():
    np.random.seed(42)
    N = 2500

    categories = ["Elektronika", "Geyim", "Qida & İçki", "Ev & Bağ", "İdman", "Kitab & Kırtasiye", "Gözəllik", "Oyuncaq"]
    cat_weights = [0.20, 0.18, 0.15, 0.12, 0.12, 0.08, 0.10, 0.05]

    products_by_cat = {
        "Elektronika":       ["Noutbuk", "Smartfon", "Planşet", "Qulaqlıq", "Smart Saat", "Kamera"],
        "Geyim":             ["Kişi Köynəyi", "Qadın Elbisəsi", "Cins Şalvar", "İdman Geyimi", "Palto", "Ayaqqabı"],
        "Qida & İçki":       ["Üzvi Çay", "Premium Qəhvə", "Zeytun Yağı", "Bal", "Quru Meyvə", "Şokolad"],
        "Ev & Bağ":          ["Divar Saatı", "Çiçək Qabı", "Yataq Örtüyü", "Pərdə", "LED İşıq", "Xalça"],
        "İdman":             ["Yoga Matı", "Dumbell Seti", "Futbol Topu", "Velosiped", "Üzgüçülük Gözlüyü", "Çadır"],
        "Kitab & Kırtasiye": ["İş Dəftəri", "Bədii Roman", "Professional Kitab", "Qrafik Planşet", "Qələm Seti", "Taxt"],
        "Gözəllik":          ["Parfüm", "Üz Kremi", "Saç Maskası", "Ruj", "Göz Kölgəsi", "Tonal Krem"],
        "Oyuncaq":           ["Lego Seti", "Oyun Konsolu", "Kukla", "RC Maşın", "Bulmaca", "Kitab Oyunu"],
    }

    price_range = {
        "Elektronika":       (250, 2800), "Geyim":         (20, 250),
        "Qida & İçki":       (5, 80),    "Ev & Bağ":      (15, 400),
        "İdman":             (30, 600),  "Kitab & Kırtasiye": (5, 120),
        "Gözəllik":          (15, 300),  "Oyuncaq":       (10, 250),
    }

    regions = ["Bakı", "Sumqayıt", "Gəncə", "Lənkəran", "Şəki", "Naxçıvan"]
    region_weights = [0.45, 0.20, 0.15, 0.08, 0.07, 0.05]

    channels = ["Onlayn", "Mağaza", "Mobil Tətbiq"]

    dates = pd.date_range("2022-01-01", "2024-12-31", freq="D")
    date_idx = np.random.choice(range(len(dates)), N)
    order_dates = dates[date_idx]

    categories_col = np.random.choice(categories, N, p=cat_weights)
    products_col = [np.random.choice(products_by_cat[c]) for c in categories_col]
    prices_col = np.array([round(np.random.uniform(*price_range[c]), 2) for c in categories_col])
    qty_col = np.random.randint(1, 6, N)
    discount_col = np.random.choice([0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30], N,
                                    p=[0.40, 0.15, 0.15, 0.12, 0.10, 0.05, 0.03])

    revenue_col = np.round(prices_col * qty_col * (1 - discount_col), 2)
    customer_ids = [f"CX{str(np.random.randint(1, 600)).zfill(5)}" for _ in range(N)]

    df = pd.DataFrame({
        "Sifariş_ID":    [f"ORD-{str(i+1).zfill(6)}" for i in range(N)],
        "Müştəri_ID":    customer_ids,
        "Tarix":         order_dates,
        "Kateqoriya":    categories_col,
        "Məhsul":        products_col,
        "Qiymət":        prices_col,
        "Miqdar":        qty_col,
        "Endirim":       discount_col,
        "Gəlir":         revenue_col,
        "Region":        np.random.choice(regions, N, p=region_weights),
        "Kanal":         np.random.choice(channels, N),
    })

    # ── Qəsdən "çirklilik" əlavə et ──
    # 1. Tipinə zidd sətir dəyərləri (Qiymət sütununda string)
    bad_idx = np.random.choice(df.index, 30, replace=False)
    df.loc[bad_idx, "Qiymət"] = ["N/A", "xəta", "—", "bilinmir", "??"] * 6

    # 2. NaN-lər
    for col, pct in [("Region", 0.04), ("Kanal", 0.03), ("Endirim", 0.05), ("Miqdar", 0.03)]:
        nan_idx = np.random.choice(df.index, int(N * pct), replace=False)
        df.loc[nan_idx, col] = np.nan

    # 3. Dublikatlar
    dup_rows = df.sample(60, random_state=1)
    df = pd.concat([df, dup_rows], ignore_index=True)

    return df.reset_index(drop=True)


@st.cache_data
def clean_data(df_raw):
    df = df_raw.copy()

    # Qiymət sütununu numerikə çevir, xətalıları NaN et
    df["Qiymət"] = pd.to_numeric(df["Qiymət"], errors="coerce")

    # Dublikatları sil
    df.drop_duplicates(inplace=True)

    # NaN-ları doldur / sil
    df["Qiymət"]  = df["Qiymət"].fillna(df["Qiymət"].median())
    df["Miqdar"]  = df["Miqdar"].fillna(df["Miqdar"].median()).astype(int)
    df["Endirim"] = df["Endirim"].fillna(df["Endirim"].median())
    df["Region"]  = df["Region"].fillna(df["Region"].mode()[0])
    df["Kanal"]   = df["Kanal"].fillna(df["Kanal"].mode()[0])

    # Gəliri yenidən hesabla
    df["Gəlir"] = (df["Qiymət"] * df["Miqdar"] * (1 - df["Endirim"])).round(2)

    # Tarix tipini düzəlt
    df["Tarix"] = pd.to_datetime(df["Tarix"])
    df["İl"]    = df["Tarix"].dt.year
    df["Ay"]    = df["Tarix"].dt.month
    df["Ay_Adı"] = df["Tarix"].dt.strftime("%b")
    df["Həftə_Günü"] = df["Tarix"].dt.day_name()

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# YARDIMÇI FUNKSIYALAR
# ─────────────────────────────────────────────
def fmt_az(n, prefix=""):
    if n >= 1_000_000:
        return f"{prefix}{n/1_000_000:.2f}M ₼"
    if n >= 1_000:
        return f"{prefix}{n/1_000:.1f}K ₼"
    return f"{prefix}{n:,.2f} ₼"


# ─────────────────────────────────────────────
# VERİ YÜKLƏ
# ─────────────────────────────────────────────
df_raw = generate_data()
df     = clean_data(df_raw)

# ─────────────────────────────────────────────
# SİDEBAR NAVİGASİYA
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Portfolio")
    st.markdown("---")

    pages = [
        "🏠  Ana Səhifə",
        "🧹  Data Təmizləmə",
        "📊  EDA — Kəşf Analizi",
        "📈  Satış Analizi",
        "🔍  Məhsul Analizi",
        "🤖  ML Modeli",
    ]
    page = st.radio("Naviqasiya", pages, label_visibility="collapsed")

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem; color:#8B9BB4; line-height:1.6;">
    <b style="color:#4BA3C3;">Dataset:</b><br>
    E-Ticarət Simulyasiyası<br>
    2022–2024 · 2,500+ sətir<br><br>
    <b style="color:#4BA3C3;">Alətlər:</b><br>
    Python · Pandas · NumPy<br>
    Plotly · Scikit-learn · Streamlit
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════
# 1. ANA SƏHİFƏ
# ═════════════════════════════════════════════
if page == pages[0]:
    st.markdown("""
    <div style="padding: 40px 0 20px 0;">
        <p style="color:#4BA3C3; font-size:0.85rem; letter-spacing:2px; text-transform:uppercase; margin:0;">Data Analitik · Portfolio</p>
        <h1 style="font-size:2.8rem; font-weight:700; color:#F0F4F8; margin:8px 0 4px 0; line-height:1.1;">
            Nicat Əliyev
        </h1>
        <p style="color:#8B9BB4; font-size:1.1rem; margin:0;">Junior Data Analyst · E-Ticarət Analitikası üzrə ixtisaslaşma</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown('<div class="section-title">Haqqımda</div>', unsafe_allow_html=True)
        st.markdown("""
        <p style="line-height:1.8; color:#C8D6E8;">
        Python ilə Data Analitikası kursunu uğurla başa vurduqdan sonra, real dünya
        e-ticarət datasından mənalı biznes dəyər çıxarma bacarığımı inkişaf etdirdim.
        Strukturlaşdırılmamış, çirkli verini — saf strategiyaya çevirməyi bacaran bir
        analitik yanaşma ilə fərqlənirəm.
        </p>
        <p style="line-height:1.8; color:#C8D6E8;">
        Bu portfolio layihəsi, tam bir analitik iş axını — veri təmizləməsindən,
        kəşf analizinə, satış/məhsul analitikasına və son olaraq maşın öyrənməsi
        modelinə qədər — real biznes konteksini nümayiş etdirir.
        </p>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Texniki Bacarıqlar</div>', unsafe_allow_html=True)
        skills = {
            "Python (Pandas, NumPy)": 88,
            "Veri Vizualizasiyası (Plotly)": 82,
            "İstatistik Analiz & EDA": 80,
            "Maşın Öyrənməsi (Scikit-learn)": 68,
            "SQL & Veritabanı": 72,
            "Streamlit / Dashboarding": 85,
        }
        for skill, pct in skills.items():
            st.markdown(f"""
            <div style="margin-bottom:12px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                    <span style="font-size:0.85rem; color:#C8D6E8;">{skill}</span>
                    <span style="font-family:'IBM Plex Mono',monospace; font-size:0.8rem; color:#4BA3C3;">{pct}%</span>
                </div>
                <div style="background:#1E3A5F; border-radius:4px; height:6px;">
                    <div style="background:linear-gradient(90deg,#2E86AB,#4BA3C3); width:{pct}%; height:6px; border-radius:4px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">Layihənin Məqsədi</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#112240; border:1px solid #1E3A5F; border-radius:8px; padding:20px;">
        <p style="color:#C8D6E8; line-height:1.7; font-size:0.9rem;">
        Bu portfolio, bir e-ticarət şirkətinin 2022–2024-cü illərə aid satış datasını
        analiz edərək aşağıdakı biznes suallarına cavab tapmağa yönəlib:
        </p>
        <ul style="color:#C8D6E8; line-height:2; font-size:0.9rem; padding-left:18px;">
            <li>Ən gəlirli kateqoriya və məhsullar hansılardır?</li>
            <li>Satış trendləri mövsümilik göstərirmi?</li>
            <li>Region bazında performans fərqləri nədir?</li>
            <li>Müştəri seqmentasiyası necə aparıla bilər?</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Dataset Baxışı</div>', unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m1.metric("Sətir sayı", f"{len(df):,}")
        m2.metric("Sütun sayı", f"{df.shape[1]}")
        m1.metric("Müştəri sayı", f"{df['Müştəri_ID'].nunique():,}")
        m2.metric("Kateqoriya sayı", f"{df['Kateqoriya'].nunique()}")

        st.markdown('<div class="section-title">İş Axını</div>', unsafe_allow_html=True)
        steps = ["① Veri Toplama & Simulyasiya", "② Data Təmizləmə", "③ EDA", "④ Biznes Analizi", "⑤ ML Modeli"]
        for i, s in enumerate(steps):
            color = "#2E86AB" if i < 2 else "#4BA3C3" if i < 4 else "#A8DADC"
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:10px; margin:8px 0;">
                <div style="width:8px; height:8px; border-radius:50%; background:{color};"></div>
                <span style="font-size:0.87rem; color:#C8D6E8;">{s}</span>
            </div>
            """, unsafe_allow_html=True)


# ═════════════════════════════════════════════
# 2. DATA TƏMİZLƏMƏ
# ═════════════════════════════════════════════
elif page == pages[1]:
    st.markdown("# 🧹 Data Təmizləmə")
    st.markdown('<p style="color:#8B9BB4;">Ham verinin yoxlanması, çirkliliklərin aşkarlanması və ardıcıl təmizləmə prosesi.</p>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Ham Veri Baxışı", "Problemlərin Analizi", "Təmizlənmiş Veri"])

    with tab1:
        st.markdown('<div class="section-title">Ham Dataset (ilk 20 sətir)</div>', unsafe_allow_html=True)
        st.dataframe(df_raw.head(20), use_container_width=True, height=300)

        c1, c2, c3 = st.columns(3)
        c1.metric("Cəmi sətir", f"{len(df_raw):,}", help="Dublikatlarla birlikdə")
        c2.metric("Dublikat sətir", "60", delta="-60 silinəcək", delta_color="inverse")
        c3.metric("Yanlış tip sütun", "1", delta="Qiymət sütunu", delta_color="inverse")

        st.info("⚠️  **Qiymət** sütununda 30 sətirdə `'N/A'`, `'xəta'`, `'—'` kimi string dəyərlər mövcuddur. Bu dəyərlər `pd.to_numeric(..., errors='coerce')` ilə NaN-a çevrilir.")

    with tab2:
        st.markdown('<div class="section-title">Boş Dəyərlərin (NaN) Analizi</div>', unsafe_allow_html=True)

        nan_counts = df_raw.isnull().sum()
        nan_pct    = (nan_counts / len(df_raw) * 100).round(2)
        nan_df     = pd.DataFrame({"Sütun": nan_counts.index, "NaN Sayı": nan_counts.values, "NaN %": nan_pct.values})
        nan_df     = nan_df[nan_df["NaN Sayı"] > 0].sort_values("NaN Sayı", ascending=False)

        # Qiymət NaN-larını əlavə et (string→NaN)
        price_nan = pd.DataFrame([{"Sütun": "Qiymət (string→NaN)", "NaN Sayı": 30, "NaN %": round(30/len(df_raw)*100, 2)}])
        nan_df = pd.concat([nan_df, price_nan], ignore_index=True)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(nan_df, use_container_width=True, hide_index=True)
        with col2:
            fig = px.bar(nan_df, x="Sütun", y="NaN Sayı",
                         color="NaN %",
                         color_continuous_scale=["#1E3A5F", "#2E86AB", "#4BA3C3"],
                         title="Sütunlara Görə Boş Dəyər Sayı")
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-title">Dublikat Analizi</div>', unsafe_allow_html=True)
        dup_count = df_raw.duplicated().sum()
        st.markdown(f"""
        <div style="background:#112240; border:1px solid #2E86AB; border-radius:8px; padding:16px;">
        <span style="color:#4BA3C3; font-family:'IBM Plex Mono',monospace; font-size:1.4rem;">{dup_count}</span>
        <span style="color:#C8D6E8; font-size:0.9rem; margin-left:8px;">tam dublikat sətir aşkarlandı → <code style="color:#A8DADC;">df.drop_duplicates()</code> ilə silindi</span>
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="section-title">Təmizlənmiş Dataset (ilk 20 sətir)</div>', unsafe_allow_html=True)
        st.dataframe(df.head(20), use_container_width=True, height=300)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sətir (təmiz)", f"{len(df):,}", delta=f"-{len(df_raw)-len(df)} dublikat silindi")
        c2.metric("NaN — Qiymət", "0 ✓", delta="median ilə dolduruldu")
        c3.metric("NaN — Region", "0 ✓", delta="mode ilə dolduruldu")
        c4.metric("Tip Xətası", "0 ✓", delta="Qiymət → float64")

        st.success("✅  Dataset tam təmizləndi. Bütün boş dəyərlər dolduruldu, dublikatlar silindi, tip xətaları aradan qaldırıldı.")

        with st.expander("📋 Tam Təmizləmə Kodu"):
            st.code("""
# 1. Qiymət sütununu numerikə çevir
df["Qiymət"] = pd.to_numeric(df["Qiymət"], errors="coerce")

# 2. Dublikatları sil
df.drop_duplicates(inplace=True)

# 3. NaN-ları doldur
df["Qiymət"]  = df["Qiymət"].fillna(df["Qiymət"].median())
df["Miqdar"]  = df["Miqdar"].fillna(df["Miqdar"].median()).astype(int)
df["Endirim"] = df["Endirim"].fillna(df["Endirim"].median())
df["Region"]  = df["Region"].fillna(df["Region"].mode()[0])
df["Kanal"]   = df["Kanal"].fillna(df["Kanal"].mode()[0])

# 4. Gəliri yenidən hesabla
df["Gəlir"] = (df["Qiymət"] * df["Miqdar"] * (1 - df["Endirim"])).round(2)

# 5. Tarix xüsusiyyətlərini çıxar
df["İl"]    = df["Tarix"].dt.year
df["Ay"]    = df["Tarix"].dt.month
df["Ay_Adı"] = df["Tarix"].dt.strftime("%b")
            """, language="python")


# ═════════════════════════════════════════════
# 3. EDA
# ═════════════════════════════════════════════
elif page == pages[2]:
    st.markdown("# 📊 EDA — Kəşf Analizi")
    st.markdown('<p style="color:#8B9BB4;">Datanın statistik xülasəsi, paylanma qrafiqləri və dəyişənlər arası əlaqələr.</p>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Statistik Xülasə", "Paylanmalar", "Korrelyasiya"])

    with tab1:
        st.markdown('<div class="section-title">.describe() — Ədədi Sütunlar</div>', unsafe_allow_html=True)
        desc = df[["Qiymət", "Miqdar", "Endirim", "Gəlir"]].describe().round(2)
        st.dataframe(desc.style.format("{:.2f}").background_gradient(cmap="Blues", axis=0),
                     use_container_width=True)

        st.markdown('<div class="section-title">.info() — Sütun Məlumatları</div>', unsafe_allow_html=True)
        info_data = {
            "Sütun":   df.columns.tolist(),
            "Tip":     [str(t) for t in df.dtypes],
            "Boş Dəyər": df.isnull().sum().values,
            "Unikal":  [df[c].nunique() for c in df.columns],
        }
        st.dataframe(pd.DataFrame(info_data), use_container_width=True, hide_index=True)

    with tab2:
        col = st.selectbox("Sütun seçin", ["Qiymət", "Miqdar", "Gəlir", "Endirim"])

        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x=col, nbins=40,
                               title=f"{col} — Paylanma Histoqramı",
                               color_discrete_sequence=["#2E86AB"])
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.box(df, x="Kateqoriya", y=col,
                          title=f"Kateqoriyaya Görə {col} Qutu Diaqramı",
                          color="Kateqoriya",
                          color_discrete_sequence=px.colors.sequential.Blues_r)
            apply_theme(fig2)
            fig2.update_layout(showlegend=False, xaxis_tickangle=-30)
            st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.scatter(df.sample(600, random_state=7), x="Qiymət", y="Gəlir",
                          color="Kateqoriya", size="Miqdar",
                          title="Qiymət vs Gəlir (600 nümunə)",
                          opacity=0.75,
                          color_discrete_sequence=px.colors.sequential.Blues_r)
        apply_theme(fig3)
        st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        num_df = df[["Qiymət", "Miqdar", "Endirim", "Gəlir", "Ay", "İl"]]
        corr   = num_df.corr().round(2)

        fig = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale=[[0, "#0A1628"], [0.5, "#2E86AB"], [1, "#A8DADC"]],
            text=corr.values.round(2),
            texttemplate="%{text}",
            textfont=dict(size=11, color="#F0F4F8"),
            showscale=True,
        ))
        fig.update_layout(title="Korrelyasiya Matrisi (Istilik Xəritəsi)", **PLOTLY_TEMPLATE["layout"])
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Gəlir ilə Qiymət arasında güclü müsbət korrelyasiya gözlənilir.")


# ═════════════════════════════════════════════
# 4. SATIŞ ANALİZİ
# ═════════════════════════════════════════════
elif page == pages[3]:
    st.markdown("# 📈 Satış Analizi")
    st.markdown('<p style="color:#8B9BB4;">Biznes KPI-ları, zaman seriyası trendləri və regional müqayisələr.</p>', unsafe_allow_html=True)

    # Filtre
    with st.expander("🔧 Filtr Seçimləri", expanded=False):
        years = sorted(df["İl"].unique())
        sel_years = st.multiselect("İl seçin", years, default=years)
        sel_cats  = st.multiselect("Kateqoriya seçin", sorted(df["Kateqoriya"].unique()), default=df["Kateqoriya"].unique().tolist())

    dff = df[df["İl"].isin(sel_years) & df["Kateqoriya"].isin(sel_cats)]

    # KPI-lar
    total_rev   = dff["Gəlir"].sum()
    total_orders= len(dff)
    aov         = dff.groupby("Sifariş_ID")["Gəlir"].sum().mean()
    avg_discount= dff["Endirim"].mean() * 100

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("💰 Ümumi Gəlir",        fmt_az(total_rev))
    k2.metric("🛒 Sifariş Sayı",       f"{total_orders:,}")
    k3.metric("📦 Orta Səbət (AOV)",   fmt_az(aov))
    k4.metric("🏷️ Ort. Endirim",      f"{avg_discount:.1f}%")

    st.markdown("---")

    # Aylıq trend
    monthly = dff.groupby(["İl", "Ay"])["Gəlir"].sum().reset_index()
    monthly["Tarix_Ox"] = pd.to_datetime(dict(year=monthly["İl"], month=monthly["Ay"], day=1))
    monthly.sort_values("Tarix_Ox", inplace=True)

    fig_trend = px.line(monthly, x="Tarix_Ox", y="Gəlir",
                        color="İl", markers=True,
                        title="Aylıq Satış Trendi (İllərə Görə)",
                        labels={"Tarix_Ox": "Tarix", "Gəlir": "Gəlir (₼)", "İl": "İl"},
                        color_discrete_sequence=["#2E86AB", "#4BA3C3", "#A8DADC"])
    apply_theme(fig_trend)
    fig_trend.update_traces(line_width=2.5)
    st.plotly_chart(fig_trend, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        reg = dff.groupby("Region")["Gəlir"].sum().reset_index().sort_values("Gəlir", ascending=True)
        fig_reg = px.bar(reg, y="Region", x="Gəlir",
                         orientation="h",
                         title="Regionlara Görə Ümumi Gəlir",
                         color="Gəlir",
                         color_continuous_scale=["#1E3A5F", "#2E86AB", "#4BA3C3"],
                         labels={"Gəlir": "Gəlir (₼)"})
        apply_theme(fig_reg)
        st.plotly_chart(fig_reg, use_container_width=True)

    with c2:
        kanal = dff.groupby(["Kanal", "İl"])["Gəlir"].sum().reset_index()
        fig_ch = px.bar(kanal, x="İl", y="Gəlir", color="Kanal",
                        barmode="group",
                        title="İllərə və Kanallara Görə Gəlir",
                        color_discrete_sequence=["#2E86AB", "#4BA3C3", "#A8DADC"])
        apply_theme(fig_ch)
        st.plotly_chart(fig_ch, use_container_width=True)

    # Haftalık istilik haritəsi
    st.markdown('<div class="section-title">Günlərə & Aylara Görə Satış Istilik Xəritəsi</div>', unsafe_allow_html=True)
    day_month = dff.groupby(["Ay", "Həftə_Günü"])["Gəlir"].mean().reset_index()
    day_order  = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    day_az     = {"Monday":"Baz.ertəsi","Tuesday":"Çərşənbə axşamı","Wednesday":"Çərşənbə",
                  "Thursday":"Cümə axşamı","Friday":"Cümə","Saturday":"Şənbə","Sunday":"Bazar"}
    day_month["Gün"] = day_month["Həftə_Günü"].map(day_az)
    pivot = day_month.pivot_table(index="Gün", columns="Ay", values="Gəlir")
    pivot = pivot.reindex([day_az[d] for d in day_order if day_az[d] in pivot.index])

    fig_heat = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"Ay {m}" for m in pivot.columns],
        y=pivot.index.tolist(),
        colorscale=[[0,"#0A1628"],[0.5,"#2E86AB"],[1,"#A8DADC"]],
        showscale=True
    ))
    fig_heat.update_layout(title="Həftə Günü × Ay — Orta Gəlir (₼)", **PLOTLY_TEMPLATE["layout"])
    st.plotly_chart(fig_heat, use_container_width=True)


# ═════════════════════════════════════════════
# 5. MƏHSUL ANALİZİ
# ═════════════════════════════════════════════
elif page == pages[4]:
    st.markdown("# 🔍 Məhsul Analizi")
    st.markdown('<p style="color:#8B9BB4;">Top məhsullar, kateqoriya paylanması və Pareto (80/20) prinsipi.</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        top_n  = st.slider("Top N məhsul", 5, 20, 10)
        top_products = df.groupby("Məhsul")["Gəlir"].sum().nlargest(top_n).reset_index()

        fig = px.bar(top_products, x="Gəlir", y="Məhsul",
                     orientation="h",
                     title=f"Top {top_n} Məhsul — Gəlir üzrə",
                     color="Gəlir",
                     color_continuous_scale=["#1E3A5F","#2E86AB","#4BA3C3"])
        apply_theme(fig)
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        cat_rev = df.groupby("Kateqoriya")["Gəlir"].sum().reset_index()
        fig2 = px.pie(cat_rev, names="Kateqoriya", values="Gəlir",
                      title="Kateqoriyaya Görə Gəlir Payı",
                      color_discrete_sequence=px.colors.sequential.Blues_r,
                      hole=0.45)
        apply_theme(fig2)
        fig2.update_traces(textposition="inside", textinfo="percent+label",
                           textfont=dict(size=11, color="#F0F4F8"))
        st.plotly_chart(fig2, use_container_width=True)

    # Pareto
    st.markdown('<div class="section-title">Pareto Analizi — 80/20 Prinsipi</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8B9BB4; font-size:0.9rem;">Məhsulların böyük əksəriyyəti ümumi gəlirin 80%-ni hansı kiçik hissəsi yaradır?</p>', unsafe_allow_html=True)

    pareto = df.groupby("Məhsul")["Gəlir"].sum().sort_values(ascending=False).reset_index()
    pareto["Kumulyativ %"] = (pareto["Gəlir"].cumsum() / pareto["Gəlir"].sum() * 100).round(2)
    pareto["Sıra"] = range(1, len(pareto)+1)
    pareto["80% Xətti"] = 80

    fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])
    fig_pareto.add_trace(
        go.Bar(x=pareto["Məhsul"], y=pareto["Gəlir"],
               name="Gəlir (₼)",
               marker_color="#2E86AB", opacity=0.85),
        secondary_y=False)
    fig_pareto.add_trace(
        go.Scatter(x=pareto["Məhsul"], y=pareto["Kumulyativ %"],
                   name="Kumulyativ %",
                   line=dict(color="#A8DADC", width=2.5),
                   mode="lines"),
        secondary_y=True)
    fig_pareto.add_trace(
        go.Scatter(x=pareto["Məhsul"], y=pareto["80% Xətti"],
                   name="80% Həddi",
                   line=dict(color="#FF6B6B", width=1.5, dash="dash"),
                   mode="lines"),
        secondary_y=True)

    fig_pareto.update_layout(
        title="Pareto Diaqramı — Məhsul Gəliri",
        **PLOTLY_TEMPLATE["layout"],
        xaxis_tickangle=-45,
    )
    fig_pareto.update_yaxes(title_text="Gəlir (₼)", secondary_y=False)
    fig_pareto.update_yaxes(title_text="Kumulyativ % →", secondary_y=True)

    st.plotly_chart(fig_pareto, use_container_width=True)

    # 80%-ə çatan məhsul sayı
    n80 = len(pareto[pareto["Kumulyativ %"] <= 80])
    pct_products = round(n80 / len(pareto) * 100, 1)
    st.info(f"📊 **Pareto nəticəsi:** Cəmi **{n80}** məhsul (ümumi məhsulların **{pct_products}%**-i) şirkətin ümumi gəlirinin **80%**-ni yaradır.")

    # Kateqoriya × Region
    st.markdown('<div class="section-title">Kateqoriya × Region Gəlir Matrisi</div>', unsafe_allow_html=True)
    pivot2 = df.pivot_table(values="Gəlir", index="Kateqoriya", columns="Region", aggfunc="sum").fillna(0).round(0)
    fig_piv = go.Figure(go.Heatmap(
        z=pivot2.values,
        x=pivot2.columns.tolist(),
        y=pivot2.index.tolist(),
        colorscale=[[0,"#0A1628"],[0.5,"#1E3A5F"],[1,"#4BA3C3"]],
        text=np.round(pivot2.values/1000,1),
        texttemplate="%{text}K",
        textfont=dict(size=10, color="#F0F4F8"),
    ))
    fig_piv.update_layout(title="Kateqoriya × Region — Ümumi Gəlir (₼)", **PLOTLY_TEMPLATE["layout"])
    st.plotly_chart(fig_piv, use_container_width=True)


# ═════════════════════════════════════════════
# 6. ML MODELİ
# ═════════════════════════════════════════════
elif page == pages[5]:
    st.markdown("# 🤖 ML Modeli")
    st.markdown('<p style="color:#8B9BB4;">Scikit-learn ilə satış proqnozu (Linear Regression) və müştəri seqmentasiyası (K-Means).</p>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📉 Satış Proqnozu (Regression)", "👥 Müştəri Seqmentasiyası (K-Means)"])

    # ── TAB 1: Linear Regression ──
    with tab1:
        st.markdown('<div class="section-title">Aylıq Gəlir Proqnozu — Linear Regression</div>', unsafe_allow_html=True)
        st.markdown('<p style="color:#8B9BB4; font-size:0.9rem;">Ay indeksini (1–36) xüsusiyyət kimi istifadə edərək aylıq ümumi gəliri proqnozlaşdırırıq.</p>', unsafe_allow_html=True)

        monthly_rev = df.groupby(["İl","Ay"])["Gəlir"].sum().reset_index()
        monthly_rev = monthly_rev.sort_values(["İl","Ay"]).reset_index(drop=True)
        monthly_rev["Ay_İndeks"] = range(1, len(monthly_rev)+1)
        monthly_rev["Ay_Kv"]    = monthly_rev["Ay_İndeks"] ** 2
        monthly_rev["Sin_Mevsim"] = np.sin(2 * np.pi * monthly_rev["Ay"] / 12)
        monthly_rev["Cos_Mevsim"] = np.cos(2 * np.pi * monthly_rev["Ay"] / 12)

        features = ["Ay_İndeks", "Ay_Kv", "Sin_Mevsim", "Cos_Mevsim"]
        X = monthly_rev[features].values
        y = monthly_rev["Gəlir"].values

        model = LinearRegression()
        model.fit(X, y)
        monthly_rev["Proqnoz"] = model.predict(X).clip(0)

        r2  = r2_score(y, monthly_rev["Proqnoz"])
        mae = mean_absolute_error(y, monthly_rev["Proqnoz"])

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("R² Skoru", f"{r2:.4f}", help="1-ə yaxın olduqca model daha yaxşıdır.")
        mc2.metric("MAE", fmt_az(mae), help="Orta Mütləq Xəta")
        mc3.metric("Cəmi Ay", f"{len(monthly_rev)}")

        fig_reg = go.Figure()
        fig_reg.add_trace(go.Scatter(
            x=monthly_rev["Ay_İndeks"], y=monthly_rev["Gəlir"],
            mode="markers+lines", name="Həqiqi Gəlir",
            line=dict(color="#4BA3C3", width=1.5),
            marker=dict(size=6)))
        fig_reg.add_trace(go.Scatter(
            x=monthly_rev["Ay_İndeks"], y=monthly_rev["Proqnoz"],
            mode="lines", name="Model Proqnozu",
            line=dict(color="#FF9F43", width=2.5, dash="dot")))
        fig_reg.update_layout(
            title="Həqiqi Gəlir vs Model Proqnozu",
            xaxis_title="Ay İndeksi", yaxis_title="Gəlir (₼)",
            **PLOTLY_TEMPLATE["layout"])
        st.plotly_chart(fig_reg, use_container_width=True)

        # İnteraktiv proqnoz
        st.markdown('<div class="section-title">🔮 İnteraktiv Proqnoz Alətı</div>', unsafe_allow_html=True)
        st.markdown('<p style="color:#8B9BB4; font-size:0.88rem;">Aşağıdakı slayderləri tənzimləyərək gələcək ay üçün proqnoz alın.</p>', unsafe_allow_html=True)

        p1, p2 = st.columns(2)
        with p1:
            future_month_idx = st.slider("Ay İndeksi (37–60 = gələcək)", 37, 60, 40)
        with p2:
            future_month_num = st.slider("Mövsüm Ayı (1–12)", 1, 12, 6)

        X_pred = np.array([[
            future_month_idx,
            future_month_idx**2,
            np.sin(2*np.pi*future_month_num/12),
            np.cos(2*np.pi*future_month_num/12)
        ]])
        pred_val = model.predict(X_pred)[0]

        st.markdown(f"""
        <div style="background:#112240; border:1px solid #2E86AB; border-radius:8px; padding:24px; text-align:center; margin-top:12px;">
            <p style="color:#8B9BB4; font-size:0.8rem; text-transform:uppercase; letter-spacing:1px; margin:0;">Proqnozlaşdırılan Aylıq Gəlir</p>
            <p style="font-family:'IBM Plex Mono',monospace; font-size:2.8rem; color:#4BA3C3; font-weight:700; margin:8px 0;">{fmt_az(max(pred_val, 0))}</p>
            <p style="color:#8B9BB4; font-size:0.82rem; margin:0;">Ay İndeksi: {future_month_idx} · Mövsüm Ayı: {future_month_num}</p>
        </div>
        """, unsafe_allow_html=True)

    # ── TAB 2: K-Means ──
    with tab2:
        st.markdown('<div class="section-title">Müştəri Seqmentasiyası — K-Means Clustering</div>', unsafe_allow_html=True)
        st.markdown('<p style="color:#8B9BB4; font-size:0.9rem;">RFM (Recency, Frequency, Monetary) xüsusiyyətləri ilə müştəriləri seqmentlərə bölürük.</p>', unsafe_allow_html=True)

        ref_date = df["Tarix"].max()
        rfm = df.groupby("Müştəri_ID").agg(
            Recency    = ("Tarix",       lambda x: (ref_date - x.max()).days),
            Frequency  = ("Sifariş_ID", "count"),
            Monetary   = ("Gəlir",      "sum")
        ).reset_index()

        scaler  = StandardScaler()
        rfm_sc  = scaler.fit_transform(rfm[["Recency","Frequency","Monetary"]])

        n_clusters = st.slider("Klaster sayı (K)", 2, 6, 4)

        kmeans  = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        rfm["Seqment"] = kmeans.fit_predict(rfm_sc).astype(str)
        rfm["Seqment"] = "Seqment " + rfm["Seqment"].astype(str)

        c1, c2 = st.columns(2)
        with c1:
            fig_km = px.scatter(rfm, x="Recency", y="Monetary",
                                color="Seqment", size="Frequency",
                                title="RFM — Recency vs Monetary",
                                labels={"Recency":"Recency (gün)", "Monetary":"Ümumi Xərcləmə (₼)"},
                                color_discrete_sequence=["#2E86AB","#4BA3C3","#A8DADC","#457B9D","#1D3557","#8ECAE6"],
                                opacity=0.8)
            apply_theme(fig_km)
            st.plotly_chart(fig_km, use_container_width=True)

        with c2:
            fig_km2 = px.scatter(rfm, x="Frequency", y="Monetary",
                                 color="Seqment", size="Recency",
                                 title="RFM — Frequency vs Monetary",
                                 labels={"Frequency":"Sifariş Sayı","Monetary":"Ümumi Xərcləmə (₼)"},
                                 color_discrete_sequence=["#2E86AB","#4BA3C3","#A8DADC","#457B9D","#1D3557","#8ECAE6"],
                                 opacity=0.8)
            apply_theme(fig_km2)
            st.plotly_chart(fig_km2, use_container_width=True)

        # Seqment xülasəsi
        st.markdown('<div class="section-title">Seqment Xülasəsi</div>', unsafe_allow_html=True)
        seg_summary = rfm.groupby("Seqment").agg(
            Müştəri_Sayı  = ("Müştəri_ID", "count"),
            Ort_Recency   = ("Recency",    "mean"),
            Ort_Frequency = ("Frequency",  "mean"),
            Ort_Monetary  = ("Monetary",   "mean"),
        ).round(1).reset_index()
        st.dataframe(seg_summary, use_container_width=True, hide_index=True)

        fig_seg = px.bar(seg_summary, x="Seqment", y="Ort_Monetary",
                         color="Seqment", title="Seqmentə Görə Orta Xərcləmə (₼)",
                         color_discrete_sequence=["#2E86AB","#4BA3C3","#A8DADC","#457B9D","#1D3557","#8ECAE6"])
        apply_theme(fig_seg)
        fig_seg.update_layout(showlegend=False)
        st.plotly_chart(fig_seg, use_container_width=True)

        st.info("💡 **Tövsiyə:** Ən yüksək Monetary dəyərli seqmentə loyallıq proqramları, ən yüksək Recency dəyərli (köhnə) seqmentə isə yenidən aktivasiya kampaniyaları tətbiq edilməlidir.")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; padding:12px 0; color:#8B9BB4; font-size:0.78rem;">
    Nicat Əliyev · Data Analitik Portfolio · 2024 &nbsp;|&nbsp;
    Python · Pandas · Plotly · Scikit-learn · Streamlit
</div>
""", unsafe_allow_html=True)
