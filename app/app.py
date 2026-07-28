import sys
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.persistence.repository import PortfolioRepository
from app.data.instrument.provider import InstrumentProvider
from app.data.currency.converter import CurrencyConverter
from app.portfolio.asset import Portfolio, Asset

# ==========================================
# PAGE & THEME CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Investment Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 15px;
        color: #E0E0E0;
    }
</style>
""", unsafe_allow_html=True)

# Shared Data Layer Services
@st.cache_resource
def get_repository():
    return PortfolioRepository()

@st.cache_resource
def get_instrument_provider():
    return InstrumentProvider()

repo = get_repository()
provider = get_instrument_provider()

SUPPORTED_CURRENCIES = ["USD", "EUR", "PLN"]

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["💼 Dashboard", "🔍 Instruments Catalog", "➕ Create Portfolio"]
)

st.sidebar.markdown("---")

# ==========================================
# PAGE 1: PORTFOLIO DASHBOARD
# ==========================================
if page == "💼 Dashboard":
    st.sidebar.subheader("💼 Portfolios")
    
    available_portfolios = repo.get_all_portfolio_names()

    if not available_portfolios:
        st.title("💼 Portfolio Dashboard")
        st.info("No portfolios found in the database. Go to '➕ Create Portfolio' to build your first portfolio!")
        st.stop()

    selected_portfolio_name = st.sidebar.selectbox("Select Portfolio", options=available_portfolios)

    if st.sidebar.button("🔄 Refresh Market Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    # Fetch domain portfolio populated with live market data
    portfolio = repo.get_by_name(selected_portfolio_name, instrument_provider=provider)

    if not portfolio or not getattr(portfolio, "_assets", []):
        st.title(f"💼 {selected_portfolio_name}")
        st.warning("This portfolio is empty or could not be loaded properly.")
        st.stop()

    # Read base portfolio metrics
    cur = portfolio.currency
    total_initial = portfolio.initial_value
    total_current = portfolio.value
    total_change = portfolio.get_value_change()
    total_percent = portfolio.get_percent_change() * 100

    assets_data = portfolio.get_assets_data() 

    df = pd.DataFrame(assets_data)

    st.title(f"💼 {portfolio.name}")
    st.caption(f"Denominated in **{cur}** | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Top KPI Banner
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Invested", f"{total_initial:,.2f} {cur}")
    kpi2.metric("Current Value", f"{total_current:,.2f} {cur}")
    kpi3.metric("Profit / Loss", f"{total_change:+,.2f} {cur}", delta=f"{total_change:+,.2f} {cur}")
    kpi4.metric("Total Return", f"{total_percent:+.2f}%", delta=f"{total_percent:+.2f}%")

    st.markdown("---")

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown('<p class="section-header">🍩 Asset Allocation</p>', unsafe_allow_html=True)
        fig_donut = px.pie(df, values='Current Value', names='Symbol', hole=0.55)
        fig_donut.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            hovertemplate=f"<b>%{{label}}</b><br>Value: %{{value:,.2f}} {cur}<br>Share: %{{percent}}"
        )
        fig_donut.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300, template="plotly_dark")
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_right:
        st.markdown('<p class="section-header">📊 Profit / Loss per Asset</p>', unsafe_allow_html=True)
        pnl_col = 'Profit / Loss' if 'Profit / Loss' in df.columns else 'Value Change'
        colors = ['#00C853' if val >= 0 else '#FF5252' for val in df[pnl_col]]
        fig_bar = go.Figure(go.Bar(x=df['Symbol'], y=df[pnl_col], marker_color=colors))
        fig_bar.update_layout(
            margin=dict(t=10, b=10, l=10, r=10), 
            height=300, 
            template="plotly_dark",
            yaxis_title=f"P/L ({cur})"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Detailed Table
    st.markdown('<p class="section-header">📜 Positions Breakdown</p>', unsafe_allow_html=True)
    st.dataframe(
        df,
        column_config={
            "Buy Price": st.column_config.NumberColumn(f"Buy Price ({cur})", format="%.2f"),
            "Current Price": st.column_config.NumberColumn(f"Live Price ({cur})", format="%.2f"),
            "Initial Value": st.column_config.NumberColumn(f"Cost Basis ({cur})", format="%.2f"),
            "Current Value": st.column_config.NumberColumn(f"Market Value ({cur})", format="%.2f"),
            "Profit / Loss": st.column_config.NumberColumn(f"Profit / Loss ({cur})", format="%+.2f"),
            "Value Change": st.column_config.NumberColumn(f"Profit / Loss ({cur})", format="%+.2f"),
            "Return (%)": st.column_config.NumberColumn("Return", format="%+.2f%%"),
        },
        hide_index=True,
        use_container_width=True
    )

# ==========================================
# PAGE 2: INSTRUMENTS CATALOG
# ==========================================
elif page == "🔍 Instruments Catalog":
    st.title("🔍 Instrument Explorer & Market Data")
    st.write("Analyze individual instruments, historical prices, fundamental ratios, and financial statements.")

    if hasattr(provider, 'instrument_symbols'):
        available_symbols = provider.instrument_symbols
    elif hasattr(provider, '_instruments'):
        available_symbols = list(provider._instruments.keys())
    else:
        available_symbols = ['PKO.WA', 'MSFT', 'AAPL']

    col_sym, col_period = st.columns([2, 1])
    with col_sym:
        selected_symbol = st.selectbox("Select Instrument to Inspect", options=available_symbols)
    with col_period:
        time_frame = st.selectbox("Historical Window", ["1 Month", "6 Months", "1 Year", "5 Years"], index=2)

    period_days_map = {"1 Month": 30, "6 Months": 180, "1 Year": 365, "5 Years": 365 * 5}
    days = period_days_map[time_frame]
    start_date = datetime.now() - timedelta(days=days)
    end_date = datetime.now()

    try:
        instrument = provider.get_instrument(selected_symbol)
        basic_info = instrument.get_basic_info()
        market_data = instrument.get_current_market_data()
        financial_metrics = instrument.get_financial_metrics()
        financial_health = instrument.get_financial_health()
    except Exception as e:
        st.error(f"Failed to load data for {selected_symbol}: {e}")
        st.stop()

    st.markdown(f"## {basic_info.get('long_name') or selected_symbol} (`{selected_symbol}`)")
    st.caption(f"Sector: **{basic_info.get('sector', 'N/A')}** | Industry: **{basic_info.get('industry', 'N/A')}** | Currency: **{basic_info.get('currency', 'USD')}**")

    st.markdown("---")

    c1, c2, c3, c4, c5 = st.columns(5)
    curr_price = market_data.get("current_price") or 0.0
    prev_close = market_data.get("previous_close") or curr_price
    price_change = curr_price - prev_close
    pct_change = (price_change / prev_close * 100) if prev_close else 0.0

    c1.metric("Current Price", f"{curr_price:,.2f} {basic_info.get('currency', '')}", delta=f"{pct_change:+.2f}%")
    c2.metric("Day Range", f"{market_data.get('day_low', 0):,.2f} - {market_data.get('day_high', 0):,.2f}")
    c3.metric("52-Wk Range", f"{market_data.get('fifty_two_week_low', 0):,.2f} - {market_data.get('fifty_two_week_high', 0):,.2f}")
    
    mcap = market_data.get("market_cap")
    mcap_str = f"${mcap/1e9:,.2f}B" if mcap and mcap > 1e9 else (f"${mcap/1e6:,.2f}M" if mcap else "N/A")
    c4.metric("Market Cap", mcap_str)
    
    pe_ratio = financial_metrics.get("trailing_pe")
    c5.metric("Trailing P/E", f"{pe_ratio:.2f}" if pe_ratio else "N/A")

    st.markdown("---")

    tab_chart, tab_fundamentals, tab_statements, tab_news = st.tabs([
        "📈 Price History", 
        "📊 Valuation & Health", 
        "📜 Financial Statements", 
        "📰 News Feed"
    ])

    with tab_chart:
        st.subheader("Historical Stock Price")
        try:
            hist_df = instrument.get_historical_market_data(start=start_date, end=end_date)
            if not hist_df.empty:
                chart_type = st.radio("Chart Type", ["Line Chart", "Candlestick"], horizontal=True)

                if chart_type == "Candlestick":
                    fig = go.Figure(data=[go.Candlestick(
                        x=hist_df.index,
                        open=hist_df['Open'],
                        high=hist_df['High'],
                        low=hist_df['Low'],
                        close=hist_df['Close'],
                        name=selected_symbol
                    )])
                else:
                    fig = px.line(hist_df, x=hist_df.index, y="Close", title=f"{selected_symbol} Close Price History")
                
                fig.update_layout(
                    height=450, 
                    template="plotly_dark", 
                    margin=dict(l=10, r=10, t=30, b=10),
                    xaxis_title="Date",
                    yaxis_title=f"Price ({basic_info.get('currency', 'USD')})"
                )
                st.plotly_chart(fig, use_container_width=True)

                fig_vol = px.bar(hist_df, x=hist_df.index, y="Volume", title="Trading Volume")
                fig_vol.update_layout(height=200, template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_vol, use_container_width=True)
            else:
                st.warning("No historical market data returned for the selected time window.")
        except Exception as e:
            st.error(f"Error rendering price chart: {e}")

    with tab_fundamentals:
        col_val, col_health = st.columns(2)
        
        with col_val:
            st.subheader("Valuation Metrics")
            val_df = pd.DataFrame([
                {"Metric": "Trailing P/E", "Value": financial_metrics.get("trailing_pe")},
                {"Metric": "Forward P/E", "Value": financial_metrics.get("forward_pe")},
                {"Metric": "PEG Ratio", "Value": financial_metrics.get("peg_ratio")},
                {"Metric": "Price-to-Book", "Value": financial_metrics.get("price_to_book")},
                {"Metric": "Dividend Yield", "Value": f"{financial_metrics.get('dividend_yield', 0) * 100:.2f}%" if financial_metrics.get('dividend_yield') else "N/A"},
                {"Metric": "Beta (Volatility)", "Value": financial_metrics.get("beta")}
            ])
            st.dataframe(val_df, hide_index=True, use_container_width=True)

        with col_health:
            st.subheader("Financial Health")
            
            def format_num(val):
                if not val: return "N/A"
                if abs(val) >= 1e9: return f"${val/1e9:,.2f}B"
                if abs(val) >= 1e6: return f"${val/1e6:,.2f}M"
                return f"${val:,.2f}"

            health_df = pd.DataFrame([
                {"Metric": "Total Revenue", "Value": format_num(financial_health.get("total_revenue"))},
                {"Metric": "Revenue Growth (YoY)", "Value": f"{financial_health.get('revenue_growth', 0) * 100:.2f}%" if financial_health.get('revenue_growth') else "N/A"},
                {"Metric": "EBITDA", "Value": format_num(financial_health.get("ebitda"))},
                {"Metric": "Profit Margin", "Value": f"{financial_health.get('profit_margin', 0) * 100:.2f}%" if financial_health.get('profit_margin') else "N/A"},
                {"Metric": "Total Debt", "Value": format_num(financial_health.get("total_debt"))},
                {"Metric": "Quick Ratio", "Value": financial_health.get("quick_ratio")},
                {"Metric": "ROE (Return on Equity)", "Value": f"{financial_health.get('return_on_equity', 0) * 100:.2f}%" if financial_health.get('return_on_equity') else "N/A"}
            ])
            st.dataframe(health_df, hide_index=True, use_container_width=True)

        if basic_info.get("summary"):
            st.subheader("Business Summary")
            st.info(basic_info.get("summary"))

    with tab_statements:
        st.subheader("Financial Statements")
        stmt_choice = st.selectbox("Select Statement", ["Income Statement", "Balance Sheet", "Cash Flow"])
        
        try:
            statements = instrument.get_financial_statements()
            key_map = {
                "Income Statement": "yearly_income_statement",
                "Balance Sheet": "yearly_balance_sheet",
                "Cash Flow": "yearly_cashflow"
            }
            raw_stmt = statements.get(key_map[stmt_choice], {})
            
            if raw_stmt:
                df_stmt = pd.DataFrame(raw_stmt)
                st.dataframe(df_stmt, use_container_width=True)
            else:
                st.info(f"No {stmt_choice} data available for {selected_symbol}.")
        except Exception as e:
            st.error(f"Error fetching financial statements: {e}")

    with tab_news:
        st.subheader(f"Latest News for {selected_symbol}")
        if st.button("📰 Fetch Latest News"):
            with st.spinner("Scraping and parsing news articles..."):
                news_items = instrument.get_news(max_workers=3)
                if news_items:
                    for item in news_items[:5]:
                        with st.expander(f"📌 {item.get('title', 'Untitled')} ({item.get('source', 'Unknown Source')})"):
                            st.caption(f"Published: {item.get('date', 'N/A')}")
                            st.write(item.get('content') or "No body content parsed.")
                else:
                    st.info("No news articles found.")

# ==========================================
# PAGE 3: PORTFOLIO CREATOR (CURRENCY-CONVERTER INTEGRATED)
# ==========================================
elif page == "➕ Create Portfolio":
    st.title("➕ Create New Portfolio")

    available_symbols = provider.instrument_symbols

    if "draft_positions" not in st.session_state:
        st.session_state.draft_positions = []

    # ------------------------------------------
    # TOP TOOLBAR: METADATA & BASE CURRENCY
    # ------------------------------------------
    c_meta1, c_meta2 = st.columns([2, 1])

    with c_meta1:
        portfolio_name_input = st.text_input(
            "Portfolio Name", 
            placeholder="e.g. Growth & Tech Portfolio 2026",
            key="p_name_input"
        )

    if "portfolio_base_currency" not in st.session_state:
        st.session_state.portfolio_base_currency = "USD"

    # Callback when portfolio currency changes
    def sync_inputs_on_base_currency_change():
        selected_symbol = st.session_state.get("pos_symbol_select", available_symbols[0])
        target_ccy = st.session_state.portfolio_base_currency
        converter = CurrencyConverter(target_ccy)
        
        try:
            inst = provider.get_instrument(selected_symbol)
            native_price = float(inst.current_price) if inst and inst.current_price else 0.0
            native_ccy = inst.get_basic_info().get('currency', 'USD') if inst else 'USD'
        except Exception:
            native_price = 0.0
            native_ccy = 'USD'

        converted_price = converter.convert(native_price, native_ccy)
        st.session_state.pos_price_input = converted_price
        st.session_state.pos_total_input = st.session_state.pos_vol_input * converted_price

    with c_meta2:
        base_currency = st.selectbox(
            "Portfolio Denomination Currency", 
            options=SUPPORTED_CURRENCIES,
            key="portfolio_base_currency",
            on_change=sync_inputs_on_base_currency_change,
            help="All assets will be converted and denominated in this currency."
        )

    active_converter = CurrencyConverter(base_currency)

    # ------------------------------------------
    # CALLBACKS FOR 3-INPUT SYNCHRONIZATION
    # ------------------------------------------
    def sync_on_instrument_change():
        selected = st.session_state.pos_symbol_select
        target_ccy = st.session_state.portfolio_base_currency
        converter = CurrencyConverter(target_ccy)
        try:
            inst = provider.get_instrument(selected)
            native_price = float(inst.current_price) if inst and inst.current_price else 0.0
            native_ccy = inst.get_basic_info().get('currency', 'USD') if inst else 'USD'
        except Exception:
            native_price = 0.0
            native_ccy = 'USD'
        
        converted_price = converter.convert(native_price, native_ccy)
        st.session_state.pos_price_input = converted_price
        st.session_state.pos_total_input = st.session_state.pos_vol_input * converted_price

    def sync_on_volume_or_price_change():
        st.session_state.pos_total_input = st.session_state.pos_vol_input * st.session_state.pos_price_input

    def sync_on_total_change():
        price = st.session_state.pos_price_input
        if price > 0:
            st.session_state.pos_vol_input = st.session_state.pos_total_input / price

    if "pos_price_input" not in st.session_state:
        try:
            init_inst = provider.get_instrument(available_symbols[0])
            init_price = float(init_inst.current_price) if init_inst and init_inst.current_price else 100.0
            init_ccy = init_inst.get_basic_info().get('currency', 'USD') if init_inst else 'USD'
        except Exception:
            init_price = 100.0
            init_ccy = 'USD'
        st.session_state.pos_price_input = active_converter.convert(init_price, init_ccy)

    if "pos_vol_input" not in st.session_state:
        st.session_state.pos_vol_input = 10.0

    if "pos_total_input" not in st.session_state:
        st.session_state.pos_total_input = st.session_state.pos_vol_input * st.session_state.pos_price_input

    st.markdown("---")

    left_col, right_col = st.columns([1, 1.5], gap="large")

    with left_col:
        st.subheader("1. Add Asset")
        
        with st.container(border=True):
            selected_symbol = st.selectbox(
                "Instrument", 
                options=available_symbols, 
                key="pos_symbol_select",
                on_change=sync_on_instrument_change
            )

            try:
                inst_obj = provider.get_instrument(selected_symbol)
                native_price = float(inst_obj.current_price) if inst_obj and inst_obj.current_price else 0.0
                native_currency = inst_obj.get_basic_info().get('currency', 'USD') if inst_obj else 'USD'
            except Exception:
                native_price = 0.0
                native_currency = 'USD'

            converted_live_price = active_converter.convert(native_price, native_currency)

            purchase_date_input = st.date_input(
                "Purchase Date",
                value=datetime.now(),
                key="pos_date_input"
            )

            st.number_input(
                "Volume / Shares", 
                min_value=0.0001, 
                step=1.0, 
                key="pos_vol_input",
                on_change=sync_on_volume_or_price_change
            )

            st.number_input(
                f"Buy Price per Share ({base_currency})", 
                min_value=0.01, 
                step=1.0,
                key="pos_price_input",
                on_change=sync_on_volume_or_price_change
            )

            st.number_input(
                f"Total Position Value ({base_currency})", 
                min_value=0.01, 
                step=10.0,
                key="pos_total_input",
                on_change=sync_on_total_change
            )

            st.caption(
                f"💡 Live price for **{selected_symbol}**: "
                f"`{converted_live_price:,.2f} {base_currency}` "
                f"*(Native: {native_price:,.2f} {native_currency})*"
            )
            st.markdown("---")

            if st.button("➕ Add to Basket", type="primary", use_container_width=True):
                volume_to_add = st.session_state.pos_vol_input
                price_to_add = st.session_state.pos_price_input
                total_to_add = st.session_state.pos_total_input
                dt_to_add = datetime.combine(purchase_date_input, datetime.min.time())

                existing_idx = next(
                    (i for i, pos in enumerate(st.session_state.draft_positions) if pos["symbol"] == selected_symbol), 
                    None
                )

                if existing_idx is not None:
                    st.session_state.draft_positions[existing_idx]["volume"] += volume_to_add
                    st.session_state.draft_positions[existing_idx]["total_cost"] += total_to_add
                    st.toast(f"Updated {selected_symbol} volume (+{volume_to_add:.2f})!", icon="🔄")
                else:
                    st.session_state.draft_positions.append({
                        "symbol": selected_symbol,
                        "volume": volume_to_add,
                        "buy_price": price_to_add,
                        "total_cost": total_to_add,
                        "purchase_date": dt_to_add,
                        "currency": base_currency
                    })
                    st.toast(f"Added {selected_symbol} to basket ({base_currency})!", icon="✅")
                
                st.rerun()

    with right_col:
        st.subheader("2. Portfolio Details")

        if st.session_state.draft_positions:
            df_draft = pd.DataFrame(st.session_state.draft_positions)
            total_val = sum(pos["total_cost"] for pos in st.session_state.draft_positions)

            st.metric(
                label=f"Total Initial Value ({base_currency})", 
                value=f"{total_val:,.2f} {base_currency}", 
                delta=f"{len(df_draft)} Position(s)"
            )

            fig_donut = px.pie(
                df_draft,
                names="symbol",
                values="total_cost",
                hole=0.55,
                title=f"Draft Asset Allocation ({base_currency})",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_donut.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                hovertemplate=f"<b>%{{label}}</b><br>Value: %{{value:,.2f}} {base_currency}<br>Share: %{{percent}}"
            )
            fig_donut.update_layout(
                height=260,
                margin=dict(l=10, r=10, t=35, b=10),
                template="plotly_dark",
                showlegend=True
            )
            st.plotly_chart(fig_donut, use_container_width=True)

            st.dataframe(
                df_draft[["symbol", "volume", "buy_price", "total_cost", "currency"]],
                column_config={
                    "symbol": "Symbol",
                    "volume": st.column_config.NumberColumn("Volume", format="%.2f"),
                    "buy_price": st.column_config.NumberColumn(f"Buy Price ({base_currency})", format="%.2f"),
                    "total_cost": st.column_config.NumberColumn(f"Total ({base_currency})", format="%.2f"),
                    "currency": "CCY"
                },
                hide_index=True,
                use_container_width=True
            )

            b_save, b_clear = st.columns([3, 1])

            with b_save:
                if st.button("🚀 Save Portfolio", type="primary", use_container_width=True):
                    if not portfolio_name_input.strip():
                        st.error("Please enter a portfolio name.")
                    else:
                        try:
                            # Instantiate domain Portfolio
                            new_portfolio = Portfolio(
                                name=portfolio_name_input.strip(), 
                                currency=base_currency
                            )

                            for pos in st.session_state.draft_positions:
                                inst = provider.get_instrument(pos["symbol"])
                                
                                # ----------------------------------------------------
                                # FIX: Get native currency and convert UI buy_price 
                                # back to instrument's native currency before saving
                                # ----------------------------------------------------
                                native_currency = inst.get_basic_info().get('currency', 'USD') if inst else 'USD'
                                native_converter = CurrencyConverter(native_currency)
                                
                                # Convert buy_price from base_currency -> native_currency
                                native_buy_price = native_converter.convert(pos["buy_price"], base_currency)

                                asset = Asset(
                                    instrument=inst, 
                                    volume=pos["volume"], 
                                    buy_price=native_buy_price,  # <--- Pass native price!
                                    purchase_date=pos["purchase_date"]
                                )
                                
                                new_portfolio.add(asset)

                            repo.save(new_portfolio)
                            st.success(f"Portfolio '{portfolio_name_input}' ({base_currency}) saved successfully!")
                            st.session_state.draft_positions = []
                            st.balloons()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Save failed: {e}")

            with b_clear:
                if st.button("🗑️ Clear", use_container_width=True):
                    st.session_state.draft_positions = []
                    st.rerun()

        else:
            st.info("🛒 Your basket is empty. Add instruments from the left panel to begin.")