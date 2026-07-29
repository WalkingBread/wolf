# AI-generated UI

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
from app.advisor.advisor import InvestingAdvisor
from app.advisor.genai import AzureModelProvider

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

@st.cache_resource
def get_investing_advisor():
    return InvestingAdvisor(AzureModelProvider())

repo = get_repository()
provider = get_instrument_provider()

advisor = get_investing_advisor()

SUPPORTED_CURRENCIES = ["USD", "EUR", "PLN"]

def calculate_historical_portfolio_performance(portfolio, provider, default_days=365):
    if not getattr(portfolio, "_assets", []):
        return pd.DataFrame(), pd.DataFrame()

    cur = portfolio.currency
    converter = CurrencyConverter(cur)

    purchase_dates = []
    for asset in portfolio._assets:
        p_date = getattr(asset, 'purchase_date', None)
        if isinstance(p_date, datetime):
            p_date = p_date.date()
        elif isinstance(p_date, str):
            p_date = datetime.strptime(p_date, "%Y-%m-%d").date()
        
        if p_date:
            purchase_dates.append(p_date)

    if purchase_dates:
        earliest_acquisition = min(purchase_dates)
    else:
        earliest_acquisition = (datetime.now() - timedelta(days=default_days)).date()

    start_date = datetime.combine(earliest_acquisition, datetime.min.time())
    end_date = datetime.now()

    asset_histories = {}
    for asset in portfolio._assets:
        try:
            hist_df = asset._instrument.get_historical_market_data(start=start_date, end=end_date)
            if not hist_df.empty and 'Close' in hist_df.columns:
                hist_df.index = pd.to_datetime(hist_df.index).date
                
                asset_pdate = getattr(asset, 'purchase_date', earliest_acquisition)
                if isinstance(asset_pdate, datetime):
                    asset_pdate = asset_pdate.date()

                asset_histories[asset.symbol] = (asset, hist_df['Close'], asset_pdate)
        except Exception:
            continue

    if not asset_histories:
        return pd.DataFrame(), pd.DataFrame()

    all_dates = sorted(list(set().union(*[series.index for _, series, _ in asset_histories.values()])))
    all_dates = [dt for dt in all_dates if dt >= earliest_acquisition]

    portfolio_daily_values = []
    asset_perf_records = []

    for dt in all_dates:
        total_day_val = 0.0
        
        for symbol, (asset, series, p_date) in asset_histories.items():
            if dt >= p_date:
                if dt in series.index:
                    price_native = float(series.loc[dt])
                else:
                    available_series = series[series.index <= dt]
                    if not available_series.empty:
                        price_native = float(available_series.iloc[-1])
                    else:
                        price_native = float(series.iloc[0])

                price_converted = converter.convert(price_native, asset.currency)
                position_val = price_converted * asset.volume
                total_day_val += position_val

                buy_price_converted = converter.convert(asset.buy_price, asset.currency)
                cost_basis = buy_price_converted * asset.volume
                pct_return = ((position_val - cost_basis) / cost_basis * 100) if cost_basis != 0 else 0.0

                asset_perf_records.append({
                    "Date": dt,
                    "Symbol": symbol,
                    "Price": price_converted,
                    "Position Value": position_val,
                    "Return (%)": pct_return
                })

        portfolio_daily_values.append({
            "Date": dt,
            "Total Portfolio Value": total_day_val
        })

    df_total_history = pd.DataFrame(portfolio_daily_values)
    df_assets_history = pd.DataFrame(asset_perf_records)

    return df_total_history, df_assets_history

st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["💼 Dashboard", "🔍 Instruments Catalog", "➕ Create Portfolio"]
)

st.sidebar.markdown("---")

if page == "💼 Dashboard":
    st.sidebar.subheader("💼 Portfolios")
    
    available_portfolios = repo.get_all_portfolio_names()

    if not available_portfolios:
        st.title("💼 Portfolio Dashboard")
        st.info("No portfolios found in the database. Go to '➕ Create Portfolio' to build your first portfolio!")
        st.stop()

    selected_portfolio_name = st.sidebar.selectbox("Select Portfolio", options=available_portfolios)

    portfolio = repo.get_by_name(selected_portfolio_name, instrument_provider=provider)

    if st.sidebar.button("🔄 Refresh Market Data", use_container_width=True):
        for asset in portfolio.assets:
            asset.instrument.refresh_data()
        st.cache_data.clear()
        st.rerun()

    if not portfolio or not getattr(portfolio, "_assets", []):
        st.title(f"💼 {selected_portfolio_name}")
        st.warning("This portfolio is empty or could not be loaded properly.")
        st.stop()

    cur = portfolio.currency
    total_initial = portfolio.initial_value
    total_current = portfolio.value
    total_change = portfolio.get_value_change()
    total_percent = portfolio.get_percent_change() * 100

    assets_data = portfolio.get_assets_data()
    
    if not assets_data:
        for asset in portfolio._assets:
            change = asset.get_value_change()
            pct_change = asset.get_percent_change() * 100
            
            assets_data.append({
                "Symbol": asset.symbol,
                "Name": getattr(asset, 'name', asset.symbol),
                "Volume": asset.volume,
                "Buy Price": asset.buy_price,
                "Current Price": asset.current_price_per_share if hasattr(asset, 'current_price_per_share') else getattr(asset._instrument, 'current_price', 0.0),
                "Initial Value": asset.initial_value,
                "Current Value": asset.value,
                "Profit / Loss": change,
                "Return (%)": pct_change
            })

    df = pd.DataFrame(assets_data)

    st.title(f"💼 {portfolio.name}")
    st.caption(f"Denominated in **{cur}** | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Invested", f"{total_initial:,.2f} {cur}")
    kpi2.metric("Current Value", f"{total_current:,.2f} {cur}")
    kpi3.metric("Profit / Loss", f"{total_change:+,.2f} {cur}", delta=f"{total_change:+,.2f} {cur}")
    kpi4.metric("Total Return", f"{total_percent:+.2f}%", delta=f"{total_percent:+.2f}%")

    st.markdown("---")

    # Allocation & P/L Columns
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
            "Return": st.column_config.NumberColumn("Return", format="%+.2f%%"),
        },
        hide_index=True,
        use_container_width=True
    )

    st.markdown('<p class="section-header">📈 Performance Over Time</p>', unsafe_allow_html=True)
    
    with st.spinner("Fetching historical market trends..."):
        df_total_history, df_assets_history = calculate_historical_portfolio_performance(portfolio, provider)

    plot_tab1, plot_tab2 = st.tabs(["💰 Total Portfolio Value", "📊 Individual Asset Return Trajectory"])

    with plot_tab1:
        if not df_total_history.empty:
            fig_portfolio_trend = px.line(
                df_total_history,
                x="Date",
                y="Total Portfolio Value",
                title=f"Total Portfolio Value Trend ({cur})"
            )
            
            fig_portfolio_trend.add_hline(
                y=total_initial, 
                line_dash="dash", 
                line_color="#FFA726",
                annotation_text=f"Initial Invested ({total_initial:,.2f} {cur})"
            )
            
            fig_portfolio_trend.update_traces(line_color="#29B6F6", line_width=2.5)
            fig_portfolio_trend.update_layout(
                height=380,
                template="plotly_dark",
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis_title="Date",
                yaxis_title=f"Portfolio Value ({cur})"
            )
            st.plotly_chart(fig_portfolio_trend, use_container_width=True)
        else:
            st.info("Unable to generate historical timeline for the total portfolio.")

    with plot_tab2:
        if not df_assets_history.empty:
            fig_assets_trend = px.line(
                df_assets_history,
                x="Date",
                y="Return (%)",
                color="Symbol",
                title="Asset Return (%) Trajectory over Time"
            )
            fig_assets_trend.add_hline(y=0, line_dash="solid", line_color="#888888")
            fig_assets_trend.update_layout(
                height=380,
                template="plotly_dark",
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis_title="Date",
                yaxis_title="Return (%)"
            )
            st.plotly_chart(fig_assets_trend, use_container_width=True)
        else:
            st.info("Unable to generate historical timeline for individual assets.")

    st.markdown("---")

elif page == "🔍 Instruments Catalog":
    st.sidebar.subheader("🔍 Instrument Selection")

    available_symbols = provider.instrument_symbols

    selected_symbol = st.sidebar.selectbox("Select Instrument", options=available_symbols)
    
    time_frame = st.sidebar.selectbox(
        "Historical Window", 
        ["1 Month", "3 Months", "6 Months", "1 Year", "5 Years"], 
        index=3
    )

    period_days_map = {
        "1 Month": 30, 
        "3 Months": 90, 
        "6 Months": 180, 
        "1 Year": 365, 
        "5 Years": 365 * 5
    }
    days = period_days_map[time_frame]
    start_date = datetime.now() - timedelta(days=days)
    end_date = datetime.now()

    # Load instrument and domain data
    try:
        instrument = provider.get_instrument(selected_symbol)
        
        # Sidebar force refresh option
        if st.sidebar.button("🔄 Refresh Instrument Data", use_container_width=True):
            instrument.refresh_data()
            st.cache_data.clear()
            st.toast(f"Data refreshed for {selected_symbol}!", icon="✅")
            st.rerun()

        basic_info = instrument.get_basic_info()
        market_data = instrument.get_current_market_data()
        financial_metrics = instrument.get_financial_metrics()
        financial_health = instrument.get_financial_health()
    except Exception as e:
        st.error(f"Failed to load data for {selected_symbol}: {e}")
        st.stop()

    st.title(f"🔍 {basic_info.get('long_name') or selected_symbol}")
    
    st.caption(
        f"**Ticker:** `{selected_symbol}` | "
        f"**Sector:** {basic_info.get('sector', 'N/A')} | "
        f"**Industry:** {basic_info.get('industry', 'N/A')} | "
        f"**Currency:** {basic_info.get('currency', 'USD')}"
    )

    st.markdown("---")

    # Top KPI Bar
    c1, c2, c3, c4, c5 = st.columns(5)
    
    curr_price = market_data.get("current_price") or 0.0
    prev_close = market_data.get("previous_close") or curr_price
    price_change = curr_price - prev_close
    pct_change = (price_change / prev_close * 100) if prev_close else 0.0

    c1.metric(
        "Current Price", 
        f"{curr_price:,.2f} {basic_info.get('currency', '')}", 
        delta=f"{pct_change:+.2f}%"
    )
    c2.metric(
        "Day Range", 
        f"{market_data.get('day_low', 0):,.2f} - {market_data.get('day_high', 0):,.2f}"
    )
    c3.metric(
        "52-Wk Range", 
        f"{market_data.get('fifty_two_week_low', 0):,.2f} - {market_data.get('fifty_two_week_high', 0):,.2f}"
    )
    
    mcap = market_data.get("market_cap")
    if mcap and mcap >= 1e9:
        mcap_str = f"${mcap/1e9:,.2f}B"
    elif mcap and mcap >= 1e6:
        mcap_str = f"${mcap/1e6:,.2f}M"
    else:
        mcap_str = "N/A"
    c4.metric("Market Cap", mcap_str)
    
    pe_ratio = financial_metrics.get("trailing_pe")
    c5.metric("Trailing P/E", f"{pe_ratio:.2f}" if pe_ratio else "N/A")

    st.markdown("---")

    tab_chart, tab_advisor, tab_fundamentals, tab_statements, tab_news = st.tabs([
        "📈 Price & Volume", 
        "🤖 AI Advisor",
        "📊 Valuation & Health", 
        "📜 Statements", 
        "📰 News"
    ])

    with tab_chart:
        try:
            hist_df = instrument.get_historical_market_data(start=start_date, end=end_date)
            if not hist_df.empty:
                c_type, _ = st.columns([1, 3])
                with c_type:
                    chart_type = st.radio("Chart View", ["Line Chart", "Candlestick"], horizontal=True)

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
                    fig = px.line(
                        hist_df, 
                        x=hist_df.index, 
                        y="Close", 
                        title=f"{selected_symbol} Price History ({time_frame})"
                    )
                
                fig.update_layout(
                    height=420, 
                    template="plotly_dark", 
                    margin=dict(l=10, r=10, t=35, b=10),
                    xaxis_title="Date",
                    yaxis_title=f"Price ({basic_info.get('currency', 'USD')})"
                )
                st.plotly_chart(fig, use_container_width=True)

                fig_vol = px.bar(hist_df, x=hist_df.index, y="Volume", title="Trading Volume")
                fig_vol.update_traces(marker_color="#29B6F6")
                fig_vol.update_layout(
                    height=180, 
                    template="plotly_dark", 
                    margin=dict(l=10, r=10, t=30, b=10),
                    xaxis_title="",
                    yaxis_title="Volume"
                )
                st.plotly_chart(fig_vol, use_container_width=True)
            else:
                st.info("No historical price data returned for this window.")
        except Exception as e:
            st.error(f"Error rendering price chart: {e}")

    with tab_advisor:
        st.markdown(f"### 🤖 AI Investment Analysis")
        st.caption(f"Multi-agent synthesis for **{basic_info.get('long_name', selected_symbol)}** (`{selected_symbol}`)")

        # Fetch advisor instance safely
        advisor = getattr(provider, "advisor", None) or st.session_state.get("advisor")
        cache_key = f"advisor_res_{selected_symbol}"
        cached_result = st.session_state.get(cache_key)

        # Overview banner container
        with st.container(border=True):
            col_info, col_actions = st.columns([3, 1], gap="medium")
            
            with col_info:
                st.markdown("**How this works:**")
                st.markdown(
                    "This system runs parallel evaluation models (**Financial Health** & **Valuation Metrics**) "
                    "and feeds their findings into a General Synthesis Analyst to produce an unbiased recommendation."
                )

            with col_actions:
                st.markdown(" ")
                run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
                if cached_result:
                    if st.button("🔄 Clear Result", use_container_width=True):
                        del st.session_state[cache_key]
                        st.rerun()

        # Handle Execution
        if run_analysis:
            if not advisor:
                st.error("⚠️ AI Advisor service is not initialized on the provider or session state.")
            else:
                with st.status(f"Analyzing {selected_symbol}...", expanded=True) as status:
                    st.write("🔍 Extracting financial health & balance sheet context...")
                    st.write("📊 Evaluating fundamental valuation metrics & ratios...")
                    st.write("🧠 Synthesizing sub-analyst decisions into final recommendation...")
                    try:
                        result = advisor.analyze_instrument(instrument)
                        st.session_state[cache_key] = result
                        status.update(label="Analysis complete!", state="complete", expanded=False)
                        st.rerun()
                    except Exception as e:
                        status.update(label="Analysis failed!", state="error", expanded=True)
                        st.error(f"Error executing advisor: {str(e)}")

        # Render Recommendation & Reasoning Output
        cached_result = st.session_state.get(cache_key)
        if cached_result:
            st.markdown("---")
            decision = str(cached_result.get("decision", "HOLD")).upper()
            reasoning_points = cached_result.get("reasoning", [])

            # Clean provider class name (e.g., AzureModelProvider)
            if hasattr(advisor, '_model_provider'):
                provider_name = advisor._model_provider.__class__.__name__
            else:
                provider_name = "LLM Synthesis Engine"

            # Redesigned Decision & Summary Hero Card
            with st.container(border=True):
                col_dec, col_meta = st.columns([1, 2], gap="large")
                
                with col_dec:
                    st.caption("SYNTHESIS DECISION")
                    if decision == "BUY":
                        st.markdown(
                            """<div style="background-color: #1b382b; border: 1px solid #2e6f40; border-radius: 8px; padding: 12px; text-align: center;">
                                <h2 style="color: #4CAF50; margin: 0; padding: 0;">🟢 BUY</h2>
                            </div>""", 
                            unsafe_allow_html=True
                        )
                    elif decision == "SELL":
                        st.markdown(
                            """<div style="background-color: #3b1c1c; border: 1px solid #7a2b2b; border-radius: 8px; padding: 12px; text-align: center;">
                                <h2 style="color: #FF5252; margin: 0; padding: 0;">🔴 SELL</h2>
                            </div>""", 
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            """<div style="background-color: #38321b; border: 1px solid #6f622e; border-radius: 8px; padding: 12px; text-align: center;">
                                <h2 style="color: #FFC107; margin: 0; padding: 0;">🟡 HOLD</h2>
                            </div>""", 
                            unsafe_allow_html=True
                        )

                with col_meta:
                    st.caption("EVALUATION METADATA")
                    st.markdown(
                        f"• Target Symbol: **`{selected_symbol}`**  \n"
                        f"• Key Findings: **{len(reasoning_points)} Points Identified**  \n"
                        f"• Engine: **`{provider_name}`**"
                    )

            st.markdown("### 💡 Decision Rationale")

            # Cleaned Bullet List Output (No Argument labels)
            if reasoning_points:
                with st.container(border=True):
                    for point in reasoning_points:
                        # Fix formatting issue: Escape raw underscores or single asterisks from LLM outputs
                        safe_point = point.replace("_", r"\_")
                        
                        st.markdown(f"• {safe_point}")
                        st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)
            else:
                st.info("No detailed reasoning delivered by the model.")

    with tab_fundamentals:
        col_val, col_health = st.columns(2, gap="large")
        
        with col_val:
            st.markdown("### 📊 Valuation Ratios")
            val_df = pd.DataFrame([
                {"Metric": "Trailing P/E", "Value": f"{financial_metrics.get('trailing_pe'):.2f}" if financial_metrics.get('trailing_pe') else "N/A"},
                {"Metric": "Forward P/E", "Value": f"{financial_metrics.get('forward_pe'):.2f}" if financial_metrics.get('forward_pe') else "N/A"},
                {"Metric": "PEG Ratio", "Value": f"{financial_metrics.get('peg_ratio'):.2f}" if financial_metrics.get('peg_ratio') else "N/A"},
                {"Metric": "Price-to-Book", "Value": f"{financial_metrics.get('price_to_book'):.2f}" if financial_metrics.get('price_to_book') else "N/A"},
                {"Metric": "Dividend Yield", "Value": f"{financial_metrics.get('dividend_yield', 0) * 100:.2f}%" if financial_metrics.get('dividend_yield') else "N/A"},
                {"Metric": "Beta (Volatility)", "Value": f"{financial_metrics.get('beta'):.2f}" if financial_metrics.get('beta') else "N/A"}
            ])
            st.dataframe(val_df, hide_index=True, use_container_width=True)

        with col_health:
            st.markdown("### 🏥 Financial Health")
            
            def format_num(val):
                if not val or pd.isna(val): return "N/A"
                if abs(val) >= 1e9: return f"${val/1e9:,.2f}B"
                if abs(val) >= 1e6: return f"${val/1e6:,.2f}M"
                return f"${val:,.2f}"

            health_df = pd.DataFrame([
                {"Metric": "Total Revenue", "Value": format_num(financial_health.get("total_revenue"))},
                {"Metric": "Revenue Growth (YoY)", "Value": f"{financial_health.get('revenue_growth', 0) * 100:.2f}%" if financial_health.get('revenue_growth') else "N/A"},
                {"Metric": "EBITDA", "Value": format_num(financial_health.get("ebitda"))},
                {"Metric": "Profit Margin", "Value": f"{financial_health.get('profit_margin', 0) * 100:.2f}%" if financial_health.get('profit_margin') else "N/A"},
                {"Metric": "Total Debt", "Value": format_num(financial_health.get("total_debt"))},
                {"Metric": "Quick Ratio", "Value": f"{financial_health.get('quick_ratio'):.2f}" if financial_health.get('quick_ratio') else "N/A"},
                {"Metric": "ROE (Return on Equity)", "Value": f"{financial_health.get('return_on_equity', 0) * 100:.2f}%" if financial_health.get('return_on_equity') else "N/A"}
            ])
            st.dataframe(health_df, hide_index=True, use_container_width=True)

        if basic_info.get("summary"):
            st.markdown("---")
            st.markdown("### 🏢 Business Profile")
            st.info(basic_info.get("summary"))

    # TAB 3: FINANCIAL STATEMENTS
    with tab_statements:
        st.markdown("### 📜 Annual Financial Statements")
        stmt_choice = st.selectbox(
            "Statement Type", 
            ["Income Statement", "Balance Sheet", "Cash Flow"]
        )
        
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
                df_stmt.columns = [
                    col.strftime("%Y-%m-%d") if hasattr(col, "strftime") else str(col) 
                    for col in df_stmt.columns
                ]
                st.dataframe(df_stmt, use_container_width=True)
            else:
                st.info(f"No {stmt_choice} data available for {selected_symbol}.")
        except Exception as e:
            st.error(f"Error fetching financial statements: {e}")

    with tab_news:
        st.markdown(f"### 📰 Latest News for {selected_symbol}")
        if st.button("📰 Fetch / Refresh News"):
            with st.spinner("Scraping and parsing articles..."):
                news_items = instrument.get_news(max_workers=3)
                if news_items:
                    for item in news_items[:5]:
                        with st.expander(f"📌 {item.get('title', 'Untitled')} ({item.get('source', 'Unknown Source')})"):
                            st.caption(f"Published: {item.get('date', 'N/A')}")
                            st.write(item.get('content') or "No body content parsed.")
                else:
                    st.info("No news articles found.")


elif page == "➕ Create Portfolio":
    st.title("➕ Create New Portfolio")

    available_symbols = provider.instrument_symbols

    if "draft_positions" not in st.session_state:
        st.session_state.draft_positions = []

    c_meta1, c_meta2 = st.columns([2, 1])

    with c_meta1:
        portfolio_name_input = st.text_input(
            "Portfolio Name", 
            placeholder="e.g. Growth & Tech Portfolio 2026",
            key="p_name_input"
        )

    if "portfolio_base_currency" not in st.session_state:
        st.session_state.portfolio_base_currency = "PLN"

    def sync_inputs_on_base_currency_change():
        selected_symbol = st.session_state.get("pos_symbol_select", available_symbols[0])
        target_ccy = st.session_state.portfolio_base_currency
        converter = CurrencyConverter(target_ccy)
        
        try:
            inst = provider.get_instrument(selected_symbol)
            native_price = float(inst.current_price) if inst and inst.current_price else 0.0
            native_ccy = inst.get_basic_info().get('currency', 'PLN') if inst else 'PLN'
        except Exception:
            native_price = 0.0
            native_ccy = 'PLN'

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
                            new_portfolio = Portfolio(
                                name=portfolio_name_input.strip(), 
                                currency=base_currency
                            )

                            for pos in st.session_state.draft_positions:
                                inst = provider.get_instrument(pos["symbol"])
                                
                                native_currency = inst.get_basic_info().get('currency', 'USD') if inst else 'USD'
                                native_converter = CurrencyConverter(native_currency)

                                native_buy_price = native_converter.convert(pos["buy_price"], base_currency)

                                asset = Asset(
                                    instrument=inst, 
                                    volume=pos["volume"], 
                                    buy_price=native_buy_price,
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