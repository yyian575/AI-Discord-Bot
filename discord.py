import discord
import os
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from discord.ext import commands
from discord.ui import View, Modal, TextInput
from dotenv import load_dotenv
import io
import asyncio
import logging
import google.generativeai as genai

# 1. 環境設定與機器人初始化
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
GEMINI_KEY = os.getenv('GEMINI_API_KEY')
logger = logging.getLogger(__name__)

# 初始化 Gemini
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
    # 使用 gemini-1.5-flash 作為預設模型 (目前尚無 2.5 版本，建議使用 1.5 或 2.0)
    model = genai.GenerativeModel('gemini-2.5-flash')
else:
    model = None

intents = discord.Intents.default()
intents.message_content = True 
bot = commands.Bot(command_prefix='!', intents=intents)

# 設定繪圖風格為 Dark Mode
plt.style.use('dark_background')
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial'] 
plt.rcParams['axes.unicode_minus'] = False

# 2. 輔助函數
def calculate_rsi(df, period=14):
    """計算 RSI 指標"""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_volume_ratio(df, period=20):
    """計算最新成交量相對於近期平均量的倍數。"""
    if len(df) < period:
        return None
    avg_volume = df['Volume'].rolling(window=period).mean().iloc[-1]
    if avg_volume == 0:
        return None
    return df['Volume'].iloc[-1] / avg_volume

def calculate_macd_status(df):
    """以 MACD 與 signal 線判斷目前偏多或偏空。"""
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    if macd.iloc[-1] >= signal.iloc[-1]:
        return "偏多"
    return "偏空"

def generate_stock_dashboard(symbol):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="6mo")
    if df.empty or len(df) < 26:
        return None
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    price_now = curr['Close']
    change_pct = ((price_now - prev['Close']) / prev['Close']) * 100
    rsi_val = calculate_rsi(df).iloc[-1]
    volume_ratio = calculate_volume_ratio(df)
    volume_ratio_text = f"{volume_ratio:.1f}x" if volume_ratio else "N/A"
    macd_status = calculate_macd_status(df)
    fig = plt.figure(figsize=(12, 10), facecolor='#121212')
    gs = gridspec.GridSpec(10, 6, figure=fig)
    cards = [
        ("現價", f"{price_now:.2f}", "white"),
        ("今日", f"{change_pct:+.2f}%", "#ef5350" if change_pct < 0 else "#26a69a"),
        ("RSI", f"{rsi_val:.1f}", "white"),
        ("MA5/20", f"{curr['MA5']:.0f} / {curr['MA20']:.0f}", "white"),
        ("量能", volume_ratio_text, "white"),
        ("MACD", macd_status, "#ffaa00")
    ]
    ax_t = fig.add_subplot(gs[0, :])
    ax_t.axis('off')
    ax_t.text(0, 0.5, f'📈 {symbol.upper()}', fontsize=24, color='white', fontweight='bold')
    for i, (label, val, col) in enumerate(cards):
        r, c = (i // 3) + 1, (i % 3) * 2
        ax_card = fig.add_subplot(gs[r, c:c+2])
        ax_card.set_facecolor('#1e1e1e')
        ax_card.set_xticks([]); ax_card.set_yticks([])
        ax_card.text(0.1, 0.65, label, color='#b0b0b0', fontsize=11)
        ax_card.text(0.1, 0.25, val, color=col, fontsize=15, fontweight='bold')
    ax1 = fig.add_subplot(gs[4:8, :])
    ax1.set_facecolor('#121212')
    ax1.plot(df.index, df['Close'], color='#00ffff', label='Close', linewidth=1.5)
    ax1.plot(df.index, df['MA5'], color='#ffff00', linestyle='--', label='MA5', alpha=0.7)
    ax1.plot(df.index, df['MA20'], color='#ff00ff', linestyle='--', label='MA20', alpha=0.7)
    ax1.fill_between(df.index, df['Close'], color='#00ffff', alpha=0.05)
    ax1.set_title(f"{symbol.upper()} 走勢圖", loc='left', color='#00ffff')
    ax1.grid(True, color='#333333', alpha=0.5)
    ax1.legend(loc='upper left', frameon=False)
    ax2 = fig.add_subplot(gs[8:10, :], sharex=ax1)
    ax2.set_facecolor('#121212')
    v_colors = ['#26a69a' if df['Close'].iloc[i] >= df['Open'].iloc[i] else '#ef5350' for i in range(len(df))]
    ax2.bar(df.index, df['Volume'], color=v_colors, alpha=0.7)
    ax2.grid(True, color='#333333', alpha=0.3)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='#121212')
    buf.seek(0)
    plt.close(fig)
    return buf

# 3. 互動介面組件
class AIScanModal(Modal):
    def __init__(self, cycle):
        super().__init__(title=f"AI {cycle}策略分析")
        self.cycle = cycle
        self.symbol_input = TextInput(
            label="股票代號",
            placeholder="例如: 2330.TW, AAPL",
            min_length=1,
            max_length=15
        )
        self.add_item(self.symbol_input)

    async def on_submit(self, interaction: discord.Interaction):
        if not model:
            await interaction.response.send_message("❌ 請先在 .env 中設定 `GEMINI_API_KEY`", ephemeral=True)
            return

        await interaction.response.defer()
        symbol = self.symbol_input.value.strip().upper()
        
        try:
            # 1. 抓取數據 (增加至 60d 以確保週末/連假後仍有足夠 20 日數據)
            ticker = yf.Ticker(symbol)
            df = await asyncio.get_event_loop().run_in_executor(None, lambda: ticker.history(period='60d'))
            
            if df.empty or len(df) < 20:
                await interaction.followup.send(f"❌ `{symbol}` 數據不足或代號錯誤 (需至少 20 日數據)", ephemeral=True)
                return

            # 2. 計算指標
            price = float(df['Close'].iloc[-1])
            ma5 = float(df['Close'].rolling(window=5).mean().iloc[-1])
            ma20 = float(df['Close'].rolling(window=20).mean().iloc[-1])
            rsi = float(calculate_rsi(df).iloc[-1])

            # 3. 撰寫 Prompt
            prompt = f"""你是一位華爾街量化交易專家。使用者欲進行【{self.cycle}】操作。
最新技術數據：收盤價={price:.2f}，5MA={ma5:.2f}，20MA={ma20:.2f}，14日RSI={rsi:.2f}。
只需要明確列出：
1. 建議進場點
2. 建議止盈點
3. 建議止損點
4. 勝率信心指數（請給出 0% ~ 100% 的具體數字，並說明原因）。"""

            # 4. 呼叫 Gemini (含錯誤攔截)
            try:
                response = await asyncio.get_event_loop().run_in_executor(None, lambda: model.generate_content(prompt))
                ai_text = response.text
            except (RuntimeError, ValueError, AttributeError) as e:
                logger.exception("Gemini API request failed")
                await interaction.followup.send("⚠️ 糟糕！目前伺服器太忙碌，或已達 Gemini 免費 API 呼叫上限，請稍後再試！")
                return

            # 5. 發送 Embed
            embed = discord.Embed(
                title=f"🤖 AI 【{self.cycle}】交易策略：{symbol}",
                description=ai_text,
                color=discord.Color.red()
            )
            embed.set_footer(text="⚠️ 備註：本系統使用 Gemini 免費版 API，僅供教學測試，非實質投資建議。")
            await interaction.followup.send(embed=embed)

        except (RuntimeError, ValueError, KeyError, IndexError) as e:
            logger.exception("AI stock analysis failed for %s", symbol)
            await interaction.followup.send(f"⚠️ 系統錯誤：{e}", ephemeral=True)

class AIStrategyView(View):
    def __init__(self):
        super().__init__(timeout=60)

    @discord.ui.button(label="短期策略 (日沖/週操作)", style=discord.ButtonStyle.primary, emoji="🔥")
    async def short_term(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(AIScanModal(cycle="短期"))

    @discord.ui.button(label="長期策略 (波段/價值投資)", style=discord.ButtonStyle.success, emoji="💎")
    async def long_term(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(AIScanModal(cycle="長期"))

class SymbolModal(Modal):
    def __init__(self, action_type):
        super().__init__(title="請輸入股票代號")
        self.action_type = action_type
        self.symbol_input = TextInput(label="股票代號", placeholder="例如: AAPL, 2330.TW", min_length=1, max_length=15)
        self.add_item(self.symbol_input)

    async def on_submit(self, interaction: discord.Interaction):
        symbol = self.symbol_input.value.strip().upper()
        await interaction.response.defer()
        try:
            ticker = yf.Ticker(symbol)
            if self.action_type == 'price':
                data = await asyncio.get_event_loop().run_in_executor(None, lambda: ticker.history(period='1d'))
                if data.empty:
                    await interaction.followup.send(f"❌ 找不到代號 `{symbol}`", ephemeral=True)
                    return
                price = data['Close'].iloc[-1]
                await interaction.followup.send(f"📈 `{symbol}` 的最新價格為：**{price:.2f}**")
            elif self.action_type == 'chart':
                image_buf = await asyncio.get_event_loop().run_in_executor(None, generate_stock_dashboard, symbol)
                if image_buf is None:
                    await interaction.followup.send(f"❌ 抓取不到 `{symbol}` 資料。", ephemeral=True)
                    return
                file = discord.File(image_buf, filename=f"{symbol}_dashboard.png")
                await interaction.followup.send(f"✅ {symbol} 儀表板已生成：", file=file)
        except (RuntimeError, ValueError, KeyError, IndexError) as e:
            logger.exception("Stock action failed for %s", symbol)
            await interaction.followup.send(f"⚠️ 錯誤：{e}", ephemeral=True)

class FinancialView(View):
    def __init__(self):
        super().__init__(timeout=None)
        self.tickers = ['2330.TW', '2317.TW', '2454.TW', '2382.TW', '2308.TW', '2881.TW', '2882.TW', '2412.TW', '1301.TW', '2002.TW', '2303.TW', '3711.TW', '2886.TW', '2891.TW', '2357.TW', '2324.TW', '3231.TW', '2356.TW', '2884.TW', '2603.TW']

    @discord.ui.button(label="查詢股價", style=discord.ButtonStyle.primary, emoji="💰", custom_id="btn_price")
    async def price_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(SymbolModal(action_type='price'))

    @discord.ui.button(label="生成圖表", style=discord.ButtonStyle.success, emoji="📊", custom_id="btn_chart")
    async def chart_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(SymbolModal(action_type='chart'))

    @discord.ui.button(label="⚡ 菁英機會掃描", style=discord.ButtonStyle.secondary, emoji="✨", custom_id="btn_scan")
    async def scan_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        results = []
        loop = asyncio.get_event_loop()
        for symbol in self.tickers:
            try:
                ticker = yf.Ticker(symbol)
                df = await loop.run_in_executor(None, ticker.history, '5d')
                if df.empty or len(df) < 5:
                    continue
                price_now = float(df['Close'].iloc[-1])
                price_prev = float(df['Close'].iloc[-2])
                change_pct = float(((price_now - price_prev) / price_prev) * 100)
                ma5 = float(df['Close'].rolling(window=5).mean().iloc[-1])
                if change_pct > 1.5 and price_now > ma5:
                    results.append({'symbol': symbol.split('.')[0], 'price': price_now, 'change': change_pct})
                await asyncio.sleep(0.1)
            except (RuntimeError, ValueError, KeyError, IndexError) as e:
                logger.warning("Skipping ticker %s during scan: %s", symbol, e)
                continue
        if not results:
            await interaction.followup.send("⚠️ **目前菁英股無明顯波動**")
            return
        embed = discord.Embed(title="⚡ 菁英機會掃描結果", color=discord.Color.gold())
        for item in results:
            indicator = "🚀 強勢噴發" if float(item['change']) > 3.0 else "📈 穩健向上"
            embed.add_field(name=f"{item['symbol']}", value=f"價: **{item['price']:.2f}** ({item['change']:+.2f}%) {indicator}", inline=True)
        await interaction.followup.send(embed=embed)

    @discord.ui.button(label="🤖 AI 策略建議", style=discord.ButtonStyle.danger, emoji="🤖", custom_id="btn_ai")
    async def ai_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_message("請選擇您的投資週期：", view=AIStrategyView(), ephemeral=True)

# 4. 機器人指令
@bot.event
async def on_ready():
    print(f'Bot已上線：{bot.user.name}')
    bot.add_view(FinancialView())

@bot.command()
async def start(ctx):
    embed = discord.Embed(title="📊 歡迎使用 AI 理財儀表板", description="請選擇下方功能開始分析。", color=discord.Color.blue())
    embed.add_field(name="💰 查詢股價", value="獲取最新市場價格。", inline=True)
    embed.add_field(name="📊 生成圖表", value="產出專業技術指標圖表。", inline=True)
    embed.add_field(name="✨ 菁英掃描", value="低延遲熱門股強勢篩選。", inline=True)
    embed.add_field(name="🤖 AI 策略", value="Gemini 專家級交易建議。", inline=True)
    embed.set_footer(text="⚠️ 投資有風險，入市需謹慎")
    await ctx.send(embed=embed, view=FinancialView())

if __name__ == "__main__":
    if TOKEN:
        bot.run(TOKEN)
    else:
        print("請在 .env 中設定 DISCORD_TOKEN")