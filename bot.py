import os
import math
import base64
import httpx
import json
import re
from groq import Groq
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

def poisson_pmf(k, lam):
    if lam <= 0: return 1.0 if k == 0 else 0.0
    log_p = -lam + k * math.log(lam)
    for i in range(1, k + 1): log_p -= math.log(i)
    return math.exp(log_p)

def run_poisson(lH, lA, max_goals=8):
    home = draw = away = over25 = btts = 0
    scores = []
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = poisson_pmf(h, lH) * poisson_pmf(a, lA)
            scores.append((h, a, p))
            if h > a: home += p
            elif h == a: draw += p
            else: away += p
            if h + a > 2: over25 += p
            if h > 0 and a > 0: btts += p
    top5 = sorted(scores, key=lambda x: -x[2])[:5]
    return {"home": home, "draw": draw, "away": away, "over25": over25, "under25": 1-over25, "btts": btts, "top5": top5}

LEAGUE_AVGS = {
    "premier league": (1.53, 1.15), "la liga": (1.44, 1.09),
    "serie a": (1.46, 1.11), "ligue 1": (1.40, 1.08),
    "bundesliga": (1.56, 1.18), "super lig": (1.50, 1.10),
    "super league": (1.38, 0.98),
}

def get_league_avg(league_name):
    for key, val in LEAGUE_AVGS.items():
        if key in league_name.lower(): return val
    return (1.45, 1.10)

def groq_ask(prompt):
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
        temperature=0.3,
    )
    return response.choices[0].message.content

def analyze_match(home_team, away_team, league="Premier League"):
    avg_home, avg_away = get_league_avg(league)

    prompt = f"""You are a football statistics expert. Based on your knowledge of the 2025-2026 season, provide stats for {home_team} (home) and {away_team} (away) in {league}.

Return ONLY this JSON, no markdown, no explanation:
{{"home_scored":1.5,"home_conceded":1.0,"away_scored":1.2,"away_conceded":1.4,"home_form":"WWDLW","away_form":"LDWWL","context":"key team news in one sentence"}}

Use realistic 2025-26 season averages. home_scored/home_conceded are for {home_team}'s home games. away_scored/away_conceded are for {away_team}'s away games."""

    stats_text = groq_ask(prompt)
    stats = {"home_scored": avg_home, "home_conceded": avg_away, "away_scored": avg_away, "away_conceded": avg_home, "home_form": "?????", "away_form": "?????", "context": ""}
    try:
        m = re.search(r'\{[^{}]*\}', stats_text, re.DOTALL)
        if m: stats.update(json.loads(m.group()))
    except: pass

    lH = max((stats["home_scored"]/avg_home) * (stats["away_conceded"]/avg_home) * avg_home, 0.3)
    lA = max((stats["away_scored"]/avg_away) * (stats["home_conceded"]/avg_away) * avg_away, 0.3)
    r = run_poisson(lH, lA)

    best_1x2 = max([("1", r["home"]), ("X", r["draw"]), ("2", r["away"])], key=lambda x: x[1])
    ou_label = "Over 2.5" if r["over25"] > 0.55 else "Under 2.5"
    ou_prob = r["over25"] if r["over25"] > 0.55 else r["under25"]
    top5_str = "".join(f"  {h}–{a}  {p*100:.1f}%\n" for h,a,p in r["top5"])
    ctx = f"\n📌 _{stats['context']}_\n" if stats.get("context") else ""

    return f"""⚽ *{home_team} vs {away_team}*
🏆 {league}{ctx}
📊 *Στατιστικά 2025/26:*
🏠 {home_team}: {stats['home_scored']:.2f} γκολ | δέχεται {stats['home_conceded']:.2f} | {stats['home_form']}
✈️ {away_team}: {stats['away_scored']:.2f} γκολ | δέχεται {stats['away_conceded']:.2f} | {stats['away_form']}

🎯 *xG: {home_team} {lH:.2f} — {away_team} {lA:.2f}*

📈 *Πιθανότητες Poisson:*
1️⃣ {home_team}: *{r['home']*100:.1f}%*
🤝 Ισοπαλία: *{r['draw']*100:.1f}%*
2️⃣ {away_team}: *{r['away']*100:.1f}%*
⚽ Over 2.5: *{r['over25']*100:.1f}%*
🔒 Under 2.5: *{r['under25']*100:.1f}%*
🔄 BTTS: *{r['btts']*100:.1f}%*

🏆 *Πιθανότερα Σκορ:*
{top5_str}
🔮 *Πρόβλεψη:*
▶️ Σημείο: *{best_1x2[0]}* ({best_1x2[1]*100:.1f}%)
▶️ Goals: *{ou_label}* ({ou_prob*100:.1f}%)
▶️ BTTS: *{'Ναι ✅' if r['btts']>0.52 else 'Όχι ❌'}* ({r['btts']*100:.1f}%)

⚠️ _Ανάλυση βάσει στατιστικών — δεν εγγυάται κέρδος._"""

def analyze_odds_image(image_bytes):
    # Groq vision
    image_b64 = base64.b64encode(image_bytes).decode()
    response = groq_client.chat.completions.create(
        model="llama-4-scout-17b-16e-instruct",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": """Αυτή είναι εικόνα με αποδόσεις ποδοσφαίρου από bookmaker.
Για κάθε αγώνα δώσε ανάλυση στα ελληνικά:
⚽ [Ομάδα Α] vs [Ομάδα Β]
📊 1=[X] | X=[X] | 2=[X]
💡 Implied: 1=[X]% | X=[X]% | 2=[X]%
🎯 Σύσταση: [ποιο σημείο και γιατί - 1 γραμμή]
⚽ Goals: [Over/Under σύσταση]
---"""}
            ]
        }],
        max_tokens=2000,
    )
    return response.choices[0].message.content

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⚽ *Football Value Analyzer*\n\nΣτείλε μου:\n📝 `Man City vs Newcastle`\n📝 `Juventus vs Como, Serie A`\n📸 Screenshot αποδόσεων\n\nPoisson Model + AI Analysis 🎯", parse_mode="Markdown")

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📖 *Οδηγίες:*\n\n`Ομάδα Α vs Ομάδα Β`\n`Ομάδα Α vs Ομάδα Β, League`\n📸 Φωτογραφία αποδόσεων\n\n⏱ ~10 δευτερόλεπτα ανά ανάλυση", parse_mode="Markdown")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    if text.startswith("/"): return
    await update.message.reply_text("🔍 Αναλύω... (~10 δευτ.)")
    try:
        league = "Premier League"
        if "," in text:
            parts = text.split(",", 1)
            match_part, league = parts[0].strip(), parts[1].strip()
        else:
            match_part = text
        if " vs " in match_part.lower():
            idx = match_part.lower().index(" vs ")
            home, away = match_part[:idx].strip(), match_part[idx+4:].strip()
        elif " - " in match_part:
            home, away = match_part.split(" - ", 1)[0].strip(), match_part.split(" - ", 1)[1].strip()
        else:
            await update.message.reply_text("❌ Στείλε π.χ.: `Man City vs Newcastle`", parse_mode="Markdown"); return
        await update.message.reply_text(analyze_match(home, away, league), parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Σφάλμα: {str(e)[:200]}")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📸 Αναλύω αποδόσεις... (~15 δευτ.)")
    try:
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        async with httpx.AsyncClient() as hc:
            image_bytes = (await hc.get(file.file_path)).content
        result = analyze_odds_image(image_bytes)
        for i in range(0, len(result), 4000):
            await update.message.reply_text(result[i:i+4000])
    except Exception as e:
        await update.message.reply_text(f"❌ Σφάλμα: {str(e)[:200]}")

def main():
    print("🤖 Football Analyzer Bot starting with Groq...")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    print("✅ Running!")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
