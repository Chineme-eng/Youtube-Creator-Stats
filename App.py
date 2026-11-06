"""
Creator Scout: YouTube creator discovery + affiliate outreach tool.

Finds mid-size YouTube channels in a niche, scores audience loyalty,
detects affiliate-friendly creators with public emails, and drafts
commission-only partnership emails with Google Gemini.

Run:  streamlit run app.py
"""

from __future__ import annotations

import html
import os
import re
import statistics
import urllib.parse
from typing import Any

import google.generativeai as genai
import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

MIN_SUBS = 1_000
MAX_SUBS = 500_000
SWEET_SPOT_MIN = 10_000
SWEET_SPOT_MAX = 90_000
SEARCH_RESULTS = 50
RECENT_VIDEOS = 5
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = (
    "You are pitching a pure affiliate/commission deal to a creator. "
    "Emphasize that because they have a highly engaged and loyal audience, "
    "they will make more money taking a 30-40% cut of the profits rather than "
    "a flat upfront fee. Mention their channel name, latest video title, the "
    "product, and the exact dollar amount per sale.\n\n"
    "Formatting rules: output a 'Subject:' line, a blank line, then the email "
    "body. Keep it under 180 words, friendly and specific, with one clear call "
    "to action. No placeholders in brackets except the sender's name if none "
    "is provided. Do not invent facts about the creator beyond what is given."
)

# Affiliate "footprints": label -> regex. Labels are shown in the UI.
AFFILIATE_PATTERNS: dict[str, re.Pattern[str]] = {
    "Amazon Storefront": re.compile(r"amazon\.com/shop", re.I),
    "Amazon short link": re.compile(r"\bamzn\.to/", re.I),
    "Promo code": re.compile(
        r"\b(promo|coupon|discount|referral)\s*code\b|\buse\s+code\b", re.I
    ),
    "Affiliate": re.compile(r"\baffiliate(s)?\b", re.I),
    "Linktree": re.compile(r"linktr\.ee", re.I),
    "Discount": re.compile(r"\b\d{1,2}\s?%\s*off\b|\bdiscount\b", re.I),
    "Affiliate network": re.compile(
        r"shareasale|impact\.com|partnerstack|awin|cj\.com|clickbank", re.I
    ),
    "Disclosure": re.compile(r"(#ad\b|#sponsored|paid partnership|commission)", re.I),
}

EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}")
# Filter out obvious false positives such as image filenames (logo@2x.png).
EMAIL_BLOCKLIST_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg")

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def get_default_secret(name: str) -> str:
    """Read a default from Streamlit secrets or environment variables."""
    try:
        value = st.secrets[name]
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, "")


def fmt_int(n: float) -> str:
    n = int(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def find_affiliate_footprints(*texts: str) -> list[str]:
    """Return labels of affiliate footprints found in any of the given texts."""
    blob = "\n".join(t for t in texts if t)
    return [label for label, pat in AFFILIATE_PATTERNS.items() if pat.search(blob)]


def find_emails(*texts: str) -> list[str]:
    """Return unique public email addresses found in the given texts."""
    blob = "\n".join(t for t in texts if t)
    found: list[str] = []
    for match in EMAIL_PATTERN.findall(blob):
        email = match.strip(".").lower()
        if email.endswith(EMAIL_BLOCKLIST_SUFFIXES) or email in found:
            continue
        found.append(email)
    return found


def chunked(items: list[Any], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


# --------------------------------------------------------------------------- #
# YouTube data layer
# --------------------------------------------------------------------------- #


def discover_creators(api_key: str, query: str, progress) -> list[dict[str, Any]]:
    """
    Full discovery pipeline:
      1. search.list           -> up to 50 channel IDs        (100 quota units)
      2. channels.list         -> exact subscriber counts     (1 unit)
      3. filter by subscribers
      4. playlistItems.list    -> 5 latest uploads / channel  (1 unit each)
      5. videos.list           -> view counts, batched        (1 unit per 50)
    """
    yt = build("youtube", "v3", developerKey=api_key, cache_discovery=False)

    progress(0.05, "Searching YouTube for channels...")
    search = (
        yt.search()
        .list(q=query, part="snippet", type="channel", maxResults=SEARCH_RESULTS)
        .execute()
    )
    channel_ids = [item["id"]["channelId"] for item in search.get("items", [])]
    if not channel_ids:
        return []

    progress(0.2, "Fetching subscriber counts...")
    channels = (
        yt.channels()
        .list(part="snippet,statistics,contentDetails", id=",".join(channel_ids))
        .execute()
        .get("items", [])
    )

    # Strict subscriber filter. Hidden counts cannot be verified, so drop them.
    kept = []
    for ch in channels:
        stats = ch.get("statistics", {})
        if stats.get("hiddenSubscriberCount") or "subscriberCount" not in stats:
            continue
        subs = int(stats["subscriberCount"])
        if MIN_SUBS <= subs <= MAX_SUBS:
            kept.append((ch, subs))
    if not kept:
        return []

    # Recent uploads per channel (playlistItems is 1 unit; search.list is 100).
    creators: list[dict[str, Any]] = []
    all_video_ids: list[str] = []
    for i, (ch, subs) in enumerate(kept):
        progress(0.3 + 0.5 * (i / len(kept)), f"Reading recent videos ({i + 1}/{len(kept)})...")
        uploads = ch["contentDetails"]["relatedPlaylists"].get("uploads")
        videos: list[dict[str, Any]] = []
        if uploads:
            try:
                items = (
                    yt.playlistItems()
                    .list(part="snippet,contentDetails", playlistId=uploads, maxResults=RECENT_VIDEOS)
                    .execute()
                    .get("items", [])
                )
            except HttpError:
                items = []
            for it in items:
                vid = it["contentDetails"]["videoId"]
                videos.append(
                    {
                        "id": vid,
                        "title": it["snippet"].get("title", ""),
                        "description": it["snippet"].get("description", ""),
                        "published": it["snippet"].get("publishedAt", ""),
                        "views": None,
                    }
                )
                all_video_ids.append(vid)

        snippet = ch["snippet"]
        creators.append(
            {
                "id": ch["id"],
                "title": snippet.get("title", "Unknown"),
                "about": snippet.get("description", ""),
                "thumbnail": snippet.get("thumbnails", {}).get("default", {}).get("url"),
                "subs": subs,
                "url": f"https://www.youtube.com/channel/{ch['id']}",
                "videos": videos,
            }
        )

    # Batch video statistics (50 IDs per call).
    progress(0.85, "Calculating audience loyalty...")
    views_by_id: dict[str, int] = {}
    for batch in chunked(all_video_ids, 50):
        resp = yt.videos().list(part="statistics", id=",".join(batch)).execute()
        for v in resp.get("items", []):
            views_by_id[v["id"]] = int(v["statistics"].get("viewCount", 0))

    for c in creators:
        for v in c["videos"]:
            v["views"] = views_by_id.get(v["id"])
        counted = [v["views"] for v in c["videos"] if v["views"] is not None]
        c["avg_views"] = statistics.fmean(counted) if counted else 0
        c["watch_pct"] = (c["avg_views"] / c["subs"] * 100) if c["subs"] else 0
        c["latest_title"] = c["videos"][0]["title"] if c["videos"] else ""

        video_text = [v["description"] for v in c["videos"]]
        c["footprints"] = find_affiliate_footprints(c["about"], *video_text)
        c["emails"] = find_emails(c["about"], *video_text)
        c["sweet_spot"] = SWEET_SPOT_MIN <= c["subs"] <= SWEET_SPOT_MAX
        c["affiliate_match"] = bool(c["emails"] and c["footprints"])

    progress(1.0, "Done")
    return creators


# --------------------------------------------------------------------------- #
# Gemini outreach
# --------------------------------------------------------------------------- #


def generate_outreach(
    api_key: str,
    model_name: str,
    creator: dict[str, Any],
    product: dict[str, Any],
    sender_name: str,
) -> str:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name=model_name, system_instruction=SYSTEM_PROMPT)

    user_prompt = f"""Write the outreach email.

Creator channel name: {creator['title']}
Creator's latest video title: {creator['latest_title'] or '(unknown)'}
Creator subscribers: {creator['subs']:,}
Average views on recent videos: {int(creator['avg_views']):,} ({creator['watch_pct']:.1f}% of subscribers watch each upload)

Product name: {product['name']}
Product description: {product['description']}
Product price: ${product['price']:,.2f}
Affiliate commission: {product['commission']:g}%
Exact payout per sale: ${product['payout']:,.2f}

Sender name: {sender_name or 'the product founder'}
"""
    response = model.generate_content(user_prompt)
    return response.text.strip()


def split_subject_body(text: str) -> tuple[str, str]:
    match = re.match(r"\s*subject:\s*(.+?)\n+(.*)", text, re.I | re.S)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "Affiliate partnership", text


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.badge {display:inline-block; padding: 2px 10px; margin: 0 6px 6px 0;
        border-radius: 999px; font-size: 0.75rem; font-weight: 600;}
.badge-sweet {background:#fff4d6; color:#8a5a00; border:1px solid #f0c75e;}
.badge-match {background:#dcf5e4; color:#146c3a; border:1px solid #7fd29f;}
.badge-muted {background:#eef0f4; color:#4a5568; border:1px solid #d5d9e0;}
.creator-name {font-size: 1.15rem; font-weight: 700; margin-bottom: 4px;}
.creator-name a {text-decoration: none;}
.subtle {color: #6b7280; font-size: 0.85rem;}
</style>
"""


def badge(text: str, kind: str) -> str:
    return f'<span class="badge badge-{kind}">{html.escape(text)}</span>'


def render_creator_card(c: dict[str, Any], product: dict[str, Any], keys: dict[str, str]) -> None:
    with st.container(border=True):
        col_img, col_info, col_stats = st.columns([1, 4, 3], vertical_alignment="center")

        with col_img:
            if c["thumbnail"]:
                st.image(c["thumbnail"], width=88)

        with col_info:
            st.markdown(
                f'<div class="creator-name"><a href="{c["url"]}" target="_blank">'
                f'{html.escape(c["title"])}</a></div>',
                unsafe_allow_html=True,
            )
            badges = ""
            if c["sweet_spot"]:
                badges += badge("⭐ Sweet Spot", "sweet")
            if c["affiliate_match"]:
                badges += badge("✅ Affiliate Match", "match")
            elif c["footprints"]:
                badges += badge("Affiliate signals", "muted")
            if c["emails"] and not c["affiliate_match"]:
                badges += badge("Email found", "muted")
            if badges:
                st.markdown(badges, unsafe_allow_html=True)
            if c["latest_title"]:
                st.markdown(
                    f'<div class="subtle">Latest: {html.escape(c["latest_title"])}</div>',
                    unsafe_allow_html=True,
                )

        with col_stats:
            m1, m2, m3 = st.columns(3)
            m1.metric("Subscribers", fmt_int(c["subs"]))
            m2.metric("Avg views", fmt_int(c["avg_views"]))
            m3.metric("Watch %", f"{c['watch_pct']:.1f}%")

        with st.expander("Contact & affiliate signals"):
            st.write("**Emails:** " + (", ".join(c["emails"]) if c["emails"] else "None found"))
            st.write(
                "**Footprints:** " + (", ".join(c["footprints"]) if c["footprints"] else "None found")
            )

        if st.button("✉️ Generate Outreach", key=f"gen_{c['id']}"):
            if not keys["gemini"]:
                st.error("Add your Gemini API key in the sidebar.")
            elif not product["name"]:
                st.error("Add a product name in the sidebar.")
            else:
                with st.spinner("Writing email..."):
                    try:
                        st.session_state.emails[c["id"]] = generate_outreach(
                            keys["gemini"], keys["model"], c, product, keys["sender"]
                        )
                    except Exception as exc:  # Gemini SDK raises various error types
                        st.error(f"Gemini request failed: {exc}")

        draft = st.session_state.emails.get(c["id"])
        if draft:
            edited = st.text_area(
                "Draft (editable)", value=draft, height=260, key=f"draft_{c['id']}"
            )
            subject, body = split_subject_body(edited)
            if c["emails"]:
                mailto = (
                    f"mailto:{c['emails'][0]}?subject={urllib.parse.quote(subject)}"
                    f"&body={urllib.parse.quote(body)}"
                )
                st.link_button("Open in email app", mailto)
            st.download_button(
                "Download .txt", edited, file_name=f"outreach_{c['id']}.txt", key=f"dl_{c['id']}"
            )


def main() -> None:
    st.set_page_config(page_title="Creator Scout", page_icon="🎯", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    st.session_state.setdefault("creators", None)
    st.session_state.setdefault("emails", {})

    # ---- Sidebar ---------------------------------------------------------- #
    with st.sidebar:
        st.header("🔑 API Keys")
        yt_key = st.text_input(
            "YouTube Data API v3 key",
            value=get_default_secret("YOUTUBE_API_KEY"),
            type="password",
        )
        gemini_key = st.text_input(
            "Google Gemini API key",
            value=get_default_secret("GEMINI_API_KEY"),
            type="password",
        )
        gemini_model = st.text_input("Gemini model", value=DEFAULT_GEMINI_MODEL)

        st.header("📦 Your Product")
        product_name = st.text_input("Product name")
        product_desc = st.text_area("Description", height=100)
        price = st.number_input("Price ($)", min_value=0.0, value=49.0, step=1.0, format="%.2f")
        commission = st.number_input(
            "Affiliate commission (%)", min_value=0.0, max_value=100.0, value=30.0, step=5.0
        )
        payout = round(price * commission / 100, 2)
        st.metric("Payout per sale", f"${payout:,.2f}")
        sender_name = st.text_input("Your name (email sign-off)")

    product = {
        "name": product_name.strip(),
        "description": product_desc.strip(),
        "price": price,
        "commission": commission,
        "payout": payout,
    }
    keys = {"gemini": gemini_key.strip(), "model": gemini_model.strip(), "sender": sender_name.strip()}

    # ---- Header + search -------------------------------------------------- #
    st.title("🎯 Creator Scout")
    st.caption(
        f"Find {fmt_int(MIN_SUBS)}-{fmt_int(MAX_SUBS)} subscriber creators with loyal audiences "
        "and affiliate-friendly channels."
    )

    with st.form("search_form", clear_on_submit=False):
        s1, s2 = st.columns([5, 1], vertical_alignment="bottom")
        query = s1.text_input("Niche or keywords", placeholder="e.g. notion productivity, home espresso")
        submitted = s2.form_submit_button("Search", use_container_width=True, type="primary")

    if submitted:
        if not yt_key.strip():
            st.error("Add your YouTube Data API key in the sidebar.")
        elif not query.strip():
            st.error("Enter a keyword or niche.")
        else:
            bar = st.progress(0.0, text="Starting...")
            try:
                st.session_state.creators = discover_creators(
                    yt_key.strip(), query.strip(), lambda p, t: bar.progress(p, text=t)
                )
                st.session_state.emails = {}
            except HttpError as exc:
                reason = getattr(exc, "reason", str(exc))
                st.error(f"YouTube API error: {reason}. Check your key and daily quota.")
                st.session_state.creators = None
            finally:
                bar.empty()

    creators = st.session_state.creators
    if creators is None:
        st.info("Enter a niche above to start discovering creators.")
        return
    if not creators:
        st.warning("No channels matched the subscriber filter. Try broader keywords.")
        return

    # ---- Results controls ------------------------------------------------- #
    f1, f2, f3 = st.columns([2, 2, 2], vertical_alignment="center")
    sort_by = f1.selectbox(
        "Sort by", ["Watch % (high to low)", "Subscribers (high to low)", "Subscribers (low to high)"]
    )
    only_sweet = f2.checkbox("⭐ Sweet Spot only")
    only_match = f3.checkbox("✅ Affiliate Match only")

    shown = [c for c in creators if (not only_sweet or c["sweet_spot"]) and (not only_match or c["affiliate_match"])]
    sorters = {
        "Watch % (high to low)": (lambda c: c["watch_pct"], True),
        "Subscribers (high to low)": (lambda c: c["subs"], True),
        "Subscribers (low to high)": (lambda c: c["subs"], False),
    }
    key_fn, reverse = sorters[sort_by]
    shown.sort(key=key_fn, reverse=reverse)

    st.markdown(
        f"**{len(shown)}** of {len(creators)} creators shown · "
        f"**{sum(c['sweet_spot'] for c in creators)}** Sweet Spot · "
        f"**{sum(c['affiliate_match'] for c in creators)}** Affiliate Match"
    )

    for c in shown:
        render_creator_card(c, product, keys)


if __name__ == "__main__":
    main()
