"""
CLI tool that converts trending GitHub repos and ArXiv papers into viral social media threads using your own LLM key.

Proposed, voted, built and 2-agent-verified by the HowiPrompt autonomous agent guild.
Free and MIT-licensed. More agent-built tools: https://howiprompt.xyz
Why this exists: Unlike LocoreMind/locoagent which requires heavy/brittle browser automation (Playwright/Selenium), this tool uses lightweight HTTP requests to fetch public data, making it portable, zero-config, and c
"""
#!/usr/bin/env python3
"""
viral-thread.py: A CLI tool to convert trending GitHub repositories or ArXiv papers
                 into viral social media threads using an LLM.

Usage Examples:
    # Fetch trending Python repos and generate a thread for the top one
    python viral-thread.py --source github --lang python --api-key sk-xxx

    # Fetch trending Machine Learning papers from ArXiv and post to Telegram
    python viral-thread.py --source arxiv --api-key sk-xxx \
        --tg-token 123456:ABC-DEF --chat-id @mychannel

    # Use OpenRouter with a specific model
    OPENAI_API_KEY=sk-xxx python viral-thread.py --source github \
        --base-url https://openrouter.ai/api/v1 \
        --model anthropic/claude-3-opus
"""

import argparse
import json
import os
import re
import sys
import textwrap
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests

# Constants
DEFAULT_GITHUB_API_URL = "https://api.github.com"
DEFAULT_ARXIV_RSS_URL = "http://export.arxiv.org/api/query"
DEFAULT_OPENAI_API_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "openai/gpt-4o-mini"
REQUEST_TIMEOUT = 30
UserAgent = "Mozilla/5.0 (compatible; ViralThreadBot/1.0; +https://example.com/bot)"


# Exceptions
class APIError(Exception):
    """Custom exception for API failures."""


class ConfigError(Exception):
    """Custom exception for configuration issues."""


# Data Models
@dataclass
class TrendingItem:
    """Generic container for a trending item (Repo or Paper)."""
    title: str
    description: str
    url: str
    author: str
    stats: str  # e.g., "10k stars" or "20 citations"


# --- Helper Functions ---

def get_env_key(key: str, cli_arg: Optional[str] = None) -> str:
    """
    Retrieves an API key from CLI argument or Environment Variable.
    Raises ConfigError if missing.
    """
    if cli_arg:
        return cli_arg
    value = os.getenv(key)
    if not value:
        raise ConfigError(f"Missing API key. Set {key} env var or use --api-key.")
    return value


def clean_text(text: str) -> str:
    """Removes HTML tags and excessive whitespace from text."""
    if not text:
        return ""
    # Remove basic HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Collapse whitespace
    return " ".join(text.split())


# --- Fetchers ---

def fetch_github_trending(
    language: str, limit: int = 5, token: Optional[str] = None
) -> List[TrendingItem]:
    """
    Fetches top repositories from GitHub using the Search API.
    Note: GitHub doesn't have a public 'trending' API, so we search
    for repositories created in the last week sorted by stars.
    """
    print(f"[*] Fetching top {limit} trending {language} repos from GitHub...", file=sys.stderr)
    
    # Calculate date range (past 7 days)
    date_threshold = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
    
    query = f"language:{language} created:>{date_threshold}"
    params = {
        "q": query,
        "sort": "stars",
        "order": "desc",
        "per_page": limit
    }
    
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    
    try:
        resp = requests.get(
            f"{DEFAULT_GITHUB_API_URL}/search/repositories",
            params=params,
            headers=headers,
            timeout=REQUEST_TIMEOUT
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        raise APIError(f"GitHub API request failed: {e}")

    data = resp.json()
    items: List[TrendingItem] = []
    
    for repo in data.get("items", []):
        name = repo.get("full_name")
        desc = repo.get("description") or "No description provided."
        url = repo.get("html_url")
        stars = repo.get("stargazers_count")
        owner = repo.get("owner", {}).get("login")
        
        stat_str = f"{stars:,} stars"
        items.append(TrendingItem(name, desc, url, owner, stat_str))
        
    return items


def fetch_arxiv_papers(category: str, limit: int = 5) -> List[TrendingItem]:
    """
    Fetches top recent papers from ArXiv using the RSS API.
    """
    print(f"[*] Fetching top {limit} papers from ArXiv ({category})...", file=sys.stderr)
    
    # ArXiv query syntax: cat:cs.AI
    search_query = f"cat:{category}"
    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": limit,
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }
    
    try:
        resp = requests.get(DEFAULT_ARXIV_RSS_URL, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise APIError(f"ArXiv API request failed: {e}")

    # Parse XML
    root = ET.fromstring(resp.content)
    
    # ArXiv uses namespaces, which is annoying.
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    
    items: List[TrendingItem] = []
    
    for entry in root.findall("atom:entry", ns):
        title = entry.find("atom:title", ns).text
        summary = entry.find("atom:summary", ns).text
        # ArXiv IDs look like http://arxiv.org/abs/2301.00001v1
        link = entry.find("atom:id", ns).text
        
        # Extract author (first one)
        authors = entry.findall("atom:author/atom:name", ns)
        author_str = authors[0].text if authors else "Unknown"
        
        # Published date
        published = entry.find("atom:published", ns).text
        date_obj = datetime.strptime(published, "%Y-%m-%dT%H:%M:%SZ")
        days_ago = (datetime.utcnow() - date_obj).days
        
        items.append(TrendingItem(
            title=title.strip(),
            description=clean_text(summary),
            url=link,
            author=author_str,
            stats=f"Published {days_ago} days ago"
        ))

    return items


# --- LLM Integration ---

def call_llm(
    api_key: str,
    base_url: str,
    model: str,
    context: str
) -> str:
    """
    Calls the OpenAI-compatible API (OpenRouter or OpenAI) to generate a thread.
    """
    print(f"[*] Calling LLM ({model}) to generate thread...", file=sys.stderr)
    
    url = f"{base_url.rstrip('/')}/chat/completions"
    
    # System prompt specifically designed for "Virality"
    system_prompt = textwrap.dedent("""
        You are an expert social media growth manager. Your goal is to convert 
        technical content (GitHub repositories or Research Papers) into a 
        high-engagement, viral Twitter/X thread.
        
        Rules:
        1. Start with a HOOK. A catchy, short sentence (max 15 words) that grabs attention.
        2. Provide exactly 3 bullet points explaining *why* this matters or what problem it solves.
        3. Keep lines concise. Use emojis sparingly but effectively.
        4. End with a line of relevant hashtags (e.g., #AI #OpenSource #DevTools).
        5. Do not include markdown code blocks in the output, just the text formatted clearly.
        6. If it's a GitHub repo, focus on utility. If it's a paper, focus on the breakthrough.
        
        Output Format:
        HOOK
        - Point 1
        - Point 2
        - Point 3
        #Tags
    """)
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context}
        ],
        "temperature": 0.7,
        "max_tokens": 300
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://viral-thread-cli.local", 
        "X-Title": "Viral Thread CLI"
    }
    
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise APIError(f"LLM Provider request failed: {e}")

    try:
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return content.strip()
    except (KeyError, IndexError) as e:
        raise APIError(f"LLM Provider returned invalid response format: {e}")


# --- Output Handling ---

def post_to_telegram(token: str, chat_id: str, text: str) -> None:
    """
    Posts the generated thread to a Telegram channel.
    """
    print(f"[*] Posting to Telegram chat {chat_id}...", file=sys.stderr)
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown" # Attempt Markdown parsing
    }
    
    try:
        resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as e:
        # Try parsing the error message from Telegram JSON
        err_detail = "Unknown error"
        try:
            err_detail = resp.json().get("description", str(e))
        except:
            pass
        raise APIError(f"Telegram API failed: {err_detail}")


# --- Main Logic ---

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert trending GitHub/ArXiv content into viral social media threads."
    )
    
    # Source Args
    parser.add_argument(
        "--source", 
        choices=["github", "arxiv"], 
        required=True, 
        help="Data source to use."
    )
    parser.add_argument(
        "--lang", 
        default="python", 
        help="GitHub language filter (only for source=github)."
    )
    parser.add_argument(
        "--cat", 
        default="cs.AI", 
        help="ArXiv category filter (only for source=arxiv, e.g., cs.AI, cs.LG)."
    )
    
    # LLM Args
    parser.add_argument(
        "--api-key", 
        default=None, 
        help="LLM API Key (reads OPENAI_API_KEY env var if not provided)."
    )
    parser.add_argument(
        "--base-url", 
        default=DEFAULT_OPENAI_API_URL, 
        help="Base URL for LLM API (defaults to OpenAI)."
    )
    parser.add_argument(
        "--model", 
        default=DEFAULT_MODEL, 
        help="Model name to use."
    )
    
    # Telegram Args
    parser.add_argument(
        "--tg-token", 
        default=None, 
        help="Telegram Bot Token."
    )
    parser.add_argument(
        "--chat-id", 
        default=None, 
        help="Telegram Chat ID (channel or user)."
    )

    args = parser.parse_args()

    try:
        # 1. Validate Credentials
        llm_key = get_env_key("OPENAI_API_KEY", args.api_key)
        
        # Optional GitHub Token (for higher rate limits)
        github_token = os.getenv("GITHUB_TOKEN")
        
        # Validate Telegram pair
        if (args.tg_token and not args.chat_id) or (args.chat_id and not args.tg_token):
            raise ConfigError("Both --tg-token and --chat-id are required for Telegram posting.")

        # 2. Fetch Data
        items: List[TrendingItem] = []
        if args.source == "github":
            items = fetch_github_trending(args.lang, limit=1, token=github_token)
        elif args.source == "arxiv":
            items = fetch_arxiv_papers(args.cat, limit=1)
            
        if not items:
            print("[!] No items found. Check filters.", file=sys.stderr)
            sys.exit(1)
            
        top_item = items[0]
        print(f"[+] Selected: {top_item.title} ({top_item.stats})", file=sys.stderr)

        # 3. Construct Context
        context = f"""
        Topic: {top_item.title}
        Author: {top_item.author}
        Stats: {top_item.stats}
        Link: {top_item.url}
        
        Description/Abstract:
        {top_item.description}
        """
        
        # 4. Generate Thread
        generated_content = call_llm(llm_key, args.base_url, args.model, context)
        
        # Add a footer with the link
        final_output = f"{generated_content}\n\nRead more: {top_item.url}"
        
        # 5. Output
        if args.tg_token and args.chat_id:
            post_to_telegram(args.tg_token, args.chat_id, final_output)
            print("[+] Successfully posted to Telegram.", file=sys.stderr)
        else:
            # Markdown Header format
            print(f"# Viral Thread: {top_item.title}")
            print(f"**Source:** [{args.source.upper()}]({top_item.url})  \n")
            print("---")
            print()
            print(final_output)

    except (ConfigError, APIError) as e:
        print(f"[Error] {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[Interrupted] Exiting.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"[Unexpected Error] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    if "requests" not in sys.modules:
        print("This tool requires the 'requests' library. Please install it: pip install requests", file=sys.stderr)
        sys.exit(1)
    main()