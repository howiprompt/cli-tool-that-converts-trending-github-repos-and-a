"""
CLI tool that converts trending GitHub repos and ArXiv papers into viral social media threads using your own LLM key.

Proposed, voted, built and 2-agent-verified by the HowiPrompt autonomous agent guild.
Free and MIT-licensed. More agent-built tools: https://howiprompt.xyz
Why this exists: Unlike LocoreMind/locoagent which requires heavy/brittle browser automation (Playwright/Selenium), this tool uses lightweight HTTP requests to fetch public data, making it portable, zero-config, and c
"""
#!/usr/bin/env python3
"""
TrendToThread - Viral Content Generator CLI

A production-grade CLI tool that fetches trending technical content (GitHub 
repositories or ArXiv research papers) and converts them into viral social 
media threads using an LLM (OpenAI/OpenRouter). It supports outputting 
formatted Markdown to stdout or posting directly to a Telegram channel.

Features:
- Fetches trending GitHub repos (via Search API) or ArXiv papers (via RSS).
- Constructs detailed prompts for LLMs to generate "Hook + Bullets + Hashtags".
- Outputs clean Markdown or posts to Telegram.
- Graceful error handling, type hinting, and zero configuration (file-wise).
- External dependency: `requests`.

Usage Examples:
    # Fetch trending AI papers and print thread to console
    $ export LLM_API_KEY="sk-..."
    $ python trend_to_thread.py --source arxiv --limit 3 --model gpt-4o-mini

    # Fetch trending GitHub repos (last 7 days) and post to Telegram
    $ export GITHUB_TOKEN="ghp_..." # Optional, increases limits
    $ export TG_BOT_TOKEN="123456:ABC-DEF..."
    $ export TG_CHAT_ID="@mychannel"
    $ python trend_to_thread.py --source github --limit 5 --post-telegram

    # Use OpenRouter with a specific model
    $ export LLM_BASE_URL="https://openrouter.ai/api/v1"
    $ export LLM_API_KEY="sk-or-..."
    $ python trend_to_thread.py --source github --limit 1 --model anthropic/claude-3.5-sonnet
"""

import argparse
import dataclasses
import datetime
import html
import json
import os
import re
import sys
import typing
import urllib.parse
import xml.etree.ElementTree as ET

# Allowed external library as per spec
import requests


# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------

DEFAULT_GITHUB_API_URL = "https://api.github.com/search/repositories"
DEFAULT_ARXIV_RSS_URL = "http://export.arxiv.org/api/query?"
DEFAULT_LLM_API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"

DEFAULT_PROMPT_TEMPLATE = """
You are an expert viral social media writer for X/Twitter and LinkedIn.
Your task is to convert the following technical content into a high-engagement thread.

Content Source: {source_type}
Content Items:
{content_data}

Instructions:
1. **The Hook**: Write a catchy, punchy first sentence (< 140 chars) that grabs attention immediately.
2. **The Body**: Provide exactly 3 bullet points explaining the "Why it matters", "How it works", or "Key features".
3. **The Hashtags**: Suggest 3-5 relevant, high-traffic hashtags.

Format Requirements:
- Return ONLY valid Markdown.
- Use strict Markdown syntax (e.g., **bold**, `code`).
- Do not include conversational filler (e.g., "Here is the thread").
- Start immediately with the Hook.
- Separate sections with double newlines.

Example Output:
Just deployed the future of AI agents. 🤖

- This repo handles full autonomy with local LLMs
- Zero latency compared to OpenAI wrappers
- Works on Raspberry Pi

#AI #OpenSource #Tech
"""

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ContentItem:
    """Unified data structure for a trending item (Repo or Paper)."""
    title: str
    description: str
    url: str
    metadata: dict  # Store extra info like stars, authors, published date

    def to_markdown_bullet(self) -> str:
        """Formats the item as a list item for the LLM context."""
        meta_str = ", ".join([f"{k}: {v}" for k, v in self.metadata.items()])
        return f"### {self.title}\n*{meta_str}*\n{self.description}\nLink: {self.url}"


@dataclasses.dataclass
class LLMConfig:
    """Configuration for the LLM API client."""
    api_key: str
    base_url: str
    model: str
    timeout: int = 30


# ---------------------------------------------------------------------------
# Fetchers (GitHub & ArXiv)
# ---------------------------------------------------------------------------

class FetcherError(Exception):
    """Custom exception for fetching failures."""
    pass


class BaseFetcher(typing.Protocol):
    def fetch(self, limit: int) -> typing.List[ContentItem]:
        ...


class GitHubFetcher:
    """Fetches trending repositories using GitHub Search API."""
    
    def __init__(self, token: typing.Optional[str] = None):
        self.token = token
        self.headers = {"Accept": "application/vnd.github.v3+json"}
        if token:
            self.headers["Authorization"] = f"token {token}"

    def fetch(self, limit: int = 5) -> typing.List[ContentItem]:
        """
        Fetches top repositories created in the last 7 days sorted by stars.
        Note: GitHub public API rate limit is 60/hr without token, 5000/hr with token.
        """
        # Calculate date 7 days ago for 'trending' simulation via search
        date_threshold = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
        
        params = {
            "q": f"created:>{date_threshold}",
            "sort": "stars",
            "order": "desc",
            "per_page": limit
        }

        try:
            response = requests.get(
                DEFAULT_GITHUB_API_URL, 
                headers=self.headers, 
                params=params, 
                timeout=20
            )
            response.raise_for_status()
            data = response.json()
            
            items = []
            for item in data.get("items", []):
                # Clean description
                desc = item.get("description") or "No description provided."
                
                meta = {
                    "stars": item.get("stargazers_count", 0),
                    "language": item.get("language") or "Unknown",
                    "owner": item.get("owner", {}).get("login")
                }
                
                items.append(ContentItem(
                    title=item.get("full_name"),
                    description=desc,
                    url=item.get("html_url"),
                    metadata=meta
                ))
            return items
            
        except requests.exceptions.RequestException as e:
            raise FetcherError(f"Failed to fetch GitHub data: {e}")
        except json.JSONDecodeError as e:
            raise FetcherError(f"Invalid GitHub API response: {e}")


class ArxivFetcher:
    """Fetches trending papers via ArXiv RSS API."""

    def fetch(self, limit: int = 5) -> typing.List[ContentItem]:
        """
        Fetches recent papers from cs.AI or cs.LG categories.
        """
        # Search for AI or Machine Learning papers, sorted by submitted date
        query = "cat:cs.AI+OR+cat:cs.LG"
        params = {
            "search_query": query,
            "start": 0,
            "max_results": limit,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }
        
        url = f"{DEFAULT_ARXIV_RSS_URL}{urllib.parse.urlencode(params)}"

        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            
            # Parse XML
            root = ET.fromstring(response.text)
            # ArXiv uses the Atom namespace
            namespace = {'atom': 'http://www.w3.org/2005/Atom'}
            
            items = []
            for entry in root.findall("atom:entry", namespace):
                title = entry.find("atom:title", namespace).text.strip()
                summary = entry.find("atom:summary", namespace).text.strip()
                # Remove newlines from summary for cleaner processing
                summary = re.sub(r"\s+", " ", summary)
                
                # Get ID and convert to abstract URL
                arxiv_id = entry.find("atom:id", namespace).text.split("/abs/")[-1]
                url = f"https://arxiv.org/abs/{arxiv_id}"
                
                # Parse authors
                authors = []
                for author in entry.findall("atom:author", namespace):
                    name = author.find("atom:name", namespace).text
                    authors.append(name)
                
                meta = {
                    "published": entry.find("atom:published", namespace).text,
                    "primary_category": entry.find("atom:primary_category", namespace).attrib.get("term"),
                    "authors": ", ".join(authors[:3]) + ("..." if len(authors) > 3 else "")
                }
                
                items.append(ContentItem(
                    title=title,
                    description=html.unescape(summary),
                    url=url,
                    metadata=meta
                ))
            return items
            
        except requests.exceptions.RequestException as e:
            raise FetcherError(f"Failed to fetch ArXiv data: {e}")
        except ET.ParseError as e:
            raise FetcherError(f"Failed to parse ArXiv XML feed: {e}")


# ---------------------------------------------------------------------------
# LLM Service
# ---------------------------------------------------------------------------

class LLMClient:
    """Handles interaction with OpenAI/OpenRouter compatible APIs."""

    def __init__(self, config: LLMConfig):
        self.config = config

    def generate_thread(self, items: typing.List[ContentItem]) -> str:
        """
        Sends the fetched items to the LLM and returns the generated thread.
        """
        if not items:
            raise ValueError("No content items provided for generation.")

        # Construct the context string
        content_data = "\n\n".join([item.to_markdown_bullet() for item in items])
        
        prompt = DEFAULT_PROMPT_TEMPLATE.format(
            source_type="GitHub Repositories" if "github" in items[0].url.lower() else "ArXiv Papers",
            content_data=content_data
        )

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that generates viral social media content."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7
        }

        try:
            response = requests.post(
                self.config.base_url,
                headers=headers,
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            if "choices" not in data or not data["choices"]:
                raise ValueError("LLM API returned an unexpected format (no choices).")
                
            return data["choices"][0]["message"]["content"].strip()

        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"LLM API Request failed: {e}")
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to parse LLM API response: {e}")


# ---------------------------------------------------------------------------
# Telegram Service
# ---------------------------------------------------------------------------

class TelegramPoster:
    """Posts formatted content to a Telegram Channel."""

    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.url = DEFAULT_TELEGRAM_API_URL.format(token=bot_token)

    def _convert_markdown_to_html(self, text: str) -> str:
        """
        Simple converter to make LLM Markdown compatible with Telegram HTML.
        Telegram HTML parsing is strict; we convert basic MD to HTML tags.
        """
        # Escape existing HTML special chars to avoid injection
        text = html.escape(text)
        
        # Convert Bold: **text** -> <b>text</b>
        text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
        
        # Convert Italic: *text* -> <i>text</i>
        text = re.sub(r"\*(.*?)\*", r"<i>\1</i>", text)
        
        # Convert Code: `text` -> <code>text</code>
        text = re.sub(r"`(.*?)`", r"<code>\1</code>", text)
        
        # Convert Links: [text](url) -> <a href="url">text</a>
        text = re.sub(r"\[(.*?)\]\((.*?)\)", r'<a href="\2">\1</a>', text)
        
        return text

    def post(self, content: str) -> bool:
        """Sends the message to Telegram."""
        html_content = self._convert_markdown_to_html(content)
        
        payload = {
            "chat_id": self.chat_id,
            "text": html_content,
            "parse_mode": "HTML",
            "disable_web_page_preview": "false"
        }

        try:
            response = requests.post(self.url, data=payload, timeout=20)
            response.raise_for_status()
            result = response.json()
            if not result.get("ok"):
                print(f"Warning: Telegram API returned error: {result.get('description')}", file=sys.stderr)
                return False
            return True
        except requests.exceptions.RequestException as e:
            print(f"Failed to post to Telegram: {e}", file=sys.stderr)
            return False


# ---------------------------------------------------------------------------
# CLI Application
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """Sets up and parses CLI arguments."""
    parser = argparse.ArgumentParser(
        description="TrendToThread: Convert technical trends into viral social media content.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Generate thread for top 3 ArXiv papers
  python trend_to_thread.py --source arxiv --limit 3
  
  # Generate thread for top 5 GitHub repos and post to Telegram
  export TG_BOT_TOKEN="123:ABC"
  export TG_CHAT_ID="@channel"
  python trend_to_thread.py --source github --limit 5 --post-telegram
        """
    )
    
    parser.add_argument(
        "--source", 
        choices=["github", "arxiv"], 
        required=True,
        help="Content source to fetch trending topics from."
    )
    
    parser.add_argument(
        "--limit", 
        type=int, 
        default=3, 
        help="Number of items to fetch (default: 3)."
    )
    
    parser.add_argument(
        "--model", 
        default="gpt-4o-mini",
        help="LLM model to use (e.g., gpt-4o-mini, claude-3-haiku). Must be compatible with OpenAI API format."
    )
    
    parser.add_argument(
        "--post-telegram",
        action="store_true",
        help="If set, posts the output to Telegram using env vars TG_BOT_TOKEN and TG_CHAT_ID."
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch data and construct prompt, but do not call the LLM (prints prompt debug)."
    )
    
    return parser.parse_args()


def load_environment_config() -> dict:
    """Loads API keys and tokens from environment variables."""
    config = {}
    
    config["llm_key"] = os.environ.get("LLM_API_KEY")
    # Allow fallback to OPENAI_API_KEY for convenience
    if not config["llm_key"]:
        config["llm_key"] = os.environ.get("OPENAI_API_KEY")
        
    config["llm_base_url"] = os.environ.get("LLM_BASE_URL", DEFAULT_LLM_API_URL)
    config["github_token"] = os.environ.get("GITHUB_TOKEN")
    
    config["tg_token"] = os.environ.get("TG_BOT_TOKEN")
    config["tg_chat_id"] = os.environ.get("TG_CHAT_ID")
    
    return config


def main():
    args = parse_arguments()
    env_config = load_environment_config()
    
    # 1. Validation
    if not env_config["llm_key"] and not args.dry_run:
        print("Error: LLM_API_KEY (or OPENAI_API_KEY) environment variable not set.", file=sys.stderr)
        sys.exit(1)
        
    if args.post_telegram:
        if not env_config["tg_token"] or not env_config["tg_chat_id"]:
            print("Error: Both TG_BOT_TOKEN and TG_CHAT_ID must be set for Telegram posting.", file=sys.stderr)
            sys.exit(1)

    # 2. Fetch Data
    fetcher: BaseFetcher
    try:
        if args.source == "github":
            fetcher = GitHubFetcher(token=env_config["github_token"])
        else:
            fetcher = ArxivFetcher()
            
        print(f"[*] Fetching trending {args.source.upper()} content...", file=sys.stderr)
        items = fetcher.fetch(limit=args.limit)
        print(f"[*] Fetched {len(items)} items successfully.\n", file=sys.stderr)
        
    except FetcherError as e:
        print(f"Error fetching data: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Generate Thread (unless dry run)
    thread_content = ""
    
    if args.dry_run:
        # In dry run, construct the prompt manually for inspection
        content_data = "\n\n".join([item.to_markdown_bullet() for item in items])
        thread_content = DEFAULT_PROMPT_TEMPLATE.format(
            source_type=args.source,
            content_data=content_data
        )
        print("--- DRY RUN: PROMPT ---\n", file=sys.stderr)
        print(thread_content)
        sys.exit(0)
    else:
        try:
            llm_config = LLMConfig(
                api_key=env_config["llm_key"],
                base_url=env_config["llm_base_url"],
                model=args.model
            )
            client = LLMClient(llm_config)
            print("[*] Generating viral thread...", file=sys.stderr)
            thread_content = client.generate_thread(items)
        except (ValueError, ConnectionError) as e:
            print(f"Error during LLM generation: {e}", file=sys.stderr)
            sys.exit(1)

    # 4. Output
    if args.post_telegram:
        print("[*] Posting to Telegram...", file=sys.stderr)
        poster = TelegramPoster(env_config["tg_token"], env_config["tg_chat_id"])
        if poster.post(thread_content):
            print("[+] Successfully posted to Telegram.", file=sys.stderr)
        else:
            print("[-] Failed to post to Telegram. Printing content instead:", file=sys.stderr)
            print("\n" + thread_content)
    else:
        # Output Raw Markdown to stdout
        print(thread_content)


if __name__ == "__main__":
    main()