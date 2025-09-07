#!/usr/bin/env python3
"""
Add a single RL implementation from a GitHub folder URL into the site:
- Updates _data/rl.json by appending or replacing the entry
- Generates a new markdown in _rl/ without regenerating everything

Usage:
  python3 add_rl_item.py --url https://github.com/<owner>/<repo>/tree/<branch>/<path>

Notes:
- Designed for the RL page only; mirrors categorization and front-matter used on the site
- Adds Policy-Based for non-DQN items, Value-Based for DQN variants
"""

import argparse
import json
import os
import posixpath
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests


BASE_PATH = Path(__file__).parent
DATA_FILE = BASE_PATH / "_data/rl.json"
MODELS_DIR = BASE_PATH / "_rl"


@dataclass
class RepoRef:
    owner: str
    repo: str
    branch: str
    path: str  # path inside repo


def parse_github_folder_url(url: str) -> RepoRef:
    """Parse GitHub folder URL like:
    https://github.com/<owner>/<repo>/tree/<branch>/<path>
    """
    m = re.match(r"^https?://github.com/([^/]+)/([^/]+)/tree/([^/]+)/(.*)$", url.rstrip('/'))
    if not m:
        raise ValueError("Unsupported URL format. Expected https://github.com/<owner>/<repo>/tree/<branch>/<path>")
    owner, repo, branch, path = m.group(1), m.group(2), m.group(3), m.group(4)
    return RepoRef(owner=owner, repo=repo, branch=branch, path=path)


def github_contents_api_url(ref: RepoRef, file_path: Optional[str] = None) -> str:
    path = ref.path if file_path is None else f"{ref.path}/{file_path}"
    return f"https://api.github.com/repos/{ref.owner}/{ref.repo}/contents/{path}?ref={ref.branch}"


def fetch_readme_text(ref: RepoRef) -> str:
    # Try common README names
    candidate_names = ["README.md", "Readme.md", "readme.md"]
    for name in candidate_names:
        url = github_contents_api_url(ref, name)
        resp = requests.get(url)
        if resp.status_code == 200:
            meta = resp.json()
            download_url = meta.get("download_url")
            if download_url:
                r2 = requests.get(download_url)
                r2.raise_for_status()
                return r2.text
    # If not found individually, list directory and find a README*
    list_url = github_contents_api_url(ref)
    resp = requests.get(list_url)
    resp.raise_for_status()
    items = resp.json()
    if isinstance(items, list):
        for it in items:
            if it.get("type") == "file" and re.match(r"(?i)^readme(\.[a-z0-9]+)?$", it.get("name", "")):
                download_url = it.get("download_url")
                if download_url:
                    r2 = requests.get(download_url)
                    r2.raise_for_status()
                    return r2.text
    return ""


def absolutize_markdown_links(markdown_text: str, ref: RepoRef) -> str:
    """Convert relative markdown/HTML links to raw.githubusercontent.com absolute URLs."""
    if not markdown_text:
        return markdown_text

    def to_absolute(url: str) -> str:
        if not url:
            return url
        u = url.strip()
        if re.match(r"^(?:https?:)?//", u) or u.startswith("mailto:") or u.startswith("#"):
            return u
        if u.startswith('/'):
            resolved = u.lstrip('/')
        else:
            joined = posixpath.join(ref.path, u) if ref.path else u
            resolved = posixpath.normpath(joined)
        return f"https://raw.githubusercontent.com/{ref.owner}/{ref.repo}/{ref.branch}/{resolved}"

    def repl_md_image(m):
        return f"![{m.group(1)}]({to_absolute(m.group(2))}{m.group(3) or ''})"

    def repl_md_link(m):
        return f"[{m.group(1)}]({to_absolute(m.group(2))}{m.group(3) or ''})"

    def repl_html_src(m):
        return f"{m.group(1)}{to_absolute(m.group(2))}{m.group(3)}"

    md_image_pattern = re.compile(r"!\[([^\]]*)\]\(([^\)\s]+)(\s+\"[^\"]*\")?\)")
    md_link_pattern = re.compile(r"(?<!\!)\[([^\]]+)\]\(([^\)\s]+)(\s+\"[^\"]*\")?\)")
    html_src_pattern = re.compile(r"(src=\")([^\"]+)(\")", re.IGNORECASE)
    html_href_pattern = re.compile(r"(href=\")([^\"]+)(\")", re.IGNORECASE)

    updated = md_image_pattern.sub(repl_md_image, markdown_text)
    updated = md_link_pattern.sub(repl_md_link, updated)
    updated = html_src_pattern.sub(repl_html_src, updated)
    updated = html_href_pattern.sub(repl_html_src, updated)
    return updated


def clean_display_name(name: str) -> str:
    name = name.replace('-', ' ').replace('_', ' ')
    words = name.split()
    out = []
    for w in words:
        wl = w.lower()
        if wl in ['dqn', 'ppo', 'a2c', 'sac', 'td3', 'ddpg', 'rnd', 'marl']:
            out.append(w.upper())
        elif wl == 'rl':
            out.append('RL')
        else:
            out.append(w.capitalize())
    return ' '.join(out)


def detect_environment(name: str, readme: str) -> str:
    nl = name.lower()
    cl = readme.lower()
    if 'atari' in nl or 'atari' in cl or 'pong' in cl:
        return 'Atari'
    if 'mujoco' in nl or 'mujoco' in cl:
        return 'MuJoCo'
    if 'lunar' in nl or 'lunarlander' in cl:
        return 'LunarLander'
    if 'taxi' in nl or 'taxi' in cl:
        return 'Taxi'
    if 'frozenlake' in nl or 'frozen' in cl:
        return 'Frozenlake'
    if 'flappybird' in nl or 'flappy' in cl:
        return 'Flappybird'
    if 'vizdoom' in nl or 'vizdoom' in cl:
        return 'Vizdoom'
    if any(env in cl for env in ['gymnasium', 'gym']):
        return 'Gymnasium'
    return 'Custom Environment'


def collect_categories_for_site(name: str, path: str, readme: str) -> List[str]:
    """Emit only site-supported labels; add Value-Based for DQN*, Policy-Based for others when appropriate."""
    nl, pl, cl = name.lower(), path.lower(), readme.lower()
    cats: List[str] = []
    # Multi-Agent
    if any(t in nl or t in pl or t in cl for t in ['marl', 'multi-agent', 'multi agent', 'ippo', 'mappo', 'self-play', 'self play']):
        cats.append('Multi-Agent')
    # Actor-Critic
    if any(t in nl for t in ['a2c', 'a3c', 'sac', 'td3', 'ddpg']) or ('actor' in nl or 'critic' in nl) or 'actor-critic' in cl:
        if 'Actor-Critic' not in cats:
            cats.append('Actor-Critic')
    # Exploration
    if 'rnd' in nl or 'exploration' in cl or 'curiosity' in cl:
        if 'Exploration' not in cats:
            cats.append('Exploration')
    # Imitation
    if any(t in nl for t in ['imitation', 'behavioral', 'cloning']) or 'dagger' in cl:
        if 'Imitation Learning' not in cats:
            cats.append('Imitation Learning')
    # Value/Policy tagging per site rules
    if any(t in nl or t in pl or t in cl for t in ['dqn', 'duel', 'q-learning', 'q learning']):
        cats.append('Value-Based')
    else:
        cats.append('Policy-Based')
    # Deduplicate while keeping order
    seen = set()
    ordered = []
    for c in cats:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered or ['Other']


def choose_primary_category(categories: List[str]) -> str:
    prio = ['Multi-Agent', 'Actor-Critic', 'Exploration', 'Imitation Learning', 'Game Environments', 'Other']
    for p in prio:
        if p in categories:
            return p
    return categories[0] if categories else 'Other'


def upsert_rl_json(entry: Dict) -> None:
    DATA_FILE.parent.mkdir(exist_ok=True)
    if DATA_FILE.exists():
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = {"rl_implementations": []}
    impls = data.get('rl_implementations', [])
    replaced = False
    for i, impl in enumerate(impls):
        if impl.get('github_url') == entry['github_url']:
            impls[i] = entry
            replaced = True
            break
    if not replaced:
        impls.append(entry)
    data['rl_implementations'] = impls
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def next_markdown_index() -> int:
    MODELS_DIR.mkdir(exist_ok=True)
    max_idx = 0
    for p in MODELS_DIR.glob('*.md'):
        m = re.match(r"^(\d+)-", p.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def slugify(name: str) -> str:
    s = re.sub(r'[^\w\s-]', '', name.lower())
    s = re.sub(r'[-\s]+', '-', s).strip('-')
    return s


def write_markdown(entry: Dict) -> Path:
    idx = next_markdown_index()
    filename = f"{idx:02d}-{slugify(entry['display_name'])}.md"
    filepath = MODELS_DIR / filename
    content = f"""---
title: "{entry['display_name']}"
excerpt: "{entry['description']}"
collection: rl
layout: rl-implementation
category: "{entry['category']}"
categories: [{', '.join([f'\"{c}\"' for c in entry.get('categories', [entry['category']])])}]
framework: "{entry['framework']}"
environment: "{entry['environment']}"
github_url: "{entry['github_url']}"
date: {entry['github_date']}
---

## Overview
{entry['description']}

## Technical Details
- **Framework**: {entry['framework']}
- **Environment**: {entry['environment']}
- **Category**: {entry['category']}

## Implementation Details

{entry['readme_content']}

## Source Code
📁 **GitHub Repository**: [{entry['display_name']}]({entry['github_url']})

View the complete implementation, training scripts, and documentation on GitHub.
"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    return filepath


def main():
    parser = argparse.ArgumentParser(description='Add single RL implementation from a GitHub folder URL')
    parser.add_argument('--url', required=True, help='GitHub folder URL (https://github.com/<owner>/<repo>/tree/<branch>/<path>)')
    args = parser.parse_args()

    ref = parse_github_folder_url(args.url)
    readme = fetch_readme_text(ref)
    processed_readme = absolutize_markdown_links(readme, ref)

    item_name = ref.path.split('/')[-1]
    display_name = clean_display_name(item_name)
    categories = collect_categories_for_site(item_name, ref.path, readme)
    category = choose_primary_category(categories)
    environment = detect_environment(item_name, readme)

    created_date = datetime.now().strftime('%Y-%m-%d')
    github_url = f"https://github.com/{ref.owner}/{ref.repo}/tree/{ref.branch}/{ref.path}"
    api_url = f"https://api.github.com/repos/{ref.owner}/{ref.repo}/contents/{ref.path}?ref={ref.branch}"

    entry = {
        "name": item_name,
        "path": ref.path,
        "display_name": display_name,
        "description": f"Implementation of {item_name} reinforcement learning algorithm",
        "readme_content": processed_readme or readme or "",
        "github_url": github_url,
        "api_url": api_url,
        "download_url": None,
        "created_date": created_date,
        "github_date": created_date,
        "category": category,
        "categories": categories,
        "framework": "PyTorch",
        "environment": environment,
    }

    upsert_rl_json(entry)
    md_path = write_markdown(entry)
    print(f"✅ Added/updated RL entry and generated: {md_path.name}")


if __name__ == '__main__':
    main()













