# Sri Lanka Reddit Coordination Analysis

This project analyses public Reddit activity in:

- r/ask_srilanka
- r/srilanka
- r/Colombo

It is designed to test observations such as:

> Are recently created accounts disproportionately involved in repetitive,
> inflammatory, or coordinated activity?

It deliberately does **not** classify people as bots. The output is a set of
observable signals and candidate clusters for human inspection.

## Data source

The default collector uses the Arctic Shift Reddit archive API:
https://arctic-shift.photon-reddit.com

Arctic Shift exposes `/api/posts/search` and `/api/comments/search`, supports
subreddit/date filtering, and allows selectable fields. See the project's API
documentation for current limits and availability.

Reddit's official API can also be used for current/live collection, but this
pipeline uses Arctic Shift because historical analysis is the main goal.

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Run

1. Edit `config.yaml`.
2. Collect data:

```bash
python src/collect_arctic.py --config config.yaml
```

3. Build account-level features and interaction edges:

```bash
python src/features.py --config config.yaml
```

4. Run text similarity / clustering:

```bash
python src/text_analysis.py --config config.yaml
```

5. Generate plots and an HTML report:

```bash
python src/report.py --config config.yaml
```

The main output is:

`reports/report.html`

## What it measures

### Account-age signals
- account age at the time of each activity
- fraction of activity from accounts <7 and <30 days old
- activity volume
- subreddit concentration
- posting/commenting bursts

### Text coordination
- TF-IDF cosine similarity
- semantic similarity using MiniLM sentence embeddings
- near-duplicate comments/posts
- clusters of semantically similar material

### Network coordination
Edges are created when users:
- reply to one another
- participate in the same thread
- repeatedly appear together in short time windows
- produce highly similar text

Network metrics include degree, weighted degree, connected components and
community structure.

### Topic analysis
The config contains transparent keyword dictionaries for:
Sinhala, Tamil, Muslim, Buddhist, Hindu, Christian, ethnic and communal terms.

This is deliberately a simple lexical analysis, not a claim about what a user
believes.

## Interpretation

Do not interpret a high score as proof of automation.

A new account can be a genuine person.
A high-volume account can be a genuine person.
Several users can independently use similar language during a breaking event.

The strongest evidence is usually a combination of:
1. unusually recent account creation,
2. repeated near-identical content,
3. synchronized timing,
4. repeated interaction among the same accounts,
5. unusual concentration on a narrow set of topics/threads.

Use the candidate table as a queue for manual investigation, not as an
automated accusation list.

## Privacy

The pipeline only uses public Reddit information returned by the data source.
Do not attempt to deanonymize users, infer real identities, or publish personal
information. Keep raw data local if you are doing research.
