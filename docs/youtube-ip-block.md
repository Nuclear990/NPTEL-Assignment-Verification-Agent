# YouTube transcript fetching: IP block issue

**Status:** Resolved — the IP block cleared on its own after enough time passed
(see "How it actually got resolved" below). Investigated 2026-09-09.

## Symptom

`process_transcripts` (`ingestion/transcripts.py`) started failing to fetch captions
for every YouTube video, regardless of course/lecture:

```
TRANSCRIPT FAILED
IpBlocked:
Could not retrieve a transcript for the video https://www.youtube.com/watch?v=bX80orqYhzU!
This is most likely caused by:
YouTube is blocking requests from your IP...
```

Failures are caught per-lecture in `fetch_week_transcripts` and don't crash the
pipeline — each failed lecture is recorded with an `"error"` field and the run
continues. The problem is that transcripts simply aren't being fetched at all.

## Root cause

YouTube is throttling/blocking the home network's IP address at the
`timedtext`/InnerTube caption endpoint. This was confirmed, not assumed:

- Swapping to a completely different library (see below) that hits a different
  code path still got blocked on the *same* video.
- An unrelated, totally different video (`dQw4w9WgXcQ`) got blocked identically.
- This rules out "this one video has no captions" or "this library has a bug" —
  it's the requesting IP itself that's rate-limited/blocked by YouTube.

This is a known limitation of scraping YouTube's caption delivery: there is no
official, unlimited API for downloading a *third-party* video's captions (see
below), so every method — official-feeling or not — ultimately hits the same
IP-based defenses.

## Things tried, and why each one failed or was ruled out

### 1. `youtube_transcript_api` (original library) — blocked
Raises `IpBlocked` directly. This is what surfaced the issue.

### 2. Official YouTube Data API v3 — ruled out, not just blocked
Considered switching to the "official" API to fix this permanently. Not viable:
`captions.download` (the only endpoint that returns actual caption text) is
restricted by Google to the **video's owner**, authenticated via OAuth. Since
these are NPTEL's videos, not ours, there is no key/token that grants access.
`captions.list` (metadata only, no text) is the only part usable with just an
API key. `youtube_transcript_api` and `yt-dlp` are both "unofficial" for a
reason — they replicate what the YouTube player itself requests, which is why
they're subject to the same IP defenses as any other scraper.

### 3. Switched to `yt-dlp` — same block, one level up
Rationale: `yt-dlp` is a large, actively-maintained project that regularly
adapts to YouTube's countermeasures, so it's the standard community workaround
when `youtube_transcript_api` gets blocked.

Result: `yt-dlp`'s metadata extraction succeeded (didn't hit `IpBlocked`), but
the actual caption file download hit `429 Too Many Requests` on the same
`timedtext` endpoint. Confirmed this was IP-wide, not video-specific, by testing
a second, unrelated video — same `429`. So the library swap avoided the harder
`IpBlocked` exception but didn't dodge the underlying network-level throttle.

Later reverted back to `youtube_transcript_api` per preference, since the
library swap wasn't actually solving the root problem — see below.

### 4. Paid rotating proxy (e.g. Webshare) — ruled out on cost
`youtube_transcript_api`'s own documentation states this is the *only reliably
effective* fix for `IpBlocked`/`RequestBlocked` (a large pool of rotating
residential IPs). Not pursued: no budget for a paid proxy service.

### 5. Free Tor SOCKS5 proxy — wired up, tested, confirmed non-viable
Implemented as an opt-in setting:
- `config.py`: reads optional `YT_PROXY` from `.env`.
- `.env`: `YT_PROXY=socks5h://127.0.0.1:9050` (commented out by default).
- `ingestion/transcripts.py`: `get_video_transcript()` passes it to
  `YouTubeTranscriptApi(proxy_config=GenericProxyConfig(...))` when set.
- `PySocks` added as a dependency (`requests`/the API need it to speak SOCKS5).

Verified the plumbing itself works: routing traffic through
`socks5h://127.0.0.1:9050` (a local Tor daemon) produced a different exit IP
each time a fresh SOCKS auth/circuit was used, confirmed via `api.ipify.org`.

Then tested the actual transcript fetch through **5 different Tor exit
circuits** — every single one came back `RequestBlocked` / `IpBlocked`. This
means the request correctly reached YouTube's backend through Tor (it's not a
connectivity failure) — it's that YouTube already blocklists most of the
public Tor exit-node IP range for exactly this kind of scraping, since Tor is
one of the most common ways people scrape it for free.

**Conclusion: the Tor wiring is correct and harmless to leave in `.env`
(disabled by default), but it does not currently work as a fix.**

### 6. Considered a paid Webshare proxy — code supports it, not purchased
`GenericProxyConfig`/`WebshareProxyConfig` (from `youtube_transcript_api`) would
have worked with this exact same `YT_PROXY` wiring — a rotating residential
proxy is the one thing that's actually durable against IP blocks, since
Webshare's own pool isn't pre-blocked the way Tor's public exit list is. Not
purchased in the end, because the block resolved on its own (#7) before it was
needed. If this issue ever recurs, this is the first thing to reach for.

### 7. How it actually got resolved: it just needed time, plus a library swap
While re-testing options, switched back to `yt-dlp` (see #3) one more time and,
on that attempt, **both** the previously-blocked video and an unrelated one
succeeded with a completely plain request — no proxy, no cookies, nothing
extra. Confirmed this wasn't a fluke by testing both videos individually.

So the actual fix was mundane: the IP-level block/throttle was temporary all
along, and enough real-world time (several days, across all the testing in
this doc) had passed for YouTube to lift it. `yt-dlp` gets kept as the library
going forward (over `youtube_transcript_api`) since it's the more actively
maintained project and was already in place when things started working again.

### 8. Tried adding Brave browser cookies to yt-dlp — made things worse, reverted
Tried `cookiesfrombrowser: ("brave",)` in `yt-dlp`'s options, on the theory that
an authenticated session might be treated more leniently than an anonymous
one. Result: it broke the *already-working* plain request, failing instead
with `ExtractorError: The page needs to be reloaded`. This is a separate, known
`yt-dlp`/YouTube issue — cookie-authenticated requests currently push `yt-dlp`
onto an internal client path that needs a "PO token" it doesn't have without
extra plugins, unrelated to IP blocking. Reverted immediately; `secretstorage`/
`jeepney` (needed only for decrypting Brave's cookie store) were removed again
since nothing uses them anymore.

**Takeaway: don't add `cookiesfrombrowser` back without also solving the PO
token requirement — it actively regresses a working setup.**

## Current state of the code

- `ingestion/transcripts.py` uses `yt-dlp` (plain, no cookies) via
  `get_video_transcript()`. Caption metadata is fetched with
  `ydl.extract_info()`; the actual `.vtt` caption file is then downloaded
  through `ydl.urlopen()` (not a separate `requests` call) so that if a proxy
  is ever enabled, it's applied to both requests, not just the metadata one.
- Optional `YT_PROXY` proxy support is still wired in via `yt-dlp`'s own
  `proxy` option, off by default (`.env`, commented out). If IP blocking
  recurs, this is where a paid residential proxy (e.g. Webshare) would be
  plugged in — no code changes needed, just uncomment and fill in the URL.
- Per-lecture failures are still caught gracefully and don't kill a run
  (`fetch_week_transcripts` in `ingestion/transcripts.py`).

## If this recurs

In rough order of effort:
1. **Wait** — this is exactly what fixed it last time. Not fast, but free and
   proven to work eventually.
2. **Mobile hotspot** — tether through phone data specifically while
   `process_transcripts` runs. Mobile carrier IPs are essentially never
   pre-blocked the way datacenter/Tor-exit IPs are, and costs nothing but
   mobile data. Manual step, not automated.
3. **Paid rotating residential proxy** (Webshare or similar) — the durable
   fix if this becomes a recurring problem rather than a one-off. `YT_PROXY`
   in `.env` is already wired for this.

Ruled out, don't retry without new information:
- **Free VPNs / Tor**: tested (Tor, 5 different exit circuits) — all
  pre-blocked by YouTube. Free VPN pools are expected to fare the same way.
- **Browser cookies for IP-block purposes**: doesn't help IP-based blocks
  (only helps age-restricted video access), and currently actively breaks
  `yt-dlp` for an unrelated PO-token reason — see #8.
- **Official YouTube Data API**: structurally can't download third-party
  captions at all (owner-only OAuth) — see #2.
