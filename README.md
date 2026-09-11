# skribbl-guesser

Real-time sketch recognition for [skribbl.io](https://skribbl.io), running locally on your GPU!

It opens skribbl.io in Chrome, watches the canvas and the hint bar, narrows the word list to what can still fit, and ranks the remaining words against the drawing with SigLIP. A panel in the page shows the ranking; click a word to send it. Every round is logged, so precision, recall, and time to solve accumulate across a session. I've been able to get 5k-7k points in each game using this tool. 

https://github.com/user-attachments/assets/fb6f9548-fd66-4825-a5bf-4929756793f1

## Quick start

Needs Python 3.10+, Chrome, and an NVIDIA GPU with a few GB of VRAM. Tested on an RTX 5080.

```
git clone https://github.com/jackkowalik/skribbl-guesser
cd skribbl-guesser
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python skribbl_guesser.py
```

The first run downloads the model (about 3.5 GB) and encodes the word lists, which takes maybe five minutes total. Both are cached, so later runs start quickly.

Chrome opens on skribbl.io. Join a game. The tool ranks but never types on its own; click a word to send it.

## How it works

Each poll (every 0.4 s) the page script reads the hint bar, the canvas, and any new chat lines, and the driver does the rest.

**Narrowing.** The hint bar gives word length, spaces, hyphens, and revealed letters, which become a regex over the word list. Every wrong guess anyone posts in chat is ruled out for the round. skribbl's "is close!" reply means the answer is one edit away from that guess, and a guess of yours that gets no such reply means it is at least two edits away; both are applied as filters. In a full room this eliminates most of the pool before the model has to decide anything.

**Ranking.** The canvas is encoded once per poll and scored against the survivors. Scores are blended with a moving average inside the round so one stroke does not flip the order. Blank canvases are skipped.

**Two word tiers.** Words that have appeared as skribbl answers are the confirmed list (`words.txt`). A frequency-ranked English list (`wordfreq`, top 30k) is a second tier: scored every frame, but only its strongest few are shown unless the confirmed pool is nearly empty. Fallback words carry a prior penalty so a confirmed word at the same image score ranks above them, show in gray, and never push a confirmed word off the board. For multi-word patterns with no confirmed match, each slot is scored on its own and the top few per slot are combined into compound candidates.

**Learning.** Any answer not already in the confirmed list is appended to `words.txt` and encoded on the spot.

**Model fit.** SigLIP so400m at 384 px handles what skribbl actually produces: rough single-color line drawings, partial drawings that fill in over the round, and drawers who just give up and write the word, which it reads as text. One image encode against a few hundred cached text embeddings runs well inside the poll interval.

## Compared with API-based guessers

Other skribbl assistants send a screenshot to a hosted vision model when you press a button and return one guess. This one, in my humble opinion, is much better:

- **Local and continuous.** SigLIP runs on your GPU. The canvas is re-scored every 0.4 s, so the ranking updates as the drawing develops rather than once per click, and there is no API key, quota, or round trip.
- **A word list.** Guesses come from a list of words skribbl has actually used, matched against the hint pattern, chat, and the close-guess rule. The model only has to order a few dozen survivors. `words.txt` is committed to the repo and I add to it as I play, so it tracks skribbl's real word set.
- **Elimination.** Every wrong guess anyone posts is removed from the pool, and skribbl's own "is close" reply is used as a filter. A public room does most of the narrowing for you.
- **Ads blocked at the network level**, including the fullscreen preroll, rather than cropped out of a screenshot.
- **Everything is fast.** Encoding is cached, the per-frame cost is one image encode and a matrix multiply, and the whole loop runs inside the poll interval.

## In the game

The top ten candidates appear as chips directly under the canvas; the full ranked list sits below the chat, where the confirmed words are green and fallback words gray. Click either to send that word. The chips disappear while you are drawing and while there is nothing to show.

The game itself is zoomed 1.5x by default so the canvas is readable on a large display, and the site's ad networks are blocked at the request level so nothing loads over the game.

## Options

```
--words PATH       word list (default words.txt)
--scale S          zoom applied to the game (default 1.5)
--no-fallback      disable the wordfreq tier
--no-frames        do not save canvas frames
--ads              do not block ad networks
```

`words.txt` is one word per line and is checked in. It started from a public list and grows every session; pull requests that add words skribbl actually used are welcome.

## Output

| File | Contents |
|---|---|
| `rounds.jsonl` | One line per round: pattern, answer, whether it was in the list, guesses sent, solved or not, time to solve, final top 5 |
| `frames/` | Every fifth scored frame, renamed with the real answer when the round ends. A labeled sketch dataset that grows as you play |
| `text_cache_*.pt` | Cached text embeddings, keyed on list contents |

## Files

- `skribbl_guesser.py`: driver, scoring, round state, stats
- `panel.js`: injected into the page: DOM reading, chat capture, panel and dock UI, ad removal

Both files carry comments on the DOM structure and behaviors that were worked out by observation, since none of it is documented.

## Built on

[open_clip](https://github.com/mlfoundations/open_clip) and the [SigLIP so400m](https://huggingface.co/timm/ViT-SO400M-14-SigLIP-384) weights, [wordfreq](https://github.com/rspeer/wordfreq), Selenium.

## License

MIT
