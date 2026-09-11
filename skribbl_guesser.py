import argparse
import base64
import hashlib
import io
import itertools
import json
import math
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image
from selenium import webdriver
from selenium.common.exceptions import NoSuchWindowException, WebDriverException
from selenium.webdriver.chrome.options import Options

MODEL = "hf-hub:timm/ViT-SO400M-14-SigLIP-384"

# Request patterns refused through DevTools when ad blocking is on. Seen in
# the page: AdSense and GPT (googlesyndication, doubleclick), the AdInPlay
# preroll player (adinplay, imasdk), and the prebid/ID sync scripts.
AD_HOSTS = [
    "*://*.googlesyndication.com/*",
    "*://*.doubleclick.net/*",
    "*://*.adinplay.com/*",
    "*://imasdk.googleapis.com/*",
    "*://fundingchoicesmessages.google.com/*",
    "*://*.atmtd.com/*",
    "*://*.liadm.com/*",
    "*://*.criteo.net/*",
    "*://*.criteo.com/*",
    "*://*.openxcdn.net/*",
    "*://*.creativecdn.com/*",
]

# Prompt ensemble for the text side. Averaging a few phrasings is worth a
# couple of points over a single "a photo of a {}" because the canvas is
# always a doodle on white, never a photo.
TEMPLATES = [
    "a drawing of a {}",
    "a doodle of a {}",
    "a simple line drawing of a {}",
    "a child's crayon drawing of a {}",
]


@dataclass
class Config:
    words_file: Path = Path("words.txt")
    panel_file: Path = Path("panel.js")
    frames_dir: Path = Path("frames")
    stats_file: Path = Path("rounds.jsonl")
    profile_dir: Path = Path("chrome-profile")
    poll_seconds: float = 0.4
    min_ink_fraction: float = 0.003
    ema_alpha: float = 0.4
    save_frames: bool = True
    top_n: int = 40
    fallback_top_n: int = 30000
    fallback_min_trusted: int = 6
    fallback_prior: float = 0.35
    fallback_keep: int = 5
    fallback_show_prob: float = 0.15
    compound_top: int = 8
    compound_encode_per_frame: int = 24
    block_ads: bool = True
    scale: float = 1.5


def parse_args():
    p = argparse.ArgumentParser(description="Zero-shot sketch recognition for skribbl.io")
    p.add_argument("--words", type=Path, default=Config.words_file, help="confirmed word list, one per line")
    p.add_argument("--no-frames", action="store_true", help="do not save canvas frames")
    p.add_argument("--no-fallback", action="store_true", help="disable the wordfreq fallback list")
    p.add_argument("--ads", action="store_true", help="do not block ad networks")
    p.add_argument("--scale", type=float, default=Config.scale, help="zoom applied to the game (default 1.5)")
    a = p.parse_args()
    return Config(
        words_file=a.words,
        save_frames=not a.no_frames,
        fallback_top_n=0 if a.no_fallback else Config.fallback_top_n,
        block_ads=not a.ads,
        scale=a.scale,
    )


class WordBank:
    """Confirmed words (seen as skribbl answers) plus an optional fallback tier.

    The commonly shared gist lists date from before skribbl's Nov 2022 word
    update, so roughly a fifth of answers in public rooms are missing from
    them. Every answer the tool sees gets appended, so the list converges.
    The loader takes the first comma or tab separated field of each line, so
    a "word,count," CSV and a plain one-word-per-line file both work.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.words = []
        self.lower = []
        self.seen = set()
        for line in cfg.words_file.read_text(encoding="utf-8").splitlines():
            self.add(line.split(",")[0].split("\t")[0].strip(), persist=False)
        self.fallback = self.load_fallback()
        self.fallback_lower = [w.lower() for w in self.fallback]

    def add(self, word, persist=True):
        key = word.lower()
        if not word or key == "word" or key in self.seen:
            return False
        self.seen.add(key)
        self.words.append(word)
        self.lower.append(key)
        if persist:
            with self.cfg.words_file.open("a", encoding="utf-8") as f:
                f.write(f"\n{word}")
        return True

    def load_fallback(self):
        # wordfreq's top N is a frequency ranking, which lines up well with the
        # common-noun space skribbl draws from. Single tokens only; the
        # confirmed list carries the multi-word entries.
        if self.cfg.fallback_top_n <= 0:
            return []
        try:
            from wordfreq import top_n_list
        except ImportError:
            print("wordfreq not installed, running without fallback list", file=sys.stderr)
            return []
        out = [w for w in top_n_list("en", self.cfg.fallback_top_n) if len(w) >= 3 and w.isalpha()]
        print(f"{len(out)} fallback words", file=sys.stderr)
        return out


class Scorer:
    def __init__(self, bank):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(MODEL, device=self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(MODEL)
        self.bank = bank
        self.text = self.load_or_encode(bank.words)
        self.extra = self.load_or_encode(["fallback"] + bank.fallback, skip_first=True) if bank.fallback else None
        self.logit_scale = self.model.logit_scale.exp().item()

    @staticmethod
    def cache_path(words):
        key = hashlib.sha1("\n".join([MODEL] + TEMPLATES + words).encode("utf-8")).hexdigest()[:16]
        return Path(f"text_cache_{key}.pt")

    def load_or_encode(self, words, skip_first=False):
        cache = self.cache_path(words)
        if cache.exists():
            print(f"loaded text embeddings from {cache}", file=sys.stderr)
            return torch.load(cache, map_location=self.device)
        feats = self.encode(words[1:] if skip_first else words)
        torch.save(feats.cpu(), cache)
        return feats

    def add_words(self, new):
        self.text = torch.cat([self.text, self.encode(new)])
        torch.save(self.text.cpu(), self.cache_path(self.bank.words))

    def autocast(self):
        if self.device == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return torch.autocast("cpu", enabled=False)

    @torch.no_grad()
    def encode(self, words, verbose=True):
        out = []
        for i in range(0, len(words), 256):
            chunk = words[i:i + 256]
            feats = None
            for t in TEMPLATES:
                tok = self.tokenizer([t.format(w) for w in chunk]).to(self.device)
                with self.autocast():
                    f = self.model.encode_text(tok).float()
                f = f / f.norm(dim=-1, keepdim=True)
                feats = f if feats is None else feats + f
            out.append(feats / feats.norm(dim=-1, keepdim=True))
            if verbose:
                print(f"encoded {min(i + 256, len(words))}/{len(words)} words", file=sys.stderr)
        return torch.cat(out)

    @torch.no_grad()
    def image_feat(self, image):
        x = self.preprocess(image).unsqueeze(0).to(self.device)
        with self.autocast():
            f = self.model.encode_image(x).float()
        return f / f.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def score_feat(self, img, feats):
        # Scaled cosine logits. SigLIP's logit_bias belongs to its sigmoid
        # loss; under a softmax it is a constant shift and does nothing.
        return ((img @ feats.T).squeeze(0) * self.logit_scale).float().cpu().numpy()


def ink_fraction(image):
    # SigLIP will confidently score an empty white canvas against whatever
    # word is nearest "white rectangle", so scoring is gated on this.
    return float((np.asarray(image.convert("L")) < 240).mean())


def edit_distance(a, b):
    # skribbl replies "<guess> is close!" when the guess is Levenshtein
    # distance 1 from the answer, so anything above 1 collapses to 2 here.
    if abs(len(a) - len(b)) > 1:
        return 2
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def pattern_regex(pattern):
    # Spaces and hyphens are revealed in the hint bar from the first second of
    # the round, so an underscore can never be a space. Matching "_" against
    # "." let "bed sheet" through for a 9-letter single word, which cost real
    # rounds before this was tightened.
    body = "".join("[^ ]" if ch == "_" else re.escape(ch) for ch in pattern)
    return re.compile(f"^{body}$", re.IGNORECASE)


def softmax_rank(items):
    # items: [(word, logit)] -> [(word, share)] sorted by share.
    if not items:
        return []
    m = max(l for _, l in items)
    exp = [(w, math.exp(l - m)) for w, l in items]
    total = sum(e for _, e in exp) or 1.0
    return sorted(((w, e / total) for w, e in exp), key=lambda kv: -kv[1])


@dataclass
class Stats:
    rounds: int = 0
    solved: int = 0
    sent: int = 0
    correct: int = 0
    in_list: int = 0
    new_words: int = 0
    solve_times: list = field(default_factory=list)

    def summary(self):
        prec = self.correct / self.sent if self.sent else 0.0
        rec = self.solved / self.rounds if self.rounds else 0.0
        avg = sum(self.solve_times) / len(self.solve_times) if self.solve_times else 0.0
        return {
            "Rounds guessed in": self.rounds,
            "Solved": f"{self.solved} ({rec:.0%})",
            "Guesses sent": self.sent,
            "Precision": f"{prec:.0%}",
            "Avg time to solve": f"{avg:.1f}s",
            "Answer was in list": self.in_list,
            "New words learned": self.new_words,
        }


class Round:
    """State for one drawing turn.

    excluded: words ruled out this round (our guesses, other players' wrong
    guesses, and close-but-wrong guesses).
    close: guesses skribbl said were close, so the answer is one edit away.
    sent_at: when we sent each guess; a guess older than a second with no
    close reply means the answer is at least two edits away from it.
    ema: per-word smoothed logit so one stroke does not flip the ranking.
    """

    def __init__(self, pattern):
        self.pattern = pattern
        self.regex = pattern_regex(pattern)
        self.start = time.time()
        self.sent = []
        self.sent_at = {}
        self.close = []
        self.excluded = set()
        self.ema = {}
        self.solved_at = None
        self.last_guess = 0.0
        self.frames = 0
        self.compounds = {}

    def update_pattern(self, pattern):
        self.pattern = pattern
        self.regex = pattern_regex(pattern)


class Guesser:
    def __init__(self, cfg):
        self.cfg = cfg
        self.bank = WordBank(cfg)
        print(f"{len(self.bank.words)} confirmed words", file=sys.stderr)
        self.scorer = Scorer(self.bank)
        self.stats = Stats()
        self.rnd = None
        self.log = []
        self.panel_js = cfg.panel_file.read_text(encoding="utf-8")
        if cfg.save_frames:
            cfg.frames_dir.mkdir(exist_ok=True)
        self.driver = self.launch()

    def launch(self):
        opts = Options()
        opts.add_argument(f"--user-data-dir={self.cfg.profile_dir.resolve()}")
        opts.add_argument("--mute-audio")
        opts.add_argument("--window-size=1700,1000")
        driver = webdriver.Chrome(options=opts)
        if self.cfg.block_ads:
            # The fullscreen preroll (AdInPlay) can start before panel.js is
            # injected, so the ad networks are refused at the request level
            # through DevTools. Nothing loads, so there is nothing to remove.
            driver.execute_cdp_cmd("Network.enable", {})
            driver.execute_cdp_cmd("Network.setBlockedURLs", {"urls": AD_HOSTS})
        driver.get("https://skribbl.io/")
        self.inject(driver)
        return driver

    def inject(self, driver):
        driver.execute_script(self.panel_js)
        driver.execute_script("window.clipbot.setScale(arguments[0]);", self.cfg.scale)

    def push(self, msg):
        self.log.insert(0, time.strftime("%H:%M:%S ") + msg)
        del self.log[8:]

    def panel(self, candidates=None, meta=""):
        self.driver.execute_script("window.clipbot.update(arguments[0]);", {
            "pattern": self.rnd.pattern if self.rnd else "",
            "meta": meta,
            "candidates": candidates or [],
            "stats": self.stats.summary(),
            "log": self.log,
        })

    def send(self, word, reason):
        if not self.driver.execute_script("return window.clipbot.send(arguments[0]);", word):
            return
        rnd = self.rnd
        rnd.sent.append(word)
        rnd.sent_at[word.lower()] = time.time()
        rnd.excluded.add(word.lower())
        rnd.last_guess = time.time()
        self.stats.sent += 1
        self.push(f"sent {word} ({reason})")

    def finish_round(self, answer):
        rnd = self.rnd
        if rnd is None:
            return
        answer = answer.strip()
        if not answer:
            self.push("round ended without an answer line")
            self.rnd = None
            return
        in_list = answer.lower() in self.bank.seen
        if answer and not in_list and self.bank.add(answer):
            self.scorer.add_words([answer])
            self.stats.new_words += 1
        self.stats.in_list += int(in_list)
        with self.cfg.stats_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "pattern": rnd.pattern, "answer": answer, "in_list": in_list,
                "sent": rnd.sent, "solved": rnd.solved_at is not None,
                "time": (rnd.solved_at - rnd.start) if rnd.solved_at else None,
                "top": sorted(rnd.ema.items(), key=lambda kv: -kv[1])[:5],
            }) + "\n")
        if self.cfg.save_frames:
            label = re.sub(r"[^a-z0-9]", "_", answer.lower())
            for f in self.cfg.frames_dir.glob(f"pending_{rnd.start:.0f}_*.png"):
                f.rename(self.cfg.frames_dir / f.name.replace("pending", label))
        self.push(f"answer: {answer}" + ("" if in_list else " (added to list)"))
        self.rnd = None

    def handle_chat(self, items, me):
        # Chat line formats seen in the wild:
        #   "<name>: <text>"             player message (kind BASE while guessing,
        #                                GUESSCHAT after that player has solved)
        #   "<name> guessed the word!"   kind JOIN, class "guessed"
        #   "<guess> is close!"          kind CLOSE, only shown for your own guesses
        #   "The word was '<answer>'"    kind JOIN, reliable end-of-round marker
        #   "<name> is drawing now!"     kind DRAWING
        # Messages from solved players are hidden from unsolved ones, so a
        # BASE-colored player line is always a failed guess by someone still
        # playing the round.
        for item in items:
            msg = item["text"]
            m = re.match(r"^The word was '(.+)'$", msg)
            if m:
                self.finish_round(m.group(1))
                continue
            if msg.endswith(" is drawing now!"):
                # Belt and braces: the hint bar hides between rounds and the
                # answer line ends the round, but if either is missed this
                # still prevents state carrying into the next drawer's turn.
                if self.rnd is not None:
                    self.finish_round("")
                continue
            rnd = self.rnd
            if rnd is None:
                continue
            if msg == f"{me} guessed the word!" or msg == "You guessed the word!":
                if rnd.solved_at is None:
                    rnd.solved_at = time.time()
                    self.stats.solved += 1
                    self.stats.correct += 1
                    self.stats.solve_times.append(rnd.solved_at - rnd.start)
                    self.push(f"correct in {rnd.solved_at - rnd.start:.1f}s")
                    self.driver.execute_script("window.clipbot.celebrate();")
                continue
            m = re.match(r"^'?(.+?)'? is close!$", msg)
            if m:
                rnd.excluded.add(m.group(1).lower())
                rnd.close.append(m.group(1).lower())
                self.push(f"close: {m.group(1)}")
            elif item["who"] and item["msg"] and item["kind"] == "BASE":
                rnd.excluded.add(item["msg"].lower())

    def candidates(self):
        rnd, bank, cfg = self.rnd, self.bank, self.cfg
        now = time.time()
        far = [g for g, t in rnd.sent_at.items() if g not in rnd.close and now - t > 1.0]

        def matches(pool):
            out = [i for i, w in enumerate(pool) if rnd.regex.match(w) and w not in rnd.excluded]
            if far:
                out = [i for i in out if all(edit_distance(pool[i], g) > 1 for g in far)]
            return out

        cands = [(bank.words[i], self.scorer.text[i], True) for i in matches(bank.lower)]
        # Fallback words are always scored. step() keeps only the strongest few
        # of them unless the confirmed pool is nearly exhausted, and they carry
        # a prior penalty, so a confirmed word at the same image score wins.
        if self.scorer.extra is not None:
            cands += [(bank.fallback[i], self.scorer.extra[i], False)
                      for i in matches(bank.fallback_lower) if bank.fallback_lower[i] not in bank.seen]
        if rnd.close:
            near = [c for c in cands if all(edit_distance(c[0].lower(), k) <= 1 for k in rnd.close)]
            if near:
                cands = near
        return cands

    def compound_candidates(self, img):
        # Multi-word answers with no confirmed match. Each slot's fallback
        # matches are scored against the drawing on their own, the top few per
        # slot are combined, and the compounds are encoded once per round.
        rnd, bank, cfg = self.rnd, self.bank, self.cfg
        parts = rnd.pattern.split(" ")
        if not 2 <= len(parts) <= 3 or self.scorer.extra is None:
            return []
        slots = []
        for part in parts:
            rx = pattern_regex(part)
            idx = [i for i, w in enumerate(bank.fallback_lower) if rx.match(w)]
            if not idx:
                return []
            p = self.scorer.score_feat(img, self.scorer.extra[idx])
            slots.append([bank.fallback[idx[i]] for i in np.argsort(-p)[:cfg.compound_top]])
        combos = [" ".join(c) for c in itertools.product(*slots)]
        combos = [c for c in combos if c.lower() not in bank.seen and c.lower() not in rnd.excluded]
        # Encoding is bounded per frame so a churning ranking early in the
        # round cannot blow the poll interval; stragglers land next poll.
        new = [c for c in combos if c not in rnd.compounds][:cfg.compound_encode_per_frame]
        if new:
            for c, f in zip(new, self.scorer.encode(new, verbose=False)):
                rnd.compounds[c] = f
        return [(c, rnd.compounds[c], False) for c in combos if c in rnd.compounds]

    def step(self, st):
        cfg, stats = self.cfg, self.stats
        self.handle_chat(st["chat"], st["me"])

        if st["revealVisible"]:
            self.finish_round(st["revealWord"])
            self.panel(meta="round over")
            return

        if not st["pattern"] or st["drawing"]:
            self.rnd = None
            self.panel(meta="you are drawing" if st["drawing"] else "")
            return

        # A new round is detected by the slot count changing. Same-length
        # consecutive rounds are separated by the "The word was" chat line,
        # which sets rnd to None in between. Revealed letters change the
        # pattern without changing its length, so those just update the regex.
        if self.rnd is None or len(st["pattern"]) != len(self.rnd.pattern):
            self.rnd = Round(st["pattern"])
            stats.rounds += 1
            self.push(f"new round {st['pattern']}")
        elif st["pattern"] != self.rnd.pattern:
            self.rnd.update_pattern(st["pattern"])
        rnd = self.rnd

        for word in st["queue"]:
            self.send(word, "manual")

        if rnd.solved_at is not None or not st["canvas"]:
            self.panel(meta="solved" if rnd.solved_at else "")
            return

        cands = self.candidates()
        trusted_n = sum(1 for c in cands if c[2])
        multiword = " " in rnd.pattern and trusted_n < cfg.fallback_min_trusted and self.scorer.extra is not None
        if not cands and not multiword:
            self.panel(meta="no candidates match")
            return
        pool = f"{trusted_n} confirmed" + (f" + {len(cands) - trusted_n} fallback" if len(cands) > trusted_n else "")

        image = Image.open(io.BytesIO(base64.b64decode(st["canvas"].split(",", 1)[1]))).convert("RGB")
        ink = ink_fraction(image)
        if ink < cfg.min_ink_fraction:
            # Nothing drawn yet. Show the pool unsorted, or by the previous
            # scores if the drawer cleared the canvas mid-round.
            share = dict(softmax_rank([(w, rnd.ema[w]) for w, _, _ in cands if w in rnd.ema]))
            rows = [{"word": w, "p": share.get(w, 0.0), "sent": w in rnd.sent, "trusted": t} for w, _, t in cands]
            rows.sort(key=lambda c: (-c["trusted"], -c["p"]))
            self.panel(rows[:cfg.top_n], meta=f"{pool}, waiting for strokes")
            return

        img = self.scorer.image_feat(image)
        if multiword:
            extra = self.compound_candidates(img)
            cands = cands + extra
            if extra:
                pool += f" + {len(extra)} compound"
        if not cands:
            self.panel(meta="no candidates match")
            return

        logits = self.scorer.score_feat(img, torch.stack([c[1] for c in cands]))
        rnd.frames += 1
        scored = list(zip(cands, logits))
        if trusted_n >= cfg.fallback_min_trusted:
            loose = sorted((s for s in scored if not s[0][2]), key=lambda s: -s[1])[:cfg.fallback_keep]
            scored = [s for s in scored if s[0][2]] + loose
        # Smooth the logits, not the probabilities: the softmax is taken over
        # whatever pool survives this frame, so smoothing after it would lag
        # every time hints or chat shrink the pool.
        log_prior = math.log(cfg.fallback_prior)
        trust = {}
        for (w, _, t), l in scored:
            l = float(l) + (0.0 if t else log_prior)
            rnd.ema[w] = cfg.ema_alpha * l + (1 - cfg.ema_alpha) * rnd.ema.get(w, l)
            trust[w] = t
        ranked = softmax_rank([(w, rnd.ema[w]) for w in trust])

        if cfg.save_frames and rnd.frames % 5 == 0:
            top = re.sub(r"[^a-z0-9]", "_", ranked[0][0].lower())
            image.save(cfg.frames_dir / f"pending_{rnd.start:.0f}_{rnd.frames:03d}_{top}.png")

        # Confirmed words are pinned: every one is shown up to the row cap, and
        # fallback words only fill leftover slots. Without this a large
        # fallback pool pushes every confirmed word off the board.
        show_loose = trusted_n < cfg.fallback_min_trusted
        confirmed = [(w, p) for w, p in ranked if trust[w]]
        loose = [(w, p) for w, p in ranked if not trust[w] and (show_loose or p >= cfg.fallback_show_prob)]
        rows = confirmed[:cfg.top_n] + loose[:max(0, cfg.top_n - len(confirmed))]
        rows.sort(key=lambda r: -r[1])
        self.panel(
            [{"word": w, "p": p, "sent": w in rnd.sent, "trusted": trust[w]} for w, p in rows],
            meta=f"{pool}, ink {ink:.1%}, {st['clock']}s left",
        )

    def run(self):
        while True:
            time.sleep(self.cfg.poll_seconds)
            try:
                st = self.driver.execute_script("return window.clipbot.read();")
                if st is not None:
                    self.step(st)
            except NoSuchWindowException:
                print("browser window closed, exiting", file=sys.stderr)
                return
            except WebDriverException as e:
                # Navigating (clicking the logo, room change) wipes the
                # injected script; re-inject. Anything else transient from
                # the driver is logged and the next poll tries again.
                if "clipbot" in str(e):
                    self.inject(self.driver)
                else:
                    print(f"webdriver: {str(e).splitlines()[0]}", file=sys.stderr)


if __name__ == "__main__":
    Guesser(parse_args()).run()
