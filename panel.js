(function () {
  if (window.clipbot) return;

  // Ad slots on the page: AdSense <ins> units, the 336x280 on the home panel
  // (.ad-side), the 728x90 above the game (.ad-1, inside #game-logo), the
  // 300x250 above chat (.ad-2), the AdInPlay preroll container and its
  // fullscreen player, the injected aswift_N iframes, and the anchor/side-rail
  // grippy. AdSense re-inserts some of these after load, hence the observer.
  // The preroll can fire before this script is injected, so the Python side
  // also blocks the ad networks at the request level through DevTools; this
  // list is the in-page backstop.
  const AD_SELECTORS = [
    'ins.adsbygoogle', '.ad-side', '.ad-1', '.ad-2', '#preroll',
    '[id^="aswift_"]', '[id^="aip"]', '.aip-player', '.grippy-host', '#game-logo'
  ];
  function killAds() {
    AD_SELECTORS.forEach(function (s) {
      document.querySelectorAll(s).forEach(function (el) { el.remove(); });
    });
  }
  killAds();
  new MutationObserver(killAds).observe(document.body, { childList: true, subtree: true });

  const css = `
    /* The panel lives inside #game-chat and splits it with the chat log.
       Fonts, sizes, and colors follow skribbl's own chat styling so it
       reads as part of the game. */
    #game-chat { display: flex !important; flex-direction: column !important; }
    #game-chat .chat-content { flex: 1 1 0 !important; min-height: 0 !important; height: auto !important; }
    #clipbot {
      flex: 1 1 0; min-height: 0; overflow-y: auto;
      margin-top: 6px; border-top: 2px solid rgba(0, 0, 0, 0.1);
      background: inherit; color: var(--COLOR_CHAT_TEXT_BASE, #333);
      font-family: Nunito, "Segoe UI", sans-serif; font-size: 13px; line-height: 1.5;
    }
    #clipbot.cb-floating {
      position: fixed; top: 24px; right: 24px; width: 320px; max-height: 80vh; z-index: 2147483000;
      background: #fff; border: 0; border-radius: 4px; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25); margin: 0;
    }
    #clipbot *, #clipbot-dock * { box-sizing: border-box; }
    #clipbot .cb-head {
      position: sticky; top: 0; z-index: 1;
      display: flex; align-items: center; gap: 6px; padding: 8px 10px;
      background: #f0f0f0; font-weight: 700;
      box-shadow: 0 1px 0 rgba(0, 0, 0, 0.1);
    }
    #clipbot .cb-head b { font-weight: 800; margin-right: auto; }
    #clipbot .cb-hint { font-size: 11px; font-weight: 400; opacity: 0.6; }
    #clipbot .cb-section { padding: 8px 10px; }
    #clipbot .cb-title { font-weight: 800; font-size: 12px; opacity: 0.6; margin-bottom: 4px; }
    #clipbot .cb-pattern {
      font-family: Inconsolata, monospace; font-size: 22px; letter-spacing: 0.15em; text-align: center;
      padding: 6px; border-radius: 3px; background: rgba(0, 0, 0, 0.05);
    }
    #clipbot .cb-meta { opacity: 0.7; margin-top: 4px; font-size: 12px; }
    #clipbot table { width: 100%; border-collapse: collapse; }
    #clipbot td { padding: 4px 4px; vertical-align: middle; }
    #clipbot tr:nth-child(odd) td { background: rgba(0, 0, 0, 0.04); }
    #clipbot tr.cb-cand { cursor: pointer; }
    #clipbot tr.cb-cand:hover td { background: rgba(0, 0, 0, 0.1); }
    #clipbot tr.cb-sent td { opacity: 0.45; text-decoration: line-through; cursor: default; }
    #clipbot tr.cb-loose td { opacity: 0.6; font-style: italic; }
    #clipbot td.cb-p { text-align: right; width: 54px; font-variant-numeric: tabular-nums; font-weight: 700; color: var(--COLOR_CHAT_TEXT_JOIN, #2a7); }
    #clipbot tr.cb-loose td.cb-p { color: inherit; font-weight: 400; }
    #clipbot .cb-bar { height: 3px; border-radius: 2px; background: var(--COLOR_CHAT_TEXT_JOIN, #2a7); margin-top: 2px; }
    #clipbot tr.cb-loose .cb-bar { background: rgba(0, 0, 0, 0.2); }
    #clipbot .cb-stats td:last-child { text-align: right; font-variant-numeric: tabular-nums; font-weight: 700; }
    #clipbot .cb-log { font-size: 12px; opacity: 0.8; }
    #clipbot .cb-log div { padding: 2px 0; }

    #clipbot-dock {
      position: fixed; z-index: 2147483000;
      display: none; flex-wrap: wrap; justify-content: center; gap: 10px;
      padding: 12px; background: #fff; border-radius: 4px; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
      font-family: Nunito, "Segoe UI", sans-serif; font-size: 22px; font-weight: 700; color: #333;
    }
    #clipbot-dock.cb-show { display: flex; }
    #clipbot-dock .cb-chip {
      padding: 10px 20px; cursor: pointer; border-radius: 4px; user-select: none; line-height: 1.3;
      background: #fff; border: 2px solid rgba(0, 0, 0, 0.15); box-shadow: 0 2px 0 rgba(0, 0, 0, 0.15);
      transition: transform 60ms, box-shadow 60ms;
    }
    #clipbot-dock .cb-chip:hover { border-color: var(--COLOR_CHAT_TEXT_JOIN, #2a7); background: rgba(0, 200, 120, 0.08); }
    #clipbot-dock .cb-chip:active { transform: translateY(2px); box-shadow: none; }
    #clipbot-dock .cb-chip small { color: var(--COLOR_CHAT_TEXT_JOIN, #2a7); margin-left: 10px; font-size: 14px; font-variant-numeric: tabular-nums; }
    #clipbot-dock .cb-chip.cb-loose { opacity: 0.6; font-weight: 400; font-style: italic; }
    #clipbot-dock .cb-chip.cb-loose small { color: inherit; }
    #clipbot-dock .cb-chip.cb-sent { opacity: 0.4; text-decoration: line-through; cursor: default; }

    .cb-confetti {
      position: fixed; top: -16px; width: 10px; height: 16px; z-index: 2147483001; pointer-events: none;
      animation-name: cb-fall; animation-timing-function: linear; animation-fill-mode: forwards;
    }
    @keyframes cb-fall {
      0%   { transform: translateY(0) rotate(0deg); opacity: 1; }
      100% { transform: translateY(105vh) rotate(720deg); opacity: 0.9; }
    }
  `;
  const style = document.createElement('style');
  style.textContent = css;
  document.head.appendChild(style);

  const root = document.createElement('div');
  root.id = 'clipbot';
  root.innerHTML = `
    <div class="cb-head"><b>Guesser</b><span class="cb-hint">click a word to guess it</span></div>
    <div class="cb-section">
      <div class="cb-pattern"></div>
      <div class="cb-meta"></div>
    </div>
    <div class="cb-section cb-cands-section">
      <div class="cb-title">Candidates</div>
      <table class="cb-cands"></table>
    </div>
    <div class="cb-section">
      <div class="cb-title">Session record</div>
      <table class="cb-stats"></table>
    </div>
    <div class="cb-section">
      <div class="cb-title">Log</div>
      <div class="cb-log"></div>
    </div>
  `;
  // Mount inside #game-chat when it exists so the panel takes the lower half
  // of the chat column; fall back to floating over the page otherwise.
  function mount() {
    const chat = document.getElementById('game-chat');
    if (chat && root.parentElement !== chat) {
      chat.appendChild(root);
      root.classList.remove('cb-floating');
    } else if (!chat && root.parentElement !== document.body) {
      document.body.appendChild(root);
      root.classList.add('cb-floating');
    }
  }
  mount();

  const dock = document.createElement('div');
  dock.id = 'clipbot-dock';
  document.body.appendChild(dock);

  const el = {
    pattern: root.querySelector('.cb-pattern'),
    meta: root.querySelector('.cb-meta'),
    cands: root.querySelector('.cb-cands'),
    stats: root.querySelector('.cb-stats'),
    log: root.querySelector('.cb-log')
  };

  // Each chat <p> is <b>prefix</b><span>text</span>. Player messages have a
  // "name: " prefix; system lines put the whole sentence in <b>. The color is
  // a CSS variable in the inline style (COLOR_CHAT_TEXT_BASE, GUESSCHAT,
  // JOIN, LEAVE, CLOSE, DRAWING, OWNER) and is the cleanest way to tell a
  // live guess from post-solve chatter.
  function parseChat(p) {
    const b = p.querySelector('b');
    const span = p.querySelector('span');
    const isPlayer = !!(b && /:\s*$/.test(b.textContent));
    const color = (p.getAttribute('style') || '').match(/COLOR_CHAT_TEXT_(\w+)/);
    return {
      text: p.textContent.trim(),
      who: isPlayer ? b.textContent.replace(/:\s*$/, '').trim() : '',
      msg: isPlayer && span ? span.textContent.trim() : '',
      kind: color ? color[1] : ''
    };
  }

  const bot = {
    mode: 'manual',
    queue: [],
    chatBuf: [],

    // The dock is shown when there are chips to show and you are not the
    // drawer, since it would cover the toolbar.
    drawing: false,
    applyLayout: function () {
      const show = !bot.drawing && dock.children.length > 0;
      dock.classList.toggle('cb-show', show);
      if (show) bot.placeDock();
    },

    // Chrome's CSS zoom scales layout and upsamples the canvas, and skribbl
    // maps pointer input through the canvas bounding rect, so drawing still
    // works at any scale.
    setScale: function (s) {
      const g = document.getElementById('game');
      if (g) g.style.zoom = s;
      bot.placeDock();
    },

    // Sit directly under the canvas, matching its width, so the chips wrap
    // into two rows right where the eye already is.
    placeDock: function () {
      const c = document.querySelector('#game-canvas');
      if (!c || !dock.classList.contains('cb-show')) return;
      const r = c.getBoundingClientRect();
      dock.style.left = r.left + 'px';
      dock.style.width = r.width + 'px';
      dock.style.top = (r.bottom + 14) + 'px';
    },

    // Two seconds of CSS confetti on a correct guess.
    celebrate: function () {
      const colors = ['#ef130b', '#ff7100', '#ffe400', '#00cc00', '#00b2ff', '#231fd3', '#a300ba'];
      for (let i = 0; i < 140; i++) {
        const piece = document.createElement('div');
        piece.className = 'cb-confetti';
        piece.style.left = (Math.random() * 100) + 'vw';
        piece.style.background = colors[i % colors.length];
        piece.style.borderRadius = '2px';
        piece.style.animationDuration = (2 + Math.random() * 1.5) + 's';
        piece.style.animationDelay = (Math.random() * 0.6) + 's';
        piece.style.transform = 'rotate(' + (Math.random() * 360) + 'deg)';
        document.body.appendChild(piece);
      }
      setTimeout(function () {
        document.querySelectorAll('.cb-confetti').forEach(function (p) { p.remove(); });
      }, 4500);
    },

    myName: function () {
      const n = document.querySelector('.player-name.me');
      return n ? n.textContent.replace(/\s*\(You\)\s*$/, '').trim() : '';
    },

    read: function () {
      // Hint bar: #game-word .hints .container holds one .hint per character.
      // Unknown letters are "_", revealed ones carry class "uncover", and a
      // space is an uncovered .hint whose text is a single space. The
      // .word-length sibling reads "5 4" for two-word answers. The container
      // stays in the DOM with display:none after the round, still holding the
      // full word, so hidden containers must be skipped or the last answer
      // reads as a fresh pattern.
      const parts = [];
      document.querySelectorAll('#game-word .hints .container').forEach(function (c) {
        if (c.offsetParent === null) return;
        let s = '';
        c.querySelectorAll('.hint').forEach(function (h) {
          const t = (h.textContent || '').replace(/\u00a0/g, ' ');
          s += t.length ? t : ' ';
        });
        parts.push(s);
      });
      const pattern = parts.join(' ').replace(/\s+/g, ' ').trim();

      // #game-word .word is only visible when you are the drawer; otherwise
      // its text is stale from your last turn.
      const wordEl = document.querySelector('#game-word .word');
      const drawing = !!(wordEl && wordEl.offsetParent !== null && wordEl.textContent.trim());
      if (drawing !== bot.drawing) { bot.drawing = drawing; bot.applyLayout(); }

      // The reveal overlay (#game-canvas .reveal) is visible for a few seconds
      // at round end with the answer in .word. The chat line is the same
      // information and arrives first, so this is a backup signal.
      const revealEl = document.querySelector('#game-canvas .reveal');
      const revealVisible = !!(revealEl && revealEl.offsetParent !== null);
      const revealWord = revealVisible ? (revealEl.querySelector('.word') || {}).textContent || '' : '';

      const clock = (document.querySelector('#game-clock .text') || {}).textContent || '';

      let canvas = null;
      if (pattern && !drawing && !revealVisible) {
        const c = document.querySelector('#game-canvas canvas');
        if (c) canvas = c.toDataURL('image/png');
      }

      return {
        pattern: pattern, drawing: drawing, revealVisible: revealVisible, revealWord: revealWord.trim(),
        chat: bot.chatBuf.splice(0), clock: clock, canvas: canvas, mode: bot.mode,
        queue: bot.queue.splice(0), me: bot.myName()
      };
    },

    // skribbl listens for the form submit; setting the value and calling
    // requestSubmit is enough. The keydown fallback is for older engines.
    send: function (text) {
      const form = document.querySelector('#game-chat form.chat-form');
      const input = form && form.querySelector('input');
      if (!input) return false;
      input.value = text;
      input.dispatchEvent(new Event('input', { bubbles: true }));
      if (form.requestSubmit) form.requestSubmit();
      else input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', keyCode: 13, bubbles: true }));
      return true;
    },

    update: function (data) {
      mount();
      el.pattern.textContent = data.pattern || '(waiting for round)';
      el.meta.textContent = data.meta || '';
      const cands = data.candidates || [];

      el.cands.innerHTML = '';
      cands.forEach(function (c) {
        const tr = document.createElement('tr');
        tr.className = 'cb-cand' + (c.sent ? ' cb-sent' : '') + (c.trusted === false ? ' cb-loose' : '');
        if (!c.sent) tr.dataset.word = c.word;
        tr.innerHTML = '<td>' + c.word + '<div class="cb-bar" style="width:' + Math.round(c.p * 100) + '%"></div></td>' +
                       '<td class="cb-p">' + (c.p * 100).toFixed(1) + '%</td>';
        el.cands.appendChild(tr);
      });

      dock.innerHTML = '';
      cands.slice(0, 10).forEach(function (c) {
        const chip = document.createElement('span');
        chip.className = 'cb-chip' + (c.sent ? ' cb-sent' : '') + (c.trusted === false ? ' cb-loose' : '');
        if (!c.sent) chip.dataset.word = c.word;
        chip.innerHTML = c.word + '<small>' + Math.round(c.p * 100) + '%</small>';
        dock.appendChild(chip);
      });
      bot.applyLayout();

      el.stats.innerHTML = '';
      Object.keys(data.stats || {}).forEach(function (k) {
        const tr = document.createElement('tr');
        tr.innerHTML = '<td>' + k + '</td><td>' + data.stats[k] + '</td>';
        el.stats.appendChild(tr);
      });

      if (data.log) el.log.innerHTML = data.log.map(function (l) { return '<div>' + l + '</div>'; }).join('');
    }
  };

  // Chat is captured as it arrives rather than re-read on each poll. skribbl
  // trims old <p> elements off the top of .chat-content once it grows, so
  // any index-based "messages since last time" approach silently drops
  // guesses in busy rooms.
  new MutationObserver(function (muts) {
    muts.forEach(function (m) {
      m.addedNodes.forEach(function (n) {
        if (n.nodeType === 1 && n.tagName === 'P' && n.parentElement && n.parentElement.classList.contains('chat-content')) {
          bot.chatBuf.push(parseChat(n));
        }
      });
    });
  }).observe(document.body, { childList: true, subtree: true });

  // Rows and chips are rebuilt every poll, and a click needs mousedown and
  // mouseup on the same element, so listen for mousedown on the containers.
  function pick(e) {
    const t = e.target.closest('[data-word]');
    if (t) { bot.queue.push(t.dataset.word); e.preventDefault(); }
  }
  el.cands.addEventListener('mousedown', pick);
  dock.addEventListener('mousedown', pick);

  window.addEventListener('resize', bot.placeDock);
  window.clipbot = bot;
})();
