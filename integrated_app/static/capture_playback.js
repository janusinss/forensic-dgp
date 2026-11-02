/* Video playback speed controls; never starts or seeks the video. */
((root) => {
  'use strict';
  const initialized = new WeakSet();

  function boundedRate(value) {
    if ((typeof value !== 'number' && typeof value !== 'string') ||
        (typeof value === 'string' && value.trim() === '')) throw new RangeError('Invalid playback speed.');
    const rate = Number(value);
    if (!Number.isFinite(rate) || rate < .25 || rate > 4) {
      throw new RangeError('Invalid playback speed.');
    }
    return rate;
  }

  function normalizeRate(value) {
    const rate = boundedRate(value);
    if (!Number.isInteger(rate * 4)) throw new RangeError('Invalid playback speed.');
    return rate;
  }

  function stepRate(current, direction) {
    const rate = boundedRate(current);
    if (direction !== -1 && direction !== 1) throw new RangeError('Invalid speed direction.');
    const quarters = rate * 4;
    const next = direction > 0 ? Math.floor(quarters) + 1 : Math.ceil(quarters) - 1;
    return Math.max(.25, Math.min(4, next / 4));
  }

  function setup(document) {
    const video = document.getElementById('capture-video');
    const select = document.getElementById('capture-speed-select');
    const minus = document.getElementById('capture-speed-minus');
    const plus = document.getElementById('capture-speed-plus');
    const error = document.getElementById('capture-speed-error');
    if (!video || !select || !minus || !plus || !error || initialized.has(select)) return;
    initialized.add(select);
    if (document.createElement && select.options && select.options.length === 0) {
      for (let i = 1; i <= 16; i++) {
        const option = document.createElement('option');
        option.value = String(i / 4);
        option.textContent = `${i / 4}×`;
        select.appendChild(option);
      }
    }
    let current = 1;
    let pending = false;
    let temporaryOption = null;
    function render() {
      if (temporaryOption) { temporaryOption.remove(); temporaryOption = null; }
      if (!Number.isInteger(current * 4) && document.createElement && select.options) {
        temporaryOption = document.createElement('option');
        temporaryOption.value = String(current);
        temporaryOption.textContent = `${current}×`;
        select.appendChild(temporaryOption);
      }
      select.value = String(current);
      select.disabled = pending;
      minus.disabled = pending || current === .25;
      plus.disabled = pending || current === 4;
    }
    function apply(value, internal = false) {
      if (pending && !internal) { render(); return; }
      try {
        const next = boundedRate(value);
        if (video.defaultPlaybackRate !== next) video.defaultPlaybackRate = next;
        if (video.playbackRate !== next) video.playbackRate = next;
        current = next;
        error.textContent = '';
      } catch (_) {
        try { if (video.defaultPlaybackRate !== current) video.defaultPlaybackRate = current; } catch (_) { /* Preserve the current speed. */ }
        try { if (video.playbackRate !== current) video.playbackRate = current; } catch (_) { /* Preserve the local error. */ }
        error.textContent = 'This playback speed is unavailable in your browser.';
      }
      render();
    }
    select.addEventListener('change', () => apply(select.value));
    minus.addEventListener('click', () => apply(stepRate(current, -1)));
    plus.addEventListener('click', () => apply(stepRate(current, 1)));
    video.addEventListener('loadedmetadata', () => apply(1, true));
    video.addEventListener('ratechange', () => {
      try {
        const next = boundedRate(video.playbackRate);
        if (video.defaultPlaybackRate !== next) video.defaultPlaybackRate = next;
        if (next !== current) error.textContent = '';
        current = next;
      } catch (_) {
        try { if (video.defaultPlaybackRate !== current) video.defaultPlaybackRate = current; } catch (_) { /* Keep the local error visible. */ }
        try { if (video.playbackRate !== current) video.playbackRate = current; } catch (_) { /* Keep the local error visible. */ }
        error.textContent = 'This playback speed is unavailable in your browser.';
      }
      render();
    });
    video.addEventListener('capture:pending', (event) => {
      pending = event.detail?.pending === true;
      render();
    });
    render();
  }

  const api = Object.freeze({ normalizeRate, stepRate, setup });
  if (typeof module === 'object' && module.exports) module.exports = api;
  else {
    root.capturePlayback = api;
    if (root.document) {
      if (root.document.readyState === 'loading') root.document.addEventListener('DOMContentLoaded', () => setup(root.document), { once: true });
      else setup(root.document);
    }
  }
})(globalThis);
