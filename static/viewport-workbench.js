(() => {
  'use strict';
  const byId = id => document.getElementById(id);
  const screens = [...document.querySelectorAll('.work-screen')];
  const form = byId('brush-form');
  const brushFeedback = byId('brush-feedback');
  const completionFeedback = byId('completion-feedback');
  let current = 'inspect', pendingPreview = false, pendingCompletion = false, pendingEstimate = false;

  // Native radios provide keyboard-accessible button choices while the existing
  // controls remain the source of truth for the restoration/completion logic.
  document.querySelectorAll('.choice-input').forEach(input => {
    const select = byId(input.dataset.select);
    input.addEventListener('change', () => {
      if (!input.checked || select.value === input.value) return;
      select.value = input.value;
      select.dispatchEvent(new Event('change', { bubbles: true }));
    });
    select.addEventListener('change', () => { input.checked = select.value === input.value; });
  });

  function fitEditor(visual) {
    const canvas = visual.querySelector('canvas');
    const caption = visual.querySelector('.visual-caption');
    if (!canvas || !canvas.width || !canvas.height || !visual.getClientRects().length) return;
    const styles = getComputedStyle(visual);
    const horizontal = parseFloat(styles.paddingLeft) + parseFloat(styles.paddingRight);
    const vertical = parseFloat(styles.paddingTop) + parseFloat(styles.paddingBottom);
    const availableWidth = Math.max(0, visual.clientWidth - horizontal);
    const availableHeight = Math.max(0, visual.clientHeight - vertical - (caption?.offsetHeight || 0) - parseFloat(styles.rowGap || 0));
    const scale = Math.min(availableWidth / canvas.width, availableHeight / canvas.height, 600 / canvas.width, 600 / canvas.height);
    if (!(scale > 0)) return;
    canvas.style.width = (canvas.width * scale) + 'px';
    canvas.style.height = (canvas.height * scale) + 'px';
  }
  // Match the editor's display size while preserving the oriented image pixels.
  const uploadPreview = byId('upload-preview');
  const inspectVisual = uploadPreview.closest('.inspect-visual');
  function fitUploadPreview() {
    if (uploadPreview.classList.contains('hidden') || !uploadPreview.naturalWidth || !uploadPreview.naturalHeight || !inspectVisual.getClientRects().length) return;
    const styles = getComputedStyle(inspectVisual);
    const availableWidth = Math.max(0, inspectVisual.clientWidth - parseFloat(styles.paddingLeft) - parseFloat(styles.paddingRight));
    const availableHeight = Math.max(0, inspectVisual.clientHeight - parseFloat(styles.paddingTop) - parseFloat(styles.paddingBottom));
    const scale = Math.min(availableWidth / uploadPreview.naturalWidth, availableHeight / uploadPreview.naturalHeight, 600 / uploadPreview.naturalWidth, 600 / uploadPreview.naturalHeight);
    if (!(scale > 0)) return;
    uploadPreview.style.width = (uploadPreview.naturalWidth * scale) + 'px';
    uploadPreview.style.height = (uploadPreview.naturalHeight * scale) + 'px';
  }
  uploadPreview.addEventListener('load', fitUploadPreview);
  new ResizeObserver(fitUploadPreview).observe(inspectVisual);
  new MutationObserver(fitUploadPreview).observe(uploadPreview, { attributes: true, attributeFilter: ['class'] });
  const editorObserver = new ResizeObserver(entries => entries.forEach(entry => fitEditor(entry.target)));
  document.querySelectorAll('.editor-visual').forEach(visual => editorObserver.observe(visual));
  function syncProgress() {
    const progress = byId('brush-progress');
    const redundant = progress.textContent.startsWith('Inspect visible facial structure before continuing.') ||
      (current === 'inspect' && (
        progress.textContent.startsWith('V2 (p2_u0224) is available.') ||
        progress.textContent === 'Inspect the original crop before continuing.'
      ));
    progress.classList.toggle('quiet-status', redundant);
  }
  new MutationObserver(syncProgress).observe(byId('brush-progress'), {childList: true, characterData: true, subtree: true});
  function showNotice(message) {
    const notice = byId('workbench-notice');
    notice.textContent = message;
    notice.classList.toggle('hidden', !message);
  }
  byId('brush-file').addEventListener('change', () => showNotice(''));
  function show(name, focus = true) {
    const screen = screens.find(item => item.dataset.screen === name);
    if (!screen || screen.closest('.hidden')) return false;
    if (name === 'options') {
      const original = byId('upload-preview').getAttribute('src');
      if (original) byId('options-preview').setAttribute('src', original);
      else byId('options-preview').removeAttribute('src');
    }
    if (name === 'options' && current === 'result') byId('brush-progress').textContent = 'Choose restoration and preparation.';
    if (name === 'inspect' && current !== 'inspect') byId('brush-progress').textContent = 'Inspect the original crop before continuing.';
    current = name;
    syncProgress();
    screens.forEach(item => item.classList.toggle('is-active', item === screen));
    const slot = screen.querySelector('.feedback-slot');
    if (slot && form.contains(screen)) slot.append(brushFeedback);
    if (slot && byId('completion-stage').contains(screen)) slot.append(completionFeedback);
    showNotice('');
    screen.querySelectorAll('.editor-visual').forEach(fitEditor);
    requestAnimationFrame(() => { screen.querySelectorAll('.editor-visual').forEach(fitEditor); if (name === 'inspect') fitUploadPreview(); });
    if (focus) {
      const heading = screen.querySelector('h1');
      if (heading) { heading.tabIndex = -1; heading.focus({ preventScroll: true }); }
    }
    return true;
  }
  function go(name) {
    if (name === 'options') {
      if (byId('upload-preview').classList.contains('hidden')) return showNotice('Choose and inspect one crop before continuing.');
      const review = byId('brush-review').value;
      if (review !== 'usable') return showNotice(review ? 'Choose a clearer, less-covered, single face crop before continuing.' : 'Review the visible facial information before continuing.');
    }
    if (name === 'options-next') {
      name = byId('brush-preparation').value === 'manual' && byId('brush-restoration').value !== 'off' ? 'brush-paint' : 'run';
    }
    if (name === 'run-back') {
      name = byId('brush-preparation').value === 'manual' && byId('brush-restoration').value !== 'off' ? 'brush-preview' : 'options';
    }
    if (name === 'run' && byId('brush-preparation').value === 'manual' && byId('brush-restoration').value !== 'off') {
      if (byId('brush-preview-section').classList.contains('hidden') || !byId('brush-confirm').checked) return showNotice('Review and confirm the current preparation preview first.');
    }
    show(name);
  }
  document.addEventListener('click', event => {
    const nav = event.target.closest('[data-nav]');
    if (nav) { go(nav.dataset.nav); return; }
    const opener = event.target.closest('[data-dialog]');
    if (opener) { byId(opener.dataset.dialog).showModal(); return; }
    const closer = event.target.closest('[data-close-dialog]');
    if (closer) { closer.closest('dialog').close(); return; }
    const zoom = event.target.closest('[data-zoom]');
    if (zoom) {
      const image = byId(zoom.dataset.zoom);
      if (!image.getAttribute('src')) return;
      const view = byId('image-dialog-view');
      view.src = image.src; view.alt = image.alt;
      byId('image-dialog-title').textContent = image.alt;
      byId('image-dialog').showModal();
    }
  });
  form.addEventListener('submit', event => {
    if (current !== 'run') { event.preventDefault(); event.stopImmediatePropagation(); }
  }, true);
  byId('brush-preview').addEventListener('click', () => { pendingPreview = true; });
  byId('completion-start').addEventListener('click', () => { pendingCompletion = true; });
  byId('completion-generate').addEventListener('click', () => { pendingEstimate = true; });
  const observe = (id, ready) => new MutationObserver(() => {
    if (!byId(id).classList.contains('hidden')) ready();
  }).observe(byId(id), { attributes: true, attributeFilter: ['class'] });
  observe('brush-preview-section', () => { if (pendingPreview) { pendingPreview = false; show('brush-preview'); } });
  observe('completion-editor-section', () => { if (pendingCompletion) { pendingCompletion = false; show('completion-paint'); } });
  observe('completion-result-section', () => { if (pendingEstimate) { pendingEstimate = false; show('completion-result'); } });
  window.addEventListener('restoration:completed', () => show('result'));
  show('inspect', false);
})();
