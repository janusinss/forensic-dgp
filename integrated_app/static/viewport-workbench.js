(() => {
  'use strict';
  const byId = id => document.getElementById(id);
  const screens = [...document.querySelectorAll('.work-screen')];
  const form = byId('brush-form');
  const brushFeedback = byId('brush-feedback');
  const completionFeedback = byId('completion-feedback');
  let current = 'inspect', pendingPreview = false, pendingEstimate = false;
  const stageTitles = {
    inspect: '', options: 'Restoration settings',
    'brush-paint': 'Protect and blur', 'brush-preview': 'Review preparation',
    run: 'Restore the picture', result: 'Review restoration',
    'completion-paint': 'Select the removal area',
    'completion-tools': 'Removal tools', 'completion-confirm': 'Confirm removal',
    'completion-result': 'Review removal'
  };
  function updateTabTitle() {
    const screen = screens.find(item => item.dataset.screen === current);
    const loading = screen?.classList.contains('is-loading');
    const stage = loading && current === 'run' ? 'Restoring…'
      : loading && current === 'completion-confirm' ? 'Generating…' : stageTitles[current];
    document.title = stage ? `Forensic DGP · ${stage}` : 'Forensic DGP';
  }
  const titleObserver = new MutationObserver(updateTabTitle);
  screens.filter(screen => ['run', 'completion-confirm'].includes(screen.dataset.screen))
    .forEach(screen => titleObserver.observe(screen, { attributes: true, attributeFilter: ['class'] }));

  // This guide describes the current screen; it does not change the workflow.
  const journey = [
    { label: 'Picture', names: ['inspect'] },
    { label: 'Settings', names: ['options', 'brush-paint', 'brush-preview'] },
    { label: 'Restore', names: ['run'] },
    { label: 'Review', names: ['result'] },
    { label: 'Removal', names: ['completion-paint', 'completion-tools', 'completion-confirm', 'completion-result'], optional: true }
  ];
  screens.forEach(screen => {
    const controls = screen.querySelector('.intake-progress') || screen.querySelector('.work-controls');
    const currentStep = journey.findIndex(step => step.names.includes(screen.dataset.screen));
    if (!controls || currentStep < 0) return;
    const progress = document.createElement('nav');
    progress.className = 'journey-progress';
    progress.setAttribute('aria-label', 'Workflow progress');
    const list = document.createElement('ol');
    journey.forEach((step, index) => {
      const item = document.createElement('li');
      item.className = index === currentStep ? 'is-current' : index < currentStep ? 'is-complete' : 'is-upcoming';
      if (index === currentStep) item.setAttribute('aria-current', 'step');
      const marker = document.createElement('span');
      marker.className = 'journey-marker';
      marker.setAttribute('aria-hidden', 'true');
      const label = document.createElement('span');
      label.className = 'journey-label';
      label.textContent = step.label;
      if (step.optional) {
        const optional = document.createElement('span');
        optional.className = 'journey-optional';
        optional.textContent = 'Optional';
        label.append(' ', optional);
      }
      item.append(marker, label);
      list.append(item);
    });
    progress.append(list);
    controls.prepend(progress);
  });

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

  // Batch fitting into one frame: measure every affected panel before writing
  // sizes, so load, navigation and resize events do not repeatedly force layout.
  const previews = ['options-preview', 'restoration-idle', 'completion-confirm-preview'].map(byId);
  const previewByVisual = new Map(previews.map(image => [image.closest('.work-visual'), image]));
  const pendingFits = new Set();
  let fitFrame = 0;
  function queueFit(visual) {
    if (!visual) return;
    pendingFits.add(visual);
    if (!fitFrame) fitFrame = requestAnimationFrame(flushFits);
  }
  function flushFits() {
    fitFrame = 0;
    const sizes = [];
    pendingFits.forEach(visual => {
      if (!visual.getClientRects().length) return;
      const image = previewByVisual.get(visual);
      const canvas = visual.querySelector('canvas:not(.ambient-shader)');
      const element = image && !image.classList.contains('hidden') ? image : canvas;
      const width = element === image ? image.naturalWidth : canvas?.width;
      const height = element === image ? image.naturalHeight : canvas?.height;
      if (!element || !width || !height) return;
      const styles = getComputedStyle(visual);
      const caption = visual.querySelector('.visual-caption');
      const gap = parseFloat(styles.rowGap) || 0;
      const availableWidth = Math.max(0, visual.clientWidth - parseFloat(styles.paddingLeft) - parseFloat(styles.paddingRight));
      const availableHeight = Math.max(0, visual.clientHeight - parseFloat(styles.paddingTop) - parseFloat(styles.paddingBottom) - (caption?.offsetHeight || 0) - (caption || canvas ? gap : 0));
      const scale = Math.min(availableWidth / width, availableHeight / height, 600 / width, 600 / height);
      if (scale > 0) sizes.push({ element, width: width * scale, height: height * scale });
    });
    pendingFits.clear();
    sizes.forEach(({ element, width, height }) => {
      // CSSOM normalizes fractional pixel strings, so compare their numeric
      // values rather than rewriting an unchanged size after serialization.
      const previousWidth = parseFloat(element.style.width), previousHeight = parseFloat(element.style.height);
      if (!Number.isFinite(previousWidth) || Math.abs(previousWidth - width) > .001) element.style.width = width + 'px';
      if (!Number.isFinite(previousHeight) || Math.abs(previousHeight - height) > .001) element.style.height = height + 'px';
    });
  }
  const fitObserver = new ResizeObserver(entries => entries.forEach(entry => queueFit(entry.target)));
  const previewObserver = new MutationObserver(entries => entries.forEach(entry => queueFit(entry.target.closest('.work-visual'))));
  new Set([...previewByVisual.keys(), ...document.querySelectorAll('.editor-visual')]).forEach(visual => fitObserver.observe(visual));
  previews.forEach(image => {
    image.addEventListener('load', () => queueFit(image.closest('.work-visual')));
    previewObserver.observe(image, { attributes: true, attributeFilter: ['class'] });
  });
  // One permanent handler also covers revisits interrupted before the reveal ends.
  screens.forEach(screen => {
    const visual = screen.querySelector('.work-visual');
    visual?.addEventListener('animationend', event => {
      if (event.target === visual && event.animationName === 'result-reveal') screen.classList.remove('result-enter');
    });
  });
  function syncProgress() {
    const progress = byId('brush-progress');
    const redundant = progress.textContent.startsWith('Check the face before continuing.') ||
      (current === 'inspect' && (
        progress.textContent.startsWith('Ready. Check your picture before continuing.') ||
        progress.textContent === 'Check your picture before continuing.'
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
  function setPreviewSource(image, source) {
    const previous = image.getAttribute('src');
    if (source) {
      if (previous !== source) image.setAttribute('src', source);
    } else if (previous !== null) image.removeAttribute('src');
  }
  function show(name, focus = true, revealResult = false) {
    const screen = screens.find(item => item.dataset.screen === name);
    if (!screen || screen.closest('.hidden')) return false;
    if (name === 'options') {
      const original = byId('upload-preview').getAttribute('src');
      setPreviewSource(byId('options-preview'), original);
    }
    if (name === 'run') {
      const manual = byId('brush-preparation').value === 'manual' && byId('brush-restoration').value !== 'off';
      const image = byId(manual ? 'brush-derived' : 'upload-preview');
      const source = image.getAttribute('src');
      setPreviewSource(byId('restoration-idle'), source);
      byId('restoration-idle').alt = manual ? 'Reviewed prepared input before restoration' : 'Original input before restoration';
    }
    if (name === 'completion-confirm') {
      const source = byId('brush-result-output').getAttribute('src');
      setPreviewSource(byId('completion-confirm-preview'), source);
    }
    if (name === 'options' && current === 'result') byId('brush-progress').textContent = 'Choose restoration and preparation.';
    if (name === 'inspect' && current !== 'inspect') byId('brush-progress').textContent = 'Check your picture before continuing.';
    window.dispatchEvent(new CustomEvent('workbench:screen', {detail:{screen:name}}));
    current = name;
    updateTabTitle();
    syncProgress();
    screens.forEach(item => { item.classList.remove('result-enter'); item.classList.toggle('is-active', item === screen); });
    if (revealResult) screen.classList.add('result-enter');
    const slot = screen.querySelector('.feedback-slot');
    if (slot && form.contains(screen)) slot.append(brushFeedback);
    if (slot && (name === 'result' || byId('completion-stage').contains(screen))) slot.append(completionFeedback);
    showNotice('');
    screen.querySelectorAll('.work-visual').forEach(queueFit);
    if (focus) {
      const heading = screen.querySelector('h1');
      if (heading) { heading.tabIndex = -1; heading.focus({ preventScroll: true }); }
    }
    return true;
  }
  function go(name) {
    if (name === 'options') {
      if (byId('upload-preview').classList.contains('hidden')) return showNotice('Choose a picture before continuing.');
      const review = byId('brush-review').value;
      if (review !== 'usable') return showNotice(review ? 'Use a clearer close-up of one face before continuing.' : 'Choose how clearly you can see the face before continuing.');
    }
    if (name === 'options-next') {
      name = byId('brush-preparation').value === 'manual' && byId('brush-restoration').value !== 'off' ? 'brush-paint' : 'run';
    }
    if (name === 'run-back') {
      name = byId('brush-preparation').value === 'manual' && byId('brush-restoration').value !== 'off' ? 'brush-preview' : 'options';
    }
    if (name === 'run' && byId('brush-preparation').value === 'manual' && byId('brush-restoration').value !== 'off') {
      if (byId('brush-preview-section').classList.contains('hidden') || !byId('brush-confirm').checked) return showNotice('Check and confirm the prepared picture before continuing.');
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
  window.addEventListener('completion:prepared', () => show('completion-paint'));
  byId('completion-generate').addEventListener('click', () => { pendingEstimate = true; });
  const observe = (id, ready) => new MutationObserver(() => {
    if (!byId(id).classList.contains('hidden')) ready();
  }).observe(byId(id), { attributes: true, attributeFilter: ['class'] });
  observe('brush-preview-section', () => { if (pendingPreview) { pendingPreview = false; show('brush-preview'); } });
  observe('completion-result-section', () => { if (pendingEstimate) { pendingEstimate = false; show('completion-result', true, true); } });
  window.addEventListener('restoration:completed', () => show('result', true, true));
  show('inspect', false);
})();
