(() => {
  'use strict';
  const byId = id => document.getElementById(id);
  const screens = [...document.querySelectorAll('.work-screen')];
  const form = byId('brush-form');
  const brushFeedback = byId('brush-feedback');
  const completionFeedback = byId('completion-feedback');
  let current = 'inspect', pendingPreview = false, pendingCompletion = false, pendingEstimate = false;

  // This guide describes the current screen; it does not change the workflow.
  const journey = [
    { label: 'Picture', names: ['inspect'] },
    { label: 'Settings', names: ['options', 'brush-paint', 'brush-tools', 'brush-preview'] },
    { label: 'Restore', names: ['run'] },
    { label: 'Review', names: ['result'] },
    { label: 'Removal', names: ['completion-intro', 'completion-paint', 'completion-tools', 'completion-confirm', 'completion-result'], optional: true }
  ];
  screens.forEach(screen => {
    const controls = screen.querySelector('.work-controls');
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

  function fitEditor(visual) {
    const canvas = visual.querySelector('canvas:not(.ambient-shader)');
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
  // Fit all input previews like the editor without changing image pixels.
  const previews = ['upload-preview', 'options-preview', 'restoration-idle', 'completion-intro-preview', 'completion-confirm-preview'].map(byId);
  function fitInputPreview(image) {
    const visual = image.closest('.work-visual');
    if (image.classList.contains('hidden') || !image.naturalWidth || !image.naturalHeight || !visual.getClientRects().length) return;
    const styles = getComputedStyle(visual);
    const caption = visual.querySelector('.visual-caption');
    const availableWidth = Math.max(0, visual.clientWidth - parseFloat(styles.paddingLeft) - parseFloat(styles.paddingRight));
    const availableHeight = Math.max(0, visual.clientHeight - parseFloat(styles.paddingTop) - parseFloat(styles.paddingBottom) - (caption?.offsetHeight || 0) - (caption ? parseFloat(styles.rowGap || 0) : 0));
    const scale = Math.min(availableWidth / image.naturalWidth, availableHeight / image.naturalHeight, 600 / image.naturalWidth, 600 / image.naturalHeight);
    if (!(scale > 0)) return;
    image.style.width = (image.naturalWidth * scale) + 'px';
    image.style.height = (image.naturalHeight * scale) + 'px';
  }
  previews.forEach(image => {
    image.addEventListener('load', () => fitInputPreview(image));
    new ResizeObserver(() => fitInputPreview(image)).observe(image.closest('.work-visual'));
    new MutationObserver(() => fitInputPreview(image)).observe(image, { attributes: true, attributeFilter: ['class'] });
  });
  const editorObserver = new ResizeObserver(entries => entries.forEach(entry => fitEditor(entry.target)));
  document.querySelectorAll('.editor-visual').forEach(visual => editorObserver.observe(visual));
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
  function show(name, focus = true, revealResult = false) {
    const screen = screens.find(item => item.dataset.screen === name);
    if (!screen || screen.closest('.hidden')) return false;
    if (name === 'options') {
      const original = byId('upload-preview').getAttribute('src');
      if (original) byId('options-preview').setAttribute('src', original);
      else byId('options-preview').removeAttribute('src');
    }
    if (name === 'run') {
      const manual = byId('brush-preparation').value === 'manual' && byId('brush-restoration').value !== 'off';
      const image = byId(manual ? 'brush-derived' : 'upload-preview');
      const source = image.getAttribute('src');
      if (source) byId('restoration-idle').setAttribute('src', source);
      else byId('restoration-idle').removeAttribute('src');
      byId('restoration-idle').alt = manual ? 'Reviewed prepared input before restoration' : 'Original input before restoration';
    }
    if (name === 'completion-intro') {
      const source = byId('brush-result-output').getAttribute('src');
      if (source) byId('completion-intro-preview').setAttribute('src', source);
      else byId('completion-intro-preview').removeAttribute('src');
    }
    if (name === 'completion-confirm') {
      const source = byId('brush-result-output').getAttribute('src');
      if (source) byId('completion-confirm-preview').setAttribute('src', source);
      else byId('completion-confirm-preview').removeAttribute('src');
    }
    if (name === 'options' && current === 'result') byId('brush-progress').textContent = 'Choose restoration and preparation.';
    if (name === 'inspect' && current !== 'inspect') byId('brush-progress').textContent = 'Check your picture before continuing.';
    current = name;
    syncProgress();
    screens.forEach(item => { item.classList.remove('result-enter'); item.classList.toggle('is-active', item === screen); });
    if (revealResult) {
      screen.classList.add('result-enter');
      const visual = screen.querySelector('.work-visual');
      if (visual) {
        const finishReveal = event => {
          if (event.target !== visual || event.animationName !== 'result-reveal') return;
          screen.classList.remove('result-enter');
          visual.removeEventListener('animationend', finishReveal);
        };
        visual.addEventListener('animationend', finishReveal);
      }
    }
    const slot = screen.querySelector('.feedback-slot');
    if (slot && form.contains(screen)) slot.append(brushFeedback);
    if (slot && byId('completion-stage').contains(screen)) slot.append(completionFeedback);
    showNotice('');
    screen.querySelectorAll('.editor-visual').forEach(fitEditor);
    requestAnimationFrame(() => { screen.querySelectorAll('.editor-visual').forEach(fitEditor); previews.filter(image => screen.contains(image)).forEach(fitInputPreview); });
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
  byId('completion-start').addEventListener('click', () => { pendingCompletion = true; });
  byId('completion-generate').addEventListener('click', () => { pendingEstimate = true; });
  const observe = (id, ready) => new MutationObserver(() => {
    if (!byId(id).classList.contains('hidden')) ready();
  }).observe(byId(id), { attributes: true, attributeFilter: ['class'] });
  observe('brush-preview-section', () => { if (pendingPreview) { pendingPreview = false; show('brush-preview'); } });
  observe('completion-editor-section', () => { if (pendingCompletion) { pendingCompletion = false; show('completion-paint'); } });
  observe('completion-result-section', () => { if (pendingEstimate) { pendingEstimate = false; show('completion-result', true, true); } });
  window.addEventListener('restoration:completed', () => show('result', true, true));
  show('inspect', false);
})();
