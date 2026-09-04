// Execute the actual browser script with canvas/DOM doubles; no browser or server.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const app = readFileSync(new URL('../object_remover/web/app.js', import.meta.url), 'utf8');
const html = readFileSync(new URL('../object_remover/web/index.html', import.meta.url), 'utf8');
const css = readFileSync(new URL('../object_remover/web/style.css', import.meta.url), 'utf8');

function harness() {
  const elements = new Map();
  function element(selector) {
    if (elements.has(selector)) return elements.get(selector);
    const context = {
      path: [], fills: [], clears: 0,
      beginPath() { this.path = []; },
      moveTo(x, y) { this.path.push(['move', x, y]); },
      lineTo(x, y) { this.path.push(['line', x, y]); },
      closePath() { this.path.push(['close']); },
      arc(x, y, r) { this.path.push(['arc', x, y, r]); },
      fill() { this.fills.push({ path: structuredClone(this.path), mode: this.globalCompositeOperation }); },
      stroke() {}, setLineDash() {},
      clearRect() { this.clears++; this.fills = []; },
      getImageData() { return structuredClone(this.fills); },
      putImageData(data) { this.fills = structuredClone(data); },
    };
    const el = {
      width: 1024, height: 512, value: '20', disabled: false,
      dataset: {}, events: {}, pointers: new Set(), context,
      classList: { toggle() {} },
      getContext() { return context; },
      getBoundingClientRect() { return { left: 10, top: 20, width: 512, height: 256 }; },
      addEventListener(name, callback) { (this.events[name] ??= []).push(callback); },
      setPointerCapture(id) { this.pointers.add(id); },
      hasPointerCapture(id) { return this.pointers.has(id); },
      releasePointerCapture(id) { this.pointers.delete(id); },
    };
    elements.set(selector, el);
    return el;
  }
  const scope = vm.createContext({
    document: { querySelector: element, querySelectorAll: () => [] },
    // Initial model-status request deliberately never settles; no live networking.
    fetch: () => new Promise(() => {}),
    console,
  });
  vm.runInContext(app, scope);
  const run = code => vm.runInContext(code, scope);
  run("file = {name: 'test.png'}; tool = 'lasso';");
  async function dispatch(type, x = 10, y = 20, id = 1, overrides = {}) {
    const event = { clientX: x, clientY: y, pointerId: id, button: 0, isPrimary: true, ...overrides };
    for (const callback of element('#mask').events[type] || []) await callback(event);
  }
  async function outline() {
    await dispatch('pointerdown', 20, 30);
    await dispatch('pointermove', 50, 30);
    await dispatch('pointermove', 50, 60);
  }
  return { element, run, dispatch, outline, mask: element('#mask').context, overlay: element('#selection').context };
}

const tests = {
  async 'lasso commits a closed, full-resolution mask only on release'() {
    const h = harness();
    await h.outline();
    assert.equal(h.mask.fills.length, 0);
    assert.equal(h.overlay.fills.length, 1);
    await h.dispatch('pointerup');
    assert.deepEqual(h.mask.fills[0], {
      path: [['move', 20, 20], ['line', 80, 20], ['line', 80, 80], ['close']], mode: 'source-over',
    });
    assert.equal(h.overlay.fills.length, 0);
    assert.equal(h.run('drawing'), false);
    assert.equal(h.element('#mask').pointers.size, 0);
  },
  async 'separate regions accumulate and each has an undo step'() {
    const h = harness();
    for (let i = 0; i < 2; i++) { await h.outline(); await h.dispatch('pointerup'); }
    assert.equal(h.mask.fills.length, 2);
    await h.element('#undo').events.click[0]();
    assert.equal(h.mask.fills.length, 1);
    assert.equal(h.run('undo.length'), 1);
  },
  async 'cancel and lost capture never apply an unfinished lasso'() {
    for (const event of ['pointercancel', 'lostpointercapture']) {
      const h = harness();
      await h.outline();
      await h.dispatch(event);
      assert.equal(h.mask.fills.length, 0);
      assert.equal(h.overlay.fills.length, 0);
      assert.equal(h.run('undo.length'), 0);
      assert.equal(h.run('drawing'), false);
    }
  },
  async 'single click and cancelled next region preserve prior mask'() {
    const h = harness();
    await h.outline(); await h.dispatch('pointerup');
    await h.dispatch('pointerdown'); await h.dispatch('pointerup');
    await h.outline(); await h.dispatch('pointercancel');
    assert.equal(h.mask.fills.length, 1);
    assert.equal(h.run('undo.length'), 1);
  },
  async 'other pointers and secondary buttons cannot mutate the stroke'() {
    const h = harness();
    await h.dispatch('pointerdown', 20, 30, 1, { button: 2 });
    assert.equal(h.run('drawing'), false);
    await h.outline();
    await h.dispatch('pointerdown', 200, 200, 2);
    await h.dispatch('pointermove', 200, 200, 2);
    await h.dispatch('pointerup', 200, 200, 2);
    assert.equal(h.run('drawing'), true);
    assert.equal(h.run('lassoPoints.length'), 3);
    await h.dispatch('pointerup');
    assert.equal(h.mask.fills.length, 1);
    assert.equal(h.run('undo.length'), 1);
  },
  async 'operation lock cancels unfinished selection and rejects new input'() {
    const h = harness();
    await h.outline();
    h.run('lockEditing(true)');
    await h.dispatch('pointerup');
    await h.dispatch('pointerdown');
    assert.equal(h.mask.fills.length, 0);
    assert.equal(h.run('drawing'), false);
    assert.equal(h.run('undo.length'), 0);
  },
  async 'brush and eraser keep their original canvas composition modes'() {
    const h = harness();
    for (const [tool, mode] of [['brush', 'source-over'], ['erase', 'destination-out']]) {
      h.run(`tool = '${tool}'`);
      await h.dispatch('pointerdown'); await h.dispatch('pointerup');
      assert.equal(h.mask.fills.at(-1).mode, mode);
      assert.equal(h.mask.fills.at(-1).path[0][0], 'arc');
    }
  },
  async 'markup selectors and overlay stacking are consistent'() {
    const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map(match => match[1]);
    assert.equal(new Set(ids).size, ids.length);
    for (const match of app.matchAll(/\$\('#([^']+)'\)/g)) assert.ok(ids.includes(match[1]), match[1]);
    assert.match(html, /data-tool="lasso"/);
    assert.ok(html.indexOf('id="selection"') > html.indexOf('id="mask"'));
    assert.match(css, /#mask, #selection\s*\{[^}]*position:\s*absolute/);
    assert.match(css, /#selection\s*\{\s*pointer-events:\s*none/);
    assert.match(css, /#mask\s*\{[^}]*touch-action:\s*none/);
  },
};

for (const [name, test] of Object.entries(tests)) {
  await test();
  console.log(`PASS ${name}`);
}
console.log(`${Object.keys(tests).length} painter contracts passed (mock canvas; no visual-browser test).`);
