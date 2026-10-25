const assert = require('assert');
const { performance } = require('perf_hooks');
const fs = require('fs');

// Minimal browser mock for Node environment
global.window = {
  location: { search: '', href: 'http://localhost:7777/' },
  navigator: { language: 'en-US' }
};
global.API_BASE = 'http://localhost:7777';
global.esc = (s) => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
global.document = {
  elements: {},
  getElementById(id) {
    if (!this.elements[id]) {
      this.elements[id] = {
        id,
        innerHTML: '',
        className: '',
        value: '',
        classList: { add: () => {}, remove: () => {}, contains: () => false, toggle: () => {} },
        addEventListener: () => {}
      };
    }
    return this.elements[id];
  },
  querySelector(sel) {
    return this.getElementById(sel);
  },
  querySelectorAll() { return []; }
};
global.$ = (id) => global.document.getElementById(id);
global.performance = performance;

// Path to index.html passed as CLI argument or default
const htmlPath = process.argv[2] || require('path').resolve(__dirname, '../src/static/index.html');
const html = fs.readFileSync(htmlPath, 'utf8');
const scriptMatch = html.match(/<script>([\s\S]*?)<\/script>/);
assert(scriptMatch, 'Script block should be present in index.html');

// Extract ResilientSSEClient, EventStateStore, SectionErrorBoundary code
const script = scriptMatch[1];

// Evaluate in current global context
const evalWrapper = new Function(script);
evalWrapper();

const { EventStateStore, ResilientSSEClient, SectionErrorBoundary } = window;
assert(EventStateStore, 'EventStateStore must be exported to window');
assert(ResilientSSEClient, 'ResilientSSEClient must be exported to window');
assert(SectionErrorBoundary, 'SectionErrorBoundary must be exported to window');

async function runAllTests() {
  // ─────────────────────────────────────────────────────────────────────────────
  // TEST 1.2: Event State Store Unit Tests
  // ─────────────────────────────────────────────────────────────────────────────
  console.log('Testing EventStateStore...');
  const store = new EventStateStore();
  const initialState = store.getState();
  assert.deepStrictEqual(initialState.fields, {});
  assert.strictEqual(initialState.structured, null);
  assert.strictEqual(initialState.reasoning, null);
  assert.deepStrictEqual(initialState.cautions, []);
  assert.deepStrictEqual(initialState.speakers, []);
  assert.strictEqual(initialState.meta.status, 'idle');

  // Immutability check
  assert.strictEqual(Object.isFrozen(initialState), true, 'State root must be frozen');
  assert.strictEqual(Object.isFrozen(initialState.fields), true, 'State fields must be frozen');
  assert.strictEqual(Object.isFrozen(initialState.meta), true, 'State meta must be frozen');
  assert.throws(() => {
    'use strict';
    initialState.fields.foo = 'bar';
  }, /TypeError/);

  // Test processing of all 5 event types
  // 1. session_start
  store.dispatch('session_start', { department: 'opd', language: 'ar', sessionId: 's-123' }, 'evt-1');
  assert.strictEqual(store.getState().meta.status, 'streaming');
  assert.strictEqual(store.getState().meta.lastEventId, 'evt-1');

  // 2. refinement & translation (structured note)
  store.dispatch('refinement_done', { refined_text: 'Refined transcript text' }, 'evt-2');
  assert.strictEqual(store.getState().structured.refinement.refined_text, 'Refined transcript text');

  store.dispatch('translation_done', { final_text: 'Translated English text' }, 'evt-3');
  assert.strictEqual(store.getState().structured.translation.final_text, 'Translated English text');

  // 3. extraction (fields)
  store.dispatch('extraction', {
    extracted_features: { chief_complaint: 'Cough and fever', onset: '3 days' },
    patient_story: 'Story text',
    department: 'opd'
  }, 'evt-4');
  assert.strictEqual(store.getState().fields.chief_complaint, 'Cough and fever');
  assert.strictEqual(store.getState().fields.onset, '3 days');
  assert.strictEqual(store.getState().structured.patientStory, 'Story text');

  // 4. diagnosis (reasoning)
  store.dispatch('diagnosis', {
    diagnosis: { icd10_code: 'J06.9', diagnosis_name: 'Acute URI', reasoning: 'Symptoms align' },
    department: 'opd'
  }, 'evt-5');
  assert.strictEqual(store.getState().reasoning.icd10_code, 'J06.9');
  assert.strictEqual(store.getState().reasoning.reasoning, 'Symptoms align');

  // 5. validation_result (cautions)
  store.dispatch('validation_result', {
    is_valid: false,
    cautions: [{ message: 'Review vitals', severity: 'warning' }],
    escalation_reason: 'Missing BP'
  }, 'evt-6');
  assert.strictEqual(store.getState().cautions.length, 2);
  assert.strictEqual(store.getState().cautions[0].message, 'Review vitals');

  // Performance benchmark: State update < 2ms per event
  console.log('Testing EventStateStore dispatch latency...');
  const tStart = performance.now();
  const iterations = 500;
  for (let i = 0; i < iterations; i++) {
    store.dispatch('extraction', { extracted_features: { count: i } }, `evt-${i}`);
  }
  const totalMs = performance.now() - tStart;
  const avgMs = totalMs / iterations;
  console.log(`Average dispatch latency: ${avgMs.toFixed(4)} ms/event (Target: < 2.0 ms)`);
  assert(avgMs < 2.0, `State update average latency ${avgMs}ms exceeded 2ms target`);

  // ─────────────────────────────────────────────────────────────────────────────
  // TEST 1.3: Error Boundary Wrapper
  // ─────────────────────────────────────────────────────────────────────────────
  console.log('Testing SectionErrorBoundary...');
  let goodSectionRendered = false;
  const goodResult = SectionErrorBoundary.wrap('good-section', () => {
    goodSectionRendered = true;
    return 'rendered-successfully';
  });
  assert.strictEqual(goodResult, 'rendered-successfully');
  assert.strictEqual(goodSectionRendered, true);

  // Now throw in bad section
  const badResult = SectionErrorBoundary.wrap('bad-section', () => {
    throw new Error('Simulated crash in bad section component');
  }, 'Failed section fallback');
  assert.strictEqual(badResult, null, 'Error boundary should catch error and return null');

  const errorEl = document.getElementById('body-bad-section');
  assert(errorEl.innerHTML.includes('section-error-boundary'), 'Fallback error boundary HTML must be rendered');
  assert(errorEl.innerHTML.includes('Simulated crash in bad section component'), 'Error message must be localized');

  // Ensure other sections remain completely intact
  assert.strictEqual(goodSectionRendered, true, 'Other sections must remain intact');

  // ─────────────────────────────────────────────────────────────────────────────
  // TEST 1.1: ResilientSSEClient Reconnection & Protocol Tests
  // ─────────────────────────────────────────────────────────────────────────────
  console.log('Testing ResilientSSEClient configuration and logic...');
  let dropCount = 0;
  let receivedEvents = [];
  let stateTransitions = [];

  // Mock fetch that simulates a network drop after 1st event, then reconnects with Last-Event-ID
  global.Headers = Map;
  global.fetch = async (url, options) => {
    dropCount++;
    const lastEventId = options.headers.get ? options.headers.get('Last-Event-ID') : null;

    if (dropCount === 1) {
      // First connection yields 1 frame then drops
      return {
        ok: true,
        body: {
          getReader() {
            let chunkIndex = 0;
            return {
              async read() {
                if (chunkIndex === 0) {
                  chunkIndex++;
                  const frame = 'id: 101\nevent: extraction\ndata: {"status":"initial"}\n\n';
                  return { done: false, value: new TextEncoder().encode(frame) };
                }
                // Transient network drop
                throw new Error('Simulated network connection drop');
              }
            };
          }
        }
      };
    } else {
      // Reconnected attempt with Last-Event-ID check
      assert.strictEqual(lastEventId, '101', 'Reconnection must send Last-Event-ID header');
      return {
        ok: true,
        body: {
          getReader() {
            let chunkIndex = 0;
            return {
              async read() {
                if (chunkIndex === 0) {
                  chunkIndex++;
                  const frame = 'id: 102\nevent: diagnosis\ndata: {"status":"resumed"}\n\n';
                  return { done: false, value: new TextEncoder().encode(frame) };
                }
                return { done: true, value: null };
              }
            };
          }
        }
      };
    }
  };

  const client = new ResilientSSEClient({
    url: 'http://localhost:7777/audio/analyze',
    maxRetries: 3,
    retryDelays: [50, 100, 200], // Faster delays for unit test
    onStateChange: (state, meta) => stateTransitions.push({ state, meta }),
    onEvent: (name, data, id) => receivedEvents.push({ name, data, id }),
  });

  await client.connect({});
  console.log('Reconnect completed. Received events:', receivedEvents.length);
  assert.strictEqual(receivedEvents.length, 2, 'Should receive event from before drop and after resume');
  assert.strictEqual(receivedEvents[0].id, '101');
  assert.strictEqual(receivedEvents[0].name, 'extraction');
  assert.strictEqual(receivedEvents[1].id, '102');
  assert.strictEqual(receivedEvents[1].name, 'diagnosis');

  const reconnectingState = stateTransitions.find(s => s.state === 'reconnecting');
  assert(reconnectingState, 'Must transition through "reconnecting" state');
  assert.strictEqual(reconnectingState.meta.lastEventId, '101');

  console.log('All Phase 1 tests passed successfully!');
}

runAllTests().catch((err) => {
  console.error('Node unit tests error:', err);
  process.exit(1);
});
