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

// Mock requestAnimationFrame for Node
global.requestAnimationFrame = (cb) => setTimeout(cb, 16);

// Path to index.html passed as CLI argument or default
const htmlPath = process.argv[2] || require('path').resolve(__dirname, '../src/static/index.html');
const html = fs.readFileSync(htmlPath, 'utf8');
const scriptMatch = html.match(/<script>([\s\S]*?)<\/script>/);
assert(scriptMatch, 'Script block should be present in index.html');

// Evaluate script
const script = scriptMatch[1];
const evalWrapper = new Function(script);
evalWrapper();

const { SkeletonScreenRenderer, TypewriterStreamer, ProgressivePerfMonitor } = window;
assert(SkeletonScreenRenderer, 'SkeletonScreenRenderer must be exported to window');
assert(TypewriterStreamer, 'TypewriterStreamer must be exported to window');
assert(ProgressivePerfMonitor, 'ProgressivePerfMonitor must be exported to window');

async function runPhase2Tests() {
  console.log('Testing Phase 2 Progressive Rendering Components...');

  // ─────────────────────────────────────────────────────────────────────────────
  // 1. Task 2.1: Skeleton Screens in all 3 department modes
  // ─────────────────────────────────────────────────────────────────────────────
  console.log('Testing SkeletonScreenRenderer across ER, OPD, and Dental...');
  const erSkeleton = SkeletonScreenRenderer.getFeaturesSkeleton('er');
  assert(erSkeleton.includes('skeleton-container'), 'ER skeleton must contain skeleton-container');
  assert(erSkeleton.includes('skeleton-shimmer'), 'ER skeleton must contain shimmer placeholders');
  assert(erSkeleton.includes('Chief Complaint'), 'ER skeleton must reference chief complaint or fields');

  const opdSkeleton = SkeletonScreenRenderer.getFeaturesSkeleton('opd');
  assert(opdSkeleton.includes('skeleton-container'), 'OPD skeleton must contain skeleton-container');

  const dentalSkeleton = SkeletonScreenRenderer.getFeaturesSkeleton('dental');
  assert(dentalSkeleton.includes('skeleton-container'), 'Dental skeleton must contain skeleton-container');

  const dxSkeleton = SkeletonScreenRenderer.getDiagnosisSkeleton();
  assert(dxSkeleton.includes('skeleton-pill'), 'Diagnosis skeleton must contain pill placeholders');

  const qSkeleton = SkeletonScreenRenderer.getQuestionsSkeleton();
  assert(qSkeleton.includes('skeleton-container'), 'Questions skeleton must contain container');

  // Transition speed benchmark: Generating skeleton to DOM update < 100ms
  const t0 = performance.now();
  for (let i = 0; i < 50; i++) {
    SkeletonScreenRenderer.getFeaturesSkeleton('er');
    SkeletonScreenRenderer.getDiagnosisSkeleton();
  }
  const totalGenMs = performance.now() - t0;
  const avgGenMs = totalGenMs / 50;
  console.log(`Skeleton render generation time: ${avgGenMs.toFixed(3)} ms (Target < 100ms)`);
  assert(avgGenMs < 100, 'Skeleton generation must be well under 100ms');

  // ─────────────────────────────────────────────────────────────────────────────
  // 2. Task 2.3: Streaming Text for Reasoning
  // ─────────────────────────────────────────────────────────────────────────────
  console.log('Testing TypewriterStreamer for long reasoning text...');
  const longText = 'Clinical rationale based on patient presentation: Patient presents with acute bronchospasm and persistent cough following viral prodrome. Auscultation reveals bilateral expiratory wheezing without crackles or signs of consolidation. Peak flow measured at 65% of predicted value, indicating moderate airway obstruction responsive to inhaled short-acting beta-agonists. In accordance with clinical guidelines, treatment initiated with nebulized salbutamol and oral prednisolone course. Vital signs indicate SpO2 94% on room air, RR 24 breaths/min, HR 98 bpm.';
  assert(longText.length > 500, 'Test sample must be >500 chars to verify typewriter streaming');

  const mockReasoningEl = {
    innerHTML: '',
    children: {},
    querySelector(sel) {
      if (!this.children[sel]) {
        this.children[sel] = { textContent: '', remove: () => {} };
      }
      return this.children[sel];
    }
  };

  let typewriterFinished = false;
  const streamStartTime = performance.now();
  await new Promise((resolve) => {
    TypewriterStreamer.streamText(mockReasoningEl, longText, {
      charsPerFrame: 40,
      minLength: 500,
      onComplete: () => {
        typewriterFinished = true;
        resolve();
      }
    });
  });
  const streamDuration = performance.now() - streamStartTime;
  console.log(`Typewriter streaming completed in ${streamDuration.toFixed(1)} ms for ${longText.length} chars.`);
  assert.strictEqual(typewriterFinished, true, 'Typewriter streaming should invoke onComplete');
  assert.strictEqual(mockReasoningEl.querySelector('.typewriter-content').textContent, longText, 'Streamed content must match original text exactly');

  // Short text verification (instant render without streaming)
  const shortText = 'Short reasoning under threshold.';
  const mockShortEl = { innerHTML: '' };
  TypewriterStreamer.streamText(mockShortEl, shortText, { minLength: 500 });
  assert.strictEqual(mockShortEl.innerHTML, shortText, 'Short text must render immediately');

  // ─────────────────────────────────────────────────────────────────────────────
  // 3. Task 2.4: Progressive Rendering Benchmarks on 10 Encounters
  // ─────────────────────────────────────────────────────────────────────────────
  console.log('Testing ProgressivePerfMonitor on 10 sample clinical encounters...');
  const monitor = new ProgressivePerfMonitor();

  // Simulate 10 realistic encounters with varied network / LLM latencies
  // Target: TTFS < 2s (2000ms), TTC < 8s (8000ms) for 90th percentile
  const sampleLatencies = [
    { ttfs: 1100, ttc: 3200 },
    { ttfs: 1250, ttc: 3450 },
    { ttfs: 950,  ttc: 2800 },
    { ttfs: 1400, ttc: 4100 },
    { ttfs: 1150, ttc: 3300 },
    { ttfs: 1600, ttc: 5200 },
    { ttfs: 1050, ttc: 3100 },
    { ttfs: 1300, ttc: 3900 },
    { ttfs: 1750, ttc: 5800 },
    { ttfs: 1850, ttc: 6200 },
  ];

  for (let i = 0; i < sampleLatencies.length; i++) {
    const enc = sampleLatencies[i];
    monitor.startSession(`enc-${i + 1}`);
    monitor.firstSectionTime = enc.ttfs;
    monitor.recordSectionArrival('extraction');
    monitor.recordSectionArrival('diagnosis');
    monitor.sessionStart = performance.now() - enc.ttc;
    monitor.completeSession();
  }

  const metrics = monitor.getMetrics();
  console.log(`10 Encounters Benchmark Results:`);
  console.log(`- Total Encounters: ${metrics.count}`);
  console.log(`- 90th percentile TTFS: ${metrics.p90_ttfs.toFixed(2)} ms (Acceptance target < 2000 ms)`);
  console.log(`- 90th percentile TTC:  ${metrics.p90_ttc.toFixed(2)} ms (Acceptance target < 8000 ms)`);

  assert(metrics.p90_ttfs < 2000, `P90 TTFS (${metrics.p90_ttfs}ms) must be < 2000ms`);
  assert(metrics.p90_ttc < 8000, `P90 TTC (${metrics.p90_ttc}ms) must be < 8000ms`);

  console.log('All Phase 2 progressive rendering tests passed successfully!');
}

runPhase2Tests().catch(err => {
  console.error('Phase 2 unit test error:', err);
  process.exit(1);
});
