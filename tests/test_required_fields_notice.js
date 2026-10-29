const assert = require('assert');
const fs = require('fs');
const path = require('path');

global.window = {
  location: { search: '', href: 'http://localhost:7777/' },
  navigator: { language: 'en-US' }
};
global.API_BASE = 'http://localhost:7777';
global.esc = value => String(value)
  .replace(/&/g, '&amp;')
  .replace(/</g, '&lt;')
  .replace(/>/g, '&gt;');

const elements = {};
const documentBody = { appendChild: element => { elements[element.id] = element; } };
global.document = {
  body: documentBody,
  getElementById(id) {
    if (!elements[id]) {
      elements[id] = {
        id,
        innerHTML: '',
        innerText: '',
        value: '',
        hidden: false,
        classList: { add() {}, remove() {}, contains() { return false; }, toggle() {} },
        addEventListener() {},
        setAttribute() {}
      };
    }
    return elements[id];
  },
  createElement() {
    return {
      innerHTML: '',
      hidden: false,
      classList: { add() {}, remove() {}, contains() { return false; }, toggle() {} },
      setAttribute(name, value) { this[name] = value; }
    };
  },
  querySelector(selector) {
    const fieldMatch = selector.match(/\[data-field="([^"]+)"\]/);
    if (fieldMatch) return { innerText: fieldValues[fieldMatch[1]] || '' };
    return this.getElementById(selector);
  },
  querySelectorAll() { return []; }
};
global.$ = id => document.getElementById(id);
global.performance = require('perf_hooks').performance;
global.requestAnimationFrame = callback => setTimeout(callback, 0);

const fieldValues = {};
const htmlPath = path.resolve(__dirname, '../src/static/index.html');
const html = fs.readFileSync(htmlPath, 'utf8');
const match = html.match(/<script>([\s\S]*?)<\/script>/);
assert(match, 'index.html should contain the application script');
const app = new Function(
  `${match[1]}; return { updateRequiredFieldsNotice, REQUIRED_CLINICAL_FIELDS };`
)();

function listMissingFields(department) {
  app.updateRequiredFieldsNotice(department, true);
  return document.getElementById('required-fields-popup');
}

const erPopup = listMissingFields('er');
assert.equal((erPopup.innerHTML.match(/<li>/g) || []).length, 6);
fieldValues.chief_complaint = 'Chest pain';
fieldValues.examination = 'Normal heart sounds';
app.updateRequiredFieldsNotice('er');
assert.equal((erPopup.innerHTML.match(/<li>/g) || []).length, 4);
for (const field of app.REQUIRED_CLINICAL_FIELDS.er) fieldValues[field] = 'Documented';
app.updateRequiredFieldsNotice('er');
assert.equal(erPopup.hidden, true, 'the notice should disappear when all ER fields are filled');

for (const field of Object.keys(fieldValues)) delete fieldValues[field];
const opdPopup = listMissingFields('opd');
assert.equal((opdPopup.innerHTML.match(/<li>/g) || []).length, 5);

for (const field of Object.keys(fieldValues)) delete fieldValues[field];
const dentalPopup = listMissingFields('dental');
assert.equal((dentalPopup.innerHTML.match(/<li>/g) || []).length, 3);
console.log('Required-field notice tests passed.');
