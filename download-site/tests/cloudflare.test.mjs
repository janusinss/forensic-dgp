import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { makeInvocation, runCloudflare } from '../scripts/cloudflare.mjs';

const site = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const account = 'b9e82b6514591358168877d8118aec18';
const configuration = {
  name: 'forensic-dgp',
  compatibility_date: '2026-10-31',
  pages_build_output_dir: './dist',
};

test('deployment targets the exact Pages project and production branch', () => {
  assert.deepEqual(makeInvocation('deploy', configuration).args,
    ['pages', 'deploy', 'dist', '--project-name', 'forensic-dgp', '--branch', 'main', '--force']);
  assert.deepEqual(makeInvocation('create', configuration).args,
    ['pages', 'project', 'create', 'forensic-dgp', '--production-branch', 'main', '--force']);
  assert.deepEqual(makeInvocation('projects', configuration).args,
    ['pages', 'project', 'list', '--json']);
  assert.deepEqual(makeInvocation('whoami', configuration).args, ['whoami']);
});

test('personal login includes Pages authorization without opening a browser', () => {
  assert.deepEqual(makeInvocation('login', configuration).args,
    ['login', '--device', '--browser', 'false', '--scopes',
      'account:read', 'user:read', 'workers_scripts:write', 'pages:write']);
});

test('account isolation removes primary and legacy credentials without changing the parent', () => {
  const parent = Object.freeze({
    KEEP_ME: 'ordinary inherited setting',
    XDG_CONFIG_HOME: 'organization-config',
    CLOUDFLARE_API_TOKEN: 'unrelated-token', CF_API_TOKEN: 'legacy-token',
    CLOUDFLARE_API_KEY: 'unrelated-key', CF_API_KEY: 'legacy-key',
    CLOUDFLARE_EMAIL: 'organization@example.invalid', CF_EMAIL: 'legacy@example.invalid',
    CLOUDFLARE_ACCOUNT_ID: 'unrelated-account', CF_ACCOUNT_ID: 'legacy-account',
  });
  const original = { ...parent };
  const { env } = makeInvocation('deploy', configuration, parent, site);
  assert.notEqual(env, parent);
  assert.deepEqual(parent, original);
  assert.equal(env.KEEP_ME, parent.KEEP_ME);
  assert.equal(env.XDG_CONFIG_HOME, path.resolve(site, '../scratch/cloudflare-personal'));
  assert.equal(env.CLOUDFLARE_ACCOUNT_ID, account);
  for (const key of ['CLOUDFLARE_API_TOKEN', 'CF_API_TOKEN', 'CLOUDFLARE_API_KEY',
    'CF_API_KEY', 'CLOUDFLARE_EMAIL', 'CF_EMAIL', 'CF_ACCOUNT_ID']) {
    assert.equal(Object.hasOwn(env, key), false, `${key} must not select unrelated credentials`);
  }
});

test('configuration drift is refused before any child process starts', () => {
  const invalid = [
    { ...configuration, name: 'another-project' },
    { ...configuration, account_id: 'another-account' },
    { ...configuration, account_id: account },
    { ...configuration, pages_build_output_dir: '../other-output' },
    { ...configuration, pages_build_output_dir: undefined },
    ...['main', 'assets', 'workers_dev'].map(key => ({ ...configuration, [key]: false })),
  ];
  for (const config of invalid) {
    let calls = 0;
    assert.throws(() => runCloudflare('deploy', {
      configuration: config, parentEnv: {}, spawn: () => { calls += 1; return { status: 0 }; },
    }));
    assert.equal(calls, 0);
  }
});

test('unsupported and destructive actions cannot spawn Wrangler', () => {
  for (const action of [undefined, '', 'delete', 'pages delete', '--help', 'deploy; delete']) {
    let calls = 0;
    assert.throws(() => runCloudflare(action, {
      configuration, parentEnv: {}, spawn: () => { calls += 1; return { status: 0 }; },
    }));
    assert.equal(calls, 0);
  }
});

test('the wrapper invokes only the local Wrangler with the isolated environment', () => {
  const calls = [];
  const status = runCloudflare('deploy', {
    configuration, parentEnv: { KEEP_ME: 'preserve' }, siteRoot: site,
    spawn: (...args) => { calls.push(args); return { status: 7 }; },
  });
  assert.equal(status, 7);
  assert.equal(calls.length, 1);
  const [command, args, options] = calls[0];
  assert.equal(command, process.execPath);
  assert.deepEqual(args, [path.join(site, 'node_modules/wrangler/bin/wrangler.js'),
    'pages', 'deploy', 'dist', '--project-name', 'forensic-dgp', '--branch', 'main', '--force']);
  assert.equal(options.cwd, site);
  assert.equal(options.stdio, 'inherit');
  assert.equal(options.env.CLOUDFLARE_ACCOUNT_ID, account);
  assert.equal(options.env.KEEP_ME, 'preserve');
  assert.equal(options.env.XDG_CONFIG_HOME, path.resolve(site, '../scratch/cloudflare-personal'));
});

test('spawn failures and absent child status cannot report deployment success', () => {
  const failure = new Error('mock child failed to start');
  assert.throws(() => runCloudflare('whoami', {
    configuration, parentEnv: {}, spawn: () => ({ error: failure, status: null }),
  }), error => error === failure);
  assert.equal(runCloudflare('whoami', {
    configuration, parentEnv: {}, spawn: () => ({ status: null }),
  }), 1);
  assert.equal(runCloudflare('whoami', {
    configuration, parentEnv: {}, spawn: () => ({ status: 0 }),
  }), 0);
});

test('checked-in hosting configuration passes the same deployment account guard', async () => {
  const config = JSON.parse(await readFile(path.join(site, 'wrangler.jsonc'), 'utf8'));
  assert.doesNotThrow(() => makeInvocation('deploy', config));
});

test('a top-level 404 page prevents Pages from treating unknown paths as the application', async () => {
  const html = await readFile(path.join(site, 'public', '404.html'), 'utf8');
  assert.match(html, /404|not found/i);
});
