import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, mkdir, readFile, writeFile, readdir, rm, symlink } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { prepareRelease } from '../scripts/prepare-release.mjs';

async function fixture(t) {
  const root = await mkdtemp(path.join(tmpdir(), 'dgp-release-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sourceDir = path.join(root, 'source');
  const siteDir = path.join(root, 'site');
  await mkdir(sourceDir); await mkdir(siteDir);
  const bytes = Buffer.from('inert fixture bytes, not an executable');
  const sha256 = createHash('sha256').update(bytes).digest('hex');
  const metadata = {
    schema: 1, version: '0.1.0', artifact_sha256: sha256, artifact_bytes: bytes.length,
    signed: false, commit: 'a'.repeat(40), internal_path: 'C:/private/build',
    files: { 'secret-internal-file': 'ignored' },
  };
  await writeFile(path.join(sourceDir, 'ForensicDGP_Setup.exe'), bytes);
  await writeFile(path.join(sourceDir, 'release.json'), JSON.stringify(metadata));
  return { root, sourceDir, siteDir, bytes, sha256, metadata };
}

async function saveMetadata(f) {
  await writeFile(path.join(f.sourceDir, 'release.json'), JSON.stringify(f.metadata));
}

async function previousRelease(f) {
  await mkdir(path.join(f.siteDir, 'public', 'downloads'), { recursive: true });
  await writeFile(path.join(f.siteDir, 'public', 'release.json'), 'previous release metadata');
  await mkdir(path.join(f.siteDir, 'src'), { recursive: true });
  await writeFile(path.join(f.siteDir, 'src', 'release.generated.json'), 'previous generated metadata');
  await writeFile(path.join(f.siteDir, 'public', 'downloads', 'ForensicDGP_Setup-0.1.0.exe'), 'previous verified executable');
}

async function assertPriorIntact(f) {
  assert.equal(await readFile(path.join(f.siteDir, 'public', 'release.json'), 'utf8'), 'previous release metadata');
  assert.equal(await readFile(path.join(f.siteDir, 'src', 'release.generated.json'), 'utf8'), 'previous generated metadata');
  assert.equal(await readFile(path.join(f.siteDir, 'public', 'downloads', 'ForensicDGP_Setup-0.1.0.exe'), 'utf8'), 'previous verified executable');
}

test('publishes only verified artifact, checksum and sanitized public metadata', async t => {
  const f = await fixture(t);
  await prepareRelease({ sourceDir: f.sourceDir, siteDir: f.siteDir });
  const publicRoot = path.join(f.siteDir, 'public');
  assert.deepEqual(await readFile(path.join(publicRoot, 'downloads', 'ForensicDGP_Setup-0.1.0.exe')), f.bytes);
  const checksum = await readFile(path.join(publicRoot, 'downloads', 'ForensicDGP_Setup-0.1.0.exe.sha256'), 'utf8');
  assert.equal(checksum, `${f.sha256}  ForensicDGP_Setup-0.1.0.exe\n`);
  const release = JSON.parse(await readFile(path.join(publicRoot, 'release.json'), 'utf8'));
  assert.deepEqual(release, {
    schema: 1, version: '0.1.0', filename: 'ForensicDGP_Setup-0.1.0.exe', bytes: f.bytes.length, sha256: f.sha256,
    signed: false, preview: true,
    downloadUrl: '/downloads/ForensicDGP_Setup-0.1.0.exe',
    checksumUrl: '/downloads/ForensicDGP_Setup-0.1.0.exe.sha256',
  });
  assert.deepEqual(JSON.parse(await readFile(path.join(f.siteDir, 'src', 'release.generated.json'), 'utf8')), release);
  assert(!JSON.stringify(release).includes('private'));
  assert(!JSON.stringify(release).includes('commit'));
});

test('tampered artifact does not replace prior public release', async t => {
  const f = await fixture(t); await previousRelease(f);
  const changed = Buffer.from(f.bytes); changed[0] ^= 1;
  await writeFile(path.join(f.sourceDir, 'ForensicDGP_Setup.exe'), changed);
  await assert.rejects(prepareRelease({ sourceDir: f.sourceDir, siteDir: f.siteDir }), /SHA-256/i);
  await assertPriorIntact(f);
});

test('size mismatch does not replace prior public release', async t => {
  const f = await fixture(t); await previousRelease(f);
  f.metadata.artifact_bytes += 1; await saveMetadata(f);
  await assert.rejects(prepareRelease({ sourceDir: f.sourceDir, siteDir: f.siteDir }), /size|bytes/i);
  await assertPriorIntact(f);
});

test('unsafe version cannot escape the downloads directory', async t => {
  for (const version of ['../escape', '0.1.0/evil', '0.1.0\\evil', '0.1.0:stream', 'v0.1.0', '0.1.0\n']) {
    const f = await fixture(t); await previousRelease(f);
    f.metadata.version = version; await saveMetadata(f);
    await assert.rejects(prepareRelease({ sourceDir: f.sourceDir, siteDir: f.siteDir }), /version/i);
    await assertPriorIntact(f);
  }
});

test('invalid integrity declarations and changed signing contract are rejected', async t => {
  for (const update of [{artifact_sha256:'bad'}, {artifact_bytes:-1}, {artifact_bytes:'22'}, {signed:true}, {schema:2}]) {
    const f = await fixture(t); await previousRelease(f);
    Object.assign(f.metadata, update); await saveMetadata(f);
    await assert.rejects(prepareRelease({ sourceDir: f.sourceDir, siteDir: f.siteDir }));
    await assertPriorIntact(f);
  }
});

test('unrelated files remain untouched on successful repeated preparation', async t => {
  const f = await fixture(t);
  const unrelated = path.join(f.siteDir, 'public', 'do-not-delete.txt');
  await mkdir(path.dirname(unrelated)); await writeFile(unrelated, 'preserve me');
  await prepareRelease({ sourceDir: f.sourceDir, siteDir: f.siteDir });
  await prepareRelease({ sourceDir: f.sourceDir, siteDir: f.siteDir });
  assert.equal(await readFile(unrelated, 'utf8'), 'preserve me');
  assert.deepEqual((await readdir(path.join(f.siteDir,'public','downloads'))).sort(), ['ForensicDGP_Setup-0.1.0.exe','ForensicDGP_Setup-0.1.0.exe.sha256']);
});

test('linked output directory is refused without touching outside data', async t => {
  const f = await fixture(t);
  const outside = path.join(f.root, 'outside'); await mkdir(outside);
  await writeFile(path.join(outside, 'preserve.txt'), 'outside');
  try { await symlink(outside, path.join(f.siteDir, 'public'), process.platform === 'win32' ? 'junction' : 'dir'); }
  catch (error) { if (error.code === 'EPERM') { t.skip('symlink creation unavailable'); return; } throw error; }
  await assert.rejects(prepareRelease({ sourceDir: f.sourceDir, siteDir: f.siteDir }), /link|ordinary/i);
  assert.deepEqual(await readdir(outside), ['preserve.txt']);
});
