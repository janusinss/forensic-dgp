/** Prepare the exact verified preview installer for a static download site. */
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { lstat, mkdir, mkdtemp, readFile, writeFile, rename, unlink, rmdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const siteRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const defaultSource = path.resolve(siteRoot, '../scratch/dist');
const maximumArtifactBytes = 25 * 1024 * 1024;
const maximumOfflineBytes = 2 * 1024 ** 3 - 1;

function isOffline(source) {
  if (source.payload !== undefined && !['online-source-bundled', 'offline-bundled'].includes(source.payload)) {
    throw new Error('Unsupported release payload contract.');
  }
  return source.payload === 'offline-bundled';
}

async function hashArtifact(target) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(target)) hash.update(chunk);
  return hash.digest('hex');
}

async function existingInfo(target) {
  try { return await lstat(target); }
  catch (error) { if (error.code === 'ENOENT') return null; throw error; }
}

async function requireOrdinary(target, kind, allowMissing = false) {
  const absolute = path.resolve(target);
  let cursor = absolute;
  let first = true;
  for (;;) {
    const info = await existingInfo(cursor);
    if (!info) {
      if (first && !allowMissing) throw new Error(`Required ${kind} is missing: ${path.basename(target)}`);
    } else {
      if (info.isSymbolicLink()) throw new Error('Release paths must be ordinary paths without links.');
      if ((first && kind === 'file' && !info.isFile()) || ((!first || kind === 'directory') && !info.isDirectory())) {
        throw new Error('Release paths must have ordinary file/directory types.');
      }
    }
    const parent = path.dirname(cursor);
    if (parent === cursor) break;
    cursor = parent;
    first = false;
  }
}

async function ensureDirectory(target) {
  await requireOrdinary(target, 'directory', true);
  await mkdir(target, { recursive: true });
  await requireOrdinary(target, 'directory');
}

function releaseMetadata(source, byteCount, sha256) {
  if (source === null || Array.isArray(source) || typeof source !== 'object' || source.schema !== 1) {
    throw new Error('Unsupported release metadata schema.');
  }
  // Restrict this first release to a plain numeric version with no path syntax.
  if (typeof source.version !== 'string' || !/^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/.test(source.version)) {
    throw new Error('Release version must be a safe numeric major.minor.patch value.');
  }
  const offline = isOffline(source);
  const maximum = offline ? maximumOfflineBytes : maximumArtifactBytes;
  if (!Number.isSafeInteger(source.artifact_bytes) || source.artifact_bytes < 1 || source.artifact_bytes > maximum) {
    throw new Error('Release artifact bytes exceed the reviewed download limit.');
  }
  if (source.artifact_bytes !== byteCount) throw new Error('Installer size differs from its release metadata.');
  if (typeof source.artifact_sha256 !== 'string' || !/^[0-9a-f]{64}$/.test(source.artifact_sha256) || source.artifact_sha256 !== sha256) {
    throw new Error('Installer SHA-256 differs from its release metadata.');
  }
  if (source.signed !== false) throw new Error('Signing status changed; review the preview download contract before publication.');
  const filename = `ForensicDGP_Setup-${source.version}.exe`;
  return {
    schema: 1,
    version: source.version,
    filename,
    bytes: byteCount,
    sha256,
    signed: false,
    preview: true,
    downloadUrl: offline
      ? `https://github.com/janusinss/forensic-dgp-downloads/releases/download/v${source.version}/${filename}`
      : `/downloads/${filename}`,
    checksumUrl: `/downloads/${filename}.sha256`,
    ...(offline ? { offline: true } : {}),
  };
}

export async function prepareRelease({ sourceDir = defaultSource, siteDir = siteRoot } = {}) {
  const source = path.resolve(sourceDir);
  const destination = path.resolve(siteDir);
  const metadataPath = path.join(source, 'release.json');
  const artifactPath = path.join(source, 'ForensicDGP_Setup.exe');
  await requireOrdinary(metadataPath, 'file');
  await requireOrdinary(artifactPath, 'file');
  const metadataInfo = await lstat(metadataPath);
  if (metadataInfo.size > 64 * 1024) throw new Error('Source release metadata exceeds its size limit.');
  const metadata = JSON.parse(await readFile(metadataPath, 'utf8'));
  if (metadata === null || Array.isArray(metadata) || typeof metadata !== 'object') throw new Error('Unsupported release metadata schema.');
  const offline = isOffline(metadata);
  const artifactInfo = await lstat(artifactPath);
  const maximum = offline ? maximumOfflineBytes : maximumArtifactBytes;
  if (artifactInfo.size < 1 || artifactInfo.size > maximum) throw new Error('Installer size exceeds the reviewed download limit.');
  const sha256 = await hashArtifact(artifactPath);
  const afterHash = await lstat(artifactPath);
  if (afterHash.size !== artifactInfo.size || afterHash.ino !== artifactInfo.ino || afterHash.mtimeMs !== artifactInfo.mtimeMs) {
    throw new Error('Installer changed while verifying its checksum.');
  }
  const release = releaseMetadata(metadata, artifactInfo.size, sha256);

  // Validation completes before public directories or previously prepared files change.
  const publicDir = path.join(destination, 'public');
  const downloadsDir = path.join(publicDir, 'downloads');
  const sourceOutput = path.join(destination, 'src');
  const filename = path.basename(release.downloadUrl);
  const targets = [
    ...(!offline ? [path.join(downloadsDir, filename)] : []),
    path.join(downloadsDir, `${filename}.sha256`),
    path.join(sourceOutput, 'release.generated.json'),
    path.join(publicDir, 'release.json'),
  ];
  await requireOrdinary(destination, 'directory');
  for (const target of targets) await requireOrdinary(target, 'file', true);
  await ensureDirectory(publicDir);
  await ensureDirectory(downloadsDir);
  await ensureDirectory(sourceOutput);
  const stage = await mkdtemp(path.join(publicDir, '.release-'));
  const staged = targets.map(target => path.join(stage, path.basename(target)));
  try {
    let offset = 0;
    if (!offline) {
      const bytes = await readFile(artifactPath);
      if (bytes.length !== release.bytes || createHash('sha256').update(bytes).digest('hex') !== sha256) {
        throw new Error('Installer changed before release staging.');
      }
      await writeFile(staged[offset++], bytes, { flag: 'wx' });
    }
    await writeFile(staged[offset++], `${sha256}  ${filename}\n`, { encoding: 'utf8', flag: 'wx' });
    for (const target of staged.slice(offset)) await writeFile(target, `${JSON.stringify(release, null, 2)}\n`, { encoding: 'utf8', flag: 'wx' });
    // Each rename replaces one owned output atomically. Publish the metadata last.
    for (let index = 0; index < staged.length; index++) {
      await requireOrdinary(targets[index], 'file', true);
      await rename(staged[index], targets[index]);
    }
  } finally {
    // Remove only this call's four known temporary paths and empty stage directory.
    for (const temporary of staged) {
      try { await unlink(temporary); }
      catch (error) { if (error.code !== 'ENOENT') throw error; }
    }
    await rmdir(stage);
  }
  return release;
}

function commandOptions(argv) {
  let sourceDir = defaultSource;
  for (let index = 0; index < argv.length; index++) {
    if (argv[index] !== '--source-dir' || index + 1 >= argv.length) throw new Error('Usage: node scripts/prepare-release.mjs [--source-dir <verified-build-directory>]');
    sourceDir = path.resolve(argv[++index]);
  }
  return { sourceDir };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    const result = await prepareRelease(commandOptions(process.argv.slice(2)));
    console.log(`Prepared Forensic DGP ${result.version}: ${result.bytes} bytes, SHA-256 ${result.sha256}`);
  } catch (error) {
    console.error(`Release preparation failed: ${error.message}`);
    process.exitCode = 1;
  }
}
