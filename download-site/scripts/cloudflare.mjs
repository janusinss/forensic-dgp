/** Deploy this Pages site using only the isolated personal-account login. */
import {spawnSync} from 'node:child_process';
import {readFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import path from 'node:path';

const site = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const account = 'b9e82b6514591358168877d8118aec18';
const project = 'forensic-dgp';

export function makeInvocation(action, configuration, parentEnv = {}, siteRoot = site) {
  if (!configuration || configuration.name !== project ||
      configuration.pages_build_output_dir !== './dist' ||
      ['account_id', 'main', 'assets', 'workers_dev'].some(key => Object.hasOwn(configuration, key))) {
    throw new Error('Expected the forensic-dgp Pages configuration in the personal Cloudflare account.');
  }
  const commands = {
    whoami: ['whoami'],
    login: ['login', '--device', '--browser', 'false', '--scopes',
      'account:read', 'user:read', 'workers_scripts:write', 'pages:write'],
    create: ['pages', 'project', 'create', project, '--production-branch', 'main', '--force'],
    projects: ['pages', 'project', 'list', '--json'],
    deploy: ['pages', 'deploy', 'dist', '--project-name', project, '--branch', 'main', '--force'],
  };
  if (!Object.hasOwn(commands, action)) throw new Error('Expected whoami, login, create, projects or deploy.');
  const env = {...parentEnv,
    XDG_CONFIG_HOME: path.resolve(siteRoot, '../scratch/cloudflare-personal'),
    CLOUDFLARE_ACCOUNT_ID: account,
  };
  for (const key of ['CLOUDFLARE_API_TOKEN', 'CLOUDFLARE_API_KEY', 'CLOUDFLARE_EMAIL',
    'CF_API_TOKEN', 'CF_API_KEY', 'CF_EMAIL', 'CF_ACCOUNT_ID']) delete env[key];
  return {args: commands[action], env};
}

export function runCloudflare(action, {configuration, parentEnv = process.env,
    spawn = spawnSync, siteRoot = site} = {}) {
  const config = configuration ?? JSON.parse(readFileSync(path.join(siteRoot, 'wrangler.jsonc'), 'utf8'));
  const {args, env} = makeInvocation(action, config, parentEnv, siteRoot);
  const cli = path.join(siteRoot, 'node_modules/wrangler/bin/wrangler.js');
  const result = spawn(process.execPath, [cli, ...args], {cwd: siteRoot, env, stdio: 'inherit'});
  if (result.error) throw result.error;
  return result.status ?? 1;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try { process.exitCode = runCloudflare(process.argv[2]); }
  catch (error) { console.error(error.message); process.exitCode = 1; }
}
