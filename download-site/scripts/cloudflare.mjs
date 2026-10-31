/** Keep this site's personal-account login separate from other Wrangler projects. */
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
const site=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const action=process.argv[2];
if(!['whoami','login','deploy'].includes(action))throw new Error('Expected whoami, login or deploy.');
const env={...process.env,XDG_CONFIG_HOME:path.resolve(site,'../scratch/cloudflare-personal')};
delete env.CLOUDFLARE_API_TOKEN;
delete env.CLOUDFLARE_API_KEY;
delete env.CLOUDFLARE_EMAIL;
delete env.CLOUDFLARE_ACCOUNT_ID;
const cli=path.join(site,'node_modules/wrangler/bin/wrangler.js');
const args=action==='login'?['login','--device','--browser','false','--scopes','account:read','user:read','workers_scripts:write']:[action];
const result=spawnSync(process.execPath,[cli,...args],{cwd:site,env,stdio:'inherit'});
if(result.error)throw result.error;
process.exitCode=result.status??1;
