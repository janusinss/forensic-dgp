"""Read-only old-archive verification beside the separately owned immutable upload.

Only downloads old archives and writes proof receipts. No delete, upload, model,
training or VM capability. It does not run the offloader's apply operation.
"""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
import offload_u96_inactive_restoration_20261021_v1 as off

def main():
 p=argparse.ArgumentParser();p.add_argument('--plan-sha',required=True);a=p.parse_args()
 plan=off.plan(a.plan_sha);off.base.connection()
 run=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',
  'Get-CimInstance Win32_Process | Select-Object Name,ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Depth 3 -Compress'],
  capture_output=True,text=True,check=True,timeout=30,creationflags=subprocess.CREATE_NO_WINDOW)
 rows=json.loads(run.stdout);evidence=[]
 # Disjoint read-only downloads can share the network with the current job's
 # exact immutable copy. Reject any other rclone or original-helper operation.
 for row in rows:
  line=(row.get('CommandLine') or '').replace('\\','/').lower()
  if row['Name'].lower()=='rclone.exe':
   assert ' copy ' in line and '--immutable' in line and '--files-from-raw' in line
   assert '20261021_u96_inactive_v1' in line and 'u96_inactive_restoration_offload_20261021_v1/files_v1.txt' in line
  if row['Name'].lower().startswith('python') and off.MARKER in line:
   assert ' copy --plan-sha '+a.plan_sha in line
  if row['Name'].lower()=='rclone.exe' or (row['Name'].lower().startswith('python') and off.MARKER in line):
   evidence.append({'pid':row['ProcessId'],'name':row['Name'],'command_sha256':hashlib.sha256(line.encode()).hexdigest()})
 off.save('disjoint_download_concurrency_v1.json',{'complete':True,'plan_sha256':a.plan_sha,
  'actual_other_transfer_operations':evidence,'this_operation':'Read-only full old-archive and exact member verification',
  'local_mutation_or_deletion':False,'distinct_cloud_objects_from_upload':True,'raw_command_lines_exported':False})
 off.verify_members(plan)

if __name__=='__main__':main()
