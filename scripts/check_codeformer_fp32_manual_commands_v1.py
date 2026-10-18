"""Parse the rendered manual commands without executing VM/SCP/training work."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess


root = Path(__file__).resolve().parents[1]
evidence = root / 'outputs/codeformer_fp32_comparison_v1_preparation'
archive = json.loads((evidence / 'execution_archive.json').read_text())
p = json.loads((root / 'outputs/cctv_codeformer_fp32_comparison_v1_vm/protocol.json').read_text())
publisher = root / 'scripts/publish_codeformer_fp32_comparison_v1.py'
tree = ast.parse(publisher.read_text())
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
node = next(n.value for n in function.body if isinstance(n, ast.Assign) and
            any(isinstance(t, ast.Name) and t.id == 'guide' for t in n.targets))
context = {'archive': archive, 'dependency_bytes': sum(p['dependency_files_bytes'].values()),
           'prefix': 'cctv-codeformer-fp32-comparison-v1',
           'protocol_sha': archive['protocol_sha256'], 'archive_sha': archive['archive_sha256']}
guide = eval(compile(ast.Expression(node), '<guide-only-literal>', 'eval'), {'__builtins__': {}}, context)
blocks = re.findall(r'```bash\n(.*?)```', guide, re.S)
for index, block in enumerate(blocks, 1):
    # -n reads/parses stdin. None of the commands within stdin are executed.
    result = subprocess.run(['C:/Program Files/Git/bin/bash.exe', '-n'], input=block,
                            text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, (index, result.stderr)
batch = re.findall(r'```bat\n(.*?)```', guide, re.S)
scp = [line for block in batch for line in block.splitlines() if line.startswith('gcloud ')]
assert len(scp) == 5 and all(line.count('janusdominic0@forensic-dgp-thesis:') == 1 for line in scp)
sha = lambda data: hashlib.sha256(data).hexdigest()
receipt = {'complete': True, 'bash_blocks_parsed_without_execution': len(blocks),
           'SCP_commands': len(scp), 'single_remote_source_per_command': True,
           'protocol_sha256': archive['protocol_sha256'], 'archive_sha256': archive['archive_sha256'],
           'publisher_source_sha256': sha(publisher.read_bytes()), 'guide_sha256': sha(guide.encode()),
           'neural_training_VM_or_SCP_calls': 0, 'checker_sha256': sha(Path(__file__).read_bytes())}
with (evidence / 'manual_command_syntax_v1.json').open('x', encoding='utf-8', newline='\n') as stream:
    json.dump(receipt, stream, indent=2); stream.write('\n')
print(json.dumps(receipt))
