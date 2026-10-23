#!/bin/bash
# NAK Drive Training Monitor - checks for new files and processes them
# Runs via cron every 30 minutes

FOLDER_ID="1eOXardO-R5LBQhS0htxPr-86hOjDKAdL"
LOCAL_DIR="/home/hatch/workspace/ai-detector/data/drive_training"
SEEN_FILE="/home/hatch/workspace/ai-detector/data/drive_training/.seen_ids"

mkdir -p "$LOCAL_DIR"
touch "$SEEN_FILE"

# Get current file list from Drive
FILES_JSON=$(/opt/hatch/bin/hatch_gws_cli drive files list --params "{\"q\":\"'$FOLDER_ID' in parents and trashed=false\",\"pageSize\":100,\"fields\":\"files(id,name)\"}" 2>/dev/null)

# Extract IDs and names
echo "$FILES_JSON" | python3 -c "
import json, sys, subprocess, os

data = json.load(sys.stdin)
files = data.get('files', [])

seen_file = '/home/hatch/workspace/ai-detector/data/drive_training/.seen_ids'
with open(seen_file, 'r') as f:
    seen = set(line.strip() for line in f if line.strip())

new_files = [f for f in files if f['id'] not in seen]

if not new_files:
    print('No new files')
    sys.exit(0)

print(f'Found {len(new_files)} new files!', flush=True)

local_dir = '/home/hatch/workspace/ai-detector/data/drive_training'
for f in new_files:
    fid = f['id']
    name = f['name']
    safe_name = ''.join(c for c in name if c.isalnum() or c in '._- ')
    out_path = os.path.join(local_dir, safe_name)
    
    print(f'Downloading: {safe_name}', flush=True)
    result = subprocess.run([
        '/opt/hatch/bin/hatch_gws_cli', 'drive', 'files', 'get',
        '--params', json.dumps({'fileId': fid, 'alt': 'media'}),
        '--output', out_path
    ], capture_output=True)
    
    if result.returncode == 0:
        print(f'  OK', flush=True)
        # Mark as seen
        with open(seen_file, 'a') as sf:
            sf.write(fid + '\n')
    else:
        print(f'  FAIL', flush=True)

print('Download complete. Processing will happen in next training cycle.', flush=True)
"
