#!/usr/bin/env bash
# Pull the finished export off the VM, then archive it to blob storage.
# Run from the project root on the workstation.
set -euo pipefail
VM_IP="${VM_IP:-20.172.179.216}"
KEY=azure/birdclef_vm_key
REMOTE=/opt/birdclef/app

mkdir -p exports/baseline data/interim
scp -i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    "azureuser@$VM_IP:$REMOTE/exports/baseline/*" exports/baseline/
scp -i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    "azureuser@$VM_IP:$REMOTE/exports/ATTRIBUTION.md" exports/ 2>/dev/null || true
scp -i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    "azureuser@$VM_IP:$REMOTE/data/interim/recordings.csv" data/interim/
scp -i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    "azureuser@$VM_IP:$REMOTE/data/interim/clips.csv" data/interim/

SA=$(cat azure/.storage_account_name)
KEYV=$(cat azure/.storage_key)
az storage blob upload-batch --account-name "$SA" --account-key "$KEYV" \
    -d exports -s exports --overwrite --only-show-errors
echo "exports pulled locally and archived to blob container 'exports'"
