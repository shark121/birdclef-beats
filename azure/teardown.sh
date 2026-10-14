#!/usr/bin/env bash
# Stop the meter. Run the first line to pause billing, the second to remove
# everything this project created. Nothing outside birdclef-rg is touched.
set -euo pipefail

case "${1:-stop}" in
  stop)
    az vm deallocate -g birdclef-rg -n birdclef-vm
    echo "VM deallocated -- compute billing stopped, disks + storage still cost pennies/day."
    ;;
  destroy)
    az group delete -n birdclef-rg --yes --no-wait
    echo "birdclef-rg deletion started -- VM, disk, NSG, public IP and storage all go."
    ;;
  *)
    echo "usage: $0 [stop|destroy]"; exit 1 ;;
esac
