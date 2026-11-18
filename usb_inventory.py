"""Enumerate USB on the Streamlit SERVER host only; never on a remote browser."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

def inventory():
    if sys.platform.startswith('linux'):
        root=Path('/sys/bus/usb/devices')
        if not root.is_dir(): raise RuntimeError('Linux USB sysfs is unavailable. Run on the host, not an isolated container.')
        found=[]
        for dev in root.iterdir():
            if not (dev/'idVendor').exists(): continue
            def read(name):
                try:return (dev/name).read_text().strip()
                except OSError:return ''
            found.append(f"{read('manufacturer')} {read('product') or 'USB device'} · {read('idVendor')}:{read('idProduct')} · port {dev.name}".strip()[:256])
        return sorted(set(found))
    if sys.platform.startswith('win'):
        command=['powershell','-NoProfile','-NonInteractive','-Command',
                 "@(Get-PnpDevice -PresentOnly -ErrorAction Stop | Where-Object { $_.InstanceId -like 'USB\\*' -or $_.Class -eq 'WPD' } | Select-Object FriendlyName,InstanceId) | ConvertTo-Json -Compress"]
        result=subprocess.run(command,capture_output=True,text=True,timeout=20)
        if result.returncode: raise RuntimeError('Windows USB enumeration failed; check PowerShell and device permissions.')
        data=json.loads(result.stdout or '[]'); data=[data] if isinstance(data,dict) else data
        return sorted(set(f"{d.get('FriendlyName') or 'USB device'} · id {hashlib.sha256(d.get('InstanceId','').encode()).hexdigest()[:10]}"[:256] for d in data))
    if sys.platform=='darwin':
        result=subprocess.run(['system_profiler','-json','SPUSBDataType'],capture_output=True,text=True,timeout=25)
        if result.returncode: raise RuntimeError('macOS USB enumeration failed.')
        data=json.loads(result.stdout); found=[]
        def walk(obj):
            if isinstance(obj,dict):
                if 'vendor_id' in obj and '_name' in obj:
                    found.append(f"{obj['_name']} · {obj.get('vendor_id','')}:{obj.get('product_id','')} · {obj.get('location_id','')}"[:256])
                for value in obj.values():walk(value)
            elif isinstance(obj,list):
                for value in obj:walk(value)
        walk(data)
        return sorted(set(found))
    raise RuntimeError('This operating system is not supported by the USB agent.')
