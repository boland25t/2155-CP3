# debug_utils.py

_last_call = {}

def callout(msg):
    print(f"\n▶️ {msg}")

def conditional_callout(msg, freq=200):
    if msg not in _last_call:
        _last_call[msg] = 0
    _last_call[msg] += 1
    if _last_call[msg] % freq == 0:
        print(f"  ↳ {msg}")
