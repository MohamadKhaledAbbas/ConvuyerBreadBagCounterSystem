import sys
sys.path.insert(0, '.')
from src.app.adaptive_frame_throttle import AdaptiveFrameThrottle

t = AdaptiveFrameThrottle(enabled=True, idle_timeout_s=900.0, hysteresis_s=60.0)
state = t.get_state()

expected_keys = {
    'enabled', 'mode', 'idle_seconds', 'idle_timeout_s', 'idle_percent',
    'time_until_degrade_s', 'degraded_since_seconds', 'hysteresis_s',
    'degraded_transitions', 'wake_transitions', 'last_wake_signal'
}

actual_keys = set(state.keys())

print('=' * 60)
print('STATE DICTIONARY VALIDATION')
print('=' * 60)
print(f'Expected keys: {len(expected_keys)}')
print(f'Actual keys: {len(actual_keys)}')
print(f'Match: {expected_keys == actual_keys}')

missing = expected_keys - actual_keys
extra = actual_keys - expected_keys

if missing:
    print(f'❌ MISSING: {missing}')
if extra:
    print(f'❌ EXTRA (should not exist): {extra}')

if expected_keys == actual_keys:
    print('✅ SUCCESS: All keys correct')
    print('\nActual state:')
    for key in sorted(state.keys()):
        print(f'  {key}: {state[key]}')
else:
    print('❌ FAILED: Keys mismatch')
    sys.exit(1)

