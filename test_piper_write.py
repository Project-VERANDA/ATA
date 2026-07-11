import tempfile
import os

fd, temp_path = tempfile.mkstemp(suffix='.wav', prefix='test_piper_', dir='/tmp')
os.close(fd)
print(f"Created: {temp_path}")

try:
    with open(temp_path, 'w') as f:
        f.write("test")
    print("✅ Write to /tmp OK")
except PermissionError as e:
    print(f"❌ Write to /tmp failed: {e}")

os.unlink(temp_path)
