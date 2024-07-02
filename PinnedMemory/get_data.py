
import subprocess

for i in range(0, 28):
    subprocess.call(['./a.out', str(2**i)])
