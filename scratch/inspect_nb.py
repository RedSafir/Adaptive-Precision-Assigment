import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('vgg16_cifar10_tf32_fp8.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code' and 'def evaluate' in ''.join(c['source']):
        print(f"===== Cell {idx} =====")
        print(''.join(c['source']))
        print()
