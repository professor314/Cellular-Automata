import json

with open('elementary_ca.ipynb', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Valid JSON: Yes")
print(f"nbformat: {data['nbformat']}.{data['nbformat_minor']}")
print(f"Total cells: {len(data['cells'])}")
print()
for i, c in enumerate(data['cells']):
    ct = c['cell_type']
    preview = c['source'][0][:70].strip() if c['source'] else '(empty)'
    print(f"  [{i:2d}] {ct:8s} | {preview}")
