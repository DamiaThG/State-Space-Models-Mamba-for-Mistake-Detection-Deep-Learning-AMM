import lmdb

LMDB_PATH = '/home/mssdmn01t05c351v/assembly-mistake-detection/data/TSM_features/C10119_rgb'

env = lmdb.open(LMDB_PATH, readonly=True, lock=False)

sequences = set()
with env.begin() as txn:
    cursor = txn.cursor()
    for key, _ in cursor:
        seq = key.decode('utf-8').split('/')[0]
        sequences.add(seq)

env.close()

print(f'Sequenze totali nell LMDB: {len(sequences)}')
print('\nPrime 10 sequenze:')
for s in sorted(sequences)[:10]:
    print(f'  {s}')
