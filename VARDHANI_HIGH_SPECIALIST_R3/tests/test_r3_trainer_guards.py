from pathlib import Path
import ast

p=Path(__file__).resolve().parents[1]/'vardhani_brain'/'r3_train_full_teacher.py'
s=p.read_text()
# The runtime chronology cursor must be updated unconditionally after the rewind guard.
t=ast.parse(s)
ifs=[n for n in ast.walk(t) if isinstance(n,ast.If) and 'ts < last_ts' in ast.unparse(n.test)]
assert len(ifs)==1
assert not any(isinstance(n,(ast.Assign,ast.AnnAssign,ast.AugAssign)) and 'last_ts' in ast.unparse(n) for n in ast.walk(ifs[0]))
assert "last_ts=ts" in s.replace(' ','')
# 2025 validation hashes/dataset must occur only after the final development checkpoint save.
pos_final=s.index("final=out/'teacher_final_development.pt'")
pos_val_hash=s.index('expected_val={')
pos_val_ds=s.index("va=R3DualChronologicalDataset")
assert pos_final < pos_val_hash < pos_val_ds
# Frozen development SHAs must be checked before CUDA and optimizer construction.
pos_dev=s.index('expected_dev={')
pos_cuda=s.index("if not torch.cuda.is_available()")
pos_opt=s.index('opt=torch.optim.AdamW')
assert pos_dev < pos_cuda < pos_opt
print('R3 heavy trainer chronology/hash/deferred-2025 guards PASS')
