import inspect
import vardhani_brain.r3_train_full_teacher as t
import vardhani_brain.r3_training_core as c
src=inspect.getsource(t)
assert 'from .r3_training_core import teacher_loss,seed_all' in src
assert 'from .training import teacher_loss,seed_all' not in src
core=inspect.getsource(c)
assert 'from .data' not in core and 'future_targets' not in core
print('R3 heavy trainer legacy-data import isolation PASS')
