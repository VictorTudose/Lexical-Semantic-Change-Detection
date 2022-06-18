#/bin/bash

python3 run.py sgns_wi tasks/sem_shift_en.json
python3 run.py sgns_wi tasks/alice_mod_test.json
python3 run.py sgns_wi tasks/subst.json

python3 run.py sgns_op tasks/sem_shift_en.json
python3 run.py sgns_op tasks/alice_mod_test.json
python3 run.py sgns_op tasks/subst.json

