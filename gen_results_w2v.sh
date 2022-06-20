#/bin/bash

python3 run.py sgns_wi tasks/sem_shift_en.json
python3 run.py sgns_op tasks/sem_shift_en.json

python3 run.py sgns_wi tasks/alice_mod_test.json
python3 run.py sgns_op tasks/alice_mod_test.json

python3 run.py sgns_wi tasks/subst.json
python3 run.py sgns_op tasks/subst.json

python3 run.py sgns_wi tasks/alice_mod_test10.json
python3 run.py sgns_op tasks/alice_mod_test10.json

python3 run.py sgns_wi tasks/alice_mod_test50.json
python3 run.py sgns_op tasks/alice_mod_test50.json

python3 run.py sgns_wi tasks/alice_mod_test100.json
python3 run.py sgns_op tasks/alice_mod_test100.json

python3 run.py sgns_wi tasks/rodica_wallachia.json
python3 run.py sgns_wi tasks/rodica_transylvania.json
python3 run.py sgns_wi tasks/rodica_bessarabia.json
python3 run.py sgns_wi tasks/rodica_moldova.json

python3 run.py sgns_op tasks/rodica_wallachia.json
python3 run.py sgns_op tasks/rodica_transylvania.json
python3 run.py sgns_op tasks/rodica_bessarabia.json
python3 run.py sgns_op tasks/rodica_moldova.json