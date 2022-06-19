from representations import sgns_op, sgns_wi, elmo_with_precomp
from sys import argv

model = sgns_wi.SGNS()
model.do_test(argv[1])

vecs1 = model.model.wv.most_similar(positive=['address1'])
vecs2 = model.model.wv.most_similar(positive=['address2'])

print(vecs1)
print(vecs2)