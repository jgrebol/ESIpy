import pickle
with open('/home/joan/PycharmProjects/ESIpy/tests/pyscf_refs.pkl', 'rb') as f:
    refs = pickle.load(f)

print(refs['7_rmp2'].keys())
print("Pops:", refs['7_rmp2']['ind']['pops'])
