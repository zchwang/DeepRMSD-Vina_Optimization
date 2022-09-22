import os

targets = [x for x in os.listdir("samples")]

content = []
for t in targets:
    rec = "samples/" + t + "/" + t + "_protein_atom_noHETATM.pdbqt"
    decoys_dpath ="samples/" + t + "/decoys"

    content.append(t + " " + rec + " " + decoys_dpath)

with open("inputs.dat", "w") as f:
    for i in content:
        f.writelines(i + "\n")