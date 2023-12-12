import string, secrets
from openbabel import pybel

all_defined_residues = ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR', 'TRP', 'SER',
                        'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP', 'GLU', 'LYS', 'ARG', 'HIS', "OTH"]
all_rec_defined_ele = ['C', 'O', 'N', 'S', 'DU']
all_lig_ele = ['C', 'O', 'N', 'P', 'S', 'Hal', 'DU']
Hal = ['F', 'Cl', 'Br', 'I']
ad4_to_ele_dict = {
    "C": "C",
    "A": "C",
    "N": "N",
    "NA": "N",
    "OA": "O",
    "S": "S",
    "SA": "S",
    "Se": "S",
    "P": "P",
    "F": "F",
    "Cl": "Cl",
    "Br": "Br",
    "I": "I",
}

def get_elementtype(e):
    if e in all_lig_ele:
        return e
    elif e in Hal:
        return 'Hal'
    else:
        return 'DU'

def convert_format(input_file, input_format, output_file, output_format):
    molecules = pybel.readfile(input_format, input_file)
    with pybel.Outputfile(output_format, output_file, overwrite=True) as output:
        for mol in molecules:
            output.write(mol)

def generate_secret(length=10):
    random_string = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(length))
    return random_string

def sdf_split(infile):
    contents = open(infile, 'r').read()
    return [c + "$$$$\n" for c in contents.split("$$$$\n")[:-1]]