import argparse, os, shutil
from collections import Counter
from Bio.PDB import PDBParser
from pdbfixer import PDBFixer
from openmm.app import PDBFile

parser_arg = argparse.ArgumentParser()
parser_arg.add_argument('--pdb_dir')
parser_arg.add_argument('--backup_dir')
args = parser_arg.parse_args()

os.makedirs(args.backup_dir, exist_ok=True)

parser = PDBParser(QUIET=True)
pdb_files = sorted(os.path.join(args.pdb_dir, f)
                   for f in os.listdir(args.pdb_dir) if f.endswith('.pdb'))

def count_atoms(path, parser):
    s = parser.get_structure('p', path)
    for model in s:
        return sum(1 for chain in model for res in chain
                   if not res.id[0].strip() for _ in res)

def remove_oxt(path):
    lines = []
    with open(path, 'r') as f:
        for line in f:
            if (line.startswith('ATOM') or line.startswith('HETATM')) \
                    and line[12:16].strip() == 'OXT':
                continue
            lines.append(line)
    with open(path, 'w') as f:
        f.writelines(lines)

counts = {p: count_atoms(p, parser) for p in pdb_files}
mode = Counter(counts.values()).most_common(1)[0][0]
anomalous = [p for p, c in counts.items() if c < mode]

if not anomalous:
    #print('[fix_atoms] Dataset consistente, nada que hacer.')
else:
    #print(f'[fix_atoms] {len(anomalous)} frame(s) anómalos. Moda: {mode} átomos.')
    for pdb_path in anomalous:
        fname = os.path.basename(pdb_path)
        try:
            shutil.copy2(pdb_path, os.path.join(args.backup_dir, fname))
            fixer = PDBFixer(filename=pdb_path)
            fixer.findMissingResidues()
            fixer.missingResidues = {}
            fixer.findNonstandardResidues()
            fixer.findMissingAtoms()
            fixer.addMissingAtoms()
            with open(pdb_path, 'w') as f:
                PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)
            remove_oxt(pdb_path)
            new_count = count_atoms(pdb_path, parser)
            #print(f'[fix_atoms] {fname}: {counts[pdb_path]} -> {new_count} átomos')
        except Exception as e:
            #print(f'[fix_atoms] {fname}: error -> {e}')