#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:     Carlota Laverón (carlota.laveronvilas@usp.ceu.es)
# *
# * CEU San Pablo University
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# **************************************************************************

"""
Runner script for geometry optimization of protein structures.
Called by ProtocolOptimization via pwchemPlugin.runScript.

For each input PDB:
  1. Read original residue numbering (before PDBFixer renumbers)
  2. PDBFixer  – fix missing atoms, add hydrogens (pH 7.4s)
  3. OpenMM    – energy minimization with AMBER14 force field (CPU) + Cα position restraints to preserve backbone
  4. Write     – output PDB with '_optimized' suffix
  5. Restore   – original residue numbering per chain
"""

import argparse
import os
import sys
import traceback


missing = []
try:
    from pdbfixer import PDBFixer
except ImportError:
    missing.append('pdbfixer')

try:
    from openmm.app import (ForceField, Simulation, PDBFile,
                            HBonds, NoCutoff, Modeller)
    from openmm import LangevinMiddleIntegrator, Platform, CustomExternalForce
    from openmm import unit
except ImportError:
    missing.append('openmm')

if missing:
    print(f'ERROR: Missing dependencies: {", ".join(missing)}')
    print('Install them with: ./scipion3 run pip install ' + ' '.join(missing))
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description='Geometry optimization runner')
    parser.add_argument('--pdb_list', type=str, required=True,
                        help='Comma-separated list of input PDB file paths')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory where optimized PDBs will be saved')
    parser.add_argument('--max_iterations', type=int, default=100,
                        help='Max OpenMM minimization iterations per phase (0 = until convergence)')
    parser.add_argument('--ca_force_constant', type=float, default=100.0,
                        help='Force constant for Cα position restraints (kcal/mol/Å²).')
    return parser.parse_args()


def _get_offsets(pdb_path):
    """
    Read the first residue number per chain from the original PDB.
    Returns a dict {chain: offset} where offset = first_resnum - 1.
    PDBFixer renumbers from 1, so we need to add this offset back.
    """
    offsets = {}
    with open(pdb_path) as f:
        for line in f:
            if line.startswith('ATOM'):
                chain = line[21] #chain 
                try:
                    resnum = int(line[22:26]) #residue number in the aa chain
                except ValueError:
                    continue
                if chain not in offsets:
                    offsets[chain] = resnum - 1
    return offsets

def optimize_one(input_pdb, output_pdb, max_iterations, ca_force_constant):

    offsets = _get_offsets(input_pdb)
    #print(f'  [Numbering] Original offsets per chain: {offsets}')

    #print(f'  [PDBFixer] Fixing: {input_pdb}')
    fixer = PDBFixer(filename=input_pdb)

    original_residue_numbers = {}
    with open(input_pdb) as f:
        for line in f:
            if line.startswith('ATOM'):
                atom_index = int(line[6:11].strip())
                chain = line[21]
                res_num = int(line[22:26].strip())
                res_name = line[17:20].strip()
                atom_name = line[12:16].strip()
                
                key = (chain, res_num, res_name, atom_name)
                original_residue_numbers[atom_index] = (chain, res_num)

    
    print(f'  [PDBFixer] Removing non-standard residues...')
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.removeHeterogens()
    fixer.addMissingHydrogens(7.4)
    
    print(f'  [PDBFixer] Done.')

    print(f'  [OpenMM] Building system (AMBER99SB, NoCutoff, CPU)...')
    ff = ForceField('amber99sb.xml', 'amber14/tip3p.xml')
    modeller = Modeller(fixer.topology, fixer.positions)
    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=NoCutoff,
        constraints=HBonds,
    )

    k_kj = ca_force_constant * 418.4
    ca_restraint_strong = CustomExternalForce(
        f'{k_kj}*((x-x0)^2+(y-y0)^2+(z-z0)^2)'
    )
    ca_restraint_strong.addPerParticleParameter('x0')
    ca_restraint_strong.addPerParticleParameter('y0')
    ca_restraint_strong.addPerParticleParameter('z0')

    positions = modeller.positions
    n_ca = 0
    for atom in modeller.topology.atoms():
        if atom.name == 'CA':
            pos = positions[atom.index]
            ca_restraint_strong.addParticle(atom.index, [
                float(pos.x),
                float(pos.y),
                float(pos.z)
            ])
            n_ca += 1

    system.addForce(ca_restraint_strong)
    print(f'  [OpenMM] Cα restraints (strong): {n_ca} atoms, '
          f'k={ca_force_constant} kcal/mol/Å² ({k_kj:.1f} kJ/mol/nm²)')

    integrator = LangevinMiddleIntegrator(
        300 * unit.kelvin,
        1 / unit.picosecond,
        0.004 * unit.picoseconds,
    )
    platform = Platform.getPlatformByName('CPU')
    simulation = Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(positions)

    #print(f'  [OpenMM]: Minimizing with strong Cα restraints (max_iterations={max_iterations})...')
    simulation.minimizeEnergy(maxIterations=max_iterations)

    state = simulation.context.getState(getPositions=True)
    state = simulation.context.getState(getPositions=True)
    with open(output_pdb, 'w') as f:
        PDBFile.writeFile(modeller.topology, state.getPositions(), f)
    #print(f'  [Output] Saved: {output_pdb}')



def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    pdb_files = [p.strip() for p in args.pdb_list.split(',') if p.strip()]
    if not pdb_files:
        print('ERROR: No PDB files provided.')
        sys.exit(1)

    print(f'Optimizing {len(pdb_files)} structure(s)...')
    print(f'Cα force constant: {args.ca_force_constant} kcal/mol/Å²')
    print(f'Two-phase optimization (strong + soft restraints)')
    failed = []

    for i, pdb_path in enumerate(pdb_files):
        basename = os.path.splitext(os.path.basename(pdb_path))[0]
        out_path = os.path.join(args.output_dir, f'{basename}_optimized.pdb')

        print(f'\n{"="*50}')
        print(f'[{i+1}/{len(pdb_files)}] Processing: {basename}')
        #print(f' Input : {pdb_path}')
        #print(f' Output : {out_path}')
        #print(f' Exists: {os.path.exists(pdb_path)}')
        #print(f' Size : {os.path.getsize(pdb_path) / 1024:.1f} KB')

        try:
            optimize_one(pdb_path, out_path, args.max_iterations,
                         args.ca_force_constant)
            #print(f'  STATUS : OK')
        except Exception as e:
            print(f'  STATUS : FAILED')
            print(f'  ERROR  : {e}')
            traceback.print_exc()
            failed.append(pdb_path)

    #print(f'\n{"="*50}')
    #print(f'Completed: {len(pdb_files) - len(failed)}/{len(pdb_files)} structures optimized.')
    if failed:
        print(f'Failed ({len(failed)}):')
        for f in failed:
            print(f'  - {f}')
        sys.exit(1)


if __name__ == '__main__':
    main()