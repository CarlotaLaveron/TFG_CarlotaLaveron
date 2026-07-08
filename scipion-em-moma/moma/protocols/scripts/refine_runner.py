import sys
import os
import argparse
import shutil
import openmm as mm
import openmm.app as app
import openmm.unit as unit

#not part of TFG


def refine_pdb(in_path, out_path, forcefield, max_iterations=500):
    try:
        pdb = app.PDBFile(in_path)

        modeller = app.Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens(forcefield)


        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=app.HBonds,
        )

        # Restraints sobre CA para conservar la geometría del ensemble
        restraint = mm.CustomExternalForce(
            'k*((x-x0)^2+(y-y0)^2+(z-z0)^2)'
        )
        restraint.addGlobalParameter(
            'k', 100.0 * unit.kilojoules_per_mole / unit.nanometer**2
        )
        restraint.addPerParticleParameter('x0')
        restraint.addPerParticleParameter('y0')
        restraint.addPerParticleParameter('z0')

        for atom in modeller.topology.atoms():
            if atom.name == 'CA':
                pos = modeller.positions[atom.index]
                restraint.addParticle(atom.index, pos)

        system.addForce(restraint)

        integrator = mm.LangevinMiddleIntegrator(
            300 * unit.kelvin,
            1 / unit.picosecond,
            0.004 * unit.picoseconds,
        )

        simulation = app.Simulation(modeller.topology, system, integrator)
        simulation.context.setPositions(modeller.positions)
        simulation.minimizeEnergy(maxIterations=max_iterations)

        positions = simulation.context.getState(getPositions=True).getPositions()

        with open(out_path, 'w') as f:
            app.PDBFile.writeFile(
                modeller.topology, positions, f,
                keepIds=True
            )


        print(f'Refined: {os.path.basename(in_path)}')

    except Exception as e:
        print(f'WARNING: could not refine {in_path}: {e}')
        shutil.copy(in_path, out_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',          required=True,  help='Directory with reconstructed PDB files')
    parser.add_argument('--output',         required=True,  help='Directory for refined PDB files')
    parser.add_argument('--max_iterations', default=50,    type=int)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    pdb_files = sorted(f for f in os.listdir(args.input) if f.endswith('.pdb'))

    if not pdb_files:
        print('No PDB files found for refinement.')
        sys.exit(0)

    total = len(pdb_files)
    forcefield = app.ForceField('amber14-all.xml')

    for i, pdb_file in enumerate(pdb_files):
        in_path  = os.path.join(args.input,  pdb_file)
        out_path = os.path.join(args.output, pdb_file)
        pct = (i + 1) / total * 100
        print(f'[Refine] Frame {i+1}/{total} ({pct:.1f}%) - {pdb_file}', flush=True)
        refine_pdb(in_path, out_path, forcefield, args.max_iterations)

    print(f'[Refine] Done. {total} frames refined.', flush=True)

