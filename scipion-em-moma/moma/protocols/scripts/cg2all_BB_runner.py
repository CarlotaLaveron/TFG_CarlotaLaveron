import argparse
import os
import glob
import subprocess
import sys
import mdtraj as md


def main():
    parser = argparse.ArgumentParser(
        description='Reconstruct backbone from PDB ensemble using cg2all.'
    )
    parser.add_argument('--input_dir', required=True,
                        help='Directory with individual CA-only PDB frames')
    parser.add_argument('--output_dir', required=True,
                        help='Directory to save reconstructed PDB frames')
    args = parser.parse_args()

    #Convert to DCD
    pdb_files = sorted(glob.glob(os.path.join(args.input_dir, 'frame_*.pdb')))
    if not pdb_files:
        print(f'ERROR: No frame PDBs found in {args.input_dir}', flush=True)
        sys.exit(1)

    topology = pdb_files[0]
    dcd_path = os.path.join(args.input_dir, 'ensemble.dcd')
    traj = md.load(pdb_files, top=topology)
    traj.save_dcd(dcd_path)
    #print(f'[cg2all] Converted {len(pdb_files)} PDBs to DCD: {dcd_path}', flush=True)

    out_dcd = os.path.join(args.output_dir, 'ensemble_reconstructed.dcd')
    out_top = os.path.join(args.output_dir, 'ensemble_topology.pdb')

    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': ''}
    cmd = [
        'convert_cg2all',
        '-p', topology,
        '-d', dcd_path,
        '-o', out_dcd,
        '-opdb', out_top,
        '--cg', 'CalphaBasedModel',
        '--fix',
        '--device', 'cpu',
    ]

    print(f'[cg2all] Running: {" ".join(cmd)}', flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if result.stdout:
        print(result.stdout, flush=True)
    if result.stderr:
        print(result.stderr, flush=True)

    if result.returncode != 0:
        print(f'ERROR: cg2all failed on {dcd_path}', flush=True)
        sys.exit(1)

    # 4. Extraer frames individuales
    traj_out = md.load_dcd(out_dcd, top=out_top)
    print(f'[cg2all] Extracted {len(traj_out)} frames', flush=True)
    for j in range(len(traj_out)):
        out_pdb = os.path.join(args.output_dir, f'frame_{j:04d}.pdb')
        traj_out[j].save_pdb(out_pdb)

    print(f'[cg2all] Done. Frames saved to {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()