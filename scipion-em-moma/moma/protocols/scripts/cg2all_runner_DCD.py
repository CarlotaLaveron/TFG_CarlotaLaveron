
import argparse
import os
import subprocess
import sys
import mdtraj as md


def main():
    parser = argparse.ArgumentParser(
        description='Reconstruct backbone from DCD using cg2all.'
    )
    parser.add_argument('--topology', required=True,
                        help='Topology PDB file (first frame of walk)')
    parser.add_argument('--dcd',      required=True,
                        help='Input DCD file (Cα-only trajectory)')
    parser.add_argument('--out_dcd',  required=True,
                        help='Output reconstructed DCD file')
    parser.add_argument('--out_top',  required=True,
                        help='Output topology PDB file')
    parser.add_argument('--output_dir',  required=True,
                    help='Directory to save individual PDB frames')
    parser.add_argument('--frame_start', type=int, default=0,
                        help='Starting frame index for naming output PDBs')
    args = parser.parse_args()

    print(f'[DEBUG] topology: {args.topology}', flush=True)
    print(f'[DEBUG] topology exists: {os.path.isfile(args.topology)}', flush=True)
    print(f'[DEBUG] dcd: {args.dcd}', flush=True)
    print(f'[DEBUG] dcd exists: {os.path.isfile(args.dcd)}', flush=True)
    print(f'[DEBUG] out_dcd: {args.out_dcd}', flush=True)
    print(f'[DEBUG] out_top: {args.out_top}', flush=True)


    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': ''}

    cmd = [
        'convert_cg2all',
        '-p',  args.topology,
        '-d',  args.dcd,
        '-o',  args.out_dcd,
        '-opdb', args.out_top,
        '--cg', 'CalphaBasedModel',
        '--fix',
        '--device', 'cpu',
    ]

    print(f'[cg2all_dcd] Running: {" ".join(cmd)}', flush=True)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
    )

    if result.stdout:
        print(result.stdout, flush=True)
    if result.stderr:
        print(result.stderr, flush=True)

    if result.returncode != 0:
        print(f'ERROR: cg2all failed on {args.dcd}', flush=True)
        sys.exit(1)

    print(f'[cg2all_dcd] Done: {args.out_dcd}', flush=True)

    
    traj = md.load_dcd(args.out_dcd, top=args.out_top)
    traj = md.load_dcd(args.out_dcd, top=args.out_top)
    print(f'[DEBUG] Extracted {len(traj)} frames from DCD', flush=True)

    for j in range(len(traj)):
        out_pdb = os.path.join(args.output_dir, f'frame_{args.frame_start + j:04d}.pdb')
        
        # Seleccionar solo átomos pesados (sin hidrógenos)
        heavy = traj.topology.select('not element H')
        frame = traj[j].atom_slice(heavy)
        
        # Escribir sin MODEL record
        frame.save_pdb(out_pdb, force_overwrite=True)
        
        # Limpiar: quitar MODEL/ENDMDL y columna de segmento P000
        with open(out_pdb) as fh:
            lines = fh.readlines()
        with open(out_pdb, 'w') as fh:
            for line in lines:
                record = line[:6].strip()
                if record in ('MODEL', 'ENDMDL'):
                    continue
                if record in ('ATOM', 'HETATM') and len(line) >= 76:
                    line = line[:72] + '    ' + line[76:]
                fh.write(line)

    print(f'[cg2all_dcd] Frames saved to {args.output_dir}', flush=True)



if __name__ == '__main__':
    main()