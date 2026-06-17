import argparse
import os
import subprocess
import sys


def run_cg2all(in_path, out_path):
    """Run convert_cg2all on a single PDB file."""
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': ''}

    cmd = [
        'convert_cg2all',
        '-p', in_path,
        '-o', out_path,
        '--cg', 'CalphaBasedModel',
        '--fix',
        '--device', 'cpu',
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f'cg2all failed on {in_path}:\n'
            f'STDOUT: {result.stdout}\n'
            f'STDERR: {result.stderr}'
        )

    return result.stdout


def main():
    parser = argparse.ArgumentParser(
        description='Reconstruct backbone from Cα-only PDB files using cg2all.'
    )
    parser.add_argument('--input_dir',  required=True,
                        help='Directory with Cα-only PDB files')
    parser.add_argument('--output_dir', required=True,
                        help='Directory for reconstructed PDB files')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pdb_files = sorted(
        f for f in os.listdir(args.input_dir) if f.endswith('.pdb')
    )

    import hashlib
    def md5(path):
        return hashlib.md5(open(path,'rb').read()).hexdigest()

    #print(f'[DEBUG] input_dir: {args.input_dir}', flush=True)
    #print(f'[DEBUG] Total PDB files: {len(pdb_files)}', flush=True)
    #print(f'[DEBUG] First 3: {pdb_files[:3]}', flush=True)
    if len(pdb_files) >= 2:
        md5_0 = md5(os.path.join(args.input_dir, pdb_files[0]))
        md5_1 = md5(os.path.join(args.input_dir, pdb_files[1]))
        #print(f'[DEBUG] md5 frame_0: {md5_0}', flush=True)
        #print(f'[DEBUG] md5 frame_1: {md5_1}', flush=True)
        #print(f'[DEBUG] Son iguales: {md5_0 == md5_1}', flush=True)


    if not pdb_files:
        print('No PDB files found for reconstruction.')
        sys.exit(0)

    total = len(pdb_files)

    for i, pdb_file in enumerate(pdb_files):
        in_path  = os.path.join(args.input_dir,  pdb_file)
        out_path = os.path.join(args.output_dir, pdb_file)
        pct = (i + 1) / total * 100

        #print(f'[cg2all] Frame {i+1}/{total} ({pct:.1f}%) - {pdb_file}', flush=True)

        try:
            run_cg2all(in_path, out_path)
            if os.path.isfile(out_path):
                atom_counts = {}
                ca_count = 0
                chains = set()
                with open(out_path) as f:
                    for line in f:
                        if line.startswith('ATOM'):
                            atom_name = line[12:16].strip()
                            chain = line[21]
                            chains.add(chain)
                            atom_counts[atom_name] = atom_counts.get(atom_name, 0) + 1
                            if atom_name == 'CA':
                                ca_count += 1
                backbone = {k: atom_counts.get(k, 0) for k in ['N', 'CA', 'C', 'O', 'CB']}
                #print(f'  [DEBUG] Chains: {sorted(chains)} | '
                # f'CA: {ca_count} | '
                # f'Backbone: {backbone} | '
                # f'Total atoms: {sum(atom_counts.values())}',
                # flush=True)
        except RuntimeError as e:
            print(f'WARNING: {e}', flush=True)
            # Si falla, copia el original sin reconstruir
            import shutil
            shutil.copy2(in_path, out_path)

    print(f'[cg2all] Done. {total} frames reconstructed.', flush=True)


if __name__ == '__main__':
    main()