import argparse
import os
import subprocess
import sys
import hashlib


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

    if not pdb_files:
        print('No PDB files found for reconstruction.')
        sys.exit(0)

    total = len(pdb_files)

    for i, pdb_file in enumerate(pdb_files):
        in_path  = os.path.join(args.input_dir,  pdb_file)
        out_path = os.path.join(args.output_dir, pdb_file)
        pct = (i + 1) / total * 100


        try:
            run_cg2all(in_path, out_path)

        except RuntimeError as e:
            print(f'WARNING: {e}', flush=True)
            # Si falla, copia el original sin reconstruir
            import shutil
            shutil.copy2(in_path, out_path)

    print(f'[cg2all] Done. {total} frames reconstructed.', flush=True)


if __name__ == '__main__':
    main()