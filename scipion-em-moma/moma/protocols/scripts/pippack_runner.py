import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description='PIPPack sidechain packing runner for Scipion integration.'
    )
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--pippack_dir', required=True)
    parser.add_argument('--num_ensembles', type=int, default=4)
    parser.add_argument('--model_weights', default=None)
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Número de PDBs por llamada a PIPPack. '
                             'Reducir si hay OOM. Default: 1.')
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    pippack_dir = os.path.abspath(args.pippack_dir)

    weights_path = os.path.abspath(args.model_weights) if args.model_weights \
                   else os.path.join(pippack_dir, 'model_weights')

    inference_script = os.path.join(pippack_dir, 'ensembled_inference.py')

    model_names = sorted(
        f.replace('_ckpt.pt', '')
        for f in os.listdir(weights_path)
        if f.endswith('_ckpt.pt') and f.startswith('pippack_model')
    )

    #print(f'[PIPPack] input_dir      : {input_dir}',    flush=True)
    #print(f'[PIPPack] output_dir     : {output_dir}',   flush=True)
    #print(f'[PIPPack] pippack_dir    : {pippack_dir}',  flush=True)
    #print(f'[PIPPack] weights_path   : {weights_path}', flush=True)
    #print(f'[PIPPack] model_names    : {model_names}',  flush=True)
    #print(f'[PIPPack] num_ensembles  : {args.num_ensembles}', flush=True)
    #print(f'[PIPPack] batch_size     : {args.batch_size}', flush=True)

    for path, label in [
        (input_dir, 'input_dir'),
        (pippack_dir, 'pippack_dir'),
        (inference_script, 'ensembled_inference.py'),
        (weights_path, 'weights_path'),
    ]:
        if not os.path.exists(path):
            print(f'ERROR: {label} not found: {path}', flush=True)
            sys.exit(1)

    if not model_names:
        print(f'ERROR: No pippack_model_*_ckpt.pt files found in {weights_path}',
              flush=True)
        sys.exit(1)

    pdb_files = sorted(
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith('.pdb')
    )

    if not pdb_files:
        print(f'ERROR: No PDB files found in {input_dir}', flush=True)
        sys.exit(1)

    print(f'[PIPPack] Found {len(pdb_files)} PDB files to process.', flush=True)
    os.makedirs(output_dir, exist_ok=True)

    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': ''}
    model_names_str = ','.join(model_names)

    import shutil

    #Error arised -> batch processing fixed it
    for batch_start in range(0, len(pdb_files), args.batch_size):
        batch = pdb_files[batch_start:batch_start + args.batch_size]
        batch_idx = batch_start // args.batch_size + 1
        n_batches = (len(pdb_files) + args.batch_size - 1) // args.batch_size

        print(f'[PIPPack] Batch {batch_idx}/{n_batches}: '
              f'frames {batch_start+1}-{batch_start+len(batch)}', flush=True)

        # Directorio temporal para este batch
        tmp_in  = os.path.join(output_dir, f'_tmp_in_{batch_start:04d}')
        tmp_out = os.path.join(output_dir, f'_tmp_out_{batch_start:04d}')
        os.makedirs(tmp_in,  exist_ok=True)
        os.makedirs(tmp_out, exist_ok=True)

        for pdb in batch:
            shutil.copy2(pdb, os.path.join(tmp_in, os.path.basename(pdb)))

        cmd = [
            sys.executable,
            inference_script,
            f'inference.weights_path={weights_path}',
            f'inference.pdb_path={tmp_in}',
            f'inference.output_dir={tmp_out}',
            f'inference.model_names=[{model_names_str}]',
            f'+inference.num_ensembles={args.num_ensembles}',
            f'inference.force_cpu=True',
        ]

        result = subprocess.run(cmd, text=True, env=env, cwd=pippack_dir)

        if result.returncode != 0:
            print(f'[PIPPack] ERROR en batch {batch_idx}, '
                  f'code {result.returncode}', flush=True)
            sys.exit(1)

        # Mover outputs al directorio final
        for out_f in os.listdir(tmp_out):
            if out_f.endswith('.pdb'):
                shutil.move(
                    os.path.join(tmp_out, out_f),
                    os.path.join(output_dir, out_f)
                )

        shutil.rmtree(tmp_in,  ignore_errors=True)
        shutil.rmtree(tmp_out, ignore_errors=True)

    out_pdbs = sorted(f for f in os.listdir(output_dir) if f.endswith('.pdb'))
    print(f'\n[PIPPack] Done. {len(out_pdbs)} all-atom PDB files written to: {output_dir}',
          flush=True)

    if len(out_pdbs) != len(pdb_files):
        print(f'WARNING: Expected {len(pdb_files)} output PDBs, '
              f'got {len(out_pdbs)}.', flush=True)


if __name__ == '__main__':
    main()