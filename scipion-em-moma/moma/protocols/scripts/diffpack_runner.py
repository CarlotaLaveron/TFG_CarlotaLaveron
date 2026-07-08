"""
diffpack_runner.py
------------------
Runs DiffPack sidechain packing on a directory of backbone PDB files.
Processes one PDB at a time to avoid OOM on CPU.
"""

import argparse
import copy
import os
import subprocess
import sys
import tempfile

import yaml


def main():
    parser = argparse.ArgumentParser(
        description='Run DiffPack sidechain packing on a directory of backbone PDBs.'
    )
    parser.add_argument('--input_dir', required=True,
                        help='Directory containing backbone PDB files')
    parser.add_argument('--output_dir', required=True,
                        help='Directory where DiffPack will write all-atom PDBs')
    parser.add_argument('--diffpack_dir', required=True,
                        help='Path to the cloned DiffPack repository')
    parser.add_argument('--config', required=True,
                        help='Path to DiffPack inference YAML config')
    parser.add_argument('--num_samples', type=int, default=4,
                        help='Number of diffusion samples per structure')
    parser.add_argument('--seed', type=int, default=2023,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    pdb_files = sorted(
        os.path.abspath(os.path.join(args.input_dir, f))
        for f in os.listdir(args.input_dir)
        if f.endswith('.pdb')
    )

    if not pdb_files:
        print(f'[DiffPack] ERROR: No PDB files found in {args.input_dir}', flush=True)
        sys.exit(1)

    #print(f'[DiffPack] Found {len(pdb_files)} backbone PDB files.', flush=True)
    #print(f'[DiffPack] Config : {args.config}', flush=True)
    #print(f'[DiffPack] Samples : {args.num_samples}', flush=True)
    #print(f'[DiffPack] Seed : {args.seed}', flush=True)

    inference_script = os.path.join(args.diffpack_dir, 'script', 'inference.py')
    if not os.path.isfile(inference_script):
        print(f'[DiffPack] ERROR: inference script not found at {inference_script}', flush=True)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)


    with open(args.config) as fh:
        cfg_base = yaml.safe_load(fh)


    checkpoint = cfg_base.get('model_checkpoint', '')
    if checkpoint:
        checkpoint = os.path.expanduser(checkpoint)
        if not os.path.isabs(checkpoint):
            checkpoint = os.path.join(args.diffpack_dir, checkpoint)
        checkpoint = os.path.abspath(checkpoint)
        if not os.path.isfile(checkpoint):
            fallback = os.path.join(args.diffpack_dir, 'model_weights',
                                    os.path.basename(checkpoint))
            if os.path.isfile(fallback):
                checkpoint = fallback
            else:
                print(f'[DiffPack] ERROR: checkpoint not found: {checkpoint}', flush=True)
                sys.exit(1)
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': ''}

    # Process one PDB at a time to avoid OOM on CPU
    for i, pdb in enumerate(pdb_files):
        #print(f'[DiffPack] Frame {i+1}/{len(pdb_files)}: {os.path.basename(pdb)}', flush=True)

        cfg_patched = copy.deepcopy(cfg_base)
        cfg_patched['engine']['gpus'] = None
        if cfg_patched['task'].get('class') == 'ConfidencePrediction':
            cfg_patched['task']['num_sample'] = args.num_samples

        cfg_patched['test_set']['sanitize'] = False
        cfg_patched['model_checkpoint'] = checkpoint

        tmp_cfg = tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False, dir=args.output_dir
        )
        yaml.dump(cfg_patched, tmp_cfg)
        tmp_cfg.close()

        cmd = [
            sys.executable,
            inference_script,
            '-c', tmp_cfg.name,
            '--seed', str(args.seed),
            '--output_dir', os.path.abspath(args.output_dir),
            '--pdb_files', pdb,
        ]

        result = subprocess.run(cmd, text=True, env=env, cwd=args.diffpack_dir)

        try:
            os.unlink(tmp_cfg.name)
        except OSError:
            pass

        if result.returncode != 0:
            print(f'[DiffPack] ERROR en frame {i+1}, code {result.returncode}', flush=True)
            sys.exit(1)

    out_pdbs = [f for f in os.listdir(args.output_dir) if f.endswith('.pdb')]
    #print(f'[DiffPack] Done. {len(out_pdbs)} all-atom PDB files in {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
