"""Build pinned upstream kernels for the requested GPU architecture only."""

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

SOURCES = (
    ('causal-conv1d', 'https://github.com/Dao-AILab/causal-conv1d.git',
     '4f6ae4e26ae5fe8af9372f8d312ab25cc4595223'),
    ('mamba-ssm', 'https://github.com/state-spaces/mamba.git',
     'e9594ce1c732d97440f0332fdc43170a2294dbfa'),
)


def main():
    from torch.utils.cpp_extension import _get_cuda_arch_flags

    flags = _get_cuda_arch_flags()
    if not flags:
        raise RuntimeError('Set TORCH_CUDA_ARCH_LIST to the target GPU capability')
    prefix = Path(sys.prefix)
    provenance = {'torch_cuda_arch_list': os.environ['TORCH_CUDA_ARCH_LIST'],
                  'nvcc_arch_flags': flags, 'sources': []}
    # NVIDIA's Conda toolkit keeps headers/libraries under targets/<triplet>,
    # while CUDA devel images expose include/ and lib64/ directly.
    toolkit = Path(os.environ['CUDA_HOME'])
    target_headers = list(toolkit.glob('targets/*/include/cuda_runtime.h'))
    build_env = {**os.environ, 'MAMBA_FORCE_BUILD': 'TRUE',
                 'MAMBA_KEEP_CUDA_BUILD': 'TRUE', 'CAUSAL_CONV1D_FORCE_BUILD': 'TRUE'}
    if not (toolkit/'include/cuda_runtime.h').is_file():
        if len(target_headers) != 1:
            raise RuntimeError(f'Cannot identify CUDA headers under {toolkit}')
        include = target_headers[0].parent
        library = include.parent/'lib'
        build_env['CPATH'] = str(include) + (':'+build_env['CPATH'] if build_env.get('CPATH') else '')
        build_env['LIBRARY_PATH'] = str(library) + (':'+build_env['LIBRARY_PATH'] if build_env.get('LIBRARY_PATH') else '')
        provenance['conda_toolkit_target'] = str(include.parent)
    # Upstream hard-codes many GPU targets. Use PyTorch's documented target
    # selection instead; kernel sources and all other build flags are unchanged.
    with tempfile.TemporaryDirectory(prefix='clas12-kernel-build-') as scratch:
        for name, url, revision in SOURCES:
            source = Path(scratch)/name
            subprocess.run(['git', 'init', '-q', str(source)], check=True)
            subprocess.run(['git', '-C', str(source), 'fetch', '--depth=1', url, revision], check=True)
            subprocess.run(['git', '-C', str(source), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
            actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
            if actual != revision:
                raise RuntimeError(f'Unexpected upstream revision: {actual}')
            setup = source/'setup.py'
            text = setup.read_text()
            marker = '    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI'
            if text.count(marker) != 1:
                raise RuntimeError(f'Pinned {name} build layout changed')
            text = text.replace(marker, f'    cc_flag = {flags!r}\n\n'+marker, 1)
            ast.parse(text)
            setup.write_text(text)
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-build-isolation',
                            '--no-deps', '--no-cache-dir', str(source)], env=build_env, check=True)
            provenance['sources'].append({'name': name, 'url': url, 'commit': revision,
                                         'build_adjustment': 'Replace cc_flag with PyTorch target architecture flags; kernel sources unchanged'})
    (prefix/'clas12-kernel-build.json').write_text(json.dumps(provenance, indent=2)+'\n')


if __name__ == '__main__':
    main()
