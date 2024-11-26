from setuptools import setup, find_packages

setup(
    name='common_code',
    version='0.1',
    packages=find_packages(),
    py_modules=[
        'get_eval',
        'data_utils'
        'eval_utils_sample_ppl_instead_loss',
        'manage_json',
        'init_seed'
        ],
    install_requires=[
        # Add any dependencies your package needs here
    ],
    entry_points={
        'console_scripts': [
            # Add any command line scripts here
        ],
    },
)