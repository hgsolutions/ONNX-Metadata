"""Copyright (c) NV5 Geospatial Solutions, Inc.
    All rights reserved.
    Unauthorized reproduction prohibited.

Application for adding custom metadata to an ONNX model.
"""
import argparse
import json
import os
import sys

from pathlib import Path
from typing import Any, Dict, Type

try:
    import onnx
except ImportError:
    MSG = '\n\nYou must first install ONNX, try -\npip install onnx==1.16.0\n\n'
    sys.stderr.write(MSG)
    sys.exit(1)

# Global Variables
METADATA = {
    "model_type": 'String: Object Detection, Pixel Segmentation, etc.',
    "model_architecture": 'String: Detectron 2, SAM, Custom, etc.',
    "number_of_classes": 'Integer: Number of trained classes, e.g., 1, 2, 3, etc.',
    "number_of_bands": 'Integer: Number of bands the model trained with, e.g., 3',
    "number_of_epochs": 'Integer: How many epochs did the model train, e.g., 100',
    "class_names": 'List[str]: Class ID ordered class names, e.g., [person, car, ...]',
    "vendor_name": 'String: Company distributing the model, e.g. NV5',
    "model_author": 'String: Original author of the model architecture, e.g., clees',
    "model_license": 'String: License for the model, e.g., Apache 2.0',
    "model_version": 'Integer: Nth version of the model distributed to NV5, e.g., 1, 2, etc.',
    "model_date": 'String: Date for which the model is ready, e.g., 2025-01-01'
}

TYPES = {
    "model_type": str, "model_architecture": str, "number_of_classes": int,
    "number_of_bands": int, "number_of_epochs": int, "class_names": list,
    "vendor_name": str, "model_author": str, "model_license": str,
    "model_version": int, "model_date": str
}

# All licensing must be verified, this is a way to
# support vendors regarding unsupported licensing.
NON_PERMISIVE_LICENSES = [
    "GPL", "AGPL", "LGPL",
    "CC BY-NC", "CC BY-NC-SA",
    "CC BY-ND", "CC BY-NC-ND",
    "GNU", "APSL", "EPL", "MPL"
]

# ANSI escape sequences for colors
COLORS = {
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "reset": "\033[0m"
}

def _error(message: str) -> None:
    """Simple helper function to report console _error and exit.

    Args:
        message (str): _error reported due to failure.
    """
    err = f'{COLORS.get("red", COLORS["reset"])}Error: {message}{COLORS["reset"]}\n'
    sys.stderr.write(err)
    sys.exit(1)


def _to_color(message: str, color: str="white") -> str:
    """Help function for changing the color of text in a terminal.

    Args:
        message (str): text where color is applied.
        color (str): color to change text too, defaults to white.
    """
    return f'{COLORS.get(color, COLORS["reset"])}{message}{COLORS["reset"]}'


def _add_metadata(
        model: onnx.ModelProto,
        dict_data: Dict[str, str]
) -> None:
    """Function for adding metadata to an ONNX model Proto.
    
    Args:
        model (ModelProto): a loaded ONNX model.
        dict_data (dict): flat dictionary with str: str key values.
    """
    del model.metadata_props[:]
    for (key, value) in dict_data.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = json.dumps(value)


def _metadata_validator(metadata: Dict[str, Any]) -> None:
    """Minimum validation of values provided in the model metadata configuration.
    Validation includes expected keys, no empty values, and class names match
    specified number of classes.

    Args:
        metadata (dict): user provided JSON configuration of metadata.
    """
    # Validate expected metadata keys
    keys = list(metadata.keys())
    meta_keys = list(METADATA.keys())
    if not set(meta_keys).issubset(keys):
        diff = set(meta_keys) - set(keys)
        _error(f'Missing configuration key(s): {str(diff)}')

    # Validate no empty values, no template values, and expected types
    for (key, value) in metadata.items():
        if value is None or value == "":
            _error(f'Invalid empty value for key: {key}')

        if key in TYPES and not isinstance(metadata[key], TYPES[key]):
            _error(f'Metadata {key} should be {TYPES[key]}.')

        if isinstance(metadata[key], str):
            if METADATA[key] in metadata[key]:
                _error(f'Configuration key {key} value matches the template, please update.')

    # Validate number_of_classes match class_names count
    n_class_names = len(metadata['class_names'])
    number_of_classes = metadata['number_of_classes']
    if n_class_names != number_of_classes:
        msg = f'Number of class_names {n_class_names} ' + \
            f'does not match number_of_classes {number_of_classes}.'
        _error(msg)

    # Check licensing
    lic = metadata['model_license'].upper()
    for npl in NON_PERMISIVE_LICENSES:
        if lic.find(npl) > 0:
            _error(f"Non commercial license {lic} detected.")


def _write_model_metadata(
        config_uri: Path,
        output_uri: Path) -> None:
    """Write metadata to a model object and save as output_uri. This function
    performs minimal validation of the configured metadata specified.
    
    Args:
        model_uri (Path): valid ONNX model path.
        config_uri (Path): valid JSON config path.
        output_uri (Path): new output ONNX path or defaults to model_uri.
    """
    # Read JSON configuration to dictionary
    with open(config_uri, 'r', encoding='utf-8') as fid:
        data = json.load(fid)

    if 'model_uri' not in data:
        _error('Key "model_uri": "/path.onnx" is missing from the configuration.')

    model_uri = Path(data['model_uri'])

    if not model_uri.is_file():
        _error(f'Invalid model file: {model_uri}')

    if model_uri.suffix != '.onnx':
        _error('Invalid model extension, expecting .onnx')

    if output_uri is None:
        output_uri = model_uri

    if 'metadata' not in data:
        _error('Key "metadata": {...} is missing from the configuration.')

    metadata: Dict[str, Any]=data['metadata']
    if not isinstance(metadata, dict):
        _error('Metadata should be a flat dictionary of key value pairs.')

    # verify metadata
    _metadata_validator(metadata)

    # Open the model, write metadata, and save the model.
    model = onnx.load(model_uri)
    _add_metadata(model, metadata)

    if len(model.metadata_props) != 0:
        onnx.save(model, output_uri)
        success = _to_color(f'Successfully wrote metadata: {output_uri}\n', 'green')
        sys.stdout.write(success)
    else:
        _error(f'Failed to write metadata: {output_uri}\n')


def _write_config_template(template_uri: str) -> None:
    """Generate a template configuration file for this utility and exit.

    Args:
        template_uri (str): JSON configuration template output URI.
    """
    template = {
        "model_uri": "/path/to/model.onnx",
        "metadata": METADATA
    }

    if template_uri.suffix != '.json':
        template_uri = f'{template_uri}.json'

    with open(template_uri, 'w', encoding='utf-8') as fid:
        fid.write(json.dumps(template, sort_keys=True, indent=4))

    abspath = os.path.abspath(template_uri)
    msg = f'Generated configuration template:\n\t{abspath}\n\n'
    sys.stdout.write(_to_color(msg, 'cyan'))
    msg = 'Update the configuration and run the following:\n'
    msg += f'\tpython {sys.argv[0]} -c {abspath}\n'
    sys.stdout.write(_to_color(msg, 'yellow'))
    sys.exit(0)


def _cli() -> Type[argparse.Namespace]:
    """Application CLI.

    Returns:
        Type[argparse.Namespace]: Argparse argument namespace.
    """
    # CLI parameter setup
    app_desc = \
        'Application writes JSON key value pairs as metadata in an ONNX model.'

    parser = argparse.ArgumentParser(
        prog='set_onnx_metadata.py',
        description=app_desc,
        epilog='Application returns 0 on success, and 1 for failure.' + \
            '\nNote: This application clears all metadata before writing new data.'
    )
    parser.add_argument(
        '-c', '--config_uri',
        help=('Specify a JSON file providing model URI and associated metadata.'),
        type=Path
    )
    parser.add_argument(
        '-o', '--output_uri',
        help=('Specify an optional output file.onnx. Defaults to config::model_uri.'),
        type=Path
    )
    parser.add_argument(
        '-m', '--make_config',
        help=('Specify a JSON file path to generate a configuration template.'),
        type=Path,
    )
    args = parser.parse_args()

    return args, parser


def main() -> None:
    """Application entry point.
    """
    # Read command line arguments
    args, parser = _cli()
    config_uri = args.config_uri
    output_uri = args.output_uri
    make_config = args.make_config

    if make_config is not None:
        _write_config_template(make_config)

    if config_uri is None:
        parser.print_help()
        sys.exit(1)

    if not config_uri.is_file():
        _error(f'Invalid configuration file: {config_uri}')

    if not config_uri.suffix == '.json':
        _error('Invalid configuration extension, expecting .json')

    _write_model_metadata(config_uri, output_uri)

    sys.exit(0)

if __name__=='__main__':
    main()
