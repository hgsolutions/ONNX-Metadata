"""Application for comparing ONNX model compatibility."""

import argparse
import itertools
import json
import os
import sys

from typing import Type
from pathlib import Path
from collections import OrderedDict

import onnxruntime as ort


# Global variables
REPORT = OrderedDict()
EXIT_STATUS_SUCCESS = 0
EXIT_STATUS_ERROR = 1


def parse_layers(container: list, layers: list, group: str) -> None:
    """Collect information from parsed layers of an ONNX model.

    Args:
        container (list): list to stor parsed results from the layer.
        layers (list): list of layer data to be parsed.
        group (str): str group name to store layers in the container.
    """
    idx = 0
    for layer in layers:
        data = {
            'id': idx,
            'name': layer.name,
            'shape': layer.shape
        }
        container[group].append(data)
        idx += 1


def get_in_out_layers(model_file: str) -> dict:
    """Instance an ONNX model using ONNX Runtime and collect model
    inputs and outputs.

    Args:
        model_file (str): An ONNX model file path containing information needed.

    Returns:
        dict: Returns a dictionary of input and output data lists.
    """
    model = ort.InferenceSession(
        model_file,
        providers=['CPUExecutionProvider']
    )
    input_layers = model.get_inputs()
    output_layers = model.get_outputs()

    layer_data = {
        'inputs': [],
        'outputs': []
    }

    parse_layers(layer_data, input_layers, 'inputs')
    parse_layers(layer_data, output_layers, 'outputs')

    return layer_data


def diff_models(a: list, b: list, group: str) -> int:
    """Compare two ONNX models and report the differences.

    Args:
        a (list): Parsed data from an ONNX model layer.
        b (list): Parsed data from an ONNX model layer.
        group (str): The associated layer group "Inputs" or "Outputs"
    """
    a_minus_b = [item for item in a if item not in b]
    b_minus_a = [item for item in b if item not in a]
    sym_diff = list(itertools.chain(a_minus_b, b_minus_a))

    status = 0
    if a == b:
        REPORT['models']['compatability'][group] = True #f'Models {group} Match...'
    else:
        REPORT['models']['compatability'][group] = False #f'Model {group} do not match'

        REPORT[group] = {}
        REPORT[group]['a_vs_b'] = 'Model A contains one or more layers missing from Model B:'
        REPORT[group]['a_layers'] = []
        for d in a_minus_b:
            REPORT[group]['a_layers'].append(d)

        REPORT[group]['b_vs_a'] = 'Model B contains one or more layers missing from Model A:'
        REPORT[group]['b_layers'] = []
        for d in b_minus_a:
            REPORT[group]['b_layers'].append(d)

        REPORT[group]['symantic_difference'] = []
        for d in sym_diff:
            REPORT[group]['symantic_difference'].append(d)

        status = 1

    return status


def cli() -> Type[argparse.Namespace]:
    """Application CLI.

    Returns:
        Type[argparse.Namespace]: Argparse argument namespace.
    """
    # CLI parameter setup
    app_desc = \
        'This application compares the input and output layers of two ONNX models.\n' + \
        'With the purpose of identifying if the two models are compatible ' + \
        'for a specific ONNX application.\n e.g., can the models be hot swapped ' + \
        'without changing pre and post processing routines.'
    parser = argparse.ArgumentParser(
        prog='diff_onnx_models',
        description=app_desc,
        epilog='Application returns 0 if models are compatible, and 1 for incompatibility.'
    )
    parser.add_argument(
        'model_a',
        help=('Specify an ONNX model compared to parameter model_b.'),
        type=Path
    )
    parser.add_argument(
        'model_b',
        help=('Specify an ONNX model compared to parameter model_a.'),
        type=Path
    )
    parser.add_argument(
        '-l', '--layer_type',
        help=('Specify whether to compare only inputs, only outputs, or defaults to both.'),
        type=str,
        required=False,
        default='both',
        choices=['inputs', 'outputs', 'both']
    )
    parser.add_argument(
        '-i', '--indent',
        help=('Specify whether to display layers in verbose JSON format.'),
        type=int,
        required=False,
        default=0
    )
    parser.add_argument(
        '-o', '--output_uri',
        help=('Specify the output report file path. If not specified a report will be written.'),
        type=Path,
        required=False
    )
    args = parser.parse_args()

    return args


def main() -> None:
    """Application main function for execution.
    """
    args = cli()

    if args.model_a.is_file():
        model_a = str(args.model_a)

    if args.model_b.is_file():
        model_b = str(args.model_b)

    if args.output_uri is not None:
        if not str(args.output_uri).endswith('.json'):
            args.output_uri = f'{args.output_uri}.json'

    REPORT['models'] = {}
    REPORT['models']['model_a'] = os.path.abspath(model_a)
    REPORT['models']['model_b'] = os.path.abspath(model_b)
    REPORT['models']['compatability'] = {}

    layer_type = args.layer_type.upper()
    indent = args.indent

    a_layers = get_in_out_layers(model_a)
    b_layers = get_in_out_layers(model_b)

    if layer_type in ('BOTH', 'INPUTS'):
        input_status = diff_models(a_layers['inputs'], b_layers['inputs'], 'Inputs')

    if layer_type in ('BOTH', 'OUTPUTS'):
        output_status = diff_models(a_layers['outputs'], b_layers['outputs'], 'Outputs')

    all_status = input_status + output_status
    status = EXIT_STATUS_SUCCESS if all_status == EXIT_STATUS_SUCCESS else EXIT_STATUS_ERROR

    REPORT['exit_status'] = status

    # Write report
    output = json.dumps(REPORT, indent=indent)
    with open(args.output_uri, 'w', encoding='utf-8') as fid:
        fid.write(output)

    sys.exit(status)

if __name__=='__main__':
    main()
